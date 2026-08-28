"""LiveKit Agents worker entrypoint: `uv run python -m openvoice.telephony.worker`.

Requires the `voice` extra (`uv sync --extra voice`) and a running LiveKit
server (self-hosted or LiveKit Cloud) reachable via `LIVEKIT_URL`,
`LIVEKIT_API_KEY`, `LIVEKIT_API_SECRET` (LiveKit's own standard env vars,
read by the SDK itself — not part of `openvoice.config.Settings`).
Inbound/outbound telephony additionally requires a SIP trunk + dispatch
rule configured on the LiveKit server (see LiveKit's SIP docs); that is
server-side configuration, not code here.

Not exercised end-to-end in this environment — no LiveKit deployment was
available to test against. See `openvoice.telephony.livekit_agent` for
what was and wasn't verified against the installed SDK.
"""

import time
import uuid
from datetime import UTC, datetime
from typing import Any

import structlog
from livekit import agents, api
from livekit.agents import AgentServer, AgentSession, JobContext, TurnHandlingOptions
from livekit.plugins import silero
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from openvoice import metrics
from openvoice.agent.conversation import ConversationManager
from openvoice.agent.prompts import build_system_prompt, build_temporal_context
from openvoice.agent.tools.base import ToolExecutor
from openvoice.agent.tools.booking import (
    BOOKING_SYSTEM_PROMPT_ADDENDUM,
    booking_tool_definitions,
    make_booking_tool_executor,
)
from openvoice.booking.service import BookingService
from openvoice.calendar.base import CalendarError
from openvoice.calendar.factory import get_calendar_provider
from openvoice.config import Settings, get_settings
from openvoice.crm.service import CRMService
from openvoice.db.models import Call, CallDirection, CallStatus, CallTranscript, SpeakerRole
from openvoice.db.session import get_sessionmaker
from openvoice.llm.base import LLMMessageRole, ToolDefinition
from openvoice.llm.factory import get_llm_provider
from openvoice.logging import configure_logging
from openvoice.notifications.factory import get_email_provider, get_sms_provider
from openvoice.observability import configure_sentry
from openvoice.stt.factory import get_stt_provider
from openvoice.tasks.summarize_call import summarize_call_task
from openvoice.telephony.livekit_agent import OpenVoiceAgent
from openvoice.tts.factory import get_tts_provider

logger = structlog.get_logger(__name__)

server = AgentServer()


async def _transfer_to_human(
    *, room_name: str, transfer_number: str | None, log: structlog.stdlib.BoundLogger
) -> None:
    """Best-effort SIP transfer to a human agent. Never raises: a failed
    transfer must not crash the call — it just stays with the bot.
    """
    if transfer_number is None:
        log.warning("transfer_to_human_requested_but_not_configured")
        return
    try:
        lk_api = api.LiveKitAPI()
        try:
            rooms = await lk_api.room.list_rooms(api.ListRoomsRequest(names=[room_name]))
            if not rooms.rooms:
                log.warning("transfer_to_human_room_not_found")
                return
            participants = await lk_api.room.list_participants(
                api.ListParticipantsRequest(room=room_name)
            )
            sip_participant = next(
                (p for p in participants.participants if p.kind == api.ParticipantInfo.Kind.SIP),
                None,
            )
            if sip_participant is None:
                log.warning("transfer_to_human_no_sip_participant")
                return
            await lk_api.sip.transfer_sip_participant(
                api.TransferSIPParticipantRequest(
                    room_name=room_name,
                    participant_identity=sip_participant.identity,
                    transfer_to=f"tel:{transfer_number}",
                )
            )
            log.info("transferred_to_human")
        finally:
            await lk_api.aclose()
    except Exception as exc:  # any transfer failure must not crash the call
        log.error("transfer_to_human_failed", error=str(exc))


def _build_booking_tools(
    *,
    settings: Settings,
    sessionmaker: async_sessionmaker[AsyncSession],
    client_id: uuid.UUID,
    call_id: uuid.UUID,
    log: structlog.stdlib.BoundLogger,
) -> tuple[list[ToolDefinition], ToolExecutor] | tuple[None, None]:
    """Build booking tools for this call, or `(None, None)` if calendar
    isn't configured. Booking is an optional sub-feature -- same as
    `openvoice.api.dependencies.get_booking_service` -- so a deployment
    without Google Calendar credentials still gets a working agent, just
    without booking actions; SMS/email confirmations are independently
    optional on top of that.
    """
    try:
        calendar = get_calendar_provider(settings)
    except CalendarError as exc:
        log.info("booking_tools_disabled_no_calendar_provider", error=str(exc))
        return None, None

    try:
        sms = get_sms_provider(settings)
    except RuntimeError:
        sms = None
    try:
        email = get_email_provider(settings)
    except RuntimeError:
        email = None

    booking_service = BookingService(
        calendar=calendar,
        sms=sms,
        email=email,
        business_hours_start=settings.booking_business_hours_start,
        business_hours_end=settings.booking_business_hours_end,
        default_duration_minutes=settings.booking_default_duration_minutes,
    )
    executor = make_booking_tool_executor(
        booking_service=booking_service,
        sessionmaker=sessionmaker,
        client_id=client_id,
        call_id=call_id,
    )
    return booking_tool_definitions(), executor


def _caller_phone_number(ctx: JobContext) -> str | None:
    """Read the caller's number off the inbound SIP participant, if any.

    `sip.phoneNumber` is the participant attribute LiveKit's SIP
    integration sets on inbound calls (see LiveKit's SIP participant
    attributes reference).

    The `isinstance` check matters in practice, not just in theory: in
    `console`/test-harness modes, LiveKit simulates a participant whose
    `attributes` isn't a real dict, so `.get(...)` returns a Mock object
    instead of `None` -- a plain truthiness check let that Mock through as
    if it were a real phone number, which then hit the database as a
    bogus query parameter (`UndefinedTableError`-adjacent crash, caught by
    running this in `console` mode for real).
    """
    for participant in ctx.room.remote_participants.values():
        phone_number = participant.attributes.get("sip.phoneNumber")
        if isinstance(phone_number, str) and phone_number:
            return phone_number
    return None


@server.rtc_session(agent_name="openvoice-agent")
async def entrypoint(ctx: JobContext) -> None:
    settings = get_settings()
    configure_logging(environment=settings.environment, log_level=settings.log_level)
    configure_sentry(settings)

    await ctx.connect()

    call_id = uuid.uuid4()
    call_start_time = time.monotonic()
    log = logger.bind(call_id=str(call_id), room=ctx.room.name)
    log.info("call_started")
    metrics.calls_started_total.inc()

    vad_provider = silero.VAD.load(min_silence_duration=settings.vad_min_silence_duration_seconds)

    crm = CRMService()
    phone_number = _caller_phone_number(ctx)

    sessionmaker = get_sessionmaker()
    async with sessionmaker() as db_session:
        client_id = None
        if phone_number is not None:
            client = await crm.get_or_create_client(
                db_session=db_session, phone_number=phone_number
            )
            client_id = client.id
            log = log.bind(client_id=str(client_id))

        call_row = Call(
            id=call_id,
            client_id=client_id,
            livekit_room_name=ctx.room.name,
            direction=CallDirection.INBOUND,
            status=CallStatus.IN_PROGRESS,
        )
        db_session.add(call_row)
        await db_session.commit()

    system_prompt = build_system_prompt(settings) + build_temporal_context(
        timezone=settings.booking_timezone
    )
    tools: list[ToolDefinition] | None = None
    tool_executor: ToolExecutor | None = None
    # Booking actions need a resolved caller identity (`client_id`) to book
    # on behalf of -- console/test-harness sessions with no SIP participant
    # have none, so they get a fully working agent minus booking, same as
    # any deployment without calendar credentials configured.
    if client_id is not None:
        tools, tool_executor = _build_booking_tools(
            settings=settings,
            sessionmaker=sessionmaker,
            client_id=client_id,
            call_id=call_id,
            log=log,
        )
        if tools is not None:
            system_prompt += BOOKING_SYSTEM_PROMPT_ADDENDUM

    conversation = ConversationManager(
        llm=get_llm_provider(settings),
        system_prompt=system_prompt,
        call_id=str(call_id),
        max_history_turns=settings.agent_max_history_turns,
        tools=tools,
        tool_executor=tool_executor,
        max_conversation_turns=settings.agent_max_conversation_turns,
        max_call_duration_seconds=settings.agent_max_call_duration_seconds,
    )

    async def on_transfer_to_human() -> None:
        await _transfer_to_human(
            room_name=ctx.room.name,
            transfer_number=settings.agent_human_transfer_number,
            log=log,
        )

    async def finalize_call(*_args: Any) -> None:
        async with sessionmaker() as db_session:
            db_call = await db_session.get(Call, call_id)
            if db_call is None:
                return
            db_call.status = CallStatus.COMPLETED
            db_call.ended_at = datetime.now(UTC)
            for i, message in enumerate(conversation.history):
                # Tool-call/tool-result messages aren't something either
                # party "said" -- skip them (and any tool-call-only
                # assistant turn with no accompanying text) so the
                # transcript stays human-readable and the post-call LLM
                # summary isn't fed raw JSON tool payloads.
                if message.role is LLMMessageRole.TOOL or not message.content:
                    continue
                is_caller = message.role is LLMMessageRole.USER
                db_session.add(
                    CallTranscript(
                        call_id=call_id,
                        sequence=i,
                        role=SpeakerRole.CALLER if is_caller else SpeakerRole.AGENT,
                        text=message.content,
                    )
                )
            await db_session.commit()
        log.info("call_finalized")
        metrics.calls_completed_total.labels(outcome="completed").inc()
        metrics.call_duration_seconds.observe(time.monotonic() - call_start_time)

        try:
            summarize_call_task.delay(str(call_id))
        except Exception as exc:  # broker unreachable etc. must not fail call teardown
            log.error("summarize_call_dispatch_failed", error=str(exc))

    ctx.add_shutdown_callback(finalize_call)

    agent = OpenVoiceAgent(
        conversation=conversation,
        stt_provider=get_stt_provider(settings),
        tts_provider=get_tts_provider(settings),
        vad_provider=vad_provider,
        system_prompt=build_system_prompt(settings),
        on_transfer_to_human=on_transfer_to_human,
    )
    session: AgentSession[None] = AgentSession(
        vad=vad_provider,
        turn_handling=TurnHandlingOptions(
            interruption={"enabled": True},
            # OpenVoiceAgent.llm_node is stateful: ConversationManager.handle_utterance
            # mutates conversation history as a side effect of being called, once
            # per confirmed turn. Preemptive generation (on by default) calls
            # llm_node speculatively, against a transcript that hasn't been
            # confirmed final yet and may still change -- for a stateful node
            # that means a second, corrective call after a changed transcript
            # would append a bogus exchange to history for words the caller
            # never actually finished saying. Disabled for correctness.
            preemptive_generation={"enabled": False},
        ),
    )

    await session.start(agent=agent, room=ctx.room)


def run() -> None:
    """Console-script entrypoint (also runnable as `python -m openvoice.telephony.worker`)."""
    agents.cli.run_app(server)


if __name__ == "__main__":
    run()
