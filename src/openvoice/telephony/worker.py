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

import uuid
from datetime import UTC, datetime
from typing import Any

import structlog
from livekit import agents, api
from livekit.agents import AgentServer, AgentSession, JobContext
from livekit.plugins import silero

from openvoice.agent.conversation import ConversationManager
from openvoice.agent.prompts import build_system_prompt
from openvoice.config import get_settings
from openvoice.crm.service import CRMService
from openvoice.db.models import Call, CallDirection, CallStatus, CallTranscript, SpeakerRole
from openvoice.db.session import get_sessionmaker
from openvoice.llm.factory import get_llm_provider
from openvoice.logging import configure_logging
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

    await ctx.connect()

    call_id = uuid.uuid4()
    log = logger.bind(call_id=str(call_id), room=ctx.room.name)
    log.info("call_started")

    conversation = ConversationManager(
        llm=get_llm_provider(settings),
        system_prompt=build_system_prompt(settings),
        call_id=str(call_id),
        max_history_turns=settings.agent_max_history_turns,
    )
    vad_provider = silero.VAD.load()

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
                is_caller = message.role.value == "user"
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
    session: AgentSession[None] = AgentSession(vad=vad_provider, allow_interruptions=True)

    await session.start(agent=agent, room=ctx.room)


def run() -> None:
    """Console-script entrypoint (also runnable as `python -m openvoice.telephony.worker`)."""
    agents.cli.run_app(server)


if __name__ == "__main__":
    run()
