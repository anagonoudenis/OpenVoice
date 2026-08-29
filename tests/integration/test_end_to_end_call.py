"""End-to-end integration test: a full simulated call from intake through
post-call summary.

Drives the same building blocks `openvoice.telephony.worker` wires
together (CRM, ConversationManager, CallPipeline, BookingService,
summarize_call) against a real PostgreSQL database, with every external
service -- LLM, STT, TTS, calendar, SMS, email -- faked. No real LiveKit,
network, or model inference is involved.
"""

import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from openvoice.agent.conversation import ConversationManager
from openvoice.booking.service import BookingService
from openvoice.crm.service import CRMService
from openvoice.db.models import Call, CallDirection, CallStatus, CallTranscript, SpeakerRole
from openvoice.stt.base import TranscriptSegment
from openvoice.tasks.summarize_call import summarize_call
from openvoice.telephony.pipeline import CallPipeline
from tests.unit.agent.fakes import FakeLLMProvider
from tests.unit.booking.fakes import FakeCalendarProvider, FakeEmailProvider, FakeSMSProvider
from tests.unit.telephony.fakes import FakeSTTProvider, FakeTTSProvider

pytestmark = pytest.mark.integration


async def _frames(chunk: bytes = b"\x00\x01") -> AsyncIterator[bytes]:
    yield chunk


def _structured(intent: str, reply: str) -> str:
    return f"{reply}\n###INTENT: {intent}"


async def test_full_call_lifecycle_intake_to_summary(db_session: AsyncSession) -> None:
    # -- 1. Call intake: caller recognized/created by phone number ----------
    crm = CRMService()
    client = await crm.get_or_create_client(db_session=db_session, phone_number="+15559991234")

    call_id = uuid.uuid4()
    call = Call(
        id=call_id,
        client_id=client.id,
        livekit_room_name="room-e2e",
        direction=CallDirection.INBOUND,
        status=CallStatus.IN_PROGRESS,
    )
    db_session.add(call)
    await db_session.commit()

    # -- 2. First turn: STT -> agent (intent + LLM reply) -> TTS -----------
    llm = FakeLLMProvider(
        responses=[_structured("support_question", "We're open Monday to Friday, 9 to 5.")]
    )
    conversation = ConversationManager(
        llm=llm, system_prompt="You are a helpful agent.", call_id=str(call_id)
    )
    stt = FakeSTTProvider([TranscriptSegment(text="What are your hours?", is_final=True)])
    tts = FakeTTSProvider()
    pipeline = CallPipeline(stt=stt, tts=tts, conversation=conversation, call_id=str(call_id))

    result = await pipeline.handle_utterance_audio(_frames())
    assert result is not None
    reply, audio_stream = result
    assert reply.text == "We're open Monday to Friday, 9 to 5."
    assert [c async for c in audio_stream] == [b"audio:We're open Monday to Friday, 9 to 5."]

    # -- 3. Second turn: the caller asks to book an appointment --------------
    llm.responses.append(_structured("booking", "You're booked for tomorrow at 10am."))
    stt._segments = [TranscriptSegment(text="Can I book an appointment?", is_final=True)]

    result = await pipeline.handle_utterance_audio(_frames())
    assert result is not None
    booking_reply, _ = result
    assert booking_reply.text == "You're booked for tomorrow at 10am."

    # -- 4. The booking itself, via BookingService --------------------------
    calendar = FakeCalendarProvider()
    sms = FakeSMSProvider()
    email = FakeEmailProvider()
    booking_service = BookingService(calendar=calendar, sms=sms, email=email)
    start = (datetime.now(UTC) + timedelta(days=1)).replace(
        hour=10, minute=0, second=0, microsecond=0
    )
    appointment = await booking_service.book_appointment(
        db_session=db_session,
        client=client,
        start=start,
        end=start + timedelta(minutes=30),
        call_id=call_id,
    )
    assert appointment.status.value == "confirmed"
    assert sms.sent[0][0] == client.phone_number
    assert calendar.created[0].event_id == appointment.calendar_event_id

    # -- 5. Call ends: transcript persisted ---------------------------------
    call.status = CallStatus.COMPLETED
    call.ended_at = datetime.now(UTC)
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

    persisted_call = await crm.get_call_with_transcript(db_session=db_session, call_id=call_id)
    assert persisted_call is not None
    assert len(persisted_call.transcripts) == 4  # 2 caller turns + 2 agent replies

    # -- 6. Post-call summary generated asynchronously (Celery in production) -
    summary_llm = FakeLLMProvider(responses=["Caller asked about hours and booked an appointment."])
    await summarize_call(db_session=db_session, llm=summary_llm, call_id=call_id)

    await db_session.refresh(call)
    assert call.summary == "Caller asked about hours and booked an appointment."

    # -- 7. Call history now shows up for the client -------------------------
    history = await crm.get_call_history(db_session=db_session, client_id=client.id)
    assert [c.id for c in history] == [call_id]
