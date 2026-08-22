"""Integration tests for `summarize_call` against a real PostgreSQL database.

Only the DB is real here; the LLM is faked (see `tests/unit/agent/fakes.py`)
-- summarization must never make a real LLM API call in tests.
"""

import uuid

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from openvoice.db.models import Call, CallDirection, CallStatus, CallTranscript, SpeakerRole
from openvoice.tasks.summarize_call import summarize_call
from tests.unit.agent.fakes import FakeLLMProvider

pytestmark = pytest.mark.integration


async def test_summarize_call_persists_llm_generated_summary(db_session: AsyncSession) -> None:
    call = Call(
        livekit_room_name="room-summary",
        direction=CallDirection.INBOUND,
        status=CallStatus.COMPLETED,
    )
    call.transcripts.append(CallTranscript(sequence=1, role=SpeakerRole.CALLER, text="Hi"))
    call.transcripts.append(
        CallTranscript(sequence=2, role=SpeakerRole.AGENT, text="How can I help?")
    )
    db_session.add(call)
    await db_session.commit()

    llm = FakeLLMProvider(responses=["Caller said hi; agent offered help."])
    await summarize_call(db_session=db_session, llm=llm, call_id=call.id)

    await db_session.refresh(call)
    assert call.summary == "Caller said hi; agent offered help."


async def test_summarize_call_is_noop_for_missing_call(db_session: AsyncSession) -> None:
    llm = FakeLLMProvider(responses=["should not be used"])

    await summarize_call(db_session=db_session, llm=llm, call_id=uuid.uuid4())

    assert llm.calls == []


async def test_summarize_call_is_noop_for_empty_transcript(db_session: AsyncSession) -> None:
    call = Call(
        livekit_room_name="room-empty",
        direction=CallDirection.INBOUND,
        status=CallStatus.COMPLETED,
    )
    db_session.add(call)
    await db_session.commit()

    llm = FakeLLMProvider(responses=["should not be used"])
    await summarize_call(db_session=db_session, llm=llm, call_id=call.id)

    assert llm.calls == []
    await db_session.refresh(call)
    assert call.summary is None
