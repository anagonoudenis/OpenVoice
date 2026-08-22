"""Integration tests for CRMService against a real PostgreSQL database."""

import uuid
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from openvoice.crm.service import CRMService
from openvoice.db.models import Call, CallDirection, CallStatus, CallTranscript, SpeakerRole

pytestmark = pytest.mark.integration


async def test_get_or_create_client_creates_new_client(db_session: AsyncSession) -> None:
    crm = CRMService()

    client = await crm.get_or_create_client(db_session=db_session, phone_number="+15559990001")

    assert client.id is not None
    assert client.phone_number == "+15559990001"


async def test_get_or_create_client_returns_existing_client(db_session: AsyncSession) -> None:
    crm = CRMService()
    first = await crm.get_or_create_client(db_session=db_session, phone_number="+15559990002")

    second = await crm.get_or_create_client(db_session=db_session, phone_number="+15559990002")

    assert second.id == first.id


async def test_get_call_history_orders_newest_first(db_session: AsyncSession) -> None:
    crm = CRMService()
    client = await crm.get_or_create_client(db_session=db_session, phone_number="+15559990003")
    now = datetime.now(UTC)
    older = Call(
        client_id=client.id,
        livekit_room_name="room-older",
        direction=CallDirection.INBOUND,
        status=CallStatus.COMPLETED,
        started_at=now - timedelta(days=1),
    )
    newer = Call(
        client_id=client.id,
        livekit_room_name="room-newer",
        direction=CallDirection.INBOUND,
        status=CallStatus.COMPLETED,
        started_at=now,
    )
    db_session.add_all([older, newer])
    await db_session.commit()

    history = await crm.get_call_history(db_session=db_session, client_id=client.id)

    assert [c.livekit_room_name for c in history] == ["room-newer", "room-older"]


async def test_get_call_with_transcript_loads_transcripts(db_session: AsyncSession) -> None:
    crm = CRMService()
    call = Call(
        livekit_room_name="room-transcript",
        direction=CallDirection.INBOUND,
        status=CallStatus.COMPLETED,
    )
    call.transcripts.append(CallTranscript(sequence=1, role=SpeakerRole.CALLER, text="Hi"))
    call.transcripts.append(CallTranscript(sequence=2, role=SpeakerRole.AGENT, text="Hello!"))
    db_session.add(call)
    await db_session.commit()

    fetched = await crm.get_call_with_transcript(db_session=db_session, call_id=call.id)

    assert fetched is not None
    assert [t.text for t in fetched.transcripts] == ["Hi", "Hello!"]


async def test_get_call_with_transcript_returns_none_when_missing(
    db_session: AsyncSession,
) -> None:
    crm = CRMService()
    result = await crm.get_call_with_transcript(db_session=db_session, call_id=uuid.uuid4())
    assert result is None
