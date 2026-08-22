"""Integration tests for ORM models against a real PostgreSQL database."""

from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from openvoice.db.models import (
    Appointment,
    AppointmentStatus,
    Call,
    CallDirection,
    CallStatus,
    CallTranscript,
    Client,
    SpeakerRole,
)

pytestmark = pytest.mark.integration


async def test_create_client(db_session: AsyncSession) -> None:
    client = Client(phone_number="+15551230001", full_name="Ada Lovelace")
    db_session.add(client)
    await db_session.commit()
    await db_session.refresh(client)

    assert client.id is not None
    assert client.created_at is not None


async def test_client_phone_number_must_be_unique(db_session: AsyncSession) -> None:
    db_session.add(Client(phone_number="+15551230002"))
    await db_session.commit()

    db_session.add(Client(phone_number="+15551230002"))
    with pytest.raises(IntegrityError):
        await db_session.commit()


async def test_call_with_transcripts_and_client(db_session: AsyncSession) -> None:
    client = Client(phone_number="+15551230003")
    db_session.add(client)
    await db_session.flush()

    call = Call(
        client_id=client.id,
        livekit_room_name="room-transcripts",
        direction=CallDirection.INBOUND,
        status=CallStatus.IN_PROGRESS,
    )
    call.transcripts.append(CallTranscript(sequence=1, role=SpeakerRole.CALLER, text="Bonjour"))
    call.transcripts.append(
        CallTranscript(sequence=2, role=SpeakerRole.AGENT, text="Comment puis-je vous aider ?")
    )
    db_session.add(call)
    await db_session.commit()

    result = await db_session.execute(select(Call).where(Call.id == call.id))
    fetched = result.scalar_one()
    assert [t.sequence for t in fetched.transcripts] == [1, 2]


async def test_transcript_sequence_unique_per_call(db_session: AsyncSession) -> None:
    call = Call(
        livekit_room_name="room-dup-seq",
        direction=CallDirection.OUTBOUND,
        status=CallStatus.IN_PROGRESS,
    )
    db_session.add(call)
    await db_session.flush()

    db_session.add(CallTranscript(call_id=call.id, sequence=1, role=SpeakerRole.CALLER, text="a"))
    await db_session.commit()

    db_session.add(CallTranscript(call_id=call.id, sequence=1, role=SpeakerRole.AGENT, text="b"))
    with pytest.raises(IntegrityError):
        await db_session.commit()


async def test_appointment_requires_ends_after_starts(db_session: AsyncSession) -> None:
    client = Client(phone_number="+15551230004")
    db_session.add(client)
    await db_session.flush()

    now = datetime.now(UTC)
    db_session.add(
        Appointment(
            client_id=client.id,
            starts_at=now + timedelta(hours=1),
            ends_at=now,
            status=AppointmentStatus.SCHEDULED,
        )
    )
    with pytest.raises(IntegrityError):
        await db_session.commit()


async def test_deleting_client_cascades_to_appointments(db_session: AsyncSession) -> None:
    now = datetime.now(UTC)
    client = Client(phone_number="+15551230005")
    client.appointments.append(
        Appointment(
            starts_at=now,
            ends_at=now + timedelta(minutes=30),
            status=AppointmentStatus.SCHEDULED,
        )
    )
    db_session.add(client)
    await db_session.commit()
    appointment_id = client.appointments[0].id

    await db_session.delete(client)
    await db_session.commit()

    result = await db_session.execute(select(Appointment).where(Appointment.id == appointment_id))
    assert result.scalar_one_or_none() is None
