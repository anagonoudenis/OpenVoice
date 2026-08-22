"""Lightweight unit tests for ORM model `__repr__`s (no database needed)."""

import uuid
from datetime import UTC, datetime, timedelta

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


def test_client_repr() -> None:
    client = Client(id=uuid.uuid4(), phone_number="+15550000000")
    assert "Client(" in repr(client)


def test_call_repr() -> None:
    call = Call(
        id=uuid.uuid4(),
        livekit_room_name="room-x",
        direction=CallDirection.INBOUND,
        status=CallStatus.IN_PROGRESS,
    )
    assert "Call(" in repr(call)


def test_appointment_repr() -> None:
    now = datetime.now(UTC)
    appointment = Appointment(
        id=uuid.uuid4(),
        starts_at=now,
        ends_at=now + timedelta(hours=1),
        status=AppointmentStatus.SCHEDULED,
    )
    assert "Appointment(" in repr(appointment)


def test_call_transcript_repr() -> None:
    transcript = CallTranscript(
        id=uuid.uuid4(), call_id=uuid.uuid4(), sequence=1, role=SpeakerRole.CALLER, text="hi"
    )
    assert "CallTranscript(" in repr(transcript)
