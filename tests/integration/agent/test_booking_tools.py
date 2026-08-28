"""Integration tests for the booking tool dispatcher against a real Postgres
database -- the ORM queries (client ownership checks, upcoming-appointment
filtering) are exactly what's under test here, so a mocked session would
defeat the point. Calendar/SMS/email stay fakes: no real Google Calendar
or Twilio credentials are available in this environment, and the
dispatcher's own logic doesn't depend on their internals.
"""

import uuid
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from openvoice.agent.tools.base import ToolExecutor
from openvoice.agent.tools.booking import make_booking_tool_executor
from openvoice.booking.service import BookingService
from openvoice.db.models import Appointment, AppointmentStatus, Client
from openvoice.llm.base import ToolCall
from tests.unit.booking.fakes import FakeCalendarProvider, FakeEmailProvider, FakeSMSProvider

pytestmark = pytest.mark.integration

_FUTURE_START = "2027-01-15T14:00:00+00:00"
_FUTURE_END = "2027-01-15T14:30:00+00:00"


async def _make_client(db_session: AsyncSession, *, phone_number: str) -> Client:
    client = Client(phone_number=phone_number)
    db_session.add(client)
    await db_session.commit()
    await db_session.refresh(client)
    return client


def _executor(
    db_engine: AsyncEngine,
    *,
    client_id: uuid.UUID,
    calendar: FakeCalendarProvider | None = None,
) -> ToolExecutor:
    booking_service = BookingService(
        calendar=calendar or FakeCalendarProvider(),
        sms=FakeSMSProvider(),
        email=FakeEmailProvider(),
    )
    sessionmaker = async_sessionmaker(db_engine, expire_on_commit=False)
    return make_booking_tool_executor(
        booking_service=booking_service,
        sessionmaker=sessionmaker,
        client_id=client_id,
        call_id=None,
    )


async def test_check_availability_returns_slots_as_json(
    db_session: AsyncSession, db_engine: AsyncEngine
) -> None:
    client = await _make_client(db_session, phone_number="+15550001")
    execute = _executor(db_engine, client_id=client.id)

    content, is_error = await execute(
        ToolCall(id="1", name="check_availability", arguments={"duration_minutes": 30})
    )

    assert is_error is False
    assert '"start"' in content


async def test_check_availability_rejects_invalid_arguments(
    db_session: AsyncSession, db_engine: AsyncEngine
) -> None:
    client = await _make_client(db_session, phone_number="+15550002")
    execute = _executor(db_engine, client_id=client.id)

    content, is_error = await execute(
        ToolCall(id="1", name="check_availability", arguments={"duration_minutes": "not-a-number"})
    )

    assert is_error is True
    assert "Invalid arguments" in content


async def test_book_appointment_requires_caller_confirmed(
    db_session: AsyncSession, db_engine: AsyncEngine
) -> None:
    client = await _make_client(db_session, phone_number="+15550003")
    execute = _executor(db_engine, client_id=client.id)

    content, is_error = await execute(
        ToolCall(
            id="1",
            name="book_appointment",
            arguments={"start": _FUTURE_START, "end": _FUTURE_END, "caller_confirmed": False},
        )
    )

    assert is_error is True
    assert "Not booked" in content


async def test_book_appointment_rejects_naive_datetime(
    db_session: AsyncSession, db_engine: AsyncEngine
) -> None:
    client = await _make_client(db_session, phone_number="+15550004")
    execute = _executor(db_engine, client_id=client.id)

    content, is_error = await execute(
        ToolCall(
            id="1",
            name="book_appointment",
            arguments={
                "start": "2027-01-15T14:00:00",  # no UTC offset
                "end": "2027-01-15T14:30:00",
                "caller_confirmed": True,
            },
        )
    )

    assert is_error is True
    assert "timezone-aware" in content


async def test_book_appointment_creates_real_appointment_row(
    db_session: AsyncSession, db_engine: AsyncEngine
) -> None:
    client = await _make_client(db_session, phone_number="+15550005")
    execute = _executor(db_engine, client_id=client.id)

    content, is_error = await execute(
        ToolCall(
            id="1",
            name="book_appointment",
            arguments={"start": _FUTURE_START, "end": _FUTURE_END, "caller_confirmed": True},
        )
    )

    assert is_error is False
    assert "Booked." in content

    appointment_id = uuid.UUID(content.split("appointment_id=")[1].split(",")[0])
    sessionmaker = async_sessionmaker(db_engine, expire_on_commit=False)
    async with sessionmaker() as verify_session:
        created = await verify_session.get(Appointment, appointment_id)
        assert created is not None
        assert created.client_id == client.id
        assert created.status is AppointmentStatus.CONFIRMED


async def test_repeating_the_same_book_appointment_call_does_not_duplicate(
    db_session: AsyncSession, db_engine: AsyncEngine
) -> None:
    """Regression test, exercised through the actual tool-dispatch path a
    live call uses: nothing stops an LLM tool-calling loop from issuing
    book_appointment twice for the same request (a retried turn, a
    confused model) -- the second call must not create a second
    appointment/calendar event.
    """
    client = await _make_client(db_session, phone_number="+15550013")
    calendar = FakeCalendarProvider()
    execute = _executor(db_engine, client_id=client.id, calendar=calendar)
    call = ToolCall(
        id="1",
        name="book_appointment",
        arguments={"start": _FUTURE_START, "end": _FUTURE_END, "caller_confirmed": True},
    )

    first_content, first_is_error = await execute(call)
    second_content, second_is_error = await execute(call)

    assert first_is_error is False
    assert second_is_error is False
    assert first_content == second_content  # same appointment_id both times
    assert len(calendar.created) == 1  # only one real calendar event was created

    sessionmaker = async_sessionmaker(db_engine, expire_on_commit=False)
    async with sessionmaker() as verify_session:
        result = await verify_session.execute(
            select(Appointment).where(Appointment.client_id == client.id)
        )
        assert len(result.scalars().all()) == 1


async def test_list_my_appointments_only_shows_this_clients_upcoming_ones(
    db_session: AsyncSession, db_engine: AsyncEngine
) -> None:
    mine = await _make_client(db_session, phone_number="+15550006")
    someone_else = await _make_client(db_session, phone_number="+15550007")

    future = datetime.now(UTC) + timedelta(days=5)
    db_session.add(
        Appointment(
            client_id=mine.id,
            starts_at=future,
            ends_at=future + timedelta(minutes=30),
            status=AppointmentStatus.CONFIRMED,
        )
    )
    db_session.add(
        Appointment(
            client_id=mine.id,
            starts_at=future,
            ends_at=future + timedelta(minutes=30),
            status=AppointmentStatus.CANCELLED,
        )
    )
    db_session.add(
        Appointment(
            client_id=someone_else.id,
            starts_at=future,
            ends_at=future + timedelta(minutes=30),
            status=AppointmentStatus.CONFIRMED,
        )
    )
    await db_session.commit()

    execute = _executor(db_engine, client_id=mine.id)
    content, is_error = await execute(ToolCall(id="1", name="list_my_appointments", arguments={}))

    assert is_error is False
    assert content.count('"appointment_id"') == 1


async def test_cancel_appointment_rejects_appointment_owned_by_another_client(
    db_session: AsyncSession, db_engine: AsyncEngine
) -> None:
    caller = await _make_client(db_session, phone_number="+15550008")
    someone_else = await _make_client(db_session, phone_number="+15550009")
    future = datetime.now(UTC) + timedelta(days=5)
    other_appointment = Appointment(
        client_id=someone_else.id,
        starts_at=future,
        ends_at=future + timedelta(minutes=30),
        status=AppointmentStatus.CONFIRMED,
    )
    db_session.add(other_appointment)
    await db_session.commit()
    await db_session.refresh(other_appointment)

    execute = _executor(db_engine, client_id=caller.id)
    content, is_error = await execute(
        ToolCall(
            id="1",
            name="cancel_appointment",
            arguments={
                "appointment_id": str(other_appointment.id),
                "caller_confirmed": True,
            },
        )
    )

    assert is_error is True
    assert "No such appointment" in content


async def test_cancel_appointment_cancels_when_owned_and_confirmed(
    db_session: AsyncSession, db_engine: AsyncEngine
) -> None:
    client = await _make_client(db_session, phone_number="+15550010")
    future = datetime.now(UTC) + timedelta(days=5)
    appointment = Appointment(
        client_id=client.id,
        starts_at=future,
        ends_at=future + timedelta(minutes=30),
        status=AppointmentStatus.CONFIRMED,
    )
    db_session.add(appointment)
    await db_session.commit()
    await db_session.refresh(appointment)

    execute = _executor(db_engine, client_id=client.id)
    _content, is_error = await execute(
        ToolCall(
            id="1",
            name="cancel_appointment",
            arguments={"appointment_id": str(appointment.id), "caller_confirmed": True},
        )
    )

    assert is_error is False

    sessionmaker = async_sessionmaker(db_engine, expire_on_commit=False)
    async with sessionmaker() as verify_session:
        refreshed = await verify_session.get(Appointment, appointment.id)
        assert refreshed is not None
        assert refreshed.status is AppointmentStatus.CANCELLED


async def test_reschedule_appointment_updates_when_owned_and_confirmed(
    db_session: AsyncSession, db_engine: AsyncEngine
) -> None:
    client = await _make_client(db_session, phone_number="+15550011")
    future = datetime.now(UTC) + timedelta(days=5)
    appointment = Appointment(
        client_id=client.id,
        starts_at=future,
        ends_at=future + timedelta(minutes=30),
        status=AppointmentStatus.CONFIRMED,
        calendar_event_id="evt-existing",
    )
    db_session.add(appointment)
    await db_session.commit()
    await db_session.refresh(appointment)

    new_start = "2027-02-01T09:00:00+00:00"
    new_end = "2027-02-01T09:30:00+00:00"
    execute = _executor(db_engine, client_id=client.id)
    _content, is_error = await execute(
        ToolCall(
            id="1",
            name="reschedule_appointment",
            arguments={
                "appointment_id": str(appointment.id),
                "new_start": new_start,
                "new_end": new_end,
                "caller_confirmed": True,
            },
        )
    )

    assert is_error is False

    sessionmaker = async_sessionmaker(db_engine, expire_on_commit=False)
    async with sessionmaker() as verify_session:
        refreshed = await verify_session.get(Appointment, appointment.id)
        assert refreshed is not None
        assert refreshed.status is AppointmentStatus.RESCHEDULED
        assert refreshed.starts_at == datetime.fromisoformat(new_start)


async def test_unknown_tool_name_returns_error(
    db_session: AsyncSession, db_engine: AsyncEngine
) -> None:
    client = await _make_client(db_session, phone_number="+15550012")
    execute = _executor(db_engine, client_id=client.id)

    content, is_error = await execute(ToolCall(id="1", name="delete_everything", arguments={}))

    assert is_error is True
    assert "Unknown tool" in content
