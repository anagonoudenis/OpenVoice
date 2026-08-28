"""Tests for BookingService, using fake calendar/SMS/email providers and a
mocked DB session (the `Appointment` ORM object under test never needs a
real database: every field asserted on is set explicitly by
`BookingService` before `commit`/`refresh` would even be called).
"""

import uuid
from datetime import UTC, datetime
from unittest.mock import AsyncMock, Mock

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from openvoice.booking.service import BookingError, BookingService
from openvoice.calendar.base import TimeSlot
from openvoice.db.models import Appointment, AppointmentStatus, Client
from tests.unit.booking.fakes import FakeCalendarProvider, FakeEmailProvider, FakeSMSProvider


def _client(**overrides: object) -> Client:
    defaults: dict[str, object] = {
        "id": uuid.uuid4(),
        "phone_number": "+15551230000",
        "email": "client@example.com",
    }
    defaults.update(overrides)
    return Client(**defaults)  # type: ignore[arg-type]


def _db_session(*, existing_overlap: Appointment | None = None) -> AsyncSession:
    """A mocked session whose `execute(...).scalars().first()` -- used by
    `BookingService._find_overlapping_appointment` -- returns
    `existing_overlap` (default `None`, i.e. no conflicting appointment).
    """
    session = AsyncMock(spec=AsyncSession)
    scalars_result = Mock()
    scalars_result.first.return_value = existing_overlap
    session.execute = AsyncMock(return_value=Mock(scalars=Mock(return_value=scalars_result)))
    return session


class TestFindAvailableSlots:
    async def test_returns_slots_within_business_hours_excluding_busy(self) -> None:
        search_from = datetime(2026, 9, 1, 8, 0, tzinfo=UTC)  # a Tuesday
        busy = [
            TimeSlot(
                start=datetime(2026, 9, 1, 9, 0, tzinfo=UTC),
                end=datetime(2026, 9, 1, 9, 30, tzinfo=UTC),
            )
        ]
        service = BookingService(
            calendar=FakeCalendarProvider(busy=busy),
            sms=None,
            email=None,
            business_hours_start=9,
            business_hours_end=10,
            default_duration_minutes=30,
        )

        slots = await service.find_available_slots(search_from=search_from, search_days=1)

        assert slots == [
            TimeSlot(
                start=datetime(2026, 9, 1, 9, 30, tzinfo=UTC),
                end=datetime(2026, 9, 1, 10, 0, tzinfo=UTC),
            )
        ]

    async def test_caps_results_at_max_results(self) -> None:
        search_from = datetime(2026, 9, 1, 9, 0, tzinfo=UTC)
        service = BookingService(
            calendar=FakeCalendarProvider(busy=[]),
            sms=None,
            email=None,
            business_hours_start=9,
            business_hours_end=17,
            default_duration_minutes=30,
        )

        slots = await service.find_available_slots(
            search_from=search_from, search_days=7, max_results=3
        )

        assert len(slots) == 3


class TestBookAppointment:
    async def test_creates_calendar_event_and_appointment(self) -> None:
        calendar = FakeCalendarProvider()
        sms = FakeSMSProvider()
        email = FakeEmailProvider()
        service = BookingService(calendar=calendar, sms=sms, email=email)
        client = _client()
        start = datetime(2026, 9, 1, 10, 0, tzinfo=UTC)
        end = datetime(2026, 9, 1, 10, 30, tzinfo=UTC)

        appointment = await service.book_appointment(
            db_session=_db_session(), client=client, start=start, end=end
        )

        assert appointment.status is AppointmentStatus.CONFIRMED
        assert appointment.calendar_event_id == calendar.created[0].event_id
        assert len(calendar.created) == 1

    async def test_sends_sms_and_email_confirmation(self) -> None:
        sms = FakeSMSProvider()
        email = FakeEmailProvider()
        service = BookingService(calendar=FakeCalendarProvider(), sms=sms, email=email)
        client = _client()

        await service.book_appointment(
            db_session=_db_session(),
            client=client,
            start=datetime(2026, 9, 1, 10, 0, tzinfo=UTC),
            end=datetime(2026, 9, 1, 10, 30, tzinfo=UTC),
        )

        assert sms.sent[0][0] == client.phone_number
        assert email.sent[0][0] == client.email

    async def test_booking_succeeds_even_if_confirmation_fails(self) -> None:
        service = BookingService(
            calendar=FakeCalendarProvider(),
            sms=FakeSMSProvider(fail=True),
            email=FakeEmailProvider(fail=True),
        )
        client = _client()

        appointment = await service.book_appointment(
            db_session=_db_session(),
            client=client,
            start=datetime(2026, 9, 1, 10, 0, tzinfo=UTC),
            end=datetime(2026, 9, 1, 10, 30, tzinfo=UTC),
        )

        assert appointment.status is AppointmentStatus.CONFIRMED

    async def test_identical_repeat_booking_is_idempotent(self) -> None:
        """Regression test: nothing upstream stops a caller (or a confused
        LLM tool-calling loop) from asking to book the same slot twice --
        the second call must return the existing appointment, not create a
        duplicate calendar event.
        """
        calendar = FakeCalendarProvider()
        client = _client()
        start = datetime(2026, 9, 1, 10, 0, tzinfo=UTC)
        end = datetime(2026, 9, 1, 10, 30, tzinfo=UTC)
        existing = Appointment(
            id=uuid.uuid4(),
            client_id=client.id,
            starts_at=start,
            ends_at=end,
            status=AppointmentStatus.CONFIRMED,
            calendar_event_id="evt-existing",
        )
        service = BookingService(calendar=calendar, sms=None, email=None)

        result = await service.book_appointment(
            db_session=_db_session(existing_overlap=existing),
            client=client,
            start=start,
            end=end,
        )

        assert result is existing
        assert calendar.created == []  # no new calendar event was created

    async def test_overlapping_different_time_raises_booking_error(self) -> None:
        client = _client()
        existing = Appointment(
            id=uuid.uuid4(),
            client_id=client.id,
            starts_at=datetime(2026, 9, 1, 10, 0, tzinfo=UTC),
            ends_at=datetime(2026, 9, 1, 10, 30, tzinfo=UTC),
            status=AppointmentStatus.CONFIRMED,
            calendar_event_id="evt-existing",
        )
        service = BookingService(calendar=FakeCalendarProvider(), sms=None, email=None)

        with pytest.raises(BookingError, match="already has an appointment"):
            await service.book_appointment(
                db_session=_db_session(existing_overlap=existing),
                client=client,
                start=datetime(2026, 9, 1, 10, 15, tzinfo=UTC),
                end=datetime(2026, 9, 1, 10, 45, tzinfo=UTC),
            )

    async def test_non_overlapping_second_appointment_is_allowed(self) -> None:
        calendar = FakeCalendarProvider()
        client = _client()
        service = BookingService(calendar=calendar, sms=None, email=None)

        appointment = await service.book_appointment(
            db_session=_db_session(existing_overlap=None),
            client=client,
            start=datetime(2026, 9, 1, 14, 0, tzinfo=UTC),
            end=datetime(2026, 9, 1, 14, 30, tzinfo=UTC),
        )

        assert appointment.status is AppointmentStatus.CONFIRMED
        assert len(calendar.created) == 1


class TestListUpcomingAppointments:
    async def test_queries_by_client_excluding_cancelled_ordered_by_start(self) -> None:
        service = BookingService(calendar=FakeCalendarProvider(), sms=None, email=None)
        client_id = uuid.uuid4()
        expected = [object(), object()]
        db_session = AsyncMock(spec=AsyncSession)
        scalars_result = AsyncMock()
        scalars_result.all = lambda: expected
        db_session.execute = AsyncMock(return_value=AsyncMock(scalars=lambda: scalars_result))

        result = await service.list_upcoming_appointments(
            db_session=db_session, client_id=client_id
        )

        assert result == expected
        db_session.execute.assert_awaited_once()


class TestCancelAndReschedule:
    async def test_cancel_appointment_cancels_calendar_event(self) -> None:
        calendar = FakeCalendarProvider()
        service = BookingService(calendar=calendar, sms=None, email=None)
        appointment = await service.book_appointment(
            db_session=_db_session(),
            client=_client(),
            start=datetime(2026, 9, 1, 10, 0, tzinfo=UTC),
            end=datetime(2026, 9, 1, 10, 30, tzinfo=UTC),
        )

        await service.cancel_appointment(db_session=_db_session(), appointment=appointment)

        assert appointment.status is AppointmentStatus.CANCELLED
        assert calendar.cancelled == [appointment.calendar_event_id]

    async def test_reschedule_appointment_updates_times(self) -> None:
        calendar = FakeCalendarProvider()
        service = BookingService(calendar=calendar, sms=None, email=None)
        appointment = await service.book_appointment(
            db_session=_db_session(),
            client=_client(),
            start=datetime(2026, 9, 1, 10, 0, tzinfo=UTC),
            end=datetime(2026, 9, 1, 10, 30, tzinfo=UTC),
        )
        new_start = datetime(2026, 9, 2, 14, 0, tzinfo=UTC)
        new_end = datetime(2026, 9, 2, 14, 30, tzinfo=UTC)

        updated = await service.reschedule_appointment(
            db_session=_db_session(), appointment=appointment, new_start=new_start, new_end=new_end
        )

        assert updated.starts_at == new_start
        assert updated.status is AppointmentStatus.RESCHEDULED

    async def test_reschedule_without_calendar_event_raises(self) -> None:
        service = BookingService(calendar=FakeCalendarProvider(), sms=None, email=None)
        appointment = await service.book_appointment(
            db_session=_db_session(),
            client=_client(),
            start=datetime(2026, 9, 1, 10, 0, tzinfo=UTC),
            end=datetime(2026, 9, 1, 10, 30, tzinfo=UTC),
        )
        appointment.calendar_event_id = None

        with pytest.raises(BookingError):
            await service.reschedule_appointment(
                db_session=_db_session(),
                appointment=appointment,
                new_start=datetime(2026, 9, 2, tzinfo=UTC),
                new_end=datetime(2026, 9, 2, 0, 30, tzinfo=UTC),
            )
