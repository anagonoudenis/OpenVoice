"""Tests for BookingService, using fake calendar/SMS/email providers and a
mocked DB session (the `Appointment` ORM object under test never needs a
real database: every field asserted on is set explicitly by
`BookingService` before `commit`/`refresh` would even be called).
"""

import uuid
from datetime import UTC, datetime
from unittest.mock import AsyncMock

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from openvoice.booking.service import BookingError, BookingService
from openvoice.calendar.base import TimeSlot
from openvoice.db.models import AppointmentStatus, Client
from tests.unit.booking.fakes import FakeCalendarProvider, FakeEmailProvider, FakeSMSProvider


def _client(**overrides: object) -> Client:
    defaults: dict[str, object] = {
        "id": uuid.uuid4(),
        "phone_number": "+15551230000",
        "email": "client@example.com",
    }
    defaults.update(overrides)
    return Client(**defaults)  # type: ignore[arg-type]


def _db_session() -> AsyncSession:
    return AsyncMock(spec=AsyncSession)


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
