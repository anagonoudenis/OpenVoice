"""Appointment booking orchestration: calendar + notifications + DB, tied together.

Covers the Phase 1 booking spec: find available slots (proposing
alternatives when the requested time is busy), book/cancel/reschedule an
appointment, and send SMS/email confirmations.
"""

import uuid
from datetime import datetime, timedelta

import structlog
from sqlalchemy.ext.asyncio import AsyncSession

from openvoice.calendar.base import BaseCalendarProvider, TimeSlot
from openvoice.db.models import Appointment, AppointmentStatus, Client
from openvoice.notifications.base import BaseEmailProvider, BaseSMSProvider, NotificationError

logger = structlog.get_logger(__name__)


class BookingError(Exception):
    """Raised for booking-logic failures (e.g. rescheduling an unbooked
    appointment) -- as opposed to `CalendarError`/`NotificationError`,
    which are lower-level provider failures.
    """


class BookingService:
    """Orchestrates appointment booking against one calendar."""

    def __init__(
        self,
        *,
        calendar: BaseCalendarProvider,
        sms: BaseSMSProvider | None,
        email: BaseEmailProvider | None,
        business_hours_start: int = 9,
        business_hours_end: int = 17,
        default_duration_minutes: int = 30,
    ) -> None:
        self._calendar = calendar
        self._sms = sms
        self._email = email
        self._business_hours_start = business_hours_start
        self._business_hours_end = business_hours_end
        self._default_duration_minutes = default_duration_minutes

    async def find_available_slots(
        self,
        *,
        search_from: datetime,
        search_days: int = 7,
        duration_minutes: int | None = None,
        max_results: int = 5,
    ) -> list[TimeSlot]:
        """Propose up to `max_results` available slots within business hours."""
        duration = timedelta(minutes=duration_minutes or self._default_duration_minutes)
        search_until = search_from + timedelta(days=search_days)
        busy = await self._calendar.list_busy_slots(start=search_from, end=search_until)

        candidates: list[TimeSlot] = []
        day = search_from.replace(
            hour=self._business_hours_start, minute=0, second=0, microsecond=0
        )
        if day < search_from:
            day += timedelta(days=1)

        while day < search_until and len(candidates) < max_results:
            day_end = day.replace(hour=self._business_hours_end, minute=0)
            slot_start = day
            while slot_start + duration <= day_end and len(candidates) < max_results:
                slot_end = slot_start + duration
                if not _overlaps_any(slot_start, slot_end, busy):
                    candidates.append(TimeSlot(start=slot_start, end=slot_end))
                slot_start += duration
            day = (day + timedelta(days=1)).replace(
                hour=self._business_hours_start, minute=0, second=0, microsecond=0
            )

        return candidates

    async def book_appointment(
        self,
        *,
        db_session: AsyncSession,
        client: Client,
        start: datetime,
        end: datetime,
        call_id: uuid.UUID | None = None,
        notes: str | None = None,
    ) -> Appointment:
        """Create the calendar event, persist the `Appointment`, and confirm by SMS/email.

        A failed confirmation message is logged, not raised: the calendar
        event and DB row already succeeded by that point, and a
        notification failure must never undo a real booking.
        """
        event = await self._calendar.create_event(
            start=start,
            end=end,
            summary=f"Appointment: {client.full_name or client.phone_number}",
        )

        appointment = Appointment(
            client_id=client.id,
            call_id=call_id,
            starts_at=start,
            ends_at=end,
            status=AppointmentStatus.CONFIRMED,
            calendar_event_id=event.event_id,
            notes=notes,
        )
        db_session.add(appointment)
        await db_session.commit()
        await db_session.refresh(appointment)

        await self._send_confirmation(client, appointment)
        return appointment

    async def cancel_appointment(
        self, *, db_session: AsyncSession, appointment: Appointment
    ) -> None:
        """Cancel the calendar event (if any) and mark the appointment cancelled."""
        if appointment.calendar_event_id is not None:
            await self._calendar.cancel_event(event_id=appointment.calendar_event_id)
        appointment.status = AppointmentStatus.CANCELLED
        await db_session.commit()

    async def reschedule_appointment(
        self,
        *,
        db_session: AsyncSession,
        appointment: Appointment,
        new_start: datetime,
        new_end: datetime,
    ) -> Appointment:
        """Move an existing appointment to a new time."""
        if appointment.calendar_event_id is None:
            raise BookingError("Cannot reschedule an appointment with no calendar event")

        await self._calendar.reschedule_event(
            event_id=appointment.calendar_event_id, start=new_start, end=new_end
        )
        appointment.starts_at = new_start
        appointment.ends_at = new_end
        appointment.status = AppointmentStatus.RESCHEDULED
        await db_session.commit()
        await db_session.refresh(appointment)
        return appointment

    async def _send_confirmation(self, client: Client, appointment: Appointment) -> None:
        when = appointment.starts_at.strftime("%A %B %d at %H:%M")
        message = f"Your appointment is confirmed for {when}."

        if self._sms is not None and client.phone_number:
            try:
                await self._sms.send_sms(to=client.phone_number, body=message)
            except NotificationError as exc:
                logger.error("appointment_sms_confirmation_failed", error=str(exc))

        if self._email is not None and client.email:
            try:
                await self._email.send_email(
                    to=client.email, subject="Appointment confirmed", body=message
                )
            except NotificationError as exc:
                logger.error("appointment_email_confirmation_failed", error=str(exc))


def _overlaps_any(start: datetime, end: datetime, busy: list[TimeSlot]) -> bool:
    return any(start < slot.end and end > slot.start for slot in busy)
