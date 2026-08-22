"""Test doubles for `BaseCalendarProvider`/`BaseSMSProvider`/`BaseEmailProvider`."""

from datetime import datetime

from openvoice.calendar.base import BaseCalendarProvider, CalendarError, CalendarEvent, TimeSlot
from openvoice.notifications.base import BaseEmailProvider, BaseSMSProvider, NotificationError


class FakeCalendarProvider(BaseCalendarProvider):
    def __init__(self, *, busy: list[TimeSlot] | None = None, fail_create: bool = False) -> None:
        self.busy = busy or []
        self.created: list[CalendarEvent] = []
        self.cancelled: list[str] = []
        self.rescheduled: list[CalendarEvent] = []
        self.fail_create = fail_create

    async def list_busy_slots(self, *, start: datetime, end: datetime) -> list[TimeSlot]:
        return self.busy

    async def create_event(
        self, *, start: datetime, end: datetime, summary: str, description: str | None = None
    ) -> CalendarEvent:
        if self.fail_create:
            raise CalendarError("fake calendar failure")
        event = CalendarEvent(
            event_id=f"evt-{len(self.created)}", start=start, end=end, summary=summary
        )
        self.created.append(event)
        return event

    async def cancel_event(self, *, event_id: str) -> None:
        self.cancelled.append(event_id)

    async def reschedule_event(
        self, *, event_id: str, start: datetime, end: datetime
    ) -> CalendarEvent:
        event = CalendarEvent(event_id=event_id, start=start, end=end, summary="rescheduled")
        self.rescheduled.append(event)
        return event


class FakeSMSProvider(BaseSMSProvider):
    def __init__(self, *, fail: bool = False) -> None:
        self.sent: list[tuple[str, str]] = []
        self.fail = fail

    async def send_sms(self, *, to: str, body: str) -> None:
        if self.fail:
            raise NotificationError("fake SMS failure")
        self.sent.append((to, body))


class FakeEmailProvider(BaseEmailProvider):
    def __init__(self, *, fail: bool = False) -> None:
        self.sent: list[tuple[str, str, str]] = []
        self.fail = fail

    async def send_email(self, *, to: str, subject: str, body: str) -> None:
        if self.fail:
            raise NotificationError("fake email failure")
        self.sent.append((to, subject, body))
