"""Abstract calendar provider interface."""

from abc import ABC, abstractmethod
from datetime import datetime

from pydantic import BaseModel


class TimeSlot(BaseModel):
    """A time window: a busy block, or a proposed/available slot."""

    start: datetime
    end: datetime


class CalendarEvent(BaseModel):
    """A booked calendar event."""

    event_id: str
    start: datetime
    end: datetime
    summary: str


class CalendarError(Exception):
    """Raised when a calendar provider fails after exhausting its retry budget."""


class BaseCalendarProvider(ABC):
    """Abstract calendar backend."""

    @abstractmethod
    async def list_busy_slots(self, *, start: datetime, end: datetime) -> list[TimeSlot]:
        """Return busy (already-booked) windows within `[start, end)`."""
        raise NotImplementedError

    @abstractmethod
    async def create_event(
        self, *, start: datetime, end: datetime, summary: str, description: str | None = None
    ) -> CalendarEvent:
        """Book a new event."""
        raise NotImplementedError

    @abstractmethod
    async def cancel_event(self, *, event_id: str) -> None:
        """Cancel an existing event."""
        raise NotImplementedError

    @abstractmethod
    async def reschedule_event(
        self, *, event_id: str, start: datetime, end: datetime
    ) -> CalendarEvent:
        """Move an existing event to a new time."""
        raise NotImplementedError
