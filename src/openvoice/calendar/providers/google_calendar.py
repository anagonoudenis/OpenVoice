"""Google Calendar provider.

`google-api-python-client`/`google-auth` are optional, heavier
dependencies — the `calendar` extra (`uv sync --extra calendar`). Like
the faster-whisper/Piper providers, this module only imports them inside
`GoogleCalendarProvider.from_settings`, so importing this module never
requires the extra to be installed. Tests inject a fake `service` via the
plain constructor.

The Google API client is a dynamically-typed resource built at runtime
via HTTP discovery — there's no concrete Python class to structurally
type it against, so `service` is typed `Any` here; that's an honest
reflection of the library's own design, not a shortcut. Every call is
synchronous/blocking and always runs via `asyncio.to_thread` so it never
blocks the event loop.
"""

import asyncio
from datetime import datetime
from typing import Any

import structlog

from openvoice.calendar.base import BaseCalendarProvider, CalendarError, CalendarEvent, TimeSlot
from openvoice.config import Settings

logger = structlog.get_logger(__name__)


class GoogleCalendarProvider(BaseCalendarProvider):
    """Calendar provider backed by the Google Calendar API."""

    def __init__(self, *, service: Any, calendar_id: str) -> None:
        self._service = service
        self._calendar_id = calendar_id

    @classmethod
    def from_settings(cls, settings: Settings) -> "GoogleCalendarProvider":
        """Build a real Google Calendar client per `settings`. Requires the `calendar` extra."""
        if settings.google_service_account_json_path is None:
            raise CalendarError(
                "GOOGLE_SERVICE_ACCOUNT_JSON_PATH is required to build the Google Calendar provider"
            )
        try:
            from google.oauth2 import service_account
            from googleapiclient.discovery import build
        except ImportError as exc:
            raise CalendarError(
                "google-api-python-client is not installed; run `uv sync --extra calendar`"
            ) from exc

        credentials = service_account.Credentials.from_service_account_file(
            settings.google_service_account_json_path,
            scopes=["https://www.googleapis.com/auth/calendar"],
        )
        service = build("calendar", "v3", credentials=credentials, cache_discovery=False)
        return cls(service=service, calendar_id=settings.google_calendar_id)

    async def list_busy_slots(self, *, start: datetime, end: datetime) -> list[TimeSlot]:
        try:
            response = await asyncio.to_thread(
                self._service.freebusy()
                .query(
                    body={
                        "timeMin": start.isoformat(),
                        "timeMax": end.isoformat(),
                        "items": [{"id": self._calendar_id}],
                    }
                )
                .execute
            )
        except Exception as exc:  # any Google API client failure becomes CalendarError
            logger.error("google_calendar_freebusy_failed", error=str(exc))
            raise CalendarError(f"Google Calendar freebusy query failed: {exc}") from exc

        busy = response["calendars"][self._calendar_id]["busy"]
        return [
            TimeSlot(
                start=datetime.fromisoformat(slot["start"]),
                end=datetime.fromisoformat(slot["end"]),
            )
            for slot in busy
        ]

    async def create_event(
        self, *, start: datetime, end: datetime, summary: str, description: str | None = None
    ) -> CalendarEvent:
        body = {
            "summary": summary,
            "description": description,
            "start": {"dateTime": start.isoformat()},
            "end": {"dateTime": end.isoformat()},
        }
        try:
            event = await asyncio.to_thread(
                self._service.events().insert(calendarId=self._calendar_id, body=body).execute
            )
        except Exception as exc:
            logger.error("google_calendar_create_event_failed", error=str(exc))
            raise CalendarError(f"Google Calendar event creation failed: {exc}") from exc

        return CalendarEvent(event_id=event["id"], start=start, end=end, summary=summary)

    async def cancel_event(self, *, event_id: str) -> None:
        try:
            await asyncio.to_thread(
                self._service.events()
                .delete(calendarId=self._calendar_id, eventId=event_id)
                .execute
            )
        except Exception as exc:
            logger.error("google_calendar_cancel_event_failed", error=str(exc), event_id=event_id)
            raise CalendarError(f"Google Calendar event cancellation failed: {exc}") from exc

    async def reschedule_event(
        self, *, event_id: str, start: datetime, end: datetime
    ) -> CalendarEvent:
        body = {"start": {"dateTime": start.isoformat()}, "end": {"dateTime": end.isoformat()}}
        try:
            event = await asyncio.to_thread(
                self._service.events()
                .patch(calendarId=self._calendar_id, eventId=event_id, body=body)
                .execute
            )
        except Exception as exc:
            logger.error(
                "google_calendar_reschedule_event_failed", error=str(exc), event_id=event_id
            )
            raise CalendarError(f"Google Calendar event reschedule failed: {exc}") from exc

        return CalendarEvent(
            event_id=event["id"], start=start, end=end, summary=event.get("summary", "")
        )
