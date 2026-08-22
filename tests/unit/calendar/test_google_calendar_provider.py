"""Tests for GoogleCalendarProvider, using a fake injected `service` double
that mimics the Google API client's chained `resource().method(...).execute()`
shape. The real `google-api-python-client` package is an optional dependency
(the `calendar` extra) and is never required just to run this test.
"""

from datetime import UTC, datetime
from typing import Any

import pytest

from openvoice.calendar.base import CalendarError
from openvoice.calendar.providers.google_calendar import GoogleCalendarProvider


class _FakeExecutable:
    def __init__(self, result: Any, *, fail: bool = False) -> None:
        self._result = result
        self._fail = fail

    def execute(self) -> Any:
        if self._fail:
            raise RuntimeError("google api error")
        return self._result


class _FakeEventsResource:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.inserted: list[dict[str, Any]] = []
        self.deleted: list[tuple[str, str]] = []
        self.patched: list[dict[str, Any]] = []

    def insert(self, *, calendarId: str, body: dict[str, Any]) -> _FakeExecutable:
        self.inserted.append({"calendarId": calendarId, "body": body})
        return _FakeExecutable({"id": "evt-1", **body}, fail=self.fail)

    def delete(self, *, calendarId: str, eventId: str) -> _FakeExecutable:
        self.deleted.append((calendarId, eventId))
        return _FakeExecutable(None, fail=self.fail)

    def patch(self, *, calendarId: str, eventId: str, body: dict[str, Any]) -> _FakeExecutable:
        self.patched.append({"calendarId": calendarId, "eventId": eventId, "body": body})
        return _FakeExecutable({"id": eventId, "summary": "Existing", **body}, fail=self.fail)


class _FakeFreebusyResource:
    def __init__(self, busy: list[dict[str, str]], *, fail: bool = False) -> None:
        self._busy = busy
        self.fail = fail

    def query(self, *, body: dict[str, Any]) -> _FakeExecutable:
        calendar_id = body["items"][0]["id"]
        return _FakeExecutable({"calendars": {calendar_id: {"busy": self._busy}}}, fail=self.fail)


class _FakeService:
    def __init__(self, *, busy: list[dict[str, str]] | None = None, fail: bool = False) -> None:
        self._events = _FakeEventsResource(fail=fail)
        self._freebusy = _FakeFreebusyResource(busy or [], fail=fail)

    def events(self) -> _FakeEventsResource:
        return self._events

    def freebusy(self) -> _FakeFreebusyResource:
        return self._freebusy


async def test_list_busy_slots_parses_response() -> None:
    busy = [{"start": "2026-09-01T09:00:00+00:00", "end": "2026-09-01T09:30:00+00:00"}]
    provider = GoogleCalendarProvider(service=_FakeService(busy=busy), calendar_id="primary")

    slots = await provider.list_busy_slots(
        start=datetime(2026, 9, 1, tzinfo=UTC), end=datetime(2026, 9, 2, tzinfo=UTC)
    )

    assert len(slots) == 1
    assert slots[0].start == datetime(2026, 9, 1, 9, 0, tzinfo=UTC)
    assert slots[0].end == datetime(2026, 9, 1, 9, 30, tzinfo=UTC)


async def test_list_busy_slots_wraps_backend_errors() -> None:
    provider = GoogleCalendarProvider(service=_FakeService(fail=True), calendar_id="primary")

    with pytest.raises(CalendarError):
        await provider.list_busy_slots(
            start=datetime(2026, 9, 1, tzinfo=UTC), end=datetime(2026, 9, 2, tzinfo=UTC)
        )


async def test_create_event_returns_event_with_id() -> None:
    service = _FakeService()
    provider = GoogleCalendarProvider(service=service, calendar_id="primary")
    start = datetime(2026, 9, 1, 10, 0, tzinfo=UTC)
    end = datetime(2026, 9, 1, 10, 30, tzinfo=UTC)

    event = await provider.create_event(start=start, end=end, summary="Checkup")

    assert event.event_id == "evt-1"
    assert event.summary == "Checkup"
    assert service.events().inserted[0]["calendarId"] == "primary"


async def test_cancel_event_calls_delete() -> None:
    service = _FakeService()
    provider = GoogleCalendarProvider(service=service, calendar_id="primary")

    await provider.cancel_event(event_id="evt-1")

    assert service.events().deleted == [("primary", "evt-1")]


async def test_reschedule_event_calls_patch_with_new_times() -> None:
    service = _FakeService()
    provider = GoogleCalendarProvider(service=service, calendar_id="primary")
    new_start = datetime(2026, 9, 2, 14, 0, tzinfo=UTC)
    new_end = datetime(2026, 9, 2, 14, 30, tzinfo=UTC)

    event = await provider.reschedule_event(event_id="evt-1", start=new_start, end=new_end)

    assert event.event_id == "evt-1"
    assert event.start == new_start
    assert service.events().patched[0]["eventId"] == "evt-1"


async def test_create_event_wraps_backend_errors() -> None:
    provider = GoogleCalendarProvider(service=_FakeService(fail=True), calendar_id="primary")

    with pytest.raises(CalendarError):
        await provider.create_event(
            start=datetime(2026, 9, 1, tzinfo=UTC),
            end=datetime(2026, 9, 1, 10, 30, tzinfo=UTC),
            summary="x",
        )
