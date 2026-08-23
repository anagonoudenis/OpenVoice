"""Tests for the calendar provider factory."""

import pytest

from openvoice.calendar.base import CalendarError
from openvoice.calendar.factory import get_calendar_provider
from openvoice.calendar.providers.google_calendar import GoogleCalendarProvider
from openvoice.config import Settings, get_settings


def _settings(monkeypatch: pytest.MonkeyPatch, **overrides: str) -> Settings:
    monkeypatch.setenv("SECRET_KEY", "test-secret-key-please-ignore")
    monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://test:test@localhost:5432/test")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    # Deterministic baseline, applied before overrides: a real local `.env`
    # may legitimately set these to something else for manual testing.
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("CALENDAR_PROVIDER", "google")
    monkeypatch.delenv("GOOGLE_SERVICE_ACCOUNT_JSON_PATH", raising=False)
    for key, value in overrides.items():
        monkeypatch.setenv(key, value)
    get_settings.cache_clear()
    return get_settings()


def test_google_provider_selected_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _settings(monkeypatch)
    sentinel = object()
    monkeypatch.setattr(
        GoogleCalendarProvider, "from_settings", classmethod(lambda cls, s: sentinel)
    )

    assert get_calendar_provider(settings) is sentinel


def test_google_provider_raises_clearly_without_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _settings(monkeypatch)

    with pytest.raises(CalendarError, match="GOOGLE_SERVICE_ACCOUNT_JSON_PATH"):
        get_calendar_provider(settings)
