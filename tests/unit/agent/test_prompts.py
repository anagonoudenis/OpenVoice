"""Tests for system prompt resolution."""

from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from openvoice.agent.prompts import build_system_prompt, build_temporal_context
from openvoice.config import Settings, get_settings


def _settings(monkeypatch: pytest.MonkeyPatch, **overrides: str) -> Settings:
    monkeypatch.setenv("SECRET_KEY", "test-secret-key-please-ignore")
    monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://test:test@localhost:5432/test")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    # Deterministic baseline, applied before overrides: a real local `.env`
    # may legitimately set these to something else for manual testing.
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.delenv("AGENT_SYSTEM_PROMPT", raising=False)
    for key, value in overrides.items():
        monkeypatch.setenv(key, value)
    get_settings.cache_clear()
    return get_settings()


def test_default_prompt_includes_company_name(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _settings(monkeypatch, AGENT_COMPANY_NAME="Acme Dental")
    prompt = build_system_prompt(settings)
    assert "Acme Dental" in prompt


def test_explicit_prompt_overrides_default(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _settings(monkeypatch, AGENT_SYSTEM_PROMPT="Custom instructions only.")
    prompt = build_system_prompt(settings)
    assert prompt == "Custom instructions only."


def test_temporal_context_includes_resolved_date_and_utc_offset() -> None:
    now = datetime(2026, 9, 1, 14, 30, tzinfo=ZoneInfo("UTC"))
    context = build_temporal_context(timezone="Europe/Paris", now=now)

    assert "Tuesday, September 01, 2026" in context
    assert "+0200" in context  # Paris is UTC+2 in September (DST)


def test_temporal_context_falls_back_to_utc_on_invalid_timezone() -> None:
    now = datetime(2026, 9, 1, 14, 30, tzinfo=ZoneInfo("UTC"))

    context = build_temporal_context(timezone="Not/A_Real_Zone", now=now)

    assert "+0000" in context
