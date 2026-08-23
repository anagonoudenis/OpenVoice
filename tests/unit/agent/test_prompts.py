"""Tests for system prompt resolution."""

import pytest

from openvoice.agent.prompts import build_system_prompt
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
