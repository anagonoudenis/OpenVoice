"""Tests for openvoice.config.Settings."""

import pytest
from pydantic import ValidationError

from openvoice.config import Environment, LLMProvider, Settings, get_settings


def test_settings_load_from_env(settings: Settings) -> None:
    assert settings.secret_key == "test-secret-key-please-ignore"
    assert settings.environment is Environment.LOCAL
    assert settings.llm_provider is LLMProvider.ANTHROPIC


def test_get_settings_is_cached(required_env: None) -> None:
    assert get_settings() is get_settings()


def test_missing_secret_key_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SECRET_KEY", raising=False)
    monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://test:test@localhost:5432/test")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    get_settings.cache_clear()
    with pytest.raises(ValidationError):
        Settings(_env_file=None)  # type: ignore[call-arg]


def test_anthropic_provider_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SECRET_KEY", "test-secret-key-please-ignore")
    monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://test:test@localhost:5432/test")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    with pytest.raises(ValidationError, match="ANTHROPIC_API_KEY"):
        Settings(_env_file=None)  # type: ignore[call-arg]


def test_self_hosted_provider_requires_base_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SECRET_KEY", "test-secret-key-please-ignore")
    monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://test:test@localhost:5432/test")
    monkeypatch.setenv("LLM_PROVIDER", "self_hosted")
    monkeypatch.delenv("SELF_HOSTED_LLM_BASE_URL", raising=False)
    with pytest.raises(ValidationError, match="SELF_HOSTED_LLM_BASE_URL"):
        Settings(_env_file=None)  # type: ignore[call-arg]


def test_elevenlabs_tts_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SECRET_KEY", "test-secret-key-please-ignore")
    monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://test:test@localhost:5432/test")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("TTS_PROVIDER", "elevenlabs")
    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
    with pytest.raises(ValidationError, match="ELEVENLABS_API_KEY"):
        Settings(_env_file=None)  # type: ignore[call-arg]
