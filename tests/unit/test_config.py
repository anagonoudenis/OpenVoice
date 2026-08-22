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


def test_openai_compatible_provider_requires_base_url_and_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SECRET_KEY", "test-secret-key-please-ignore")
    monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://test:test@localhost:5432/test")
    monkeypatch.setenv("LLM_PROVIDER", "openai_compatible")
    monkeypatch.delenv("OPENAI_COMPATIBLE_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_COMPATIBLE_MODEL", raising=False)
    with pytest.raises(ValidationError, match="OPENAI_COMPATIBLE_BASE_URL"):
        Settings(_env_file=None)  # type: ignore[call-arg]


def test_openai_compatible_provider_requires_model_even_with_base_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SECRET_KEY", "test-secret-key-please-ignore")
    monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://test:test@localhost:5432/test")
    monkeypatch.setenv("LLM_PROVIDER", "openai_compatible")
    monkeypatch.setenv("OPENAI_COMPATIBLE_BASE_URL", "https://api.deepseek.com/v1")
    monkeypatch.delenv("OPENAI_COMPATIBLE_MODEL", raising=False)
    with pytest.raises(ValidationError, match="OPENAI_COMPATIBLE_MODEL"):
        Settings(_env_file=None)  # type: ignore[call-arg]


def test_blank_optional_env_values_are_treated_as_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    """Regression test: `.env.example` ships every optional field as `KEY=`
    (no value) by design. Without `env_ignore_empty=True`, pydantic-settings
    treats that as the literal empty string rather than "unset", which
    silently breaks every `is None` check downstream (e.g.
    `build_system_prompt`, `GoogleCalendarProvider.from_settings`) instead
    of falling through to the field's real default.
    """
    monkeypatch.setenv("SECRET_KEY", "test-secret-key-please-ignore")
    monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://test:test@localhost:5432/test")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("AGENT_SYSTEM_PROMPT", "")
    monkeypatch.setenv("GOOGLE_SERVICE_ACCOUNT_JSON_PATH", "")

    settings = Settings(_env_file=None)  # type: ignore[call-arg]

    assert settings.agent_system_prompt is None
    assert settings.google_service_account_json_path is None


def test_elevenlabs_tts_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SECRET_KEY", "test-secret-key-please-ignore")
    monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://test:test@localhost:5432/test")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("TTS_PROVIDER", "elevenlabs")
    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
    with pytest.raises(ValidationError, match="ELEVENLABS_API_KEY"):
        Settings(_env_file=None)  # type: ignore[call-arg]
