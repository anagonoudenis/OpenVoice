"""Tests for the LLM provider factory."""

import pytest

from openvoice.config import Settings, get_settings
from openvoice.llm.factory import get_llm_provider
from openvoice.llm.providers.anthropic import AnthropicLLMProvider
from openvoice.llm.providers.openai_compatible import OpenAICompatibleLLMProvider


def _settings(monkeypatch: pytest.MonkeyPatch, **overrides: str) -> Settings:
    monkeypatch.setenv("SECRET_KEY", "test-secret-key-please-ignore")
    monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://test:test@localhost:5432/test")
    for key, value in overrides.items():
        monkeypatch.setenv(key, value)
    get_settings.cache_clear()
    return get_settings()


def test_anthropic_provider_selected_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _settings(monkeypatch, ANTHROPIC_API_KEY="test-key")
    provider = get_llm_provider(settings)
    assert isinstance(provider, AnthropicLLMProvider)


def test_openai_provider_selected_via_config(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _settings(monkeypatch, LLM_PROVIDER="openai", OPENAI_API_KEY="test-key")
    provider = get_llm_provider(settings)
    assert isinstance(provider, OpenAICompatibleLLMProvider)


def test_self_hosted_provider_selected_via_config(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _settings(
        monkeypatch,
        LLM_PROVIDER="self_hosted",
        SELF_HOSTED_LLM_BASE_URL="http://localhost:8001/v1",
    )
    provider = get_llm_provider(settings)
    assert isinstance(provider, OpenAICompatibleLLMProvider)
