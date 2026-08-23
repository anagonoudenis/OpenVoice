"""Tests for the LLM provider factory."""

import pytest

from openvoice.config import Settings, get_settings
from openvoice.llm.factory import get_llm_provider
from openvoice.llm.providers.anthropic import AnthropicLLMProvider
from openvoice.llm.providers.openai_compatible import OpenAICompatibleLLMProvider


def _settings(monkeypatch: pytest.MonkeyPatch, **overrides: str) -> Settings:
    monkeypatch.setenv("SECRET_KEY", "test-secret-key-please-ignore")
    monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://test:test@localhost:5432/test")
    # Deterministic baseline, applied before overrides: a real local `.env`
    # may legitimately set LLM_PROVIDER to something else for manual
    # testing, which would otherwise leak into "default" test cases here
    # (real env vars take priority over `.env` file values, but nothing
    # here overrides them unless we do it explicitly).
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
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


def test_openai_compatible_provider_selected_via_config(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _settings(
        monkeypatch,
        LLM_PROVIDER="openai_compatible",
        OPENAI_COMPATIBLE_BASE_URL="https://api.deepseek.com/v1",
        OPENAI_COMPATIBLE_MODEL="deepseek-chat",
        OPENAI_COMPATIBLE_API_KEY="test-key",
    )
    provider = get_llm_provider(settings)
    assert isinstance(provider, OpenAICompatibleLLMProvider)
    # The API key must actually reach the client -- this used to be hardcoded
    # to None for this provider, which silently broke any hosted
    # OpenAI-compatible service (DeepSeek, Kimi, Qwen, ...) that requires auth.
    assert provider._client.api_key == "test-key"


def test_openai_compatible_provider_works_without_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A local vLLM/Ollama server typically needs no auth at all."""
    settings = _settings(
        monkeypatch,
        LLM_PROVIDER="openai_compatible",
        OPENAI_COMPATIBLE_BASE_URL="http://localhost:8001/v1",
        OPENAI_COMPATIBLE_MODEL="llama-3.1-8b",
    )
    provider = get_llm_provider(settings)
    assert isinstance(provider, OpenAICompatibleLLMProvider)
