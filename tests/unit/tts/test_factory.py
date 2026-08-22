"""Tests for the TTS provider factory."""

from typing import Any

import pytest

from openvoice.config import Settings, get_settings
from openvoice.tts.factory import get_tts_provider
from openvoice.tts.providers.elevenlabs import ElevenLabsTTSProvider
from openvoice.tts.providers.piper import PiperTTSProvider


def _settings(monkeypatch: pytest.MonkeyPatch, **overrides: str) -> Settings:
    monkeypatch.setenv("SECRET_KEY", "test-secret-key-please-ignore")
    monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://test:test@localhost:5432/test")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    for key, value in overrides.items():
        monkeypatch.setenv(key, value)
    get_settings.cache_clear()
    return get_settings()


def test_piper_selected_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _settings(monkeypatch)
    sentinel = object()
    monkeypatch.setattr(PiperTTSProvider, "from_settings", classmethod(lambda cls, s: sentinel))

    result: Any = get_tts_provider(settings)

    assert result is sentinel


def test_elevenlabs_provider_selected_via_config(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _settings(
        monkeypatch, TTS_PROVIDER="elevenlabs", ELEVENLABS_API_KEY="test-elevenlabs-key"
    )

    provider = get_tts_provider(settings)

    assert isinstance(provider, ElevenLabsTTSProvider)


def test_coqui_provider_not_implemented(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _settings(monkeypatch, TTS_PROVIDER="coqui")

    with pytest.raises(NotImplementedError):
        get_tts_provider(settings)
