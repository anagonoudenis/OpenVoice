"""Tests for the STT provider factory."""

from typing import Any

import pytest

from openvoice.config import Settings, get_settings
from openvoice.stt.factory import get_stt_provider
from openvoice.stt.providers.faster_whisper import FasterWhisperSTTProvider


def _settings(monkeypatch: pytest.MonkeyPatch, **overrides: str) -> Settings:
    monkeypatch.setenv("SECRET_KEY", "test-secret-key-please-ignore")
    monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://test:test@localhost:5432/test")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    for key, value in overrides.items():
        monkeypatch.setenv(key, value)
    get_settings.cache_clear()
    return get_settings()


def test_faster_whisper_selected_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _settings(monkeypatch)
    sentinel = object()
    monkeypatch.setattr(
        FasterWhisperSTTProvider, "from_settings", classmethod(lambda cls, s: sentinel)
    )

    result: Any = get_stt_provider(settings)

    assert result is sentinel


def test_deepgram_provider_not_implemented(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _settings(monkeypatch, STT_PROVIDER="deepgram")

    with pytest.raises(NotImplementedError):
        get_stt_provider(settings)
