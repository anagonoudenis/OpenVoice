"""Piper TTS provider (default: Apache-2.0, CPU-friendly, no vendor lock-in).

`piper-tts` is an optional, heavier dependency — the `voice` extra
(`uv sync --extra voice`). Like the faster-whisper STT provider, this
module only imports it inside `PiperTTSProvider.from_settings`, so
importing this module never requires the extra to be installed. Tests
inject a fake voice via the plain constructor.

Piper's synthesis call is blocking/CPU-bound; it always runs via
`asyncio.to_thread` so it never blocks the event loop serving a live call.
"""

import asyncio
from collections.abc import AsyncIterator, Iterable
from typing import Protocol

import structlog

from openvoice.config import Settings
from openvoice.tts.base import BaseTTSProvider, TTSError

logger = structlog.get_logger(__name__)


class _PiperVoice(Protocol):
    """Structural type for `piper.PiperVoice`, avoiding a hard import."""

    def synthesize_stream_raw(self, text: str) -> Iterable[bytes]: ...


class PiperTTSProvider(BaseTTSProvider):
    """TTS provider backed by a local Piper voice model."""

    def __init__(self, *, voice: _PiperVoice) -> None:
        self._voice = voice

    @classmethod
    def from_settings(cls, settings: Settings) -> "PiperTTSProvider":
        """Load a real Piper voice per `settings`. Requires the `voice` extra."""
        try:
            from piper import PiperVoice
        except ImportError as exc:
            raise TTSError("piper-tts is not installed; run `uv sync --extra voice`") from exc

        voice = PiperVoice.load(settings.piper_voice_model_path)
        return cls(voice=voice)

    async def synthesize(
        self, text: str, *, voice: str | None = None, sample_rate: int = 16000
    ) -> AsyncIterator[bytes]:
        try:
            chunks = await asyncio.to_thread(lambda: list(self._voice.synthesize_stream_raw(text)))
        except Exception as exc:  # any Piper/onnxruntime failure becomes TTSError
            logger.error("piper_synthesize_failed", error=str(exc))
            raise TTSError(f"Piper synthesis failed: {exc}") from exc

        for chunk in chunks:
            yield chunk
