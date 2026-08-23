"""Piper TTS provider (default: Apache-2.0, CPU-friendly, no vendor lock-in).

`piper-tts` is an optional, heavier dependency — the `voice` extra
(`uv sync --extra voice`). Like the faster-whisper STT provider, this
module only imports it inside `PiperTTSProvider.from_settings`, so
importing this module never requires the extra to be installed. Tests
inject a fake voice via the plain constructor.

The `_PiperVoice`/`_PiperAudioChunk` protocols below (`synthesize(text) ->
Iterable[chunk with .audio_int16_bytes]`) were verified against the real
installed `piper-tts==1.7.0` API, not guessed.

Piper's synthesis call is blocking/CPU-bound; it always runs via
`asyncio.to_thread` so it never blocks the event loop serving a live call.

Piper voice models don't synthesize at an arbitrary caller-chosen rate --
they always produce audio at their own native `config.sample_rate` (most
voices, including the bundled default `en_US-amy-medium`, are 22050 Hz).
`BaseTTSProvider.synthesize`'s contract is to yield PCM16 *at the
requested* `sample_rate` (see `ElevenLabsTTSProvider`, which asks its API
for that rate directly); silently ignoring the mismatch and yielding
22050 Hz audio labeled as 16000 Hz plays back ~27% too slow and pitched
down, which is exactly what a real voice call sounded like before this
was caught by actually listening to it (see CHANGELOG). Fixed by
resampling to the requested rate with simple linear interpolation --
good enough for speech at these rates, and avoids pulling in a dedicated
resampling dependency for one call site.
"""

import asyncio
from collections.abc import AsyncIterator, Iterable
from typing import Protocol

import numpy as np
import structlog

from openvoice.config import Settings
from openvoice.tts.base import BaseTTSProvider, TTSError

logger = structlog.get_logger(__name__)


class _PiperAudioChunk(Protocol):
    @property
    def audio_int16_bytes(self) -> bytes: ...


class _PiperVoiceConfig(Protocol):
    @property
    def sample_rate(self) -> int: ...


class _PiperVoice(Protocol):
    """Structural type for `piper.PiperVoice`, avoiding a hard import."""

    @property
    def config(self) -> _PiperVoiceConfig: ...

    def synthesize(self, text: str) -> Iterable[_PiperAudioChunk]: ...


def _resample_pcm16(pcm: bytes, *, from_rate: int, to_rate: int) -> bytes:
    """Linearly resample mono PCM16 `pcm` from `from_rate` to `to_rate`."""
    if from_rate == to_rate or not pcm:
        return pcm

    samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float32)
    target_count = max(1, round(samples.size * to_rate / from_rate))
    resampled = np.interp(
        np.linspace(0, samples.size - 1, target_count),
        np.arange(samples.size),
        samples,
    )
    return resampled.astype(np.int16).tobytes()


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
            chunks = await asyncio.to_thread(lambda: list(self._voice.synthesize(text)))
        except Exception as exc:  # any Piper/onnxruntime failure becomes TTSError
            logger.error("piper_synthesize_failed", error=str(exc))
            raise TTSError(f"Piper synthesis failed: {exc}") from exc

        native_rate = self._voice.config.sample_rate
        for chunk in chunks:
            yield _resample_pcm16(
                chunk.audio_int16_bytes, from_rate=native_rate, to_rate=sample_rate
            )
