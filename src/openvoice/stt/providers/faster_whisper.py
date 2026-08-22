"""faster-whisper STT provider.

`faster-whisper` (and its native `ctranslate2` backend) is an optional,
heavy dependency — the `voice` extra (`uv sync --extra voice`). This
module never imports it at module load time; it's only imported inside
`FasterWhisperSTTProvider.from_settings`, so importing
`openvoice.stt.providers.faster_whisper` (and anything that transitively
imports it) never requires the extra to be installed unless a provider is
actually constructed from real settings. Tests inject a fake model via the
plain constructor instead.

faster-whisper's `transcribe()` is a blocking, CPU-bound call — it always
runs via `asyncio.to_thread` so it never blocks the event loop serving a
live call.
"""

import asyncio
from collections.abc import AsyncIterator, Iterable
from typing import Any, Protocol

import numpy as np
import structlog

from openvoice.config import Settings
from openvoice.stt.base import BaseSTTProvider, STTError, TranscriptSegment

logger = structlog.get_logger(__name__)


class _Segment(Protocol):
    text: str


class _WhisperModel(Protocol):
    """Structural type for `faster_whisper.WhisperModel`, avoiding a hard import."""

    def transcribe(
        self, audio: np.ndarray, *, language: str | None = None
    ) -> tuple[Iterable[_Segment], Any]: ...


class FasterWhisperSTTProvider(BaseSTTProvider):
    """STT provider backed by a local faster-whisper model."""

    def __init__(self, *, model: _WhisperModel, language: str | None = None) -> None:
        self._model = model
        self._language = language

    @classmethod
    def from_settings(cls, settings: Settings) -> "FasterWhisperSTTProvider":
        """Load a real faster-whisper model per `settings`. Requires the `voice` extra."""
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise STTError("faster-whisper is not installed; run `uv sync --extra voice`") from exc

        model = WhisperModel(
            settings.faster_whisper_model_size,
            device=settings.faster_whisper_device,
            compute_type=settings.faster_whisper_compute_type,
        )
        return cls(model=model)

    async def transcribe_stream(
        self, audio_frames: AsyncIterator[bytes], *, sample_rate: int = 16000
    ) -> AsyncIterator[TranscriptSegment]:
        raw = b"".join([frame async for frame in audio_frames])
        if not raw:
            return

        audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0

        try:
            segments, _info = await asyncio.to_thread(
                self._model.transcribe, audio, language=self._language
            )
            text = "".join(segment.text for segment in segments)
        except Exception as exc:  # any faster-whisper/ctranslate2 failure becomes STTError
            logger.error("faster_whisper_transcribe_failed", error=str(exc))
            raise STTError(f"faster-whisper transcription failed: {exc}") from exc

        yield TranscriptSegment(text=text.strip(), is_final=True, language=self._language)
