"""faster-whisper STT provider.

`faster-whisper` (and its native `ctranslate2` backend) is an optional,
heavy dependency — the `voice` extra (`uv sync --extra voice`). This
module never imports it at module load time; it's only imported inside
`FasterWhisperSTTProvider.from_settings`, so importing
`openvoice.stt.providers.faster_whisper` (and anything that transitively
imports it) never requires the extra to be installed unless a provider is
actually constructed from real settings. Tests inject a fake model via the
plain constructor instead.

faster-whisper's `transcribe()` returns almost instantly with a *lazy*
generator: the actual blocking, CPU-bound decoding only happens once
it's iterated, not when `transcribe()` is called. Both the call and the
iteration that consumes it are run inside the same `asyncio.to_thread`
worker function (see `_transcribe` below) so none of that work runs on
the event loop -- doing the iteration outside the thread call, which
this module used to do, silently defeats the entire point of
`asyncio.to_thread` and was caught by an actual live call going
mysteriously quiet (empty transcripts) rather than merely slow, likely
because ctranslate2's own execution engine has thread-affinity
expectations `asyncio.to_thread`'s worker-thread reuse can violate when
the generator is driven from a different thread than the one that
created it.
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

        def _transcribe() -> str:
            # `model.transcribe()` returns almost immediately with a *lazy*
            # generator -- the actual CPU-bound Whisper decoding only runs
            # once it's iterated. Consuming it here, inside this function
            # (which `asyncio.to_thread` runs in a worker thread), keeps
            # *all* of that work off the event loop. Joining the segments
            # outside this function -- which is what this code used to do
            # -- runs the real transcription work back on the event loop
            # instead, blocking every other concurrent call (VAD framing,
            # other utterances, ...) for the whole transcription: silently
            # defeating the entire point of `asyncio.to_thread`, and -- since
            # ctranslate2's execution engine has its own internal threading
            # model -- a plausible source of the transcript coming back
            # empty or wrong rather than merely slow.
            segments, _info = self._model.transcribe(audio, language=self._language)
            return "".join(segment.text for segment in segments)

        try:
            text = await asyncio.to_thread(_transcribe)
        except Exception as exc:  # any faster-whisper/ctranslate2 failure becomes STTError
            logger.error("faster_whisper_transcribe_failed", error=str(exc))
            raise STTError(f"faster-whisper transcription failed: {exc}") from exc

        yield TranscriptSegment(text=text.strip(), is_final=True, language=self._language)
