"""Abstract speech-to-text provider interface.

Mirrors the shape of `openvoice.llm.base`: business logic depends only on
`BaseSTTProvider`, obtained from `openvoice.stt.factory.get_stt_provider`.
"""

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator

from pydantic import BaseModel


class TranscriptSegment(BaseModel):
    """One recognized segment of speech."""

    text: str
    is_final: bool
    language: str | None = None


class STTError(Exception):
    """Raised when an STT provider fails after exhausting its retry/fallback budget."""


class BaseSTTProvider(ABC):
    """Abstract speech-to-text backend.

    Audio is consumed as an async stream of raw PCM16 mono frames (the
    format LiveKit delivers), representing one VAD-delimited utterance —
    the caller is responsible for segmenting continuous audio into
    utterances before calling this. Implementations yield one or more
    `TranscriptSegment`s as recognition progresses.
    """

    @abstractmethod
    def transcribe_stream(
        self, audio_frames: AsyncIterator[bytes], *, sample_rate: int = 16000
    ) -> AsyncIterator[TranscriptSegment]:
        """Transcribe a stream of raw PCM16 mono audio frames."""
        raise NotImplementedError
