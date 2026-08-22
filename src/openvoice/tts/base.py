"""Abstract text-to-speech provider interface.

Mirrors `openvoice.llm.base` and `openvoice.stt.base`: business logic
depends only on `BaseTTSProvider`, obtained from
`openvoice.tts.factory.get_tts_provider`.
"""

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator


class TTSError(Exception):
    """Raised when a TTS provider fails after exhausting its retry budget."""


class BaseTTSProvider(ABC):
    """Abstract text-to-speech backend.

    Synthesizes raw PCM16 mono audio (the format LiveKit expects to play
    back), streamed as chunks so playback can start before the full
    utterance has been synthesized.
    """

    @abstractmethod
    def synthesize(
        self, text: str, *, voice: str | None = None, sample_rate: int = 16000
    ) -> AsyncIterator[bytes]:
        """Synthesize `text` to speech, yielding raw PCM16 audio chunks."""
        raise NotImplementedError
