"""Test doubles for `BaseSTTProvider`/`BaseTTSProvider`, shared across telephony tests."""

from collections.abc import AsyncIterator

from openvoice.stt.base import BaseSTTProvider, TranscriptSegment
from openvoice.tts.base import BaseTTSProvider


class FakeSTTProvider(BaseSTTProvider):
    """Drains the given audio frames, then yields a fixed list of segments."""

    def __init__(self, segments: list[TranscriptSegment]) -> None:
        self._segments = segments
        self.received_sample_rates: list[int] = []

    async def transcribe_stream(
        self, audio_frames: AsyncIterator[bytes], *, sample_rate: int = 16000
    ) -> AsyncIterator[TranscriptSegment]:
        self.received_sample_rates.append(sample_rate)
        async for _ in audio_frames:
            pass
        for segment in self._segments:
            yield segment


class FakeTTSProvider(BaseTTSProvider):
    """Records synthesized text and yields a single deterministic audio chunk."""

    def __init__(self) -> None:
        self.synthesized_text: list[str] = []

    async def synthesize(
        self, text: str, *, voice: str | None = None, sample_rate: int = 16000
    ) -> AsyncIterator[bytes]:
        self.synthesized_text.append(text)
        yield f"audio:{text}".encode()
