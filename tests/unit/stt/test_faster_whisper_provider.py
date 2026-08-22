"""Tests for FasterWhisperSTTProvider, using a fake injected model.

The real `faster-whisper` package is an optional dependency (the `voice`
extra) and is never required just to run this test: the provider's
constructor accepts any object matching its structural `_WhisperModel`
protocol, so these tests inject a lightweight fake instead.
"""

from collections.abc import AsyncIterator, Iterable
from dataclasses import dataclass

import numpy as np
import pytest

from openvoice.stt.base import STTError, TranscriptSegment
from openvoice.stt.providers.faster_whisper import FasterWhisperSTTProvider


@dataclass
class _FakeSegment:
    text: str


class _FakeModel:
    def __init__(self, segments: list[str], *, fail: bool = False) -> None:
        self._segments = segments
        self._fail = fail
        self.received_audio: np.ndarray | None = None

    def transcribe(
        self, audio: np.ndarray, *, language: str | None = None
    ) -> tuple[Iterable[_FakeSegment], object]:
        if self._fail:
            raise RuntimeError("boom")
        self.received_audio = audio
        return ([_FakeSegment(text=s) for s in self._segments], object())


async def _frames(chunks: list[bytes]) -> AsyncIterator[bytes]:
    for chunk in chunks:
        yield chunk


async def test_transcribe_stream_joins_segments() -> None:
    model = _FakeModel(["Hello ", "world"])
    provider = FasterWhisperSTTProvider(model=model)

    pcm = np.array([0, 1000, -1000], dtype=np.int16).tobytes()
    results = [seg async for seg in provider.transcribe_stream(_frames([pcm]))]

    assert results == [TranscriptSegment(text="Hello world", is_final=True, language=None)]
    assert model.received_audio is not None
    assert model.received_audio.dtype == np.float32


async def test_transcribe_stream_yields_nothing_for_empty_audio() -> None:
    model = _FakeModel([])
    provider = FasterWhisperSTTProvider(model=model)

    results = [seg async for seg in provider.transcribe_stream(_frames([]))]

    assert results == []


async def test_transcribe_stream_wraps_backend_errors() -> None:
    model = _FakeModel([], fail=True)
    provider = FasterWhisperSTTProvider(model=model)
    pcm = np.array([0, 1, 2], dtype=np.int16).tobytes()

    with pytest.raises(STTError):
        [seg async for seg in provider.transcribe_stream(_frames([pcm]))]
