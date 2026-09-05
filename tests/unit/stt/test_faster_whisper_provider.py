"""Tests for FasterWhisperSTTProvider, using a fake injected model.

The real `faster-whisper` package is an optional dependency (the `voice`
extra) and is never required just to run this test: the provider's
constructor accepts any object matching its structural `_WhisperModel`
protocol, so these tests inject a lightweight fake instead.
"""

import threading
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


async def test_transcribe_generator_is_consumed_off_the_event_loop() -> None:
    """Regression test: `model.transcribe()` returns a *lazy* generator --
    the real, blocking decode work only happens once it's iterated, not
    when `transcribe()` is called. Consuming it back on the event loop
    (which this code used to do) silently defeats `asyncio.to_thread`
    entirely, and was the leading suspect behind a real live call
    producing empty transcripts rather than merely slow ones. This fakes
    a genuinely lazy generator and records which thread actually drives
    it, to prove the fix keeps *all* of that work off the event loop.
    """
    main_thread_id = threading.get_ident()
    consuming_thread_ids: list[int] = []

    def _lazy_segments() -> Iterable[_FakeSegment]:
        for text in ["Hello ", "world"]:
            consuming_thread_ids.append(threading.get_ident())
            yield _FakeSegment(text=text)

    class _LazyModel:
        def transcribe(
            self, audio: np.ndarray, *, language: str | None = None
        ) -> tuple[Iterable[_FakeSegment], object]:
            # Returning a generator here, not an already-materialized
            # list, is what makes this test actually exercise the bug:
            # nothing has run yet at this point, unlike `_FakeModel` above.
            return (_lazy_segments(), object())

    provider = FasterWhisperSTTProvider(model=_LazyModel())
    pcm = np.array([0, 1000, -1000], dtype=np.int16).tobytes()

    results = [seg async for seg in provider.transcribe_stream(_frames([pcm]))]

    assert results == [TranscriptSegment(text="Hello world", is_final=True, language=None)]
    assert consuming_thread_ids  # the generator was actually driven to completion
    assert all(tid != main_thread_id for tid in consuming_thread_ids)


async def test_transcribe_stream_wraps_backend_errors() -> None:
    model = _FakeModel([], fail=True)
    provider = FasterWhisperSTTProvider(model=model)
    pcm = np.array([0, 1, 2], dtype=np.int16).tobytes()

    with pytest.raises(STTError):
        [seg async for seg in provider.transcribe_stream(_frames([pcm]))]
