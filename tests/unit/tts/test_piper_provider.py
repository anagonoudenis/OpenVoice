"""Tests for PiperTTSProvider, using a fake injected voice.

The real `piper-tts` package is an optional dependency (the `voice`
extra) and is never required just to run this test: the provider's
constructor accepts any object matching its structural `_PiperVoice`
protocol.
"""

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import pytest

from openvoice.tts.base import TTSError
from openvoice.tts.providers.piper import PiperTTSProvider


@dataclass
class _FakeAudioChunk:
    audio_int16_bytes: bytes


@dataclass
class _FakeVoiceConfig:
    sample_rate: int = 16000


class _FakeVoice:
    def __init__(
        self, chunks: list[bytes], *, fail: bool = False, sample_rate: int = 16000
    ) -> None:
        self._chunks = chunks
        self._fail = fail
        self.received_text: str | None = None
        self.config = _FakeVoiceConfig(sample_rate=sample_rate)

    def synthesize(self, text: str) -> Iterable[_FakeAudioChunk]:
        if self._fail:
            raise RuntimeError("boom")
        self.received_text = text
        return (_FakeAudioChunk(audio_int16_bytes=c) for c in self._chunks)


def _pcm16(*samples: int) -> bytes:
    return np.array(samples, dtype=np.int16).tobytes()


async def test_synthesize_yields_chunks_in_order() -> None:
    voice = _FakeVoice([b"chunk1", b"chunk2"])
    provider = PiperTTSProvider(voice=voice)

    chunks = [c async for c in provider.synthesize("Hello")]

    assert chunks == [b"chunk1", b"chunk2"]
    assert voice.received_text == "Hello"


async def test_synthesize_wraps_backend_errors() -> None:
    voice = _FakeVoice([], fail=True)
    provider = PiperTTSProvider(voice=voice)

    with pytest.raises(TTSError):
        [c async for c in provider.synthesize("Hello")]


async def test_synthesize_resamples_from_the_voices_native_rate() -> None:
    """Regression test: Piper voices (e.g. the bundled en_US-amy-medium)
    synthesize at their own native rate (often 22050 Hz), not whatever
    `sample_rate` the caller asks for. Labeling 22050 Hz audio as 16000 Hz
    without resampling plays back too slow and pitched down -- this was
    caught by actually listening to a real call, not by any prior test.
    """
    voice = _FakeVoice([_pcm16(*range(0, 2205, 1))], sample_rate=22050)
    provider = PiperTTSProvider(voice=voice)

    chunks = [c async for c in provider.synthesize("Hello", sample_rate=16000)]

    resampled = np.frombuffer(chunks[0], dtype=np.int16)
    # 2205 samples at 22050 Hz is 100ms; at 16000 Hz that's ~1600 samples.
    assert 1590 <= resampled.size <= 1610


async def test_synthesize_is_a_no_op_when_rates_already_match() -> None:
    voice = _FakeVoice([b"chunk1"], sample_rate=16000)
    provider = PiperTTSProvider(voice=voice)

    chunks = [c async for c in provider.synthesize("Hello", sample_rate=16000)]

    assert chunks == [b"chunk1"]
