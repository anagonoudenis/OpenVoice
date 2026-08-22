"""Tests for PiperTTSProvider, using a fake injected voice.

The real `piper-tts` package is an optional dependency (the `voice`
extra) and is never required just to run this test: the provider's
constructor accepts any object matching its structural `_PiperVoice`
protocol.
"""

from collections.abc import Iterable
from dataclasses import dataclass

import pytest

from openvoice.tts.base import TTSError
from openvoice.tts.providers.piper import PiperTTSProvider


@dataclass
class _FakeAudioChunk:
    audio_int16_bytes: bytes


class _FakeVoice:
    def __init__(self, chunks: list[bytes], *, fail: bool = False) -> None:
        self._chunks = chunks
        self._fail = fail
        self.received_text: str | None = None

    def synthesize(self, text: str) -> Iterable[_FakeAudioChunk]:
        if self._fail:
            raise RuntimeError("boom")
        self.received_text = text
        return (_FakeAudioChunk(audio_int16_bytes=c) for c in self._chunks)


async def test_synthesize_yields_chunks_in_order() -> None:
    voice = _FakeVoice([b"chunk1", b"chunk2"])
    provider = PiperTTSProvider(voice=voice)

    chunks = [c async for c in provider.synthesize("Bonjour")]

    assert chunks == [b"chunk1", b"chunk2"]
    assert voice.received_text == "Bonjour"


async def test_synthesize_wraps_backend_errors() -> None:
    voice = _FakeVoice([], fail=True)
    provider = PiperTTSProvider(voice=voice)

    with pytest.raises(TTSError):
        [c async for c in provider.synthesize("Bonjour")]
