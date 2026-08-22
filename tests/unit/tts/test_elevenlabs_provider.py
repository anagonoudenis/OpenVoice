"""Tests for ElevenLabsTTSProvider, mocking the HTTP layer with pytest-httpx."""

import pytest
from pytest_httpx import HTTPXMock

from openvoice.tts.base import TTSError
from openvoice.tts.providers.elevenlabs import ElevenLabsTTSProvider


@pytest.fixture
def provider() -> ElevenLabsTTSProvider:
    return ElevenLabsTTSProvider(
        api_key="test-key",
        voice_id="voice-1",
        model_id="eleven_turbo_v2_5",
        timeout_seconds=5.0,
        max_retries=3,
    )


async def test_synthesize_returns_audio_bytes(
    provider: ElevenLabsTTSProvider, httpx_mock: HTTPXMock
) -> None:
    httpx_mock.add_response(content=b"raw-pcm-audio")

    chunks = [c async for c in provider.synthesize("Bonjour")]

    assert chunks == [b"raw-pcm-audio"]
    request = httpx_mock.get_requests()[0]
    assert request.headers["xi-api-key"] == "test-key"
    assert "/text-to-speech/voice-1" in str(request.url)


async def test_synthesize_uses_override_voice(
    provider: ElevenLabsTTSProvider, httpx_mock: HTTPXMock
) -> None:
    httpx_mock.add_response(content=b"audio")

    [c async for c in provider.synthesize("Bonjour", voice="voice-2")]

    request = httpx_mock.get_requests()[0]
    assert "/text-to-speech/voice-2" in str(request.url)


async def test_synthesize_retries_on_503_then_succeeds(
    provider: ElevenLabsTTSProvider, httpx_mock: HTTPXMock
) -> None:
    httpx_mock.add_response(status_code=503)
    httpx_mock.add_response(content=b"audio-after-retry")

    chunks = [c async for c in provider.synthesize("Bonjour")]

    assert chunks == [b"audio-after-retry"]
    assert len(httpx_mock.get_requests()) == 2


async def test_synthesize_raises_after_exhausting_retries(
    provider: ElevenLabsTTSProvider, httpx_mock: HTTPXMock
) -> None:
    httpx_mock.add_response(status_code=503)
    httpx_mock.add_response(status_code=503)
    httpx_mock.add_response(status_code=503)

    with pytest.raises(TTSError):
        [c async for c in provider.synthesize("Bonjour")]

    assert len(httpx_mock.get_requests()) == 3


async def test_synthesize_does_not_retry_auth_error(
    provider: ElevenLabsTTSProvider, httpx_mock: HTTPXMock
) -> None:
    httpx_mock.add_response(status_code=401)

    with pytest.raises(TTSError):
        [c async for c in provider.synthesize("Bonjour")]

    assert len(httpx_mock.get_requests()) == 1
