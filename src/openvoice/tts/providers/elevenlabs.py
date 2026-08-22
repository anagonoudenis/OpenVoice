"""ElevenLabs TTS provider — a modern cloud voice option.

Talks to the ElevenLabs REST API directly over `httpx` (no vendor SDK
dependency needed), applying the same timeout + tenacity-retry pattern as
the LLM providers: transport errors and 429/5xx responses are retried with
exponential backoff; 4xx (auth, bad request) fail immediately.
"""

from collections.abc import AsyncIterator

import httpx
import structlog
from tenacity import AsyncRetrying, retry_if_exception_type, stop_after_attempt, wait_exponential

from openvoice.tts.base import BaseTTSProvider, TTSError

logger = structlog.get_logger(__name__)

_BASE_URL = "https://api.elevenlabs.io/v1"
_RETRYABLE_STATUS_CODES = frozenset({429, 500, 502, 503, 504})


class _RetryableStatusError(Exception):
    """Internal signal that a response's status code warrants a retry."""

    def __init__(self, response: httpx.Response) -> None:
        self.response = response
        super().__init__(f"Retryable status {response.status_code}")


class ElevenLabsTTSProvider(BaseTTSProvider):
    """TTS provider backed by the ElevenLabs cloud API."""

    def __init__(
        self,
        *,
        api_key: str,
        voice_id: str,
        model_id: str,
        timeout_seconds: float = 15.0,
        max_retries: int = 3,
    ) -> None:
        self._client = httpx.AsyncClient(
            base_url=_BASE_URL, headers={"xi-api-key": api_key}, timeout=timeout_seconds
        )
        self._voice_id = voice_id
        self._model_id = model_id
        self._retryer = AsyncRetrying(
            reraise=True,
            stop=stop_after_attempt(max_retries),
            wait=wait_exponential(multiplier=0.5, min=0.5, max=8),
            retry=retry_if_exception_type((httpx.TransportError, _RetryableStatusError)),
        )

    async def synthesize(
        self, text: str, *, voice: str | None = None, sample_rate: int = 16000
    ) -> AsyncIterator[bytes]:
        voice_id = voice or self._voice_id
        try:
            response: httpx.Response = await self._retryer(
                self._post, voice_id=voice_id, text=text, sample_rate=sample_rate
            )
        except (httpx.HTTPError, _RetryableStatusError) as exc:
            logger.error("elevenlabs_synthesize_failed", error=str(exc), voice_id=voice_id)
            raise TTSError(f"ElevenLabs synthesis failed: {exc}") from exc

        yield response.content

    async def _post(self, *, voice_id: str, text: str, sample_rate: int) -> httpx.Response:
        response = await self._client.post(
            f"/text-to-speech/{voice_id}",
            params={"output_format": f"pcm_{sample_rate}"},
            json={"text": text, "model_id": self._model_id},
        )
        if response.status_code in _RETRYABLE_STATUS_CODES:
            raise _RetryableStatusError(response)
        response.raise_for_status()
        return response
