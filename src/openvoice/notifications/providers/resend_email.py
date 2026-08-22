"""Resend email provider.

Talks to the Resend REST API directly over `httpx` (no `resend` SDK
dependency), following the same timeout + tenacity-retry pattern as the
other notification/LLM/TTS providers.
"""

import httpx
import structlog
from tenacity import AsyncRetrying, retry_if_exception_type, stop_after_attempt, wait_exponential

from openvoice.notifications.base import BaseEmailProvider, NotificationError

logger = structlog.get_logger(__name__)

_BASE_URL = "https://api.resend.com"
_RETRYABLE_STATUS_CODES = frozenset({429, 500, 502, 503, 504})


class _RetryableStatusError(Exception):
    """Internal signal that a response's status code warrants a retry."""

    def __init__(self, response: httpx.Response) -> None:
        self.response = response
        super().__init__(f"Retryable status {response.status_code}")


class ResendEmailProvider(BaseEmailProvider):
    """Email provider backed by the Resend REST API."""

    def __init__(
        self,
        *,
        api_key: str,
        from_email: str,
        timeout_seconds: float = 10.0,
        max_retries: int = 3,
    ) -> None:
        self._client = httpx.AsyncClient(
            base_url=_BASE_URL,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=timeout_seconds,
        )
        self._from_email = from_email
        self._retryer = AsyncRetrying(
            reraise=True,
            stop=stop_after_attempt(max_retries),
            wait=wait_exponential(multiplier=0.5, min=0.5, max=8),
            retry=retry_if_exception_type((httpx.TransportError, _RetryableStatusError)),
        )

    async def send_email(self, *, to: str, subject: str, body: str) -> None:
        try:
            await self._retryer(self._post, to=to, subject=subject, body=body)
        except (httpx.HTTPError, _RetryableStatusError) as exc:
            logger.error("resend_email_failed", error=str(exc), to=to)
            raise NotificationError(f"Resend email send failed: {exc}") from exc

    async def _post(self, *, to: str, subject: str, body: str) -> httpx.Response:
        response = await self._client.post(
            "/emails",
            json={"from": self._from_email, "to": [to], "subject": subject, "text": body},
        )
        if response.status_code in _RETRYABLE_STATUS_CODES:
            raise _RetryableStatusError(response)
        response.raise_for_status()
        return response
