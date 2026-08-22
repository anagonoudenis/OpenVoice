"""Twilio SMS provider.

Talks to Twilio's REST API directly over `httpx` (no `twilio` SDK
dependency), following the same timeout + tenacity-retry pattern as the
LLM/TTS providers: transport errors and 429/5xx responses are retried
with exponential backoff; 4xx (auth, bad request) fail immediately.
"""

import httpx
import structlog
from tenacity import AsyncRetrying, retry_if_exception_type, stop_after_attempt, wait_exponential

from openvoice.notifications.base import BaseSMSProvider, NotificationError

logger = structlog.get_logger(__name__)

_BASE_URL = "https://api.twilio.com/2010-04-01"
_RETRYABLE_STATUS_CODES = frozenset({429, 500, 502, 503, 504})


class _RetryableStatusError(Exception):
    """Internal signal that a response's status code warrants a retry."""

    def __init__(self, response: httpx.Response) -> None:
        self.response = response
        super().__init__(f"Retryable status {response.status_code}")


class TwilioSMSProvider(BaseSMSProvider):
    """SMS provider backed by the Twilio REST API."""

    def __init__(
        self,
        *,
        account_sid: str,
        auth_token: str,
        from_number: str,
        timeout_seconds: float = 10.0,
        max_retries: int = 3,
    ) -> None:
        self._client = httpx.AsyncClient(
            base_url=_BASE_URL, auth=(account_sid, auth_token), timeout=timeout_seconds
        )
        self._account_sid = account_sid
        self._from_number = from_number
        self._retryer = AsyncRetrying(
            reraise=True,
            stop=stop_after_attempt(max_retries),
            wait=wait_exponential(multiplier=0.5, min=0.5, max=8),
            retry=retry_if_exception_type((httpx.TransportError, _RetryableStatusError)),
        )

    async def send_sms(self, *, to: str, body: str) -> None:
        try:
            await self._retryer(self._post, to=to, body=body)
        except (httpx.HTTPError, _RetryableStatusError) as exc:
            logger.error("twilio_sms_failed", error=str(exc), to=to)
            raise NotificationError(f"Twilio SMS send failed: {exc}") from exc

    async def _post(self, *, to: str, body: str) -> httpx.Response:
        response = await self._client.post(
            f"/Accounts/{self._account_sid}/Messages.json",
            data={"To": to, "From": self._from_number, "Body": body},
        )
        if response.status_code in _RETRYABLE_STATUS_CODES:
            raise _RetryableStatusError(response)
        response.raise_for_status()
        return response
