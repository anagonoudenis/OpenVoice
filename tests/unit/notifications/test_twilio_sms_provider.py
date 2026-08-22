"""Tests for TwilioSMSProvider, mocking the HTTP layer with pytest-httpx."""

import pytest
from pytest_httpx import HTTPXMock

from openvoice.notifications.base import NotificationError
from openvoice.notifications.providers.twilio_sms import TwilioSMSProvider


@pytest.fixture
def provider() -> TwilioSMSProvider:
    return TwilioSMSProvider(
        account_sid="AC123",
        auth_token="secret",
        from_number="+15550001111",
        timeout_seconds=5.0,
        max_retries=3,
    )


async def test_send_sms_posts_expected_payload(
    provider: TwilioSMSProvider, httpx_mock: HTTPXMock
) -> None:
    httpx_mock.add_response(status_code=201)

    await provider.send_sms(to="+15559998888", body="Your appointment is confirmed.")

    request = httpx_mock.get_requests()[0]
    assert "/Accounts/AC123/Messages.json" in str(request.url)
    body = request.read().decode()
    assert "To=%2B15559998888" in body
    assert "From=%2B15550001111" in body


async def test_send_sms_retries_on_503_then_succeeds(
    provider: TwilioSMSProvider, httpx_mock: HTTPXMock
) -> None:
    httpx_mock.add_response(status_code=503)
    httpx_mock.add_response(status_code=201)

    await provider.send_sms(to="+15559998888", body="hi")

    assert len(httpx_mock.get_requests()) == 2


async def test_send_sms_raises_after_exhausting_retries(
    provider: TwilioSMSProvider, httpx_mock: HTTPXMock
) -> None:
    httpx_mock.add_response(status_code=503)
    httpx_mock.add_response(status_code=503)
    httpx_mock.add_response(status_code=503)

    with pytest.raises(NotificationError):
        await provider.send_sms(to="+15559998888", body="hi")


async def test_send_sms_does_not_retry_auth_error(
    provider: TwilioSMSProvider, httpx_mock: HTTPXMock
) -> None:
    httpx_mock.add_response(status_code=401)

    with pytest.raises(NotificationError):
        await provider.send_sms(to="+15559998888", body="hi")

    assert len(httpx_mock.get_requests()) == 1
