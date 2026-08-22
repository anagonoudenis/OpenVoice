"""Tests for ResendEmailProvider, mocking the HTTP layer with pytest-httpx."""

import json

import pytest
from pytest_httpx import HTTPXMock

from openvoice.notifications.base import NotificationError
from openvoice.notifications.providers.resend_email import ResendEmailProvider


@pytest.fixture
def provider() -> ResendEmailProvider:
    return ResendEmailProvider(
        api_key="re_test_key",
        from_email="bookings@example.com",
        timeout_seconds=5.0,
        max_retries=3,
    )


async def test_send_email_posts_expected_payload(
    provider: ResendEmailProvider, httpx_mock: HTTPXMock
) -> None:
    httpx_mock.add_response(status_code=200, json={"id": "abc"})

    await provider.send_email(to="client@example.com", subject="Confirmed", body="See you Tuesday.")

    request = httpx_mock.get_requests()[0]
    assert request.headers["Authorization"] == "Bearer re_test_key"
    payload = json.loads(request.read())
    assert payload["from"] == "bookings@example.com"
    assert payload["to"] == ["client@example.com"]
    assert payload["subject"] == "Confirmed"


async def test_send_email_retries_on_503_then_succeeds(
    provider: ResendEmailProvider, httpx_mock: HTTPXMock
) -> None:
    httpx_mock.add_response(status_code=503)
    httpx_mock.add_response(status_code=200, json={"id": "abc"})

    await provider.send_email(to="client@example.com", subject="s", body="b")

    assert len(httpx_mock.get_requests()) == 2


async def test_send_email_raises_after_exhausting_retries(
    provider: ResendEmailProvider, httpx_mock: HTTPXMock
) -> None:
    httpx_mock.add_response(status_code=503)
    httpx_mock.add_response(status_code=503)
    httpx_mock.add_response(status_code=503)

    with pytest.raises(NotificationError):
        await provider.send_email(to="client@example.com", subject="s", body="b")


async def test_send_email_does_not_retry_auth_error(
    provider: ResendEmailProvider, httpx_mock: HTTPXMock
) -> None:
    httpx_mock.add_response(status_code=401)

    with pytest.raises(NotificationError):
        await provider.send_email(to="client@example.com", subject="s", body="b")

    assert len(httpx_mock.get_requests()) == 1
