"""Tests for the SMS/email provider factories."""

import pytest

from openvoice.config import Settings, get_settings
from openvoice.notifications.factory import get_email_provider, get_sms_provider
from openvoice.notifications.providers.resend_email import ResendEmailProvider
from openvoice.notifications.providers.twilio_sms import TwilioSMSProvider


def _settings(monkeypatch: pytest.MonkeyPatch, **overrides: str) -> Settings:
    monkeypatch.setenv("SECRET_KEY", "test-secret-key-please-ignore")
    monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://test:test@localhost:5432/test")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    # Deterministic baseline, applied before overrides: a real local `.env`
    # may legitimately set these to something else for manual testing.
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("SMS_PROVIDER", "twilio")
    monkeypatch.setenv("EMAIL_PROVIDER", "resend")
    monkeypatch.delenv("TWILIO_ACCOUNT_SID", raising=False)
    monkeypatch.delenv("TWILIO_AUTH_TOKEN", raising=False)
    monkeypatch.delenv("TWILIO_FROM_NUMBER", raising=False)
    monkeypatch.delenv("RESEND_API_KEY", raising=False)
    monkeypatch.delenv("RESEND_FROM_EMAIL", raising=False)
    for key, value in overrides.items():
        monkeypatch.setenv(key, value)
    get_settings.cache_clear()
    return get_settings()


def test_twilio_provider_built_when_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _settings(
        monkeypatch,
        TWILIO_ACCOUNT_SID="AC123",
        TWILIO_AUTH_TOKEN="secret",
        TWILIO_FROM_NUMBER="+15550001111",
    )
    assert isinstance(get_sms_provider(settings), TwilioSMSProvider)


def test_twilio_provider_raises_clearly_when_not_configured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _settings(monkeypatch)
    with pytest.raises(RuntimeError, match="TWILIO_ACCOUNT_SID"):
        get_sms_provider(settings)


def test_resend_provider_built_when_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _settings(
        monkeypatch,
        RESEND_API_KEY="re_test_key",
        RESEND_FROM_EMAIL="bookings@example.com",
    )
    assert isinstance(get_email_provider(settings), ResendEmailProvider)


def test_resend_provider_raises_clearly_when_not_configured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _settings(monkeypatch)
    with pytest.raises(RuntimeError, match="RESEND_API_KEY"):
        get_email_provider(settings)
