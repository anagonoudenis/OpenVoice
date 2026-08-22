"""SMS/email provider factories. Select the implementation from `Settings`."""

from openvoice.config import EmailProvider, Settings, SMSProvider
from openvoice.notifications.base import BaseEmailProvider, BaseSMSProvider
from openvoice.notifications.providers.resend_email import ResendEmailProvider
from openvoice.notifications.providers.twilio_sms import TwilioSMSProvider


def get_sms_provider(settings: Settings) -> BaseSMSProvider:
    """Build the SMS provider configured in `settings`."""
    if settings.sms_provider is SMSProvider.TWILIO:
        if not (settings.twilio_account_sid and settings.twilio_auth_token):
            raise RuntimeError(
                "TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN are required when SMS_PROVIDER=twilio"
            )
        if not settings.twilio_from_number:
            raise RuntimeError("TWILIO_FROM_NUMBER is required when SMS_PROVIDER=twilio")
        return TwilioSMSProvider(
            account_sid=settings.twilio_account_sid,
            auth_token=settings.twilio_auth_token,
            from_number=settings.twilio_from_number,
            timeout_seconds=settings.notification_request_timeout_seconds,
            max_retries=settings.notification_max_retries,
        )

    raise ValueError(f"Unsupported SMS provider: {settings.sms_provider}")  # pragma: no cover


def get_email_provider(settings: Settings) -> BaseEmailProvider:
    """Build the email provider configured in `settings`."""
    if settings.email_provider is EmailProvider.RESEND:
        if not (settings.resend_api_key and settings.resend_from_email):
            raise RuntimeError(
                "RESEND_API_KEY and RESEND_FROM_EMAIL are required when EMAIL_PROVIDER=resend"
            )
        return ResendEmailProvider(
            api_key=settings.resend_api_key,
            from_email=settings.resend_from_email,
            timeout_seconds=settings.notification_request_timeout_seconds,
            max_retries=settings.notification_max_retries,
        )

    raise ValueError(f"Unsupported email provider: {settings.email_provider}")  # pragma: no cover
