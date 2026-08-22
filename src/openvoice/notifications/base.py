"""Abstract SMS/email provider interfaces."""

from abc import ABC, abstractmethod


class NotificationError(Exception):
    """Raised when a notification provider fails after exhausting its retry budget."""


class BaseSMSProvider(ABC):
    """Abstract SMS backend."""

    @abstractmethod
    async def send_sms(self, *, to: str, body: str) -> None:
        """Send an SMS. Raises `NotificationError` on unrecoverable failure."""
        raise NotImplementedError


class BaseEmailProvider(ABC):
    """Abstract email backend."""

    @abstractmethod
    async def send_email(self, *, to: str, subject: str, body: str) -> None:
        """Send an email. Raises `NotificationError` on unrecoverable failure."""
        raise NotImplementedError
