"""FastAPI dependency providers.

Every provider factory call goes through here, so route handlers depend
on abstractions (`BaseCalendarProvider`, etc.), never a concrete class,
matching the rest of the codebase's pluggable-provider pattern.
"""

from functools import lru_cache

from fastapi import Depends

from openvoice.booking.service import BookingService
from openvoice.calendar.factory import get_calendar_provider
from openvoice.config import Settings, get_settings
from openvoice.crm.service import CRMService
from openvoice.notifications.factory import get_email_provider, get_sms_provider

__all__ = [
    "get_booking_service",
    "get_crm_service",
    "get_settings_dependency",
]


def get_settings_dependency() -> Settings:
    return get_settings()


@lru_cache
def get_crm_service() -> CRMService:
    return CRMService()


def get_booking_service(
    settings: Settings = Depends(get_settings_dependency),
) -> BookingService:
    """Build a `BookingService`. Notification providers are optional: if their
    credentials aren't configured, booking still works, just without
    SMS/email confirmations (logged, not raised, at send time).
    """
    calendar = get_calendar_provider(settings)

    try:
        sms = get_sms_provider(settings)
    except RuntimeError:
        sms = None

    try:
        email = get_email_provider(settings)
    except RuntimeError:
        email = None

    return BookingService(
        calendar=calendar,
        sms=sms,
        email=email,
        business_hours_start=settings.booking_business_hours_start,
        business_hours_end=settings.booking_business_hours_end,
        default_duration_minutes=settings.booking_default_duration_minutes,
    )
