"""Calendar provider factory. Selects the implementation from `Settings.calendar_provider`."""

from openvoice.calendar.base import BaseCalendarProvider
from openvoice.calendar.providers.google_calendar import GoogleCalendarProvider
from openvoice.config import CalendarProvider, Settings


def get_calendar_provider(settings: Settings) -> BaseCalendarProvider:
    """Build the calendar provider configured in `settings`."""
    if settings.calendar_provider is CalendarProvider.GOOGLE:
        return GoogleCalendarProvider.from_settings(settings)

    raise ValueError(  # pragma: no cover
        f"Unsupported calendar provider: {settings.calendar_provider}"
    )
