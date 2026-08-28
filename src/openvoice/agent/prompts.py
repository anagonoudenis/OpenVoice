"""System prompt resolution.

The system prompt is never hardcoded into business logic: it's either
supplied verbatim via `Settings.agent_system_prompt` (per-company config,
e.g. one `.env` per deployment) or falls back to a generic default filled
in with `Settings.agent_company_name`.
"""

from datetime import datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import structlog

from openvoice.config import Settings

logger = structlog.get_logger(__name__)

_DEFAULT_SYSTEM_PROMPT_TEMPLATE = """\
You are a professional, friendly phone agent for {company_name}.

Your job on every call:
- Understand what the caller needs (booking an appointment, a support \
question, or an urgent issue) and help them directly when you can.
- Keep responses short and natural — this is a spoken conversation, not \
a chat window. One or two sentences per turn unless the caller asks for \
detail.
- Always reply in the same language the caller is speaking, even if these \
instructions are in English. Switch languages immediately if the caller \
switches.
- If you cannot resolve the caller's request, or they explicitly ask for \
a human, say so plainly and let the call be transferred.
- Never invent information (appointment slots, policies, prices) you \
don't actually have access to.
"""


def build_system_prompt(settings: Settings) -> str:
    """Resolve the system prompt to use for a call, per `settings`."""
    if settings.agent_system_prompt is not None:
        return settings.agent_system_prompt
    return _DEFAULT_SYSTEM_PROMPT_TEMPLATE.format(company_name=settings.agent_company_name)


def build_temporal_context(*, timezone: str, now: datetime | None = None) -> str:
    """A short block telling the model the current date/time, so it can
    resolve relative expressions ("tomorrow afternoon", "next Monday") the
    caller says into real, timezone-aware datetimes for booking tool
    calls. Falls back to UTC (logged, not raised) on a misconfigured
    timezone name -- a bad `BOOKING_TIMEZONE` value must not crash every
    single call.
    """
    try:
        tz = ZoneInfo(timezone)
    except ZoneInfoNotFoundError:
        logger.error("invalid_booking_timezone_falling_back_to_utc", timezone=timezone)
        tz = ZoneInfo("UTC")

    reference = (now or datetime.now(tz)).astimezone(tz)
    return (
        f"\nCurrent date and time: {reference.strftime('%A, %B %d, %Y, %H:%M')} "
        f"({reference.tzname()}, UTC offset {reference.strftime('%z')}). Use this to "
        'resolve relative dates and times the caller mentions (e.g. "tomorrow '
        'afternoon", "next Monday"), and always include this UTC offset in any '
        "datetime you pass to a tool.\n"
    )
