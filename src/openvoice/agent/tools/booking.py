"""Booking tools: lets the conversational agent take real booking actions
(check availability, book/cancel/reschedule, list upcoming appointments)
via native LLM tool-calling, instead of only talking about them.

Mutating tools (`book_appointment`, `cancel_appointment`,
`reschedule_appointment`) require `caller_confirmed: true` in their
arguments -- enforced here in code, not just requested via prompt text:
the dispatcher rejects the call outright if it's missing or false,
forcing the model to have asked the caller to confirm the exact
time/action before it can actually execute it. This can't fully prevent
a model from lying about having asked (no code-level guardrail can,
short of a separate human-in-the-loop UI), but it does prevent the most
common failure mode -- the model silently booking/cancelling the instant
it resolves a date, based on a possibly-mistranscribed request -- from
being possible without at least one extra explicit step.

Cancel/reschedule also re-check that the resolved appointment actually
belongs to the caller's own `client_id` before acting on it: nothing
should let a caller mutate an appointment merely by guessing or
mentioning someone else's ID.
"""

import json
import uuid
from datetime import UTC, datetime

import structlog
from pydantic import BaseModel, ValidationError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from openvoice.agent.tools.base import ToolExecutionResult, ToolExecutor
from openvoice.booking.service import BookingError, BookingService
from openvoice.calendar.base import CalendarError
from openvoice.db.models import Appointment, Client
from openvoice.llm.base import ToolCall, ToolDefinition
from openvoice.notifications.base import NotificationError

logger = structlog.get_logger(__name__)

BOOKING_SYSTEM_PROMPT_ADDENDUM = """

You can take real booking actions using the tools available to you:
- check_availability and list_my_appointments are informational -- use them freely.
- book_appointment, cancel_appointment, and reschedule_appointment actually change \
the calendar. Before calling one of these, say the exact date, time, and action out \
loud and wait for the caller to clearly confirm (e.g. "yes", "that's right"). Only \
then call the tool with caller_confirmed set to true. Never guess a confirmation the \
caller didn't give, and never call a mutating tool speculatively.
"""

_ISO_DESCRIPTION = "Timezone-aware ISO 8601 datetime, e.g. 2026-09-01T14:00:00+02:00"


class _CheckAvailabilityArgs(BaseModel):
    duration_minutes: int | None = None
    search_days: int | None = None


class _BookAppointmentArgs(BaseModel):
    start: str
    end: str
    notes: str | None = None
    caller_confirmed: bool = False


class _AppointmentIdArgs(BaseModel):
    appointment_id: str
    caller_confirmed: bool = False


class _RescheduleAppointmentArgs(BaseModel):
    appointment_id: str
    new_start: str
    new_end: str
    caller_confirmed: bool = False


def booking_tool_definitions() -> list[ToolDefinition]:
    """Tool schemas exposed to the model when booking is configured for this call."""
    return [
        ToolDefinition(
            name="check_availability",
            description="Find open appointment slots within business hours.",
            parameters={
                "type": "object",
                "properties": {
                    "duration_minutes": {
                        "type": "integer",
                        "description": "Desired appointment length in minutes.",
                    },
                    "search_days": {
                        "type": "integer",
                        "description": "How many days ahead to search.",
                    },
                },
                "required": [],
            },
        ),
        ToolDefinition(
            name="list_my_appointments",
            description=(
                "List the caller's upcoming (non-cancelled) appointments, with their IDs."
            ),
            parameters={"type": "object", "properties": {}, "required": []},
        ),
        ToolDefinition(
            name="book_appointment",
            description=(
                "Book a new appointment. Only call this after the caller has explicitly "
                "confirmed the date and time out loud."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "start": {"type": "string", "description": _ISO_DESCRIPTION},
                    "end": {"type": "string", "description": _ISO_DESCRIPTION},
                    "notes": {"type": "string", "description": "Optional note about the visit."},
                    "caller_confirmed": {
                        "type": "boolean",
                        "description": "True only if the caller just confirmed this exact booking.",
                    },
                },
                "required": ["start", "end", "caller_confirmed"],
            },
        ),
        ToolDefinition(
            name="cancel_appointment",
            description=(
                "Cancel an existing appointment. Call list_my_appointments first if you "
                "don't already have its ID. Only call this after the caller has explicitly "
                "confirmed the cancellation out loud."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "appointment_id": {"type": "string"},
                    "caller_confirmed": {
                        "type": "boolean",
                        "description": "True only if the caller just confirmed this cancellation.",
                    },
                },
                "required": ["appointment_id", "caller_confirmed"],
            },
        ),
        ToolDefinition(
            name="reschedule_appointment",
            description=(
                "Move an existing appointment to a new time. Call list_my_appointments "
                "first if you don't already have its ID. Only call this after the caller "
                "has explicitly confirmed the new date and time out loud."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "appointment_id": {"type": "string"},
                    "new_start": {"type": "string", "description": _ISO_DESCRIPTION},
                    "new_end": {"type": "string", "description": _ISO_DESCRIPTION},
                    "caller_confirmed": {
                        "type": "boolean",
                        "description": "True only if the caller just confirmed this new time.",
                    },
                },
                "required": ["appointment_id", "new_start", "new_end", "caller_confirmed"],
            },
        ),
    ]


def make_booking_tool_executor(
    *,
    booking_service: BookingService,
    sessionmaker: async_sessionmaker[AsyncSession],
    client_id: uuid.UUID,
    call_id: uuid.UUID | None,
) -> ToolExecutor:
    """Build a `ToolExecutor` bound to one call's resolved caller and DB access."""

    async def execute(tool_call: ToolCall) -> ToolExecutionResult:
        try:
            async with sessionmaker() as db_session:
                if tool_call.name == "check_availability":
                    return await _check_availability(booking_service, tool_call.arguments)
                if tool_call.name == "list_my_appointments":
                    return await _list_my_appointments(booking_service, db_session, client_id)
                if tool_call.name == "book_appointment":
                    return await _book_appointment(
                        booking_service, db_session, tool_call.arguments, client_id, call_id
                    )
                if tool_call.name == "cancel_appointment":
                    return await _cancel_appointment(
                        booking_service, db_session, tool_call.arguments, client_id
                    )
                if tool_call.name == "reschedule_appointment":
                    return await _reschedule_appointment(
                        booking_service, db_session, tool_call.arguments, client_id
                    )
        except (BookingError, CalendarError, NotificationError) as exc:
            logger.error("booking_tool_failed", tool=tool_call.name, error=str(exc))
            return f"Could not complete this action: {exc}", True

        logger.error("unknown_booking_tool_requested", tool=tool_call.name)
        return f"Unknown tool: {tool_call.name}", True

    return execute


async def _check_availability(
    booking_service: BookingService, raw_arguments: dict[str, object]
) -> ToolExecutionResult:
    try:
        args = _CheckAvailabilityArgs.model_validate(raw_arguments)
    except ValidationError as exc:
        return f"Invalid arguments: {exc}", True

    slots = await booking_service.find_available_slots(
        search_from=datetime.now(UTC),
        search_days=args.search_days or 7,
        duration_minutes=args.duration_minutes,
    )
    if not slots:
        return "No available slots found in the searched window.", False
    return (
        json.dumps([{"start": s.start.isoformat(), "end": s.end.isoformat()} for s in slots]),
        False,
    )


async def _list_my_appointments(
    booking_service: BookingService, db_session: AsyncSession, client_id: uuid.UUID
) -> ToolExecutionResult:
    appointments = await booking_service.list_upcoming_appointments(
        db_session=db_session, client_id=client_id
    )
    if not appointments:
        return "No upcoming appointments.", False
    return (
        json.dumps(
            [
                {
                    "appointment_id": str(a.id),
                    "start": a.starts_at.isoformat(),
                    "end": a.ends_at.isoformat(),
                    "status": a.status.value,
                }
                for a in appointments
            ]
        ),
        False,
    )


async def _book_appointment(
    booking_service: BookingService,
    db_session: AsyncSession,
    raw_arguments: dict[str, object],
    client_id: uuid.UUID,
    call_id: uuid.UUID | None,
) -> ToolExecutionResult:
    try:
        args = _BookAppointmentArgs.model_validate(raw_arguments)
    except ValidationError as exc:
        return f"Invalid arguments: {exc}", True

    if not args.caller_confirmed:
        return (
            "Not booked: you must say the exact date and time out loud and get the "
            "caller's explicit confirmation before calling this tool with "
            "caller_confirmed set to true.",
            True,
        )

    start, end = _parse_datetime(args.start), _parse_datetime(args.end)
    if start is None or end is None:
        return "Invalid start/end: must be timezone-aware ISO 8601 datetimes.", True

    client = await db_session.get(Client, client_id)
    if client is None:
        return "Caller record not found.", True

    appointment = await booking_service.book_appointment(
        db_session=db_session,
        client=client,
        start=start,
        end=end,
        call_id=call_id,
        notes=args.notes,
    )
    return (
        f"Booked. appointment_id={appointment.id}, start={appointment.starts_at.isoformat()}, "
        f"end={appointment.ends_at.isoformat()}.",
        False,
    )


async def _cancel_appointment(
    booking_service: BookingService,
    db_session: AsyncSession,
    raw_arguments: dict[str, object],
    client_id: uuid.UUID,
) -> ToolExecutionResult:
    try:
        args = _AppointmentIdArgs.model_validate(raw_arguments)
    except ValidationError as exc:
        return f"Invalid arguments: {exc}", True

    if not args.caller_confirmed:
        return (
            "Not cancelled: get the caller's explicit confirmation before calling this "
            "tool with caller_confirmed set to true.",
            True,
        )

    appointment = await _get_owned_appointment(db_session, args.appointment_id, client_id)
    if appointment is None:
        return "No such appointment for this caller.", True

    await booking_service.cancel_appointment(db_session=db_session, appointment=appointment)
    return f"Cancelled appointment {appointment.id}.", False


async def _reschedule_appointment(
    booking_service: BookingService,
    db_session: AsyncSession,
    raw_arguments: dict[str, object],
    client_id: uuid.UUID,
) -> ToolExecutionResult:
    try:
        args = _RescheduleAppointmentArgs.model_validate(raw_arguments)
    except ValidationError as exc:
        return f"Invalid arguments: {exc}", True

    if not args.caller_confirmed:
        return (
            "Not rescheduled: get the caller's explicit confirmation before calling this "
            "tool with caller_confirmed set to true.",
            True,
        )

    new_start, new_end = _parse_datetime(args.new_start), _parse_datetime(args.new_end)
    if new_start is None or new_end is None:
        return "Invalid new_start/new_end: must be timezone-aware ISO 8601 datetimes.", True

    appointment = await _get_owned_appointment(db_session, args.appointment_id, client_id)
    if appointment is None:
        return "No such appointment for this caller.", True

    updated = await booking_service.reschedule_appointment(
        db_session=db_session, appointment=appointment, new_start=new_start, new_end=new_end
    )
    return (
        f"Rescheduled. appointment_id={updated.id}, start={updated.starts_at.isoformat()}, "
        f"end={updated.ends_at.isoformat()}.",
        False,
    )


async def _get_owned_appointment(
    db_session: AsyncSession, appointment_id_raw: str, client_id: uuid.UUID
) -> Appointment | None:
    """Look up an appointment by ID, but only if it belongs to this caller."""
    try:
        appointment_id = uuid.UUID(appointment_id_raw)
    except ValueError:
        return None

    appointment = await db_session.get(Appointment, appointment_id)
    if appointment is None or appointment.client_id != client_id:
        return None
    return appointment


def _parse_datetime(raw: str) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(raw)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        # A naive datetime is ambiguous (which timezone?) -- reject rather
        # than silently assume one and risk booking the wrong hour.
        return None
    return parsed
