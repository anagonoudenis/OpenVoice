"""Pydantic request/response schemas for the REST API."""

import uuid
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from openvoice.db.models import AppointmentStatus, CallDirection, CallStatus, SpeakerRole


class ClientOut(BaseModel):
    """A client, as returned by the API."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    phone_number: str
    full_name: str | None
    email: str | None
    notes: str | None
    created_at: datetime
    updated_at: datetime


class CallTranscriptOut(BaseModel):
    """One transcript turn, as returned by the API."""

    model_config = ConfigDict(from_attributes=True)

    sequence: int
    role: SpeakerRole
    text: str
    spoken_at: datetime


class CallOut(BaseModel):
    """A call, as returned by the API."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    client_id: uuid.UUID | None
    direction: CallDirection
    status: CallStatus
    started_at: datetime
    ended_at: datetime | None
    summary: str | None


class CallWithTranscriptOut(CallOut):
    """A call including its full transcript."""

    transcripts: list[CallTranscriptOut]


class AppointmentOut(BaseModel):
    """A booked appointment, as returned by the API."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    client_id: uuid.UUID
    call_id: uuid.UUID | None
    starts_at: datetime
    ends_at: datetime
    status: AppointmentStatus
    calendar_event_id: str | None
    notes: str | None


class TimeSlotOut(BaseModel):
    """A proposed/available appointment slot."""

    start: datetime
    end: datetime


class AppointmentCreateRequest(BaseModel):
    """Request body to book a new appointment."""

    phone_number: str = Field(
        ..., description="Caller's phone number; the client is looked up or created by this."
    )
    starts_at: datetime
    ends_at: datetime
    notes: str | None = None


class AppointmentRescheduleRequest(BaseModel):
    """Request body to move an existing appointment."""

    starts_at: datetime
    ends_at: datetime


class HealthDependency(BaseModel):
    """The health of one external dependency."""

    name: str
    healthy: bool
    required: bool
    detail: str | None = None


class HealthResponse(BaseModel):
    """The overall health of the service and its dependencies."""

    status: Literal["ok", "degraded"]
    dependencies: list[HealthDependency]
