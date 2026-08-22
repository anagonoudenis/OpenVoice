"""Appointment endpoints: available slots, booking, cancellation, rescheduling."""

import uuid
from datetime import UTC, datetime

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from openvoice.api.dependencies import get_booking_service, get_crm_service, get_settings_dependency
from openvoice.api.schemas import (
    AppointmentCreateRequest,
    AppointmentOut,
    AppointmentRescheduleRequest,
    TimeSlotOut,
)
from openvoice.booking.service import BookingError, BookingService
from openvoice.config import Settings
from openvoice.crm.service import CRMService
from openvoice.db.models import Appointment
from openvoice.db.session import get_db_session

router = APIRouter(prefix="/appointments", tags=["appointments"])


@router.get("/available-slots", response_model=list[TimeSlotOut])
async def available_slots(
    duration_minutes: int | None = None,
    search_days: int | None = None,
    booking: BookingService = Depends(get_booking_service),
    settings: Settings = Depends(get_settings_dependency),
) -> list[TimeSlotOut]:
    """Propose available slots within business hours."""
    slots = await booking.find_available_slots(
        search_from=datetime.now(UTC),
        search_days=search_days or settings.booking_search_days_ahead,
        duration_minutes=duration_minutes,
    )
    return [TimeSlotOut(start=slot.start, end=slot.end) for slot in slots]


@router.post("", response_model=AppointmentOut, status_code=201)
async def create_appointment(
    payload: AppointmentCreateRequest,
    db_session: AsyncSession = Depends(get_db_session),
    booking: BookingService = Depends(get_booking_service),
    crm: CRMService = Depends(get_crm_service),
) -> Appointment:
    """Book an appointment. The client is looked up (or created) by phone number."""
    client = await crm.get_or_create_client(
        db_session=db_session, phone_number=payload.phone_number
    )
    return await booking.book_appointment(
        db_session=db_session,
        client=client,
        start=payload.starts_at,
        end=payload.ends_at,
        notes=payload.notes,
    )


@router.delete("/{appointment_id}", status_code=204)
async def cancel_appointment(
    appointment_id: uuid.UUID,
    db_session: AsyncSession = Depends(get_db_session),
    booking: BookingService = Depends(get_booking_service),
) -> None:
    appointment = await db_session.get(Appointment, appointment_id)
    if appointment is None:
        raise HTTPException(status_code=404, detail="Appointment not found")
    await booking.cancel_appointment(db_session=db_session, appointment=appointment)


@router.patch("/{appointment_id}", response_model=AppointmentOut)
async def reschedule_appointment(
    appointment_id: uuid.UUID,
    payload: AppointmentRescheduleRequest,
    db_session: AsyncSession = Depends(get_db_session),
    booking: BookingService = Depends(get_booking_service),
) -> Appointment:
    appointment = await db_session.get(Appointment, appointment_id)
    if appointment is None:
        raise HTTPException(status_code=404, detail="Appointment not found")
    try:
        return await booking.reschedule_appointment(
            db_session=db_session,
            appointment=appointment,
            new_start=payload.starts_at,
            new_end=payload.ends_at,
        )
    except BookingError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
