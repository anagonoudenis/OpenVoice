"""Call endpoints."""

import uuid

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from openvoice.api.dependencies import get_crm_service
from openvoice.api.schemas import CallWithTranscriptOut
from openvoice.crm.service import CRMService
from openvoice.db.models import Call
from openvoice.db.session import get_db_session

router = APIRouter(prefix="/calls", tags=["calls"])


@router.get("/{call_id}", response_model=CallWithTranscriptOut)
async def get_call(
    call_id: uuid.UUID,
    db_session: AsyncSession = Depends(get_db_session),
    crm: CRMService = Depends(get_crm_service),
) -> Call:
    """A single call, including its full transcript."""
    call = await crm.get_call_with_transcript(db_session=db_session, call_id=call_id)
    if call is None:
        raise HTTPException(status_code=404, detail="Call not found")
    return call
