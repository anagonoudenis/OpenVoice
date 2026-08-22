"""Client endpoints."""

import uuid

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from openvoice.api.dependencies import get_crm_service
from openvoice.api.schemas import CallOut, ClientOut
from openvoice.crm.service import CRMService
from openvoice.db.models import Call, Client
from openvoice.db.session import get_db_session

router = APIRouter(prefix="/clients", tags=["clients"])


@router.get("", response_model=list[ClientOut])
async def list_clients(
    limit: int = 50, offset: int = 0, db_session: AsyncSession = Depends(get_db_session)
) -> list[Client]:
    """List clients, newest first."""
    result = await db_session.execute(
        select(Client).order_by(Client.created_at.desc()).limit(limit).offset(offset)
    )
    return list(result.scalars().all())


@router.get("/{client_id}", response_model=ClientOut)
async def get_client(
    client_id: uuid.UUID, db_session: AsyncSession = Depends(get_db_session)
) -> Client:
    client = await db_session.get(Client, client_id)
    if client is None:
        raise HTTPException(status_code=404, detail="Client not found")
    return client


@router.get("/{client_id}/calls", response_model=list[CallOut])
async def get_client_calls(
    client_id: uuid.UUID,
    db_session: AsyncSession = Depends(get_db_session),
    crm: CRMService = Depends(get_crm_service),
) -> list[Call]:
    """Call history for a client, newest first."""
    client = await db_session.get(Client, client_id)
    if client is None:
        raise HTTPException(status_code=404, detail="Client not found")
    return await crm.get_call_history(db_session=db_session, client_id=client_id)
