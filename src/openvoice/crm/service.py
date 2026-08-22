"""CRM operations shared by the call pipeline and the REST API."""

import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from openvoice.db.models import Call, Client


class CRMService:
    """Client lookup/creation and call-history queries."""

    async def get_or_create_client(self, *, db_session: AsyncSession, phone_number: str) -> Client:
        """Look up a client by phone number, creating one if this is a new caller."""
        result = await db_session.execute(select(Client).where(Client.phone_number == phone_number))
        client = result.scalar_one_or_none()
        if client is not None:
            return client

        client = Client(phone_number=phone_number)
        db_session.add(client)
        await db_session.commit()
        await db_session.refresh(client)
        return client

    async def get_call_history(
        self, *, db_session: AsyncSession, client_id: uuid.UUID, limit: int = 50
    ) -> list[Call]:
        """Most recent calls for a client, newest first."""
        result = await db_session.execute(
            select(Call)
            .where(Call.client_id == client_id)
            .order_by(Call.started_at.desc())
            .limit(limit)
        )
        return list(result.scalars().all())

    async def get_call_with_transcript(
        self, *, db_session: AsyncSession, call_id: uuid.UUID
    ) -> Call | None:
        """A single call with its full transcript eagerly loaded."""
        result = await db_session.execute(
            select(Call).where(Call.id == call_id).options(selectinload(Call.transcripts))
        )
        return result.scalar_one_or_none()
