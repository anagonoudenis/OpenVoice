"""Integration tests for the REST API against a real PostgreSQL database.

Uses FastAPI's `dependency_overrides` to point `get_db_session` at the
real test database (`db_session` fixture) and `get_booking_service` at a
fake calendar/notification stack, so these tests never need real Google
Calendar/Twilio/Resend credentials.
"""

from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from openvoice.api.dependencies import get_booking_service
from openvoice.booking.service import BookingService
from openvoice.db.models import Call, CallDirection, CallStatus, CallTranscript, SpeakerRole
from openvoice.db.session import get_db_session
from openvoice.main import app
from tests.unit.booking.fakes import FakeCalendarProvider, FakeEmailProvider, FakeSMSProvider

pytestmark = pytest.mark.integration


@pytest.fixture
async def client(db_session: AsyncSession) -> AsyncIterator[AsyncClient]:
    async def _override_db_session() -> AsyncIterator[AsyncSession]:
        yield db_session

    def _override_booking_service() -> BookingService:
        return BookingService(
            calendar=FakeCalendarProvider(), sms=FakeSMSProvider(), email=FakeEmailProvider()
        )

    app.dependency_overrides[get_db_session] = _override_db_session
    app.dependency_overrides[get_booking_service] = _override_booking_service
    try:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as async_client:
            yield async_client
    finally:
        app.dependency_overrides.clear()


async def test_list_clients_empty(client: AsyncClient) -> None:
    response = await client.get("/clients")
    assert response.status_code == 200
    assert response.json() == []


async def test_get_client_not_found(client: AsyncClient) -> None:
    response = await client.get("/clients/00000000-0000-0000-0000-000000000000")
    assert response.status_code == 404


async def test_client_and_call_history_flow(client: AsyncClient, db_session: AsyncSession) -> None:
    call = Call(
        livekit_room_name="room-api-1",
        direction=CallDirection.INBOUND,
        status=CallStatus.COMPLETED,
    )
    call.transcripts.append(CallTranscript(sequence=1, role=SpeakerRole.CALLER, text="Hi"))
    db_session.add(call)
    await db_session.commit()

    call_response = await client.get(f"/calls/{call.id}")
    assert call_response.status_code == 200
    body = call_response.json()
    assert body["id"] == str(call.id)
    assert body["transcripts"][0]["text"] == "Hi"


async def test_call_not_found(client: AsyncClient) -> None:
    response = await client.get("/calls/00000000-0000-0000-0000-000000000000")
    assert response.status_code == 404


async def test_book_cancel_appointment_flow(client: AsyncClient) -> None:
    start = (datetime.now(UTC) + timedelta(days=1)).replace(microsecond=0)
    end = start + timedelta(minutes=30)

    create_response = await client.post(
        "/appointments",
        json={
            "phone_number": "+15559990010",
            "starts_at": start.isoformat(),
            "ends_at": end.isoformat(),
        },
    )
    assert create_response.status_code == 201
    appointment = create_response.json()
    assert appointment["status"] == "confirmed"

    cancel_response = await client.delete(f"/appointments/{appointment['id']}")
    assert cancel_response.status_code == 204


async def test_health_endpoint_reports_database_status(client: AsyncClient) -> None:
    response = await client.get("/health")
    assert response.status_code == 200
    body = response.json()
    dependency_names = {d["name"] for d in body["dependencies"]}
    assert {"database", "redis", "livekit"} <= dependency_names
    database = next(d for d in body["dependencies"] if d["name"] == "database")
    assert database["healthy"] is True


async def test_available_slots_endpoint(client: AsyncClient) -> None:
    response = await client.get("/appointments/available-slots?duration_minutes=30&search_days=2")
    assert response.status_code == 200
    slots = response.json()
    assert len(slots) > 0
    assert "start" in slots[0]
    assert "end" in slots[0]
