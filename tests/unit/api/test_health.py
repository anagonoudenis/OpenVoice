"""Tests for the /health dependency checks, mocking DB/Redis/network."""

import asyncio
from unittest.mock import AsyncMock, Mock

import pytest

from openvoice.api.routers import health
from openvoice.config import Settings, get_settings


def _settings(monkeypatch: pytest.MonkeyPatch) -> Settings:
    monkeypatch.setenv("SECRET_KEY", "test-secret-key-please-ignore")
    monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://test:test@localhost:5432/test")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    get_settings.cache_clear()
    return get_settings()


async def test_check_database_healthy() -> None:
    db_session = AsyncMock()
    result = await health._check_database(db_session)
    assert result.healthy is True
    assert result.required is True


async def test_check_database_unhealthy_on_error() -> None:
    db_session = AsyncMock()
    db_session.execute.side_effect = RuntimeError("connection refused")
    result = await health._check_database(db_session)
    assert result.healthy is False
    assert "connection refused" in (result.detail or "")


async def test_check_redis_healthy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _settings(monkeypatch)
    fake_client = AsyncMock()
    monkeypatch.setattr(health.redis, "from_url", lambda *a, **kw: fake_client)

    result = await health._check_redis(settings)

    assert result.healthy is True
    fake_client.ping.assert_awaited_once()
    fake_client.aclose.assert_awaited_once()


async def test_check_redis_unhealthy_on_error(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _settings(monkeypatch)
    fake_client = AsyncMock()
    fake_client.ping.side_effect = RuntimeError("no route to host")
    monkeypatch.setattr(health.redis, "from_url", lambda *a, **kw: fake_client)

    result = await health._check_redis(settings)

    assert result.healthy is False
    fake_client.aclose.assert_awaited_once()


async def test_check_livekit_not_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LIVEKIT_URL", raising=False)
    result = await health._check_livekit()
    assert result.healthy is True
    assert result.required is False
    assert result.detail == "not configured"


async def test_check_livekit_healthy_when_reachable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LIVEKIT_URL", "ws://localhost:7880")

    writer = Mock()
    writer.close = Mock()  # real asyncio.StreamWriter.close() is synchronous
    writer.wait_closed = AsyncMock()

    async def fake_open_connection(host: str, port: int) -> tuple[object, AsyncMock]:
        return object(), writer

    monkeypatch.setattr(asyncio, "open_connection", fake_open_connection)

    result = await health._check_livekit()

    assert result.healthy is True
    assert result.required is False


async def test_check_livekit_unhealthy_when_unreachable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LIVEKIT_URL", "ws://localhost:7880")

    async def fake_open_connection(host: str, port: int) -> tuple[object, object]:
        raise ConnectionRefusedError("refused")

    monkeypatch.setattr(asyncio, "open_connection", fake_open_connection)

    result = await health._check_livekit()

    assert result.healthy is False
    assert result.required is False


async def test_check_livekit_unparseable_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LIVEKIT_URL", "not a url")
    result = await health._check_livekit()
    assert result.healthy is False
