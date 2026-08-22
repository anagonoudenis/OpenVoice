"""Health check endpoint: verifies the service and its dependencies are reachable."""

import asyncio
import os
from urllib.parse import urlparse

import redis.asyncio as redis
import structlog
from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from openvoice.api.dependencies import get_settings_dependency
from openvoice.api.schemas import HealthDependency, HealthResponse
from openvoice.config import Settings
from openvoice.db.session import get_db_session

logger = structlog.get_logger(__name__)

router = APIRouter(tags=["health"])

_TCP_CHECK_TIMEOUT_SECONDS = 3.0


async def _check_database(db_session: AsyncSession) -> HealthDependency:
    try:
        await db_session.execute(text("SELECT 1"))
    except Exception as exc:  # any DB failure means "unhealthy", not a crashed endpoint
        logger.error("health_check_database_failed", error=str(exc))
        return HealthDependency(name="database", healthy=False, required=True, detail=str(exc))
    return HealthDependency(name="database", healthy=True, required=True)


async def _check_redis(settings: Settings) -> HealthDependency:
    client = redis.from_url(settings.redis_url, socket_connect_timeout=_TCP_CHECK_TIMEOUT_SECONDS)
    try:
        await client.ping()
    except Exception as exc:
        logger.error("health_check_redis_failed", error=str(exc))
        return HealthDependency(name="redis", healthy=False, required=True, detail=str(exc))
    finally:
        # redis-py's bundled type stubs lag the runtime API: `aclose()`
        # exists and is the non-deprecated method (`close()` is deprecated
        # since redis-py 5.0.1), but isn't declared on the stub.
        await client.aclose()  # type: ignore[attr-defined]
    return HealthDependency(name="redis", healthy=True, required=True)


async def _check_livekit() -> HealthDependency:
    """Best-effort TCP reachability check.

    `LIVEKIT_URL` is read directly from the environment (LiveKit's own SDK
    env var, not part of `Settings` -- see `openvoice.telephony.worker`'s
    module docstring) and is optional: the API process doesn't itself
    need LiveKit, only the telephony worker does, so this is informational
    and never marked `required`.
    """
    livekit_url = os.environ.get("LIVEKIT_URL")
    if not livekit_url:
        return HealthDependency(
            name="livekit", healthy=True, required=False, detail="not configured"
        )

    parsed = urlparse(livekit_url.replace("ws://", "http://").replace("wss://", "https://"))
    host = parsed.hostname
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    if host is None:
        return HealthDependency(
            name="livekit",
            healthy=False,
            required=False,
            detail=f"unparseable LIVEKIT_URL: {livekit_url}",
        )

    try:
        _reader, writer = await asyncio.wait_for(
            asyncio.open_connection(host, port), timeout=_TCP_CHECK_TIMEOUT_SECONDS
        )
        writer.close()
        await writer.wait_closed()
    except Exception as exc:
        logger.error("health_check_livekit_failed", error=str(exc))
        return HealthDependency(name="livekit", healthy=False, required=False, detail=str(exc))
    return HealthDependency(name="livekit", healthy=True, required=False)


@router.get("/health", response_model=HealthResponse)
async def health(
    db_session: AsyncSession = Depends(get_db_session),
    settings: Settings = Depends(get_settings_dependency),
) -> HealthResponse:
    """Liveness + dependency check: DB and Redis are required for `status`
    to be "ok"; LiveKit reachability is reported but informational only.
    """
    dependencies = [
        await _check_database(db_session),
        await _check_redis(settings),
        await _check_livekit(),
    ]
    overall = "ok" if all(d.healthy for d in dependencies if d.required) else "degraded"
    return HealthResponse(status=overall, dependencies=dependencies)
