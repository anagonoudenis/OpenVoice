"""FastAPI application entrypoint."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI

from openvoice.config import get_settings
from openvoice.logging import configure_logging

logger = structlog.get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Application startup/shutdown hook."""
    settings = get_settings()
    configure_logging(environment=settings.environment, log_level=settings.log_level)
    logger.info("app_starting", environment=settings.environment.value)
    yield
    logger.info("app_stopping")


def create_app() -> FastAPI:
    """Application factory. Settings are validated as a side effect of import."""
    settings = get_settings()

    app = FastAPI(
        title="OpenVoice",
        description="Open-source AI voice agent for calls, booking, and support.",
        version="0.1.0",
        debug=settings.debug,
        lifespan=lifespan,
    )

    @app.get("/health", tags=["health"])
    async def health() -> dict[str, str]:
        """Liveness probe.

        Deep dependency checks (DB, Redis, LiveKit) are added once those
        clients exist; see docs/ROADMAP.md.
        """
        return {"status": "ok"}

    return app


app = create_app()
