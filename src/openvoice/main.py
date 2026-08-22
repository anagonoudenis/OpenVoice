"""FastAPI application entrypoint."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from openvoice.api.routers import appointments, calls, clients, health
from openvoice.booking.service import BookingError
from openvoice.calendar.base import CalendarError
from openvoice.config import get_settings
from openvoice.logging import configure_logging
from openvoice.notifications.base import NotificationError

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

    app.include_router(health.router)
    app.include_router(clients.router)
    app.include_router(calls.router)
    app.include_router(appointments.router)

    @app.exception_handler(CalendarError)
    async def _calendar_error_handler(_request: Request, exc: CalendarError) -> JSONResponse:
        logger.error("calendar_error", error=str(exc))
        return JSONResponse(status_code=503, content={"detail": f"Calendar unavailable: {exc}"})

    @app.exception_handler(NotificationError)
    async def _notification_error_handler(
        _request: Request, exc: NotificationError
    ) -> JSONResponse:
        logger.error("notification_error", error=str(exc))
        return JSONResponse(
            status_code=502, content={"detail": f"Notification delivery failed: {exc}"}
        )

    @app.exception_handler(BookingError)
    async def _booking_error_handler(_request: Request, exc: BookingError) -> JSONResponse:
        return JSONResponse(status_code=409, content={"detail": str(exc)})

    return app


app = create_app()
