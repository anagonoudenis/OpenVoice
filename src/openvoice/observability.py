"""Sentry error tracking setup.

This app runs as three separate OS processes -- the FastAPI API server
(`openvoice.main`), the LiveKit telephony worker
(`openvoice.telephony.worker`), and the Celery worker
(`openvoice.tasks.celery_app`) -- each of which can crash independently.
`configure_sentry` is the one place that knows how to initialize Sentry,
called from each entrypoint's own startup so none of them silently fails
without anyone finding out (`Settings.sentry_dsn` existed but nothing
called `sentry_sdk.init()` until this was written -- see CHANGELOG).
"""

import structlog

from openvoice.config import Settings

logger = structlog.get_logger(__name__)


def configure_sentry(settings: Settings) -> None:
    """Initialize Sentry error tracking, if `SENTRY_DSN` is configured.

    A no-op when unset, not an error: Sentry is optional instrumentation,
    never a hard requirement to run this app. Safe to call more than once
    per process (e.g. once per job in a warmed `livekit-agents` worker
    process) -- `sentry_sdk.init()` just reconfigures the global client.
    """
    if settings.sentry_dsn is None:
        return

    import sentry_sdk

    sentry_sdk.init(
        dsn=settings.sentry_dsn,
        environment=settings.environment.value,
        # Errors are always captured regardless of this rate; traces are
        # relatively cheap and genuinely useful for spotting latency
        # regressions in a latency-sensitive voice pipeline.
        traces_sample_rate=0.1,
    )
    logger.info("sentry_configured", environment=settings.environment.value)
