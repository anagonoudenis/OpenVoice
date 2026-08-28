"""Celery application instance.

Configured from `openvoice.config.Settings` (Redis as both broker and
result backend) — a single source of truth for configuration, consistent
with the rest of the app. Run a worker with:

    uv run celery -A openvoice.tasks.celery_app worker --loglevel=info
"""

from celery import Celery

from openvoice.config import get_settings
from openvoice.observability import configure_sentry


def create_celery_app() -> Celery:
    """Build the Celery app. Called once at import time below."""
    settings = get_settings()
    configure_sentry(settings)
    app = Celery(
        "openvoice",
        broker=settings.redis_url,
        backend=settings.redis_url,
        include=["openvoice.tasks.summarize_call"],
    )
    app.conf.update(
        task_serializer="json",
        result_serializer="json",
        accept_content=["json"],
        task_time_limit=120,
        task_soft_time_limit=90,
        # Celery/Kombu's own defaults retry a broker connection up to 100
        # times (~4s apiece) before giving up. `summarize_call_task.delay()`
        # is called synchronously from the live call-teardown path
        # (`openvoice.telephony.worker.finalize_call`) and is wrapped in a
        # try/except specifically so a broker outage can't break call
        # teardown -- but with the default retry budget, that `.delay()`
        # call itself could block for minutes before raising, stalling the
        # worker process from picking up its next call. The post-call
        # summary is best-effort (failure is only logged, never raised) --
        # one quick retry is enough to ride out a single transient blip
        # without turning a real Redis outage into a multi-minute hang.
        broker_connection_retry_on_startup=True,
        broker_connection_timeout=2.0,
        broker_connection_max_retries=1,
    )
    return app


celery_app = create_celery_app()
