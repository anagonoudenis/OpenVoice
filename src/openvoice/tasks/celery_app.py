"""Celery application instance.

Configured from `openvoice.config.Settings` (Redis as both broker and
result backend) — a single source of truth for configuration, consistent
with the rest of the app. Run a worker with:

    uv run celery -A openvoice.tasks.celery_app worker --loglevel=info
"""

from celery import Celery

from openvoice.config import get_settings


def create_celery_app() -> Celery:
    """Build the Celery app. Called once at import time below."""
    settings = get_settings()
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
    )
    return app


celery_app = create_celery_app()
