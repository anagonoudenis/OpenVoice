"""Structured JSON logging setup (structlog, routed through stdlib logging).

Call :func:`configure_logging` once at process startup. Everywhere else,
get a logger with ``structlog.get_logger(__name__)`` and bind a
``call_id`` (or other correlation id) as early as possible so it flows
through every log line for that request/call.

Logging is routed through the stdlib ``logging`` module (not
``structlog.PrintLoggerFactory``) so that third-party libraries using
plain stdlib logging (uvicorn, sqlalchemy, ...) are rendered through the
same formatter as our own structured logs.
"""

import logging
import sys

import structlog

from openvoice.config import Environment

_SHARED_PROCESSORS: list[structlog.types.Processor] = [
    structlog.contextvars.merge_contextvars,
    structlog.stdlib.add_log_level,
    structlog.stdlib.add_logger_name,
    structlog.processors.TimeStamper(fmt="iso"),
    structlog.processors.StackInfoRenderer(),
    structlog.processors.format_exc_info,
]


def configure_logging(*, environment: Environment, log_level: str = "INFO") -> None:
    """Configure structlog + stdlib logging for the process.

    In local/test environments logs are rendered human-readable; in
    staging/production they are emitted as single-line JSON for ingestion
    by log aggregators.
    """
    renderer: structlog.types.Processor
    if environment in (Environment.LOCAL, Environment.TEST):
        renderer = structlog.dev.ConsoleRenderer()
    else:
        renderer = structlog.processors.JSONRenderer()

    structlog.configure(
        processors=[
            *_SHARED_PROCESSORS,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            logging.getLevelNamesMapping().get(log_level.upper(), logging.INFO)
        ),
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        processors=[structlog.stdlib.ProcessorFormatter.remove_processors_meta, renderer],
        foreign_pre_chain=_SHARED_PROCESSORS,
    )
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.handlers = [handler]
    root_logger.setLevel(log_level.upper())
