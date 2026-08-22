"""Tests for openvoice.logging.configure_logging."""

import json
import logging

import pytest
import structlog

from openvoice.config import Environment
from openvoice.logging import configure_logging


def test_configure_logging_local_renders_human_readable(
    capsys: pytest.CaptureFixture[str],
) -> None:
    configure_logging(environment=Environment.LOCAL, log_level="DEBUG")
    structlog.get_logger("test").info("hello", call_id="abc-123")
    captured = capsys.readouterr()
    assert "hello" in captured.out
    assert "abc-123" in captured.out


def test_configure_logging_production_renders_json(
    capsys: pytest.CaptureFixture[str],
) -> None:
    configure_logging(environment=Environment.PRODUCTION, log_level="INFO")
    structlog.get_logger("test").info("hello", call_id="abc-123")
    captured = capsys.readouterr()
    payload = json.loads(captured.out.strip().splitlines()[-1])
    assert payload["event"] == "hello"
    assert payload["call_id"] == "abc-123"
    assert payload["level"] == "info"


def test_configure_logging_sets_root_log_level() -> None:
    configure_logging(environment=Environment.TEST, log_level="WARNING")
    assert logging.getLogger().level == logging.WARNING
