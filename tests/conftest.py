"""Shared pytest fixtures."""

import os
from collections.abc import Iterator

import pytest

# Set before any `openvoice.*` module is imported (including during test
# collection), so that Settings() validation never blocks importing the
# app. Individual tests still override these via the `required_env`
# fixture or monkeypatch when they need specific values.
os.environ.setdefault("SECRET_KEY", "test-secret-key-please-ignore")
os.environ.setdefault("DATABASE_URL", "postgresql+asyncpg://test:test@localhost:5432/test")
os.environ.setdefault("ANTHROPIC_API_KEY", "test-anthropic-key")

from openvoice.config import Settings, get_settings


@pytest.fixture
def required_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Set the minimal env vars needed for ``Settings`` to validate."""
    monkeypatch.setenv("SECRET_KEY", "test-secret-key-please-ignore")
    monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://test:test@localhost:5432/test")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-anthropic-key")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def settings(required_env: None) -> Settings:
    """A validated Settings instance built from the minimal test env."""
    return get_settings()
