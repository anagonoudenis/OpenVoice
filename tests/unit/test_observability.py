"""Tests for Sentry setup."""

from unittest.mock import MagicMock

import pytest

from openvoice.config import Settings, get_settings
from openvoice.observability import configure_sentry


def _settings(**overrides: object) -> Settings:
    get_settings.cache_clear()
    defaults: dict[str, object] = {
        "secret_key": "test-secret-key-please-ignore",
        "database_url": "postgresql+asyncpg://test:test@localhost:5432/test",
        "anthropic_api_key": "test-key",
    }
    defaults.update(overrides)
    return Settings(**defaults)  # type: ignore[arg-type]


def test_configure_sentry_is_a_no_op_when_dsn_unset() -> None:
    # Must not raise, and must not require the `sentry_sdk` import to
    # succeed with a real network-capable init -- unset DSN means Sentry
    # is simply never touched.
    configure_sentry(_settings())


def test_configure_sentry_initializes_when_dsn_set(monkeypatch: pytest.MonkeyPatch) -> None:
    import sentry_sdk

    mock_init = MagicMock()
    monkeypatch.setattr(sentry_sdk, "init", mock_init)

    configure_sentry(_settings(sentry_dsn="https://example@o0.ingest.sentry.io/0"))

    mock_init.assert_called_once()
    assert mock_init.call_args.kwargs["dsn"] == "https://example@o0.ingest.sentry.io/0"
