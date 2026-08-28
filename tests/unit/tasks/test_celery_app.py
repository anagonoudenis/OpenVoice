"""Tests for the Celery app's broker-connection configuration.

Regression test: the default Celery/Kombu broker-connection retry budget
(up to 100 attempts) let a Redis outage stall `.delay()` calls for
minutes, blocking the live call-teardown path
(`openvoice.telephony.worker.finalize_call`) that dispatches the
post-call summary task. See CHANGELOG.
"""

from openvoice.tasks.celery_app import celery_app


def test_broker_connection_fails_fast_on_outage() -> None:
    assert celery_app.conf.broker_connection_timeout <= 2.0
    assert celery_app.conf.broker_connection_max_retries <= 1
