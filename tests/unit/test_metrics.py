"""Tests for the Prometheus metrics registered for the voice pipeline."""

from prometheus_client import generate_latest

from openvoice import metrics


def test_all_metrics_are_registered_and_render() -> None:
    output = generate_latest().decode()
    assert "openvoice_calls_started_total" in output
    assert "openvoice_calls_completed_total" in output
    assert "openvoice_call_duration_seconds" in output
    assert "openvoice_llm_errors_total" in output
    assert "openvoice_tool_calls_total" in output


def test_calls_started_counter_increments() -> None:
    before = generate_latest()

    metrics.calls_started_total.inc()

    after = generate_latest()
    assert before != after
