"""Prometheus metrics for the voice pipeline.

Exposed at `/metrics` (see `openvoice.main`) via `prometheus_client`'s own
ASGI app -- serving them never goes through business logic. Metric
objects are created once at import time (the `prometheus_client`
convention); instrumented call sites just call `.inc()`/`.observe()`.

Deliberately a small, targeted set tied to what actually matters for a
voice agent's reliability -- call volume/duration and the two failure
modes most likely to affect a real caller (LLM failures, tool-call
failures) -- not a general-purpose analytics layer.
"""

from prometheus_client import Counter, Histogram

calls_started_total = Counter(
    "openvoice_calls_started_total", "Calls that reached the telephony worker's entrypoint."
)

calls_completed_total = Counter(
    "openvoice_calls_completed_total",
    "Calls that reached call teardown (finalize_call ran).",
    ["outcome"],
)

call_duration_seconds = Histogram(
    "openvoice_call_duration_seconds",
    "Wall-clock duration of a call, from entrypoint to teardown.",
    buckets=(5, 15, 30, 60, 120, 300, 600, 900, 1800),
)

llm_errors_total = Counter(
    "openvoice_llm_errors_total", "LLMProviderError occurrences during a live call."
)

tool_calls_total = Counter(
    "openvoice_tool_calls_total",
    "Agent tool-calls, by tool name and outcome.",
    ["tool", "outcome"],
)
