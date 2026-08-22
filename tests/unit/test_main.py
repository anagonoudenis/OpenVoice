"""Tests for the FastAPI app factory and health endpoint."""

from fastapi.testclient import TestClient

from openvoice.main import create_app


def test_health_endpoint_returns_dependency_report(required_env: None) -> None:
    # No real Postgres/Redis in this unit test env, so "degraded" (rather than
    # "ok") is the expected, correct response here -- the full green-path
    # check against real infra lives in tests/integration/test_api.py.
    client = TestClient(create_app())
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] in {"ok", "degraded"}
    assert {d["name"] for d in body["dependencies"]} == {"database", "redis", "livekit"}


def test_lifespan_runs_on_startup_and_shutdown(required_env: None) -> None:
    # TestClient used as a context manager drives the lifespan
    # (startup/shutdown) hook, exercising configure_logging + both log lines.
    with TestClient(create_app()) as client:
        response = client.get("/health")
        assert response.status_code == 200
