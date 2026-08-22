"""Tests for the FastAPI app factory and health endpoint."""

from fastapi.testclient import TestClient

from openvoice.main import create_app


def test_health_endpoint_returns_ok(required_env: None) -> None:
    client = TestClient(create_app())
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_lifespan_runs_on_startup_and_shutdown(required_env: None) -> None:
    # TestClient used as a context manager drives the lifespan
    # (startup/shutdown) hook, exercising configure_logging + both log lines.
    with TestClient(create_app()) as client:
        response = client.get("/health")
        assert response.status_code == 200
