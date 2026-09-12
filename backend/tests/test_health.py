from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_health_check() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_response_includes_a_request_id_header_for_correlation() -> None:
    response = client.get("/health")

    assert "x-request-id" in response.headers
    assert response.headers["x-request-id"].startswith("req-")


def test_readiness_check_reports_not_ready_without_a_database() -> None:
    # No DATABASE_URL is configured in the test environment, so the
    # readiness probe must safely report "not_ready" (never raise, never
    # leak connection details).
    response = client.get("/health/ready")

    assert response.status_code == 503
    assert response.json() == {"status": "not_ready"}