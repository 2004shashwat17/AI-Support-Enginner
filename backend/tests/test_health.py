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