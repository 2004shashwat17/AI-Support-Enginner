from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_ingests_text_document() -> None:
    response = client.post(
        "/documents/ingest",
        files={"file": ("faq.txt", b"How do I reset my password?", "text/plain")},
    )

    assert response.status_code == 201
    body = response.json()
    assert body["filename"] == "faq.txt"
    assert body["metadata"]["source_type"] == "text"
    assert body["metadata"]["character_count"] == 27
    assert "document_id" in body
    assert "content" not in body


def test_returns_bad_request_for_empty_document() -> None:
    response = client.post(
        "/documents/ingest",
        files={"file": ("empty.txt", b"   \n", "text/plain")},
    )

    assert response.status_code == 400
    assert response.json() == {"detail": "The document contains no usable text."}


def test_returns_unsupported_media_type_for_other_extensions() -> None:
    response = client.post(
        "/documents/ingest",
        files={"file": ("policy.pdf", b"Refund policy", "application/pdf")},
    )

    assert response.status_code == 415
    assert response.json() == {"detail": "Only .txt documents are supported."}
