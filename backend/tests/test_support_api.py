from uuid import UUID

from fastapi.testclient import TestClient

from app.api.dependencies import get_rag_service
from app.main import app
from app.models.rag import Citation, RAGResponse, RetrievedChunkResponse
from app.services.rag import RAGGenerationError


class SuccessfulRAGService:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    async def answer(
        self,
        question: str,
        *,
        top_k: int | None = None,
        document_id: UUID | None = None,
    ) -> RAGResponse:
        self.calls.append(
            {"question": question, "top_k": top_k, "document_id": document_id}
        )
        citation = Citation(
            chunk_id=UUID(int=1),
            document_id=UUID(int=100),
            chunk_index=0,
            metadata={"source_filename": "guide.txt"},
        )
        chunk = RetrievedChunkResponse(
            chunk_id=citation.chunk_id,
            document_id=citation.document_id,
            chunk_index=0,
            content="Reset your password from Settings.",
            metadata=citation.metadata,
            embedding_model="test-model",
            cosine_distance=0.1,
            cosine_similarity=0.9,
        )
        return RAGResponse(
            answer="Reset your password from Settings.",
            citations=[citation],
            retrieved_chunks=[chunk],
        )


class FailingRAGService(SuccessfulRAGService):
    async def answer(
        self,
        question: str,
        *,
        top_k: int | None = None,
        document_id: UUID | None = None,
    ) -> RAGResponse:
        raise RAGGenerationError("secret provider details")


def test_support_endpoint_returns_structured_rag_response() -> None:
    service = SuccessfulRAGService()
    app.dependency_overrides[get_rag_service] = lambda: service
    client = TestClient(app)

    try:
        response = client.post(
            "/api/v1/support/ask",
            json={"question": "How do I reset my password?", "top_k": 3},
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json()["answer"] == "Reset your password from Settings."
    assert response.json()["citations"][0]["chunk_index"] == 0
    assert service.calls == [
        {
            "question": "How do I reset my password?",
            "top_k": 3,
            "document_id": None,
        }
    ]


def test_support_endpoint_does_not_leak_internal_failure_details() -> None:
    app.dependency_overrides[get_rag_service] = lambda: FailingRAGService()
    client = TestClient(app)

    try:
        response = client.post(
            "/api/v1/support/ask",
            json={"question": "How do I reset my password?"},
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 503
    assert response.json() == {
        "detail": "The support service is temporarily unavailable."
    }
    assert "secret provider details" not in response.text