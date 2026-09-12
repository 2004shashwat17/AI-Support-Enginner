import asyncio
from uuid import UUID

import pytest

from app.db.repositories.retrieval import RetrievedChunk
from app.models.rag import LLMGroundedAnswer
from app.services.llm import LLMProviderError
from app.services.rag import (
    INSUFFICIENT_KNOWLEDGE_ANSWER,
    InvalidRAGQuestionError,
    MalformedRAGResponseError,
    RAGGenerationError,
    RAGRetrievalError,
    RAGService,
)


class RecordingRetriever:
    def __init__(self, results: list[RetrievedChunk]) -> None:
        self.results = results
        self.calls: list[dict[str, object]] = []

    async def search(
        self,
        query: str,
        *,
        top_k: int | None = None,
        document_id: UUID | None = None,
        embedding_model: str | None = None,
    ) -> list[RetrievedChunk]:
        self.calls.append(
            {"query": query, "top_k": top_k, "document_id": document_id}
        )
        return self.results


class FailingRetriever(RecordingRetriever):
    async def search(self, *args: object, **kwargs: object) -> list[RetrievedChunk]:
        raise ConnectionError("database unavailable")


class RecordingLLMProvider:
    def __init__(self, response: LLMGroundedAnswer) -> None:
        self.response = response
        self.calls: list[dict[str, str]] = []

    async def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
    ) -> LLMGroundedAnswer:
        self.calls.append(
            {"system_prompt": system_prompt, "user_prompt": user_prompt}
        )
        return self.response


class FailingLLMProvider(RecordingLLMProvider):
    async def generate(self, *, system_prompt: str, user_prompt: str) -> LLMGroundedAnswer:
        raise LLMProviderError("provider unavailable")


def make_chunk(index: int, content: str) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=UUID(int=index + 1),
        document_id=UUID(int=100),
        chunk_index=index,
        content=content,
        metadata={"source_filename": "guide.txt", "rank": index},
        embedding_model="test-model",
        cosine_distance=index / 10,
        cosine_similarity=1 - (index / 10),
    )


def test_generates_grounded_response_and_citations() -> None:
    chunks = [make_chunk(0, "Reset from Settings."), make_chunk(1, "Use email recovery.")]
    retriever = RecordingRetriever(chunks)
    provider = RecordingLLMProvider(
        LLMGroundedAnswer(
            answer="Reset it from Settings.",
            cited_source_ids=["S1"],
        )
    )
    service = RAGService(retriever, provider)

    response = asyncio.run(
        service.answer("How do I reset my password?", top_k=2, document_id=UUID(int=100))
    )

    assert response.answer == "Reset it from Settings."
    assert response.citations[0].chunk_id == chunks[0].chunk_id
    assert [chunk.chunk_id for chunk in response.retrieved_chunks] == [
        chunk.chunk_id for chunk in chunks
    ]
    assert retriever.calls == [
        {
            "query": "How do I reset my password?",
            "top_k": 2,
            "document_id": UUID(int=100),
        }
    ]


def test_context_preserves_chunk_content_and_order() -> None:
    chunks = [make_chunk(0, "First exact content."), make_chunk(1, "Second exact content.")]
    provider = RecordingLLMProvider(
        LLMGroundedAnswer(answer="Answer.", cited_source_ids=["S2"])
    )

    response = asyncio.run(RAGService(RecordingRetriever(chunks), provider).answer("Question?"))

    prompt = provider.calls[0]["user_prompt"]
    assert prompt.index("[S1]\nFirst exact content.") < prompt.index(
        "[S2]\nSecond exact content."
    )
    assert response.citations[0].chunk_id == chunks[1].chunk_id
    assert response.citations[0].metadata == chunks[1].metadata


def test_rejects_blank_question_without_dependencies() -> None:
    retriever = RecordingRetriever([])
    provider = RecordingLLMProvider(LLMGroundedAnswer(answer="unused", cited_source_ids=[]))

    with pytest.raises(InvalidRAGQuestionError, match="cannot be empty"):
        asyncio.run(RAGService(retriever, provider).answer("  \n"))

    assert retriever.calls == []
    assert provider.calls == []


def test_returns_insufficient_answer_without_llm_when_no_chunks() -> None:
    provider = RecordingLLMProvider(LLMGroundedAnswer(answer="unused", cited_source_ids=[]))

    response = asyncio.run(RAGService(RecordingRetriever([]), provider).answer("Unknown?"))

    assert response.answer == INSUFFICIENT_KNOWLEDGE_ANSWER
    assert response.citations == []
    assert response.retrieved_chunks == []
    assert provider.calls == []


def test_rejects_unknown_generated_source_id() -> None:
    provider = RecordingLLMProvider(
        LLMGroundedAnswer(answer="Answer.", cited_source_ids=["S9"])
    )

    with pytest.raises(MalformedRAGResponseError, match="do not match"):
        asyncio.run(
            RAGService(RecordingRetriever([make_chunk(0, "Content")]), provider).answer(
                "Question?"
            )
        )


def test_wraps_llm_failure() -> None:
    provider = FailingLLMProvider(
        LLMGroundedAnswer(answer="unused", cited_source_ids=[])
    )

    with pytest.raises(RAGGenerationError, match="generation failed"):
        asyncio.run(
            RAGService(RecordingRetriever([make_chunk(0, "Content")]), provider).answer(
                "Question?"
            )
        )


def test_wraps_retrieval_failure() -> None:
    provider = RecordingLLMProvider(LLMGroundedAnswer(answer="unused", cited_source_ids=[]))

    with pytest.raises(RAGRetrievalError, match="retrieval failed"):
        asyncio.run(RAGService(FailingRetriever([]), provider).answer("Question?"))

    assert provider.calls == []
