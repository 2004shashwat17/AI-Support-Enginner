import asyncio
from dataclasses import replace
from uuid import UUID

import pytest

from app.db.repositories.retrieval import RetrievedChunk
from app.models.rag import LLMGroundedAnswer
from app.services.guardrails import EvidenceGuardrail, EvidenceStatus
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


# Guardrail: strong evidence -> normal grounded answer, tagged sufficient.
def test_evidence_status_sufficient_for_grounded_answer() -> None:
    chunks = [make_chunk(0, "Reset it from Settings.")]
    provider = RecordingLLMProvider(
        LLMGroundedAnswer(answer="Reset it from Settings.", cited_source_ids=["S1"])
    )

    response = asyncio.run(RAGService(RecordingRetriever(chunks), provider).answer("Q?"))

    assert response.evidence_status is EvidenceStatus.SUFFICIENT_EVIDENCE
    assert response.answer == "Reset it from Settings."


# Guardrail: no evidence -> safe fallback without calling the LLM.
def test_evidence_status_insufficient_and_no_llm_call_when_no_chunks() -> None:
    provider = RecordingLLMProvider(LLMGroundedAnswer(answer="unused", cited_source_ids=[]))

    response = asyncio.run(RAGService(RecordingRetriever([]), provider).answer("Unknown?"))

    assert response.answer == INSUFFICIENT_KNOWLEDGE_ANSWER
    assert response.evidence_status is EvidenceStatus.INSUFFICIENT_EVIDENCE
    assert provider.calls == []


# Guardrail: weak evidence (below configured similarity threshold) skips the LLM call.
def test_evidence_status_insufficient_for_weak_similarity_skips_llm() -> None:
    weak_chunk = replace(make_chunk(0, "Loosely related content."), cosine_similarity=0.1)
    provider = RecordingLLMProvider(LLMGroundedAnswer(answer="unused", cited_source_ids=[]))
    guardrail = EvidenceGuardrail(min_cosine_similarity=0.95)
    service = RAGService(RecordingRetriever([weak_chunk]), provider, guardrail=guardrail)

    response = asyncio.run(service.answer("Question?"))

    assert response.answer == INSUFFICIENT_KNOWLEDGE_ANSWER
    assert response.evidence_status is EvidenceStatus.INSUFFICIENT_EVIDENCE
    assert provider.calls == []


# Guardrail: malformed retrieval scores are treated as insufficient evidence.
def test_evidence_status_insufficient_for_malformed_scores() -> None:
    malformed_chunk = replace(make_chunk(0, "Content"), cosine_similarity=float("nan"))
    provider = RecordingLLMProvider(LLMGroundedAnswer(answer="unused", cited_source_ids=[]))

    response = asyncio.run(
        RAGService(RecordingRetriever([malformed_chunk]), provider).answer("Question?")
    )

    assert response.evidence_status is EvidenceStatus.INSUFFICIENT_EVIDENCE
    assert provider.calls == []


# Guardrail: generated answer with no citations is treated as ungrounded and overridden.
def test_missing_citations_forces_safe_fallback_response() -> None:
    chunks = [make_chunk(0, "Reset it from Settings.")]
    provider = RecordingLLMProvider(
        LLMGroundedAnswer(answer="It works somehow.", cited_source_ids=[])
    )

    response = asyncio.run(RAGService(RecordingRetriever(chunks), provider).answer("Question?"))

    assert response.answer == INSUFFICIENT_KNOWLEDGE_ANSWER
    assert response.citations == []
    assert response.evidence_status is EvidenceStatus.INSUFFICIENT_EVIDENCE
    # The LLM was still called -- retrieval evidence looked sufficient beforehand.
    assert len(provider.calls) == 1


# Generation failure remains an explicit error, not a silently-swallowed guardrail state.
def test_generation_failure_still_raises_instead_of_returning_a_response() -> None:
    provider = FailingLLMProvider(LLMGroundedAnswer(answer="unused", cited_source_ids=[]))

    with pytest.raises(RAGGenerationError):
        asyncio.run(
            RAGService(RecordingRetriever([make_chunk(0, "Content")]), provider).answer("Q?")
        )

