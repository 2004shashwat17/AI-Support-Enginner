from math import nan
from uuid import UUID

import pytest

from app.db.repositories.retrieval import RetrievedChunk
from app.services.guardrails import (
    EvidenceGuardrail,
    EvidenceStatus,
    assess_generation_grounding,
)


def make_chunk(
    index: int,
    *,
    cosine_similarity: float | None = None,
    rerank_score: float | None = None,
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=UUID(int=index + 1),
        document_id=UUID(int=100),
        chunk_index=index,
        content=f"content {index}",
        metadata={"source_filename": "guide.txt"},
        embedding_model="test-model",
        cosine_similarity=cosine_similarity,
        rerank_score=rerank_score,
    )


# Strong evidence.
def test_sufficient_evidence_when_chunks_present_and_no_thresholds_configured() -> None:
    guardrail = EvidenceGuardrail()

    assessment = guardrail.assess([make_chunk(0)])

    assert assessment.status is EvidenceStatus.SUFFICIENT_EVIDENCE
    assert assessment.chunk_count == 1


# No evidence.
def test_insufficient_evidence_for_empty_chunk_list() -> None:
    guardrail = EvidenceGuardrail()

    assessment = guardrail.assess([])

    assert assessment.status is EvidenceStatus.INSUFFICIENT_EVIDENCE
    assert assessment.chunk_count == 0


# Weak evidence via configured similarity threshold.
def test_insufficient_evidence_when_top_similarity_below_threshold() -> None:
    guardrail = EvidenceGuardrail(min_cosine_similarity=0.5)

    assessment = guardrail.assess([make_chunk(0, cosine_similarity=0.2)])

    assert assessment.status is EvidenceStatus.INSUFFICIENT_EVIDENCE


def test_sufficient_evidence_when_top_similarity_meets_threshold() -> None:
    guardrail = EvidenceGuardrail(min_cosine_similarity=0.5)

    assessment = guardrail.assess([make_chunk(0, cosine_similarity=0.9)])

    assert assessment.status is EvidenceStatus.SUFFICIENT_EVIDENCE


# Weak evidence via configured rerank score threshold.
def test_insufficient_evidence_when_top_rerank_score_below_threshold() -> None:
    guardrail = EvidenceGuardrail(min_rerank_score=0.5)

    assessment = guardrail.assess([make_chunk(0, rerank_score=0.1)])

    assert assessment.status is EvidenceStatus.INSUFFICIENT_EVIDENCE


# Fewer chunks than the configured minimum.
def test_insufficient_evidence_when_below_minimum_chunk_count() -> None:
    guardrail = EvidenceGuardrail(min_chunk_count=2)

    assessment = guardrail.assess([make_chunk(0)])

    assert assessment.status is EvidenceStatus.INSUFFICIENT_EVIDENCE


def test_rejects_non_positive_minimum_chunk_count() -> None:
    with pytest.raises(ValueError):
        EvidenceGuardrail(min_chunk_count=0)


# Malformed retrieval scores (NaN) must not be trusted.
def test_insufficient_evidence_for_non_finite_similarity_score() -> None:
    guardrail = EvidenceGuardrail()

    assessment = guardrail.assess([make_chunk(0, cosine_similarity=nan)])

    assert assessment.status is EvidenceStatus.INSUFFICIENT_EVIDENCE


def test_insufficient_evidence_for_non_finite_rerank_score() -> None:
    guardrail = EvidenceGuardrail()

    assessment = guardrail.assess([make_chunk(0, rerank_score=float("inf"))])

    assert assessment.status is EvidenceStatus.INSUFFICIENT_EVIDENCE


# Conflicting evidence: multiple chunks, only the top one is checked against thresholds.
def test_only_top_chunk_is_checked_against_thresholds() -> None:
    guardrail = EvidenceGuardrail(min_cosine_similarity=0.5)
    chunks = [
        make_chunk(0, cosine_similarity=0.9),
        make_chunk(1, cosine_similarity=0.1),
    ]

    assessment = guardrail.assess(chunks)

    assert assessment.status is EvidenceStatus.SUFFICIENT_EVIDENCE


# Missing citations after generation.
def test_generation_grounding_insufficient_when_no_citations() -> None:
    assessment = assess_generation_grounding(0)

    assert assessment.status is EvidenceStatus.INSUFFICIENT_EVIDENCE


def test_generation_grounding_sufficient_when_citations_present() -> None:
    assessment = assess_generation_grounding(2)

    assert assessment.status is EvidenceStatus.SUFFICIENT_EVIDENCE
