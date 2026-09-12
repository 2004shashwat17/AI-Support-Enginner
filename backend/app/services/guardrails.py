"""Evidence-quality guardrails for the RAG answer pipeline.

Pipeline:

    USER QUESTION -> RETRIEVAL -> RERANKING -> EVIDENCE QUALITY -> ANSWER OR
    INSUFFICIENT KNOWLEDGE

This module decides whether the system has enough evidence to safely attempt
a grounded answer, and whether a generated answer was actually grounded once
produced. It intentionally uses explicit, discrete states instead of a single
numeric "confidence" score:

- retrieval relevance (cosine similarity / rerank score) is a coarse,
  uncalibrated ranking signal from the underlying model, not a probability
  that the answer is correct.
- answer confidence would require a calibrated model (e.g. verified against
  labeled outcomes); no such calibration exists here, so this module never
  reports one.
- groundedness is measured by evaluation/generation.py as key-fact overlap
  with retrieved context, a separate, independent signal.
- answerability is a property of the *question* (does the knowledge base
  contain an answer at all), tracked in evaluation/models.py.

This module governs a different, narrower question: given the evidence
actually retrieved for a request, is it safe to let the LLM attempt an
answer, and did the resulting answer actually cite that evidence.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from math import isfinite

from app.db.repositories.retrieval import RetrievedChunk


class EvidenceStatus(StrEnum):
    SUFFICIENT_EVIDENCE = "sufficient_evidence"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    RETRIEVAL_FAILED = "retrieval_failed"
    GENERATION_FAILED = "generation_failed"
    NEEDS_CLARIFICATION = "needs_clarification"


@dataclass(frozen=True, slots=True)
class EvidenceAssessment:
    status: EvidenceStatus
    reason: str
    chunk_count: int


DEFAULT_MIN_CHUNK_COUNT = 1


class EvidenceGuardrail:
    """Assesses retrieved evidence before an LLM call is attempted.

    `min_cosine_similarity` and `min_rerank_score` are optional, disabled by
    default. If enabled, they are coarse relative thresholds on the top
    result only -- not calibrated probabilities of correctness.
    """

    def __init__(
        self,
        *,
        min_chunk_count: int = DEFAULT_MIN_CHUNK_COUNT,
        min_cosine_similarity: float | None = None,
        min_rerank_score: float | None = None,
    ) -> None:
        if min_chunk_count < 1:
            raise ValueError("min_chunk_count must be at least 1.")
        self._min_chunk_count = min_chunk_count
        self._min_cosine_similarity = min_cosine_similarity
        self._min_rerank_score = min_rerank_score

    def assess(self, chunks: Sequence[RetrievedChunk]) -> EvidenceAssessment:
        if not chunks:
            return EvidenceAssessment(
                EvidenceStatus.INSUFFICIENT_EVIDENCE,
                "No evidence was retrieved.",
                0,
            )

        for chunk in chunks:
            for score in (chunk.cosine_similarity, chunk.rerank_score):
                if score is not None and not isfinite(score):
                    return EvidenceAssessment(
                        EvidenceStatus.INSUFFICIENT_EVIDENCE,
                        "Retrieved evidence contained a malformed score.",
                        len(chunks),
                    )

        if len(chunks) < self._min_chunk_count:
            return EvidenceAssessment(
                EvidenceStatus.INSUFFICIENT_EVIDENCE,
                f"Fewer than {self._min_chunk_count} evidence chunk(s) were retrieved.",
                len(chunks),
            )

        top = chunks[0]
        if (
            self._min_cosine_similarity is not None
            and top.cosine_similarity is not None
            and top.cosine_similarity < self._min_cosine_similarity
        ):
            return EvidenceAssessment(
                EvidenceStatus.INSUFFICIENT_EVIDENCE,
                "Top retrieval similarity is below the configured threshold.",
                len(chunks),
            )

        if (
            self._min_rerank_score is not None
            and top.rerank_score is not None
            and top.rerank_score < self._min_rerank_score
        ):
            return EvidenceAssessment(
                EvidenceStatus.INSUFFICIENT_EVIDENCE,
                "Top rerank score is below the configured threshold.",
                len(chunks),
            )

        return EvidenceAssessment(
            EvidenceStatus.SUFFICIENT_EVIDENCE,
            "Evidence passed guardrail checks.",
            len(chunks),
        )


def assess_generation_grounding(citation_count: int) -> EvidenceAssessment:
    """Post-generation guardrail: a generated answer with zero citations
    cannot be treated as grounded, even if retrieval evidence looked sufficient.
    """
    if citation_count == 0:
        return EvidenceAssessment(
            EvidenceStatus.INSUFFICIENT_EVIDENCE,
            "Generated answer cited no retrieved sources.",
            citation_count,
        )
    return EvidenceAssessment(
        EvidenceStatus.SUFFICIENT_EVIDENCE,
        "Generated answer cited retrieved sources.",
        citation_count,
    )
