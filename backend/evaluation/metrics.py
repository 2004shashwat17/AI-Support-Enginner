from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol
from uuid import UUID

from evaluation.models import EvaluationCase, EvaluationDataset, SourceReference


class RetrievedSource(Protocol):
    chunk_id: UUID
    document_id: UUID
    chunk_index: int
    metadata: Mapping[str, str | int]


@dataclass(frozen=True, slots=True)
class RetrievalMetrics:
    k: int
    evaluated_cases: int
    recall_at_k: float
    precision_at_k: float
    hit_rate_at_k: float


def calculate_retrieval_metrics(
    dataset: EvaluationDataset,
    retrieved_by_case: Mapping[str, Sequence[RetrievedSource]],
    *,
    k: int,
) -> RetrievalMetrics:
    if k <= 0:
        raise ValueError("K must be greater than zero.")

    recalls: list[float] = []
    precisions: list[float] = []
    hits: list[float] = []

    for case in dataset.cases:
        if not case.relevant_sources:
            continue

        retrieved = list(retrieved_by_case.get(case.case_id, ()))[:k]
        covered_sources = sum(
            any(_source_matches(expected, result) for result in retrieved)
            for expected in case.relevant_sources
        )
        relevant_results = sum(
            any(_source_matches(expected, result) for expected in case.relevant_sources)
            for result in retrieved
        )

        recalls.append(covered_sources / len(case.relevant_sources))
        precisions.append(relevant_results / len(retrieved) if retrieved else 0.0)
        hits.append(float(covered_sources > 0))

    if not recalls:
        return RetrievalMetrics(k, 0, 0.0, 0.0, 0.0)

    return RetrievalMetrics(
        k=k,
        evaluated_cases=len(recalls),
        recall_at_k=sum(recalls) / len(recalls),
        precision_at_k=sum(precisions) / len(precisions),
        hit_rate_at_k=sum(hits) / len(hits),
    )


def _source_matches(expected: SourceReference, result: RetrievedSource) -> bool:
    source_filename = result.metadata.get("source_filename")
    return all(
        (
            expected.chunk_id is None or expected.chunk_id == result.chunk_id,
            expected.document_id is None or expected.document_id == result.document_id,
            expected.chunk_index is None or expected.chunk_index == result.chunk_index,
            expected.source_filename is None
            or expected.source_filename == source_filename,
        )
    )
