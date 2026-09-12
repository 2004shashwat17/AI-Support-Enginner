from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol
from uuid import UUID

from app.db.repositories.retrieval import RetrievedChunk
from app.models.rag import RAGResponse
from evaluation.generation import GenerationSignals, evaluate_generation
from evaluation.metrics import RetrievalMetrics, calculate_retrieval_metrics
from evaluation.models import EvaluationDataset


class EvaluationRetriever(Protocol):
    async def search(
        self,
        query: str,
        *,
        top_k: int | None = None,
        document_id: UUID | None = None,
        embedding_model: str | None = None,
    ) -> list[RetrievedChunk]: ...


class EvaluationRAGService(Protocol):
    async def answer(
        self,
        question: str,
        *,
        top_k: int | None = None,
        document_id: UUID | None = None,
    ) -> RAGResponse: ...


@dataclass(frozen=True, slots=True)
class GenerationSummary:
    evaluated_cases: int
    groundedness: float
    answer_relevance: float
    citation_correctness: float
    answerability_accuracy: float


@dataclass(frozen=True, slots=True)
class EvaluationReport:
    dataset_size: int
    retrieval: tuple[RetrievalMetrics, ...]
    generation: GenerationSummary | None


@dataclass(frozen=True, slots=True)
class RetrievalComparisonReport:
    dataset_size: int
    strategies: dict[str, tuple[RetrievalMetrics, ...]]


async def run_evaluation(
    dataset: EvaluationDataset,
    retriever: EvaluationRetriever,
    *,
    k_values: Sequence[int] = (1, 3, 5),
    rag_service: EvaluationRAGService | None = None,
) -> EvaluationReport:
    if not k_values or any(k <= 0 for k in k_values):
        raise ValueError("K values must contain positive integers.")

    maximum_k = max(k_values)
    retrieved_by_case: dict[str, list[RetrievedChunk]] = {}
    generation_signals: list[GenerationSignals] = []

    for case in dataset.cases:
        retrieved_by_case[case.case_id] = await retriever.search(
            case.question,
            top_k=maximum_k,
        )
        if rag_service is not None:
            response = await rag_service.answer(case.question, top_k=maximum_k)
            generation_signals.append(evaluate_generation(case, response))

    retrieval_metrics = tuple(
        calculate_retrieval_metrics(dataset, retrieved_by_case, k=k)
        for k in k_values
    )
    generation = _summarize_generation(generation_signals)
    return EvaluationReport(len(dataset.cases), retrieval_metrics, generation)


async def run_retrieval_comparison(
    dataset: EvaluationDataset,
    retrievers: Mapping[str, EvaluationRetriever],
    *,
    k_values: Sequence[int] = (1, 3, 5),
) -> RetrievalComparisonReport:
    if not retrievers:
        raise ValueError("At least one retrieval strategy is required.")

    strategies: dict[str, tuple[RetrievalMetrics, ...]] = {}
    for strategy, retriever in retrievers.items():
        report = await run_evaluation(dataset, retriever, k_values=k_values)
        strategies[strategy] = report.retrieval
    return RetrievalComparisonReport(len(dataset.cases), strategies)


def _summarize_generation(
    signals: Sequence[GenerationSignals],
) -> GenerationSummary | None:
    if not signals:
        return None

    count = len(signals)
    return GenerationSummary(
        evaluated_cases=count,
        groundedness=sum(signal.groundedness for signal in signals) / count,
        answer_relevance=sum(signal.answer_relevance for signal in signals) / count,
        citation_correctness=(
            sum(signal.citation_correctness for signal in signals) / count
        ),
        answerability_accuracy=(
            sum(signal.answerability_correct for signal in signals) / count
        ),
    )
