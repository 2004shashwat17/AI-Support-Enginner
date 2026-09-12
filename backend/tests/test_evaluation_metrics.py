from pathlib import Path
from uuid import UUID

import pytest

from app.db.repositories.retrieval import RetrievedChunk
from evaluation.dataset import load_evaluation_dataset
from evaluation.metrics import calculate_retrieval_metrics


DATASET_PATH = (
    Path(__file__).parents[1] / "evaluation" / "dataset" / "rag_dataset.json"
)


def result(
    filename: str,
    chunk_index: int,
    *,
    chunk_id: UUID | None = None,
    document_id: UUID | None = None,
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id or UUID(int=chunk_index + 1),
        document_id=document_id or UUID(int=100),
        chunk_index=chunk_index,
        content="content",
        metadata={"source_filename": filename},
        embedding_model="test-model",
        cosine_distance=0.1,
        cosine_similarity=0.9,
    )


def test_calculates_recall_precision_and_hit_rate_at_k() -> None:
    dataset = load_evaluation_dataset(DATASET_PATH)
    case = next(
        case for case in dataset.cases if case.case_id == "password-reset-and-recovery"
    )
    focused_dataset = dataset.model_copy(update={"cases": [case]})
    retrieved = {
        case.case_id: [
            result(
                "password-guide.txt",
                0,
                chunk_id=UUID("fdb00a4e-a795-51f4-9876-85923c0f7405"),
                document_id=UUID("9bf25f95-1316-5324-8ce8-b21abb6a4b60"),
            ),
            result("unrelated.txt", 0),
        ]
    }

    at_one = calculate_retrieval_metrics(focused_dataset, retrieved, k=1)
    at_two = calculate_retrieval_metrics(focused_dataset, retrieved, k=2)

    assert at_one.recall_at_k == 0.5
    assert at_one.precision_at_k == 1.0
    assert at_one.hit_rate_at_k == 1.0
    assert at_two.recall_at_k == 0.5
    assert at_two.precision_at_k == 0.5


def test_empty_retrieval_has_zero_metrics() -> None:
    dataset = load_evaluation_dataset(DATASET_PATH)

    metrics = calculate_retrieval_metrics(dataset, {}, k=3)

    assert metrics.recall_at_k == 0.0
    assert metrics.precision_at_k == 0.0
    assert metrics.hit_rate_at_k == 0.0
    assert metrics.evaluated_cases == 5


def test_rejects_invalid_k() -> None:
    with pytest.raises(ValueError, match="greater than zero"):
        calculate_retrieval_metrics(load_evaluation_dataset(DATASET_PATH), {}, k=0)
