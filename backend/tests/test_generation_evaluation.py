from pathlib import Path
from uuid import UUID

from app.models.rag import Citation, RAGResponse, RetrievedChunkResponse
from app.services.rag import INSUFFICIENT_KNOWLEDGE_ANSWER
from evaluation.dataset import load_evaluation_dataset
from evaluation.generation import (
    evaluate_generation,
    validate_citations,
    validate_source_labels,
)


DATASET_PATH = (
    Path(__file__).parents[1] / "evaluation" / "dataset" / "rag_dataset.json"
)


def chunk(chunk_id: int = 1) -> RetrievedChunkResponse:
    return RetrievedChunkResponse(
        chunk_id=UUID(int=chunk_id),
        document_id=UUID(int=100),
        chunk_index=0,
        content="Reset the password from Settings.",
        metadata={"source_filename": "guide.txt"},
        embedding_model="test-model",
        cosine_distance=0.1,
        cosine_similarity=0.9,
    )


def citation(chunk_id: int = 1) -> Citation:
    return Citation(
        chunk_id=UUID(int=chunk_id),
        document_id=UUID(int=100),
        chunk_index=0,
        metadata={"source_filename": "guide.txt"},
    )


def case(case_id: str):
    dataset = load_evaluation_dataset(DATASET_PATH)
    return next(item for item in dataset.cases if item.case_id == case_id)


def test_validates_known_unknown_duplicate_and_empty_source_labels() -> None:
    validation = validate_source_labels(["S1", "S1", "S3"], retrieved_count=2)

    assert validation.valid_labels == ("S1",)
    assert validation.duplicate_labels == ("S1",)
    assert validation.unknown_labels == ("S3",)
    assert validate_source_labels([], retrieved_count=0).valid_labels == ()


def test_valid_citation_maps_to_retrieved_chunk() -> None:
    response = RAGResponse(
        answer="Reset it from Settings.",
        citations=[citation()],
        retrieved_chunks=[chunk()],
    )

    validation = validate_citations(case("password-reset-settings"), response)

    assert validation.correctness == 1.0
    assert validation.unknown_chunk_ids == ()
    assert validation.mismatched_chunk_ids == ()


def test_detects_citation_for_chunk_not_retrieved_and_duplicates() -> None:
    response = RAGResponse(
        answer="Reset it from Settings.",
        citations=[citation(), citation(), citation(9)],
        retrieved_chunks=[chunk()],
    )

    validation = validate_citations(case("password-reset-settings"), response)

    assert validation.correctness == 1 / 3
    assert validation.duplicate_chunk_ids == (str(UUID(int=1)),)
    assert validation.unknown_chunk_ids == (str(UUID(int=9)),)


def test_detects_citation_metadata_that_does_not_match_retrieved_chunk() -> None:
    altered_citation = citation()
    altered_citation.metadata = {"source_filename": "invented.txt"}
    response = RAGResponse(
        answer="Reset it from Settings.",
        citations=[altered_citation],
        retrieved_chunks=[chunk()],
    )

    validation = validate_citations(case("password-reset-settings"), response)

    assert validation.correctness == 0.0
    assert validation.mismatched_chunk_ids == (str(UUID(int=1)),)


def test_unanswerable_fallback_with_empty_citations_scores_correctly() -> None:
    response = RAGResponse(
        answer=INSUFFICIENT_KNOWLEDGE_ANSWER,
        citations=[],
        retrieved_chunks=[],
    )

    signals = evaluate_generation(case("support-phone-number"), response)

    assert signals.groundedness == 1.0
    assert signals.answer_relevance == 1.0
    assert signals.citation_correctness == 1.0
    assert signals.answerability_correct


def test_answerable_response_reports_heuristic_generation_signals() -> None:
    response = RAGResponse(
        answer="Reset it from Settings.",
        citations=[citation()],
        retrieved_chunks=[chunk()],
    )

    signals = evaluate_generation(case("password-reset-settings"), response)

    assert signals.groundedness == 1.0
    assert signals.answer_relevance == 1.0
    assert signals.citation_correctness == 1.0
