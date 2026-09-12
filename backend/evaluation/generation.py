import re
from dataclasses import dataclass

from app.models.rag import RAGResponse
from app.services.rag import INSUFFICIENT_KNOWLEDGE_ANSWER
from evaluation.models import Answerability, EvaluationCase


SOURCE_LABEL_PATTERN = re.compile(r"^S([1-9][0-9]*)$")


@dataclass(frozen=True, slots=True)
class SourceLabelValidation:
    valid_labels: tuple[str, ...]
    unknown_labels: tuple[str, ...]
    duplicate_labels: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class CitationValidation:
    correctness: float
    unknown_chunk_ids: tuple[str, ...]
    duplicate_chunk_ids: tuple[str, ...]
    mismatched_chunk_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class GenerationSignals:
    groundedness: float
    answer_relevance: float
    citation_correctness: float
    answerability_correct: bool


def validate_source_labels(
    labels: list[str],
    *,
    retrieved_count: int,
) -> SourceLabelValidation:
    seen: set[str] = set()
    valid: list[str] = []
    unknown: list[str] = []
    duplicates: list[str] = []

    for label in labels:
        match = SOURCE_LABEL_PATTERN.fullmatch(label)
        if match is None or int(match.group(1)) > retrieved_count:
            unknown.append(label)
        elif label in seen:
            duplicates.append(label)
        else:
            seen.add(label)
            valid.append(label)

    return SourceLabelValidation(tuple(valid), tuple(unknown), tuple(duplicates))


def validate_citations(
    case: EvaluationCase,
    response: RAGResponse,
) -> CitationValidation:
    retrieved_by_id = {
        chunk.chunk_id: chunk for chunk in response.retrieved_chunks
    }
    seen: set[str] = set()
    unknown: list[str] = []
    duplicates: list[str] = []
    mismatched: list[str] = []
    valid_count = 0

    for citation in response.citations:
        chunk_id = str(citation.chunk_id)
        retrieved = retrieved_by_id.get(citation.chunk_id)
        if retrieved is None:
            unknown.append(chunk_id)
        elif chunk_id in seen:
            duplicates.append(chunk_id)
        elif (
            citation.document_id != retrieved.document_id
            or citation.chunk_index != retrieved.chunk_index
            or citation.metadata != retrieved.metadata
        ):
            seen.add(chunk_id)
            mismatched.append(chunk_id)
        else:
            seen.add(chunk_id)
            valid_count += 1

    if not response.citations:
        correctness = (
            1.0 if case.answerability is Answerability.UNANSWERABLE else 0.0
        )
    else:
        correctness = valid_count / len(response.citations)

    return CitationValidation(
        correctness,
        tuple(unknown),
        tuple(duplicates),
        tuple(mismatched),
    )


def evaluate_generation(
    case: EvaluationCase,
    response: RAGResponse,
) -> GenerationSignals:
    citation_validation = validate_citations(case, response)

    if case.answerability is Answerability.UNANSWERABLE:
        refused = response.answer.strip() == INSUFFICIENT_KNOWLEDGE_ANSWER
        return GenerationSignals(
            groundedness=float(refused),
            answer_relevance=float(refused),
            citation_correctness=citation_validation.correctness,
            answerability_correct=refused and not response.citations,
        )

    answer = _normalize(response.answer)
    context = _normalize(" ".join(chunk.content for chunk in response.retrieved_chunks))
    facts_in_answer = [
        fact for fact in case.expected_key_facts if _normalize(fact) in answer
    ]
    answer_relevance = len(facts_in_answer) / len(case.expected_key_facts)
    grounded_facts = sum(_normalize(fact) in context for fact in facts_in_answer)
    groundedness = grounded_facts / len(facts_in_answer) if facts_in_answer else 0.0

    return GenerationSignals(
        groundedness=groundedness,
        answer_relevance=answer_relevance,
        citation_correctness=citation_validation.correctness,
        answerability_correct=response.answer.strip() != INSUFFICIENT_KNOWLEDGE_ANSWER,
    )


def _normalize(text: str) -> str:
    return " ".join(text.casefold().split())
