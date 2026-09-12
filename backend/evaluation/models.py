from enum import StrEnum
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Answerability(StrEnum):
    ANSWERABLE = "answerable"
    UNANSWERABLE = "unanswerable"


class SourceReference(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chunk_id: UUID | None = None
    document_id: UUID | None = None
    chunk_index: int | None = Field(default=None, ge=0)
    source_filename: str | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def require_identifier(self) -> "SourceReference":
        if not any(
            value is not None
            for value in (
                self.chunk_id,
                self.document_id,
                self.chunk_index,
                self.source_filename,
            )
        ):
            raise ValueError("A source reference requires at least one identifier.")
        return self


class EvaluationCase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    case_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]*$")
    category: str = Field(min_length=1)
    question: str = Field(min_length=1)
    answerability: Answerability
    expected_answer: str = Field(min_length=1)
    expected_key_facts: list[str]
    relevant_sources: list[SourceReference]
    notes: str | None = None

    @model_validator(mode="after")
    def validate_ground_truth(self) -> "EvaluationCase":
        if self.answerability is Answerability.ANSWERABLE:
            if not self.expected_key_facts:
                raise ValueError("Answerable cases require expected key facts.")
            if not self.relevant_sources:
                raise ValueError("Answerable cases require relevant sources.")
        elif self.relevant_sources:
            raise ValueError("Unanswerable cases cannot declare relevant sources.")
        return self


class EvaluationDataset(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: str = Field(min_length=1)
    description: str = Field(min_length=1)
    cases: list[EvaluationCase] = Field(min_length=1)

    @model_validator(mode="after")
    def require_unique_case_ids(self) -> "EvaluationDataset":
        case_ids = [case.case_id for case in self.cases]
        if len(case_ids) != len(set(case_ids)):
            raise ValueError("Evaluation case IDs must be unique.")
        return self
