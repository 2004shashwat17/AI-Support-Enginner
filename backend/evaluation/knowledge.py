import json
from pathlib import Path
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from app.services.document_ingestion import MetadataValue


class KnowledgeSnapshotError(ValueError):
    """Raised when a knowledge snapshot cannot be loaded safely."""


class SnapshotDocument(BaseModel):
    model_config = ConfigDict(extra="forbid")

    document_id: UUID
    filename: str = Field(pattern=r"^[^/\\]+\.txt$")
    title: str = Field(min_length=1)
    content: str = Field(min_length=1)
    metadata: dict[str, MetadataValue] = Field(default_factory=dict)

    @model_validator(mode="after")
    def reject_blank_content(self) -> "SnapshotDocument":
        if not self.content.strip():
            raise ValueError("Snapshot document content cannot be blank.")
        return self


class KnowledgeSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: str = Field(min_length=1)
    namespace: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]*$")
    chunk_size: int = Field(gt=0)
    chunk_overlap: int = Field(ge=0)
    documents: list[SnapshotDocument] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_snapshot(self) -> "KnowledgeSnapshot":
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("Snapshot chunk overlap must be smaller than chunk size.")

        document_ids = [document.document_id for document in self.documents]
        filenames = [document.filename for document in self.documents]
        if len(document_ids) != len(set(document_ids)):
            raise ValueError("Snapshot document IDs must be unique.")
        if len(filenames) != len(set(filenames)):
            raise ValueError("Snapshot filenames must be unique.")
        return self


def load_knowledge_snapshot(path: Path) -> KnowledgeSnapshot:
    try:
        raw_snapshot = json.loads(path.read_text(encoding="utf-8"))
        return KnowledgeSnapshot.model_validate(raw_snapshot)
    except OSError as exc:
        raise KnowledgeSnapshotError(f"Could not read knowledge snapshot: {path}") from exc
    except json.JSONDecodeError as exc:
        raise KnowledgeSnapshotError("Knowledge snapshot is not valid JSON.") from exc
    except ValidationError as exc:
        raise KnowledgeSnapshotError("Knowledge snapshot schema is invalid.") from exc
