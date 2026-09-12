from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from app.services.document_ingestion import MetadataValue
from app.services.guardrails import EvidenceStatus


class Citation(BaseModel):
    chunk_id: UUID
    document_id: UUID
    chunk_index: int
    metadata: dict[str, MetadataValue]


class RetrievedChunkResponse(BaseModel):
    chunk_id: UUID
    document_id: UUID
    chunk_index: int
    content: str
    metadata: dict[str, MetadataValue]
    embedding_model: str
    cosine_distance: float | None = None
    cosine_similarity: float | None = None
    rerank_score: float | None = None
    original_rank: int | None = None


class RAGResponse(BaseModel):
    answer: str = Field(min_length=1)
    citations: list[Citation]
    evidence_status: EvidenceStatus = EvidenceStatus.SUFFICIENT_EVIDENCE
    retrieved_chunks: list[RetrievedChunkResponse]


class LLMGroundedAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answer: str = Field(min_length=1)
    cited_source_ids: list[str]
