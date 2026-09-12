from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol
from uuid import UUID

from app.services.document_chunking import DocumentChunk, chunk_document
from app.services.document_ingestion import IngestedDocument, ingest_text_document
from app.services.embeddings import EmbeddedChunk
from evaluation.knowledge import KnowledgeSnapshot


class SnapshotEmbedder(Protocol):
    async def embed_chunks(
        self,
        chunks: Sequence[DocumentChunk],
    ) -> list[EmbeddedChunk]: ...


class SnapshotChunkStore(Protocol):
    def replace_documents(
        self,
        document_ids: set[UUID],
        chunks: Sequence[EmbeddedChunk],
    ) -> None: ...


@dataclass(frozen=True, slots=True)
class SeedPlan:
    documents: tuple[IngestedDocument, ...]
    chunks: tuple[DocumentChunk, ...]
    document_ids: frozenset[UUID]


@dataclass(frozen=True, slots=True)
class SeedResult:
    document_count: int
    chunk_count: int
    document_ids: frozenset[UUID]
    chunk_ids: tuple[UUID, ...]


def build_seed_plan(snapshot: KnowledgeSnapshot) -> SeedPlan:
    documents: list[IngestedDocument] = []
    chunks: list[DocumentChunk] = []

    for source in snapshot.documents:
        ingested = ingest_text_document(source.filename, source.content.encode("utf-8"))
        document = ingested.model_copy(
            update={
                "document_id": source.document_id,
                "metadata": {
                    **ingested.metadata,
                    **source.metadata,
                    "title": source.title,
                    "evaluation_namespace": snapshot.namespace,
                    "snapshot_version": snapshot.version,
                },
            }
        )
        documents.append(document)
        chunks.extend(
            chunk_document(
                document,
                chunk_size=snapshot.chunk_size,
                chunk_overlap=snapshot.chunk_overlap,
            )
        )

    return SeedPlan(
        documents=tuple(documents),
        chunks=tuple(chunks),
        document_ids=frozenset(document.document_id for document in documents),
    )


async def seed_evaluation_knowledge(
    snapshot: KnowledgeSnapshot,
    embedder: SnapshotEmbedder,
    store: SnapshotChunkStore,
) -> SeedResult:
    plan = build_seed_plan(snapshot)
    embedded_chunks = await embedder.embed_chunks(plan.chunks)

    expected_chunk_ids = [chunk.chunk_id for chunk in plan.chunks]
    embedded_chunk_ids = [chunk.chunk_id for chunk in embedded_chunks]
    if embedded_chunk_ids != expected_chunk_ids:
        raise ValueError("Embedding output does not preserve the seed chunk order and IDs.")

    store.replace_documents(set(plan.document_ids), embedded_chunks)
    return SeedResult(
        document_count=len(plan.documents),
        chunk_count=len(embedded_chunks),
        document_ids=plan.document_ids,
        chunk_ids=tuple(embedded_chunk_ids),
    )
