import asyncio
from collections.abc import Sequence
from pathlib import Path
from uuid import UUID

import pytest

from app.services.document_chunking import DocumentChunk
from app.services.embeddings import EmbeddedChunk
from evaluation.knowledge import load_knowledge_snapshot
from evaluation.seeding import seed_evaluation_knowledge


SNAPSHOT_PATH = (
    Path(__file__).parents[1]
    / "evaluation"
    / "knowledge"
    / "support_knowledge.json"
)


class DeterministicEmbedder:
    async def embed_chunks(
        self,
        chunks: Sequence[DocumentChunk],
    ) -> list[EmbeddedChunk]:
        return [
            EmbeddedChunk(
                chunk_id=chunk.chunk_id,
                document_id=chunk.document_id,
                chunk_index=chunk.chunk_index,
                content=chunk.content,
                embedding=(float(chunk.chunk_index), 1.0, 0.0),
                embedding_model="test-model",
                metadata=dict(chunk.metadata),
            )
            for chunk in chunks
        ]


class RecordingStore:
    def __init__(self) -> None:
        self.calls: list[tuple[set[UUID], list[EmbeddedChunk]]] = []
        self.rows: dict[UUID, EmbeddedChunk] = {}

    def replace_documents(
        self,
        document_ids: set[UUID],
        chunks: Sequence[EmbeddedChunk],
    ) -> None:
        self.calls.append((set(document_ids), list(chunks)))
        self.rows = {
            chunk_id: chunk
            for chunk_id, chunk in self.rows.items()
            if chunk.document_id not in document_ids
        }
        self.rows.update({chunk.chunk_id: chunk for chunk in chunks})


def test_seeding_is_idempotent_and_preserves_unrelated_rows() -> None:
    snapshot = load_knowledge_snapshot(SNAPSHOT_PATH)
    store = RecordingStore()
    unrelated = EmbeddedChunk(
        chunk_id=UUID(int=999),
        document_id=UUID(int=998),
        chunk_index=0,
        content="Unrelated",
        embedding=(0.0, 0.0, 1.0),
        embedding_model="test-model",
        metadata={},
    )
    store.rows[unrelated.chunk_id] = unrelated

    first = asyncio.run(
        seed_evaluation_knowledge(snapshot, DeterministicEmbedder(), store)
    )
    second = asyncio.run(
        seed_evaluation_knowledge(snapshot, DeterministicEmbedder(), store)
    )

    assert first.chunk_ids == second.chunk_ids
    assert len(store.rows) == 5
    assert store.rows[unrelated.chunk_id] == unrelated
    assert store.calls[0][0] == set(first.document_ids)
    assert store.calls[1][0] == set(second.document_ids)


class ReorderingEmbedder(DeterministicEmbedder):
    async def embed_chunks(
        self,
        chunks: Sequence[DocumentChunk],
    ) -> list[EmbeddedChunk]:
        embedded = await super().embed_chunks(chunks)
        return list(reversed(embedded))


def test_rejects_embedding_output_that_changes_chunk_order() -> None:
    snapshot = load_knowledge_snapshot(SNAPSHOT_PATH)
    store = RecordingStore()

    with pytest.raises(ValueError, match="preserve the seed chunk order"):
        asyncio.run(seed_evaluation_knowledge(snapshot, ReorderingEmbedder(), store))

    assert store.calls == []
