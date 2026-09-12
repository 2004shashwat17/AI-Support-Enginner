import json
from pathlib import Path
from uuid import UUID

import pytest

from evaluation.dataset import load_evaluation_dataset
from evaluation.knowledge import KnowledgeSnapshotError, load_knowledge_snapshot
from evaluation.seeding import build_seed_plan


SNAPSHOT_PATH = (
    Path(__file__).parents[1]
    / "evaluation"
    / "knowledge"
    / "support_knowledge.json"
)
DATASET_PATH = (
    Path(__file__).parents[1] / "evaluation" / "dataset" / "rag_dataset.json"
)
EXPECTED_DOCUMENT_IDS = {
    UUID("9bf25f95-1316-5324-8ce8-b21abb6a4b60"),
    UUID("ae0dabd1-ab28-5273-a48a-aa8919147356"),
    UUID("a4d2ee51-2f9c-5f5c-93ff-a48d814cd1f7"),
    UUID("741fd888-d438-5284-af5b-e0e8e11c56e4"),
}
EXPECTED_CHUNK_IDS = {
    UUID("fdb00a4e-a795-51f4-9876-85923c0f7405"),
    UUID("2ab6d70e-1b4d-5c3d-b64b-ac0db8c60e65"),
    UUID("33aa05cf-4958-537d-ab0d-0bb060314cf5"),
    UUID("1ee6547c-0ffb-5ad4-9809-d437c35af7ac"),
}


def test_loads_reviewed_snapshot_with_stable_document_ids() -> None:
    snapshot = load_knowledge_snapshot(SNAPSHOT_PATH)

    assert len(snapshot.documents) == 4
    assert {document.document_id for document in snapshot.documents} == (
        EXPECTED_DOCUMENT_IDS
    )
    assert all(document.content.strip() for document in snapshot.documents)


def test_transforms_snapshot_to_internal_documents() -> None:
    snapshot = load_knowledge_snapshot(SNAPSHOT_PATH)

    plan = build_seed_plan(snapshot)

    assert plan.document_ids == frozenset(EXPECTED_DOCUMENT_IDS)
    assert len(plan.documents) == 4
    assert all(
        document.metadata["evaluation_namespace"] == "evaluation-baseline-v1"
        for document in plan.documents
    )


def test_chunk_ids_are_deterministic_and_match_committed_references() -> None:
    snapshot = load_knowledge_snapshot(SNAPSHOT_PATH)

    first_plan = build_seed_plan(snapshot)
    second_plan = build_seed_plan(snapshot)

    assert [chunk.chunk_id for chunk in first_plan.chunks] == [
        chunk.chunk_id for chunk in second_plan.chunks
    ]
    assert {chunk.chunk_id for chunk in first_plan.chunks} == EXPECTED_CHUNK_IDS


def test_evaluation_sources_reference_snapshot_ids() -> None:
    plan = build_seed_plan(load_knowledge_snapshot(SNAPSHOT_PATH))
    dataset = load_evaluation_dataset(DATASET_PATH)
    chunk_ids = {chunk.chunk_id for chunk in plan.chunks}

    for case in dataset.cases:
        for source in case.relevant_sources:
            assert source.document_id in plan.document_ids
            assert source.chunk_id in chunk_ids


def test_rejects_missing_snapshot(tmp_path: Path) -> None:
    with pytest.raises(KnowledgeSnapshotError, match="Could not read"):
        load_knowledge_snapshot(tmp_path / "missing.json")


def test_rejects_invalid_snapshot(tmp_path: Path) -> None:
    path = tmp_path / "invalid.json"
    path.write_text(
        json.dumps(
            {
                "version": "1.0",
                "namespace": "evaluation",
                "chunk_size": 10,
                "chunk_overlap": 10,
                "documents": [],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(KnowledgeSnapshotError, match="schema is invalid"):
        load_knowledge_snapshot(path)
