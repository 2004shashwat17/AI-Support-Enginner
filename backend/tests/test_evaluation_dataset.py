import json
from pathlib import Path

import pytest

from evaluation.dataset import EvaluationDatasetError, load_evaluation_dataset
from evaluation.models import Answerability


DATASET_PATH = (
    Path(__file__).parents[1] / "evaluation" / "dataset" / "rag_dataset.json"
)


def test_loads_versioned_evaluation_dataset() -> None:
    dataset = load_evaluation_dataset(DATASET_PATH)

    assert len(dataset.cases) == 10
    assert {case.answerability for case in dataset.cases} == {
        Answerability.ANSWERABLE,
        Answerability.UNANSWERABLE,
    }


def test_rejects_malformed_evaluation_record(tmp_path: Path) -> None:
    malformed = {
        "version": "1.0",
        "description": "Malformed dataset",
        "cases": [
            {
                "case_id": "missing-ground-truth",
                "category": "direct_factual",
                "question": "Question?",
                "answerability": "answerable",
                "expected_answer": "Answer.",
                "expected_key_facts": [],
                "relevant_sources": []
            }
        ]
    }
    path = tmp_path / "malformed.json"
    path.write_text(json.dumps(malformed), encoding="utf-8")

    with pytest.raises(EvaluationDatasetError, match="schema is invalid"):
        load_evaluation_dataset(path)
