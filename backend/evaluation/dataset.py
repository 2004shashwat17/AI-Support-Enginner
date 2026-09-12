import json
from pathlib import Path

from pydantic import ValidationError

from evaluation.models import EvaluationDataset


class EvaluationDatasetError(ValueError):
    """Raised when an evaluation dataset cannot be loaded safely."""


def load_evaluation_dataset(path: Path) -> EvaluationDataset:
    try:
        raw_dataset = json.loads(path.read_text(encoding="utf-8"))
        return EvaluationDataset.model_validate(raw_dataset)
    except OSError as exc:
        raise EvaluationDatasetError(f"Could not read evaluation dataset: {path}") from exc
    except json.JSONDecodeError as exc:
        raise EvaluationDatasetError("Evaluation dataset is not valid JSON.") from exc
    except ValidationError as exc:
        raise EvaluationDatasetError("Evaluation dataset schema is invalid.") from exc
