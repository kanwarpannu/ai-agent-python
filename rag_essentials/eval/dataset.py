from __future__ import annotations

from pathlib import Path

import yaml

from ..exceptions import ConfigurationError
from .models import EvalCase


def load_dataset(path: str | Path) -> list[EvalCase]:
    """Load and validate the YAML evaluation dataset.

    The file is a list of mappings, each with ``id`` and ``question`` (required)
    and optional ``reference_answer`` / ``relevant_pages``.
    """
    dataset_path = Path(path)
    if not dataset_path.exists():
        raise ConfigurationError(f"Eval dataset not found: {dataset_path}")

    raw = yaml.safe_load(dataset_path.read_text(encoding="utf-8"))
    if not isinstance(raw, list) or not raw:
        raise ConfigurationError(
            f"Eval dataset {dataset_path} must be a non-empty YAML list of cases"
        )

    try:
        cases = [EvalCase.model_validate(item) for item in raw]
    except Exception as exc:  # pragma: no cover - surfaced to the CLI
        raise ConfigurationError(f"Invalid eval dataset {dataset_path}: {exc}") from exc

    ids = [c.id for c in cases]
    if len(set(ids)) != len(ids):
        raise ConfigurationError(f"Duplicate case ids in {dataset_path}: {ids}")

    return cases
