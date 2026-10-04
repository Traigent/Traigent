from __future__ import annotations

import json
from pathlib import Path

from traigent.metrics.ragas_metrics import POPULAR_RAGAS_METRICS

REPO_ROOT = Path(__file__).resolve().parents[3]
RAGAS_EXAMPLES = REPO_ROOT / "examples" / "advanced" / "ragas"


def _load_jsonl(path: Path) -> list[dict]:
    # The datasets are tracked in git; a missing file is a regression, not a skip.
    assert path.is_file(), f"Tracked RAGAS example dataset is missing: {path}"
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def test_ragas_basics_dataset_has_required_fields() -> None:
    rows = _load_jsonl(RAGAS_EXAMPLES / "basics" / "evaluation_set.jsonl")
    assert len(rows) >= 2
    for row in rows:
        assert "input" in row and "question" in row["input"]
        assert "output" in row
        assert "retrieved_contexts" in row
        assert "reference_contexts" in row


def test_ragas_with_llm_dataset_has_required_fields() -> None:
    rows = _load_jsonl(RAGAS_EXAMPLES / "with_llm" / "evaluation_set.jsonl")
    assert len(rows) >= 2
    for row in rows:
        assert "input" in row and "question" in row["input"]
        assert "output" in row
        assert "retrieved_contexts" in row
        assert "reference_contexts" in row


def test_ragas_column_map_dataset_matches_custom_keys() -> None:
    rows = _load_jsonl(RAGAS_EXAMPLES / "column_map" / "evaluation_set.jsonl")
    assert len(rows) >= 2
    for row in rows:
        assert "input" in row and "prompt" in row["input"]
        assert "output" in row
        assert "gold_contexts" in row
        assert "reference_answer" in row


def test_ragas_examples_skip_when_metrics_missing() -> None:
    # Ensure tests don't fail when ragas extras are absent.
    assert isinstance(POPULAR_RAGAS_METRICS, tuple)
