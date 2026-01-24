from __future__ import annotations

import json
from pathlib import Path

from traigent.metrics.ragas_metrics import POPULAR_RAGAS_METRICS


def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def test_ragas_basics_dataset_has_required_fields() -> None:
    path = Path("examples/advanced/ragas/basics/evaluation_set.jsonl")
    rows = _load_jsonl(path)
    assert len(rows) >= 2
    for row in rows:
        assert "input" in row and "question" in row["input"]
        assert "output" in row
        assert "retrieved_contexts" in row
        assert "reference_contexts" in row


def test_ragas_with_llm_dataset_has_reference_field() -> None:
    path = Path("examples/advanced/ragas/with_llm/evaluation_set.jsonl")
    rows = _load_jsonl(path)
    assert len(rows) >= 2
    for row in rows:
        assert "reference" in row
        assert "retrieved_contexts" in row
        assert "reference_contexts" in row


def test_ragas_column_map_dataset_matches_custom_keys() -> None:
    path = Path("examples/advanced/ragas/column_map/evaluation_set.jsonl")
    rows = _load_jsonl(path)
    assert len(rows) == 1
    row = rows[0]
    assert "gold_contexts" in row
    assert "reference_answer" in row


def test_ragas_examples_skip_when_metrics_missing() -> None:
    # Ensure tests don't fail when ragas extras are absent.
    assert isinstance(POPULAR_RAGAS_METRICS, tuple)
