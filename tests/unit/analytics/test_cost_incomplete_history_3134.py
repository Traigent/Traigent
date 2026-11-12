"""#3134: damaged history lines must not drive cost-opportunity claims."""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from traigent.analytics.cost_optimization import CostOptimizationAI
from traigent.analytics.meta_learning import OptimizationHistory


def _line(i: int) -> str:
    return json.dumps(
        {
            "optimization_id": f"o{i}",
            "function_name": "f",
            "algorithm": "grid",
            "configuration_space": {"p": [1]},
            "objectives": ["accuracy"],
            "dataset_size": 1,
            "best_score": 0.5,
            "total_trials": 1,
            "duration_seconds": 1,
            "status": "completed",
            "timestamp": datetime.now(UTC).isoformat(),
            "metadata": {"cost_reduction": 50},
        }
    )


def _opportunities(history_text: str, tmp_path: Path) -> list[dict]:
    path = tmp_path / "h.jsonl"
    path.write_text(history_text, encoding="utf-8")
    history = OptimizationHistory(storage_path=str(path))
    ai = CostOptimizationAI()
    ai.optimization_history = history
    return ai.analyze_cost_patterns()["optimization_opportunities"], history


def test_complete_history_still_reports_pattern_replication(tmp_path: Path) -> None:
    opps, history = _opportunities(_line(1) + "\n", tmp_path)
    assert history.records and history._skipped_lines == 0
    assert any(o["type"] == "pattern_replication" for o in opps)


def test_damaged_line_counted_and_opportunities_withheld(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.WARNING):
        opps, history = _opportunities(
            _line(1) + "\n{torn\n" + _line(2) + "\n", tmp_path
        )
    assert history._skipped_lines == 1
    assert not any(o["type"] == "pattern_replication" for o in opps)
    assert "incomplete" in caplog.text


def test_fake_history_with_skipped_lines_withholds() -> None:
    ai = CostOptimizationAI()
    ai.optimization_history = SimpleNamespace(
        records=[{"cost_reduction": 50}], _skipped_lines=2
    )
    # records are dicts here; _get_cost_reduction reads dicts directly.
    assert not any(
        o["type"] == "pattern_replication"
        for o in ai.analyze_cost_patterns()["optimization_opportunities"]
    )
    ai.optimization_history = SimpleNamespace(
        records=[{"cost_reduction": 50}], _skipped_lines=0
    )
    assert any(
        o["type"] == "pattern_replication"
        for o in ai.analyze_cost_patterns()["optimization_opportunities"]
    )
