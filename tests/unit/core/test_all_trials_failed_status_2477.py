"""#2477: a local run whose every executed trial failed is not COMPLETED.

``optimize()`` returned ``status=COMPLETED`` with ``best_config=None`` and
``success_rate=0.0`` when every trial failed, so a status check (or an
example script's exit code) counted a total failure as a pass. The #1691
downgrade cannot reach this case: its predicate skips failed trials.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import traigent
from traigent.api.decorators import EvaluationOptions
from traigent.api.types import OptimizationStatus


def _dataset(tmp_path: Path) -> str:
    data = tmp_path / "qa.jsonl"
    data.write_text(
        "".join(
            json.dumps({"input": {"q": q}, "output": q.upper()}) + "\n"
            for q in ("a", "b", "c")
        )
    )
    return str(data)


def _run(tmp_path: Path, metric):
    @traigent.optimize(
        evaluation=EvaluationOptions(
            eval_dataset=_dataset(tmp_path), metric_functions={"accuracy": metric}
        ),
        objectives=["accuracy"],
        configuration_space={"temperature": [0.0, 0.5]},
        offline=True,
        algorithm="grid",
    )
    def answer(q: str) -> str:
        traigent.get_config()
        return q.upper()

    return answer.optimize_sync(max_trials=2)


@pytest.fixture(autouse=True)
def _offline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TRAIGENT_OFFLINE_MODE", "true")
    monkeypatch.setenv("TRAIGENT_DATASET_ROOT", str(tmp_path))


def test_every_trial_failing_marks_the_run_failed(tmp_path: Path) -> None:
    def broken(output, expected, **_):
        raise RuntimeError("judge down")

    result = _run(tmp_path, broken)

    assert result.trials and all(not t.is_successful for t in result.trials)
    assert result.best_config is None
    assert result.status == OptimizationStatus.FAILED
    assert "ALL_TRIALS_FAILED" in result.warning_codes
    assert any("judge down" in w for w in result.warnings)


def test_a_successful_trial_keeps_the_run_completed(tmp_path: Path) -> None:
    def exact(output, expected, **_):
        return float(output == expected)

    result = _run(tmp_path, exact)

    assert result.status == OptimizationStatus.COMPLETED
    assert "ALL_TRIALS_FAILED" not in result.warning_codes
