"""Cross-SDK conformance: multi-objective normalization and selection.

The vendored fixture (TraigentSchema at the ref pinned in
``fixtures.lock.json``) is the answer key. Every assertion goes through public
APIs: ``OptimizationResult.score_trials`` with an ``ObjectiveSchema`` for
per-trial scoring, and ``traigent.optimize(offline=True, algorithm="grid")``
for end-to-end best-configuration selection. The fixture's ``comparison``
section is binding: absolute tolerance 1e-9, trial order, unique-maximum best
trial, and a validation error (before scoring) for a dominated weight vector.
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest

import traigent
from traigent.api.types import OptimizationResult, TrialResult, TrialStatus
from traigent.core.objectives import ObjectiveDefinition, ObjectiveSchema
from traigent.evaluators.base import Dataset, EvaluationExample, ExampleResult

from .test_conformance_grid import FIXTURES_DIR, LOCK_PATH, _verified_fixture_path

FIXTURE_ID = "multi-objective.normalization-selection.v1"
TOL = 1e-9


def _load() -> dict[str, Any]:
    path = _verified_fixture_path(LOCK_PATH, FIXTURES_DIR, FIXTURE_ID)
    data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    assert data["fixtureId"] == FIXTURE_ID
    return data


FIXTURE = _load()
CASES: list[dict[str, Any]] = FIXTURE["cases"]
SCORED = [c for c in CASES if "validationError" not in c["expected"]]
REJECTED = [c for c in CASES if c["expected"].get("validationError")]
SELECTED = [c for c in SCORED if c["expected"]["bestTrialId"] is not None]


def _schema(case: dict[str, Any]) -> ObjectiveSchema:
    return ObjectiveSchema.from_objectives(
        [
            ObjectiveDefinition(
                name=o["metric"],
                orientation=o["direction"],
                weight=float(o["weight"]),
            )
            for o in case["input"]["objectives"]
        ]
    )


def _result(case: dict[str, Any]) -> OptimizationResult:
    trials = [
        TrialResult(
            trial_id=t["trial_id"],
            config={"trial": t["trial_id"]},
            metrics=dict(t["metrics"]),
            status=TrialStatus.COMPLETED,
            duration=1.0,
            timestamp=datetime.now(),
        )
        for t in case["input"]["trials"]
    ]
    return OptimizationResult(
        trials=trials,
        best_config=trials[0].config,
        best_score=0.0,
        optimization_id=f"fixture_{case['id']}",
        duration=float(len(trials)),
        convergence_info={},
        status="completed",
        objectives=[o["metric"] for o in case["input"]["objectives"]],
        algorithm="fixture",
        timestamp=datetime.now(),
    )


def test_fixture_covers_expected_cases() -> None:
    ids = {c["id"] for c in CASES}
    assert len(CASES) == 8 and len(ids) == 8
    assert len(REJECTED) == 1 and len(SCORED) == 7


@pytest.mark.parametrize("case", SCORED, ids=lambda c: c["id"])
def test_per_trial_scores_via_score_trials(case: dict[str, Any]) -> None:
    schema = _schema(case)
    expected = case["expected"]

    for metric, want in expected["normalizedWeights"].items():
        assert schema.get_normalized_weight(metric) == pytest.approx(want, abs=TOL)

    actual = _result(case).score_trials(objective_schema=schema)
    assert [a["trial_id"] for a in actual] == [
        t["trial_id"] for t in expected["trial_scores"]
    ]
    for got, want in zip(actual, expected["trial_scores"], strict=True):
        for metric, value in want["normalized"].items():
            assert metric in got["normalized"], (got["trial_id"], metric)
            assert got["normalized"][metric] == pytest.approx(value, abs=TOL), (
                case["id"],
                got["trial_id"],
                metric,
            )
        assert got["weighted"] == pytest.approx(want["weighted"], abs=TOL), (
            case["id"],
            got["trial_id"],
        )

    best_id = expected["bestTrialId"]
    if best_id is not None:
        top = max(a["weighted"] for a in actual)
        winners = [a["trial_id"] for a in actual if a["weighted"] == top]
        assert winners == [best_id]


@pytest.mark.parametrize("case", SELECTED, ids=lambda c: c["id"])
def test_best_config_via_public_optimize(
    case: dict[str, Any], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TRAIGENT_OFFLINE_MODE", "true")
    monkeypatch.setenv("TRAIGENT_MOCK_LLM", "true")
    monkeypatch.setenv("TRAIGENT_RESULTS_FOLDER", str(tmp_path / "results"))
    # Fixture metrics named "cost" are read as spend by the budget gate; raise
    # the run limit so the gate does not truncate the grid being compared.
    monkeypatch.setenv("TRAIGENT_RUN_COST_LIMIT", "1000")

    metrics_by_trial = {
        t["trial_id"]: {k: float(v) for k, v in t["metrics"].items()}
        for t in case["input"]["trials"]
    }

    def evaluator(func: Any, config: dict[str, Any], example: Any) -> ExampleResult:
        return ExampleResult(
            example_id="fixture",
            input_data=example.input_data,
            expected_output=example.expected_output,
            actual_output="ok",
            metrics=dict(metrics_by_trial[config["trial"]]),
            execution_time=0.001,
            success=True,
        )

    dataset = Dataset([EvaluationExample({"text": "a"}, "ok")], name="parity_mo")

    @traigent.optimize(
        eval_dataset=dataset,
        objectives=_schema(case),
        configuration_space={"trial": list(metrics_by_trial)},
        custom_evaluator=evaluator,
        injection_mode="parameter",
        offline=True,
        max_trials=len(metrics_by_trial),
        algorithm="grid",
    )
    def stub(text: str, config: dict[str, Any]) -> str:
        return "ok"

    result = asyncio.run(stub.optimize())

    assert len(result.trials) == len(metrics_by_trial)
    assert result.best_config == {"trial": case["expected"]["bestTrialId"]}


@pytest.mark.parametrize("case", REJECTED, ids=lambda c: c["id"])
def test_dominance_guard_rejects_before_scoring(case: dict[str, Any]) -> None:
    # [199, 1] -> 0.995 > 0.99: construction fails, so nothing can be scored.
    with pytest.raises(ValueError):
        _schema(case)


def test_dominance_guard_boundary_accepted() -> None:
    boundary = next(c for c in SCORED if c["id"] == "dominance_guard_boundary_allowed")
    weights = [o["weight"] for o in boundary["input"]["objectives"]]
    assert weights == [99, 1]
    assert _schema(boundary).get_normalized_weight("quality") == pytest.approx(
        0.99, abs=TOL
    )
