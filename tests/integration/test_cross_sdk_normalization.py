"""Cross-SDK normalization parity test.

Asserts the Python SDK's ``score_trials()`` and weight-rescaling logic match
the numerical contract pinned in ``tests/fixtures/multi_objective_normalization.json``.

The traigent-js SDK has a parallel test
(``traigent-js/tests/cross-sdk/multi-objective-normalization.test.ts``)
asserting its repo-local mirror of the same fixture. Drift on either side
fails its own suite; agreement on the fixture is the cross-SDK contract.

Contract source: TraigentSchema/optimization/multi_objective_semantics_schema.json v1.0.0.
"""
from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

import pytest

from traigent.api.types import OptimizationResult, TrialResult, TrialStatus
from traigent.core.objectives import ObjectiveDefinition, ObjectiveSchema


def _locate_fixture() -> Path:
    """Find the normalization fixture.

    CI uses the repo-local fixture. Local enterprise parity runs may override
    with TRAIGENT_CROSS_SDK_FIXTURE_PATH or use the workspace-level
    integrations/traigent-cross-sdk-benchmarks copy when present.
    """
    override = os.environ.get("TRAIGENT_CROSS_SDK_FIXTURE_PATH")
    if override and Path(override).exists():
        return Path(override)

    local = Path(__file__).resolve().parent.parent / "fixtures" / "multi_objective_normalization.json"
    if local.exists():
        return local

    cursor = Path(__file__).resolve()
    for _ in range(10):
        candidate = (
            cursor.parent
            / "integrations"
            / "traigent-cross-sdk-benchmarks"
            / "fixtures"
            / "multi_objective_normalization.json"
        )
        if candidate.exists():
            return candidate
        if cursor.parent == cursor:
            break
        cursor = cursor.parent
    raise FileNotFoundError(
        "Could not locate tests/fixtures/multi_objective_normalization.json; set "
        "TRAIGENT_CROSS_SDK_FIXTURE_PATH to a compatible fixture copy."
    )


FIXTURE_PATH = _locate_fixture()
FIXTURE = json.loads(FIXTURE_PATH.read_text())
TOLERANCE = float(FIXTURE["_meta"]["tolerance"])


def _build_schema(case_objectives: list[dict]) -> ObjectiveSchema:
    """Build an ObjectiveSchema from the fixture objectives.

    Mirrors the JS SDK's `normalizeObjectiveWeights` chokepoint: the schema
    constructor performs sum-to-one rescaling and dominance-guard validation.
    """
    objs = [
        ObjectiveDefinition(
            name=o["metric"],
            orientation=o["direction"],  # "maximize" | "minimize"
            weight=float(o["weight"]),
        )
        for o in case_objectives
    ]
    return ObjectiveSchema.from_objectives(objs)


def _build_optimization_result(
    case_id: str, case_trials: list[dict], case_objectives: list[dict]
) -> OptimizationResult:
    trials = [
        TrialResult(
            trial_id=t["trial_id"],
            config={},
            metrics=dict(t["metrics"]),
            status=TrialStatus.COMPLETED,
            duration=1.0,
            timestamp=datetime.now(),
        )
        for t in case_trials
    ]
    first_trial = trials[0] if trials else None
    return OptimizationResult(
        trials=trials,
        best_config=first_trial.config if first_trial else {},
        best_score=0.0,
        optimization_id=f"fixture_{case_id}",
        duration=float(len(trials)),
        convergence_info={},
        status="completed",
        objectives=[o["metric"] for o in case_objectives],
        algorithm="fixture",
        timestamp=datetime.now(),
    )


_normal_cases = [c for c in FIXTURE["cases"] if not c.get("expects_validation_error")]
_violation_cases = [c for c in FIXTURE["cases"] if c.get("expects_validation_error")]


@pytest.mark.parametrize("case", _normal_cases, ids=lambda c: c["id"])
def test_per_trial_normalized_values_match_fixture(case: dict) -> None:
    """Per-trial normalized values and weighted scores match the fixture."""
    schema = _build_schema(case["objectives"])
    result = _build_optimization_result(case["id"], case["trials"], case["objectives"])

    actual = result.score_trials(objective_schema=schema)
    expected = case["expected"]["trial_scores"]

    assert len(actual) == len(expected), (
        f"{case['id']}: trial count mismatch (got {len(actual)}, want {len(expected)})"
    )

    for got, want in zip(actual, expected):
        assert got["trial_id"] == want["trial_id"], (
            f"{case['id']}: trial order mismatch"
        )
        for metric, expected_value in want["normalized"].items():
            assert metric in got["normalized"], (
                f"{case['id']}/{got['trial_id']}: missing normalized {metric}"
            )
            assert got["normalized"][metric] == pytest.approx(
                expected_value, abs=TOLERANCE
            ), (
                f"{case['id']}/{got['trial_id']}/{metric}: "
                f"got {got['normalized'][metric]}, want {expected_value}"
            )
        assert got["weighted"] == pytest.approx(want["weighted"], abs=TOLERANCE), (
            f"{case['id']}/{got['trial_id']}: weighted score "
            f"got {got['weighted']}, want {want['weighted']}"
        )


@pytest.mark.parametrize("case", _normal_cases, ids=lambda c: c["id"])
def test_weights_rescale_to_sum_to_one(case: dict) -> None:
    """ObjectiveSchema construction rescales raw weights to sum-to-one."""
    schema = _build_schema(case["objectives"])
    total = sum(schema.weights_normalized.values())
    assert total == pytest.approx(1.0, abs=TOLERANCE)

    raw_total = sum(o["weight"] for o in case["objectives"])
    for raw in case["objectives"]:
        expected = raw["weight"] / raw_total
        assert schema.weights_normalized[raw["metric"]] == pytest.approx(
            expected, abs=TOLERANCE
        )


@pytest.mark.parametrize("case", _violation_cases, ids=lambda c: c["id"])
def test_dominance_guard_violations_raise(case: dict) -> None:
    """Cases marked `expects_validation_error` must raise at schema build."""
    with pytest.raises(ValueError):
        _build_schema(case["objectives"])
