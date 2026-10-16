"""Cross-SDK conformance: optimize stop limits (max trials, cost cap, examples).

The vendored fixture (TraigentSchema at the ref pinned in
``fixtures.lock.json``) is the answer key; expectations come from it only.
Every run goes through the public ``@traigent.optimize(...)`` decorator with
``algorithm="grid"`` fully offline and a custom evaluator that returns the
fixture's scripted accuracy and cost, bound to the grid configuration (not to
call order). The fixture's ``comparison`` block is binding: absolute tolerance
1e-9 on best accuracy, and for ``rejects`` cases only that the SDK rejects
(no error class, message or rejection point is asserted).
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

import traigent
from traigent.evaluators.base import Dataset, EvaluationExample, ExampleResult

from .test_conformance_grid import FIXTURES_DIR, LOCK_PATH, _verified_fixture_path

FIXTURE_ID = "optimize.stop-limits.v1"
TOL = 1e-9

# Fixture stop-reason vocabulary -> Python SDK ``OptimizationResult.stop_reason``.
STOP_REASON_MAP = {
    "max_trials": "max_trials_reached",
    "cost_limit": "cost_limit",
    "max_examples": "max_samples_reached",
}


def _load() -> dict[str, Any]:
    path = _verified_fixture_path(LOCK_PATH, FIXTURES_DIR, FIXTURE_ID)
    data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    assert data["fixtureId"] == FIXTURE_ID
    return data


FIXTURE = _load()
CASES: list[dict[str, Any]] = FIXTURE["cases"]
RUNS = [c for c in CASES if not c["expected"].get("rejects")]
REJECTS = [c for c in CASES if c["expected"].get("rejects")]


def _reject_variants(case: dict[str, Any]) -> list[tuple[str, Any, dict[str, Any]]]:
    """Return (label, value, input-overrides) for each invalid value to reject."""
    inp = case["input"]
    out: list[tuple[str, Any, dict[str, Any]]] = []
    for value in inp.get("maxTrialsValues", []):
        out.append(("maxTrials", value, {"maxTrials": value}))
    for value in inp.get("costCapValues", []):
        # JSON cannot hold NaN: the fixture encodes it as the string "NaN".
        real = float("nan") if value == "NaN" else value
        out.append(("costCap", real, {"costCap": real}))
    if not out:
        out.append(("scriptedTrials", None, {}))
    return out


def _run(
    case_input: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> tuple[Any, dict[str, int]]:
    """Drive the public optimize API; return (result, rows evaluated per variant)."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TRAIGENT_OFFLINE_MODE", "true")
    monkeypatch.setenv("TRAIGENT_MOCK_LLM", "true")
    monkeypatch.setenv("TRAIGENT_RESULTS_FOLDER", str(tmp_path / "results"))

    space = case_input["configurationSpace"]
    assert list(space) == ["variant"]
    variants = list(space["variant"])
    scripted = dict(zip(variants, case_input["scriptedTrials"], strict=True))
    rows_seen: dict[str, int] = {}

    def evaluator(func: Any, config: dict[str, Any], example: Any) -> ExampleResult:
        script = scripted[config["variant"]]
        rows_seen[config["variant"]] = rows_seen.get(config["variant"], 0) + 1
        return ExampleResult(
            example_id=f"row{rows_seen[config['variant']]}",
            input_data=example.input_data,
            expected_output=example.expected_output,
            actual_output="ok",
            metrics={"accuracy": float(script["accuracy"]), "cost": script["cost"]},
            execution_time=0.001,
            success=True,
        )

    rows = int(case_input.get("datasetRows", 1))
    dataset = Dataset(
        [EvaluationExample({"text": f"r{i}"}, "ok") for i in range(rows)],
        name="parity_stop_limits",
    )

    kwargs: dict[str, Any] = {
        "max_trials": case_input.get("maxTrials"),
        "cost_limit": case_input["costCap"] if "costCap" in case_input else None,
        "cost_approved": True,
    }
    if "maxTotalExamples" in case_input:
        kwargs["max_total_examples"] = case_input["maxTotalExamples"]

    @traigent.optimize(
        eval_dataset=dataset,
        objectives=["accuracy"],
        configuration_space={"variant": variants},
        custom_evaluator=evaluator,
        injection_mode="parameter",
        offline=True,
        algorithm="grid",
        **{k: v for k, v in kwargs.items() if v is not None},
    )
    def stub(text: str, config: dict[str, Any]) -> str:
        return "ok"

    return asyncio.run(stub.optimize()), rows_seen


def test_fixture_covers_expected_cases() -> None:
    ids = [c["id"] for c in CASES]
    assert ids == [
        "max-trials-below-space",
        "max-trials-invalid",
        "cost-cap-crossing-trial-counts",
        "cost-cap-first-trial-exceeds",
        "cost-cap-invalid",
        "max-examples-truncation",
        "negative-trial-cost",
    ]
    assert len(RUNS) == 4 and len(REJECTS) == 3
    assert "1e-9" in FIXTURE["comparison"]["numericTolerance"]


@pytest.mark.parametrize("case", RUNS, ids=lambda c: c["id"])
def test_stop_limits_via_public_optimize(
    case: dict[str, Any], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    expected = case["expected"]
    result, rows_seen = _run(case["input"], monkeypatch, tmp_path)

    assert len(result.trials) == expected["trialCount"], case["id"]
    assert result.stop_reason == STOP_REASON_MAP[expected["stopReason"]], case["id"]

    accuracies = [t.metrics["accuracy"] for t in result.trials]
    assert max(accuracies) == pytest.approx(expected["bestAccuracy"], abs=TOL)

    if "rowsEvaluatedPerTrial" in expected:
        variants = case["input"]["configurationSpace"]["variant"]
        per_trial = [rows_seen[v] for v in variants[: expected["trialCount"]]]
        assert per_trial == expected["rowsEvaluatedPerTrial"]
        assert sum(rows_seen.values()) == expected["totalRowsEvaluated"]


@pytest.mark.parametrize("case", REJECTS, ids=lambda c: c["id"])
def test_invalid_values_are_rejected(
    case: dict[str, Any], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Only that the SDK rejects, per value; no class, message or point asserted.
    for _label, _value, override in _reject_variants(case):
        run_input = {**case["input"], **override}
        with pytest.raises(Exception), pytest.MonkeyPatch.context() as mp:  # noqa: B017, PT011
            _run(run_input, mp, tmp_path)
