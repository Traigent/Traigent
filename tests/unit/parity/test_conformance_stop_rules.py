"""Cross-SDK conformance: optimize stop rules.

The vendored fixture (TraigentSchema at the ref pinned in
``fixtures.lock.json``) is the answer key; expectations come from it only.
Rules covered: crashed trials, the default cost cap, plateau stopping, hard
limit precedence over plateau, the arithmetic-mean pre-trial cost estimate and
refusal of a cost cap without reported costs.

Every run goes through the public ``@traigent.optimize(...)`` decorator with
``algorithm="grid"`` fully offline and a custom evaluator that returns the
fixture's scripted accuracy and cost, bound to the grid configuration (not to
call order). A scripted trial with ``raises: true`` raises inside the
evaluator. The fixture's ``comparison`` block is binding: absolute tolerance
1e-9 on best accuracy, and for ``rejects`` cases only that the SDK rejects (no
error class, message or rejection point is asserted).
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

FIXTURE_ID = "optimize.stop-rules.v1"
TOL = 1e-9

# Fixture stop-reason vocabulary -> Python SDK ``OptimizationResult.stop_reason``.
STOP_REASON_MAP = {
    "max_trials": "max_trials_reached",
    "cost_limit": "cost_limit",
    "max_examples": "max_samples_reached",
    "plateau": "plateau",
}
# Fixture trial-status vocabulary -> Python SDK ``TrialStatus`` values.
STATUS_MAP = {"completed": "completed", "failed": "failed"}


class ScriptedTrialError(RuntimeError):
    """Raised inside the evaluator for a scripted ``raises: true`` trial."""


def _load() -> dict[str, Any]:
    path = _verified_fixture_path(LOCK_PATH, FIXTURES_DIR, FIXTURE_ID)
    data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    assert data["fixtureId"] == FIXTURE_ID
    return data


FIXTURE = _load()
CASES: list[dict[str, Any]] = FIXTURE["cases"]
RUNS = [c for c in CASES if not c["expected"].get("rejects")]
REJECTS = [c for c in CASES if c["expected"].get("rejects")]


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
    # The "default cap" cases require that no run cost limit is named. An
    # earlier test in the same process may have leaked one into os.environ
    # (tests/unit/identity/no_grant_snapshot.py used setdefault), which would
    # replace the 2.0 default cap, so the ambient value is removed here.
    monkeypatch.delenv("TRAIGENT_RUN_COST_LIMIT", raising=False)

    space = case_input["configurationSpace"]
    assert list(space) == ["variant"]
    assert case_input["algorithm"] == "grid"
    assert case_input["concurrency"] == 1
    variants = list(space["variant"])
    scripted = dict(zip(variants, case_input["scriptedTrials"], strict=True))
    rows_seen: dict[str, int] = {}

    def evaluator(func: Any, config: dict[str, Any], example: Any) -> ExampleResult:
        script = scripted[config["variant"]]
        rows_seen[config["variant"]] = rows_seen.get(config["variant"], 0) + 1
        if script.get("raises"):
            raise ScriptedTrialError(f"scripted failure for {config['variant']}")
        metrics: dict[str, Any] = {"accuracy": float(script["accuracy"])}
        if "cost" in script:
            metrics["cost"] = script["cost"]
        return ExampleResult(
            example_id=f"row{rows_seen[config['variant']]}",
            input_data=example.input_data,
            expected_output=example.expected_output,
            actual_output="ok",
            metrics=metrics,
            execution_time=0.001,
            success=True,
        )

    rows = int(case_input.get("datasetRows", 1))
    dataset = Dataset(
        [EvaluationExample({"text": f"r{i}"}, "ok") for i in range(rows)],
        name="parity_stop_rules",
    )

    kwargs: dict[str, Any] = {"max_trials": case_input.get("maxTrials")}
    if "costCap" in case_input:
        kwargs["cost_limit"] = case_input["costCap"]
        kwargs["cost_approved"] = True
    if "maxTotalExamples" in case_input:
        kwargs["max_total_examples"] = case_input["maxTotalExamples"]
    if "plateau" in case_input:
        plateau = case_input["plateau"]
        kwargs["plateau_window"] = plateau["window"]
        kwargs["plateau_epsilon"] = plateau["tolerance"]

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


def test_lock_entry_pins_the_vendored_fixture() -> None:
    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    entry = lock["fixtures"][FIXTURE_ID]
    assert entry["path"] == "optimize/stop-rules.v1.json"
    assert (FIXTURES_DIR / entry["path"]).is_file()
    # Pinned to the TraigentSchema commit that carries the fixture: a 40-hex
    # SHA once the schema PR merges, the explicit placeholder until then.
    ref = entry["schemaRef"]
    assert ref == "PENDING-MERGE-SHA-OF-TRAIGENTSCHEMA-547" or (
        len(ref) == 40 and all(c in "0123456789abcdef" for c in ref)
    )


def test_vendored_fixture_edit_fails_lock_check(tmp_path: Path) -> None:
    edited = tmp_path / "optimize" / "stop-rules.v1.json"
    edited.parent.mkdir()
    original = (FIXTURES_DIR / "optimize" / "stop-rules.v1.json").read_bytes()
    edited.write_bytes(original + b" ")
    with pytest.raises(AssertionError, match="sha256"):
        _verified_fixture_path(LOCK_PATH, tmp_path, FIXTURE_ID)


def test_fixture_covers_expected_cases() -> None:
    ids = [c["id"] for c in CASES]
    assert ids == [
        "crashed-trial-continues",
        "default-cap-applies",
        "plateau-fires",
        "plateau-tolerance-zero",
        "plateau-not-fired-when-improving",
        "tie-max-trials-over-plateau",
        "tie-max-examples-over-plateau",
        "estimator-uses-mean",
        "plateau-within-nonzero-tolerance",
        "default-cap-crossing-trial-counts",
        "estimate-exactly-equals-cap-starts",
        "crashed-trial-counts-toward-max-trials",
        "cost-cap-no-cost-reported-refused",
    ]
    assert len(RUNS) == 12 and len(REJECTS) == 1
    assert "1e-9" in FIXTURE["comparison"]["numericTolerance"]


@pytest.mark.parametrize("case", RUNS, ids=lambda c: c["id"])
def test_stop_rules_via_public_optimize(
    case: dict[str, Any], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    expected = case["expected"]
    result, rows_seen = _run(case["input"], monkeypatch, tmp_path)

    if "trialCount" in expected:
        assert len(result.trials) == expected["trialCount"], case["id"]
    if "trialStatuses" in expected:
        statuses = [
            str(getattr(t.status, "value", t.status)).lower() for t in result.trials
        ]
        assert statuses == [STATUS_MAP[s] for s in expected["trialStatuses"]], case[
            "id"
        ]
    if "stopReason" in expected:
        assert result.stop_reason == STOP_REASON_MAP[expected["stopReason"]], case["id"]

    if "bestAccuracy" in expected:
        completed = [
            t.metrics["accuracy"]
            for t in result.trials
            if str(getattr(t.status, "value", t.status)).lower() == "completed"
        ]
        assert max(completed) == pytest.approx(expected["bestAccuracy"], abs=TOL)

    if expected.get("runCompletes"):
        assert result is not None

    if "rowsEvaluatedPerTrial" in expected:
        variants = case["input"]["configurationSpace"]["variant"]
        per_trial = [rows_seen[v] for v in variants[: expected["trialCount"]]]
        assert per_trial == expected["rowsEvaluatedPerTrial"]
        assert sum(rows_seen.values()) == expected["totalRowsEvaluated"]


@pytest.mark.parametrize("case", REJECTS, ids=lambda c: c["id"])
def test_cost_cap_without_reported_cost_is_rejected(
    case: dict[str, Any], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Only that the SDK rejects; no class, message or point asserted.
    with pytest.raises(Exception):  # noqa: B017, PT011
        _run(case["input"], monkeypatch, tmp_path)
