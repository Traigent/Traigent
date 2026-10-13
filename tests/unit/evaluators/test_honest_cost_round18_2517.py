"""Round 18 (#2517): a partly-unmeasured trial fails closed with a fix."""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

import pytest

import traigent
from traigent.api.decorators import EvaluationOptions, ExecutionOptions
from traigent.core.cost_enforcement import (
    CostEnforcer,
    CostEnforcerConfig,
)
from traigent.utils.exceptions import OptimizationError

_COST_ENV = (
    "TRAIGENT_STRICT_COST_ACCOUNTING",
    "TRAIGENT_REQUIRE_COST_TRACKING",
    "TRAIGENT_RUN_COST_LIMIT",
)


def _dataset(tmp_path: Path) -> str:
    path = tmp_path / "ds.jsonl"
    with open(path, "w") as f:
        for i in range(5):
            f.write(json.dumps({"input": {"q": f"q{i}"}, "output": "a"}) + "\n")
    return str(path)


def _run(tmp_path: Path, monkeypatch, *, wrap: bool, objectives, **extra):
    for key in _COST_ENV:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("TRAIGENT_MOCK_LLM", "true")
    monkeypatch.setenv("TRAIGENT_OFFLINE_MODE", "true")
    monkeypatch.setenv("TRAIGENT_DATASET_ROOT", str(tmp_path))

    @traigent.optimize(
        configuration_space={"model": ["gpt-4o-mini", "gpt-4o"]},
        objectives=objectives,
        evaluation=EvaluationOptions(eval_dataset=_dataset(tmp_path)),
        execution=ExecutionOptions(offline=True),
        max_trials=2,
        **extra,
    )
    def agent(q: str):
        traigent.get_config()
        if q == "q4":
            if wrap:
                return traigent.with_usage("a", total_cost=0)
            return "a"  # guard-rail fast path: no LLM call made
        return traigent.with_usage(
            "a", total_cost=0.01, input_tokens=100, output_tokens=10
        )

    return asyncio.run(agent.optimize())


def test_cost_objective_with_plain_return_example_raises_actionable_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(OptimizationError) as exc:
        _run(tmp_path, monkeypatch, wrap=False, objectives=["accuracy", "cost"])
    message = str(exc.value)
    assert "example_4" in message
    assert "with_usage" in message
    assert "TRAIGENT_STRICT_COST_ACCOUNTING=false" in message


def test_wrapping_the_no_call_path_completes_at_measured_cost(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result = _run(tmp_path, monkeypatch, wrap=True, objectives=["accuracy", "cost"])
    costs = [t.metrics.get("cost") for t in result.trials]
    assert costs == [pytest.approx(0.04), pytest.approx(0.04)]


def test_flagged_trial_counts_as_unmeasured_and_charges_floor(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    for key in _COST_ENV:
        monkeypatch.delenv(key, raising=False)
    enforcer = CostEnforcer(CostEnforcerConfig(limit=10.0))
    permit = enforcer.acquire_permit()
    with caplog.at_level(logging.WARNING):
        enforcer.track_cost(
            0.04,
            permit=permit,
            cost_unmeasured=True,
            unmeasured_examples=["example_4"],
        )
    assert enforcer.get_status().unmeasured_trial_count == 1
    assert enforcer.accumulated_cost == pytest.approx(0.04)
    assert "example_4" in caplog.text


def test_cost_limit_only_run_continues_charges_floor_and_warns(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setenv("TRAIGENT_RUN_COST_LIMIT", "5.0")
    with caplog.at_level(logging.WARNING):
        result = _run(tmp_path, monkeypatch, wrap=False, objectives=["accuracy"])
    assert len(result.trials) == 2
    assert all(t.metrics.get("cost_unmeasured") for t in result.trials)
    assert "example_4" in caplog.text


def test_unmeasured_example_ids_never_return_content() -> None:
    from traigent.utils.trial_costs import unmeasured_example_ids

    rows = [
        {"example_id": "a", "metrics": {"total_cost": 0.01}, "success": True},
        {"example_id": "b", "metrics": {"accuracy": 1.0}, "success": True},
        {"example_id": "c", "metrics": {}, "success": False},
        {"example_id": "", "metrics": {"cost_unmeasured": 1.0}, "success": True},
    ]
    assert unmeasured_example_ids(rows) == ["b", "#3"]
    assert unmeasured_example_ids(rows, limit=1) == ["b"]
    assert unmeasured_example_ids(None) == []
