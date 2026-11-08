"""Duplicate values in a list knob must not run duplicate trials (#2954).

Grid search used to enumerate the list as written, so ``[1, 1, 2]`` ran three
trials for two configurations, while random search deduplicated by its
tried-config identity and reported a different stop reason. Values are now
deduplicated with the identity random search already uses (first occurrence
kept), so both algorithms see the same space and the same cardinality.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

import traigent
from traigent.evaluators.base import Dataset, EvaluationExample, ExampleResult
from traigent.optimizers.grid import GridSearchOptimizer
from traigent.optimizers.random import RandomSearchOptimizer
from traigent.utils.discrete_domains import (
    discrete_cardinality_for_config_param,
    discrete_values_for_config_param,
)


@pytest.mark.parametrize(
    ("space", "expected"),
    [
        ([1, 1, 2], (1, 2)),
        ([2, 1, 2, 1], (2, 1)),
        (["a", "a"], ("a",)),
        ([{"k": 1}, {"k": 1}, {"k": 2}], ({"k": 1}, {"k": 2})),
        ({"type": "categorical", "choices": ["x", "y", "x"]}, ("x", "y")),
    ],
)
def test_discrete_helpers_dedupe_first_occurrence(space: Any, expected: tuple) -> None:
    assert discrete_values_for_config_param(space) == expected
    assert discrete_cardinality_for_config_param(space) == len(expected)


def test_bool_and_float_stay_distinct_from_int() -> None:
    # Same identity as random search's tried-config hash: True is not 1, 1.0 is not 1.
    values = discrete_values_for_config_param([1.0, 1, True, 1])
    assert values == (1.0, 1, True)
    assert [type(v) for v in values] == [float, int, bool]


def test_grid_enumerates_each_distinct_value_once() -> None:
    grid = GridSearchOptimizer({"a": [1, 1, 2], "b": ["x", "x"]}, ["accuracy"])
    assert grid.total_combinations == 2
    assert grid.config_space_cardinality == 2
    assert grid.get_remaining_combinations() == [
        {"a": 1, "b": "x"},
        {"a": 2, "b": "x"},
    ]


def test_random_cardinality_counts_distinct_values() -> None:
    opt = RandomSearchOptimizer({"a": [1, 1, 2]}, ["accuracy"], max_trials=10)
    assert opt.config_space_cardinality == 2
    seen = [opt.suggest_next_trial([]) for _ in range(2)]
    assert sorted(c["a"] for c in seen) == [1, 2]
    assert opt.should_stop([]) is True


def _run(algorithm: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TRAIGENT_OFFLINE_MODE", "true")
    monkeypatch.setenv("TRAIGENT_MOCK_LLM", "true")
    monkeypatch.setenv("TRAIGENT_RESULTS_FOLDER", str(tmp_path / "results"))
    monkeypatch.delenv("TRAIGENT_RUN_COST_LIMIT", raising=False)

    def evaluator(func: Any, config: dict[str, Any], example: Any) -> ExampleResult:
        return ExampleResult(
            example_id="row",
            input_data=example.input_data,
            expected_output=example.expected_output,
            actual_output="ok",
            metrics={"accuracy": 0.5, "cost": 0.01},
            execution_time=0.001,
            success=True,
        )

    @traigent.optimize(
        eval_dataset=Dataset([EvaluationExample({"text": "r"}, "ok")], name="dups"),
        objectives=["accuracy"],
        configuration_space={"a": [1, 1, 2]},
        custom_evaluator=evaluator,
        injection_mode="parameter",
        offline=True,
        algorithm=algorithm,
        max_trials=10,
        cost_limit=1000.0,
        cost_approved=True,
    )
    def stub(text: str, config: dict[str, Any]) -> str:
        return "ok"

    return asyncio.run(stub.optimize())


@pytest.mark.parametrize("algorithm", ["grid", "random"])
def test_end_to_end_runs_two_trials_and_same_stop_reason(
    algorithm: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result = _run(algorithm, tmp_path, monkeypatch)
    assert sorted(t.config["a"] for t in result.trials) == [1, 2]
    assert result.stop_reason == "optimizer"
