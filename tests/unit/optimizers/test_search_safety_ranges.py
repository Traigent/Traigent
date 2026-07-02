"""Regression coverage for finite range semantics and search safety."""

from __future__ import annotations

import pytest

from traigent.api.config_space import ConfigSpace
from traigent.api.constraints import require
from traigent.api.parameter_ranges import Range
from traigent.api.validation_protocol import SatStatus
from traigent.core.trial_lifecycle import TrialLifecycle
from traigent.optimizers.grid import GridSearchOptimizer
from traigent.optimizers.random import RandomSearchOptimizer
from traigent.utils.discrete_domains import discrete_cardinality_for_config_param
from traigent.utils.exceptions import OptimizationError


def test_grid_search_rejects_large_space_before_product_materialization(monkeypatch):
    """Large finite grids fail before cartesian product allocation."""

    def fail_product(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise AssertionError("grid product should not be materialized")

    monkeypatch.setattr("traigent.optimizers.grid.itertools.product", fail_product)

    with pytest.raises(OptimizationError, match="exceeding the safety cap"):
        GridSearchOptimizer(
            {f"p{i}": list(range(3)) for i in range(4)},
            ["accuracy"],
            max_grid_combinations=10,
        )


def test_grid_search_rejects_large_stepped_range_before_value_materialization(
    monkeypatch: pytest.MonkeyPatch,
):
    """Large stepped range dicts fail on cardinality before tuple allocation."""

    def fail_stepped_float_values(
        _low: float, _high: float, _step: float
    ) -> tuple[float, ...]:
        raise AssertionError("stepped values should not be materialized before cap")

    monkeypatch.setattr(
        "traigent.utils.discrete_domains.stepped_float_values",
        fail_stepped_float_values,
    )

    with pytest.raises(OptimizationError, match="exceeding the safety cap"):
        GridSearchOptimizer(
            {"temperature": {"type": "float", "low": 0.0, "high": 20.0, "step": 0.001}},
            ["accuracy"],
            max_grid_combinations=10_000,
        )


def test_grid_search_enumerates_stepped_float_range_below_cap():
    optimizer = GridSearchOptimizer(
        {"temperature": Range(0.0, 1.0, step=0.5).to_config_value()},
        ["accuracy"],
    )

    assert optimizer.total_combinations == 3
    assert optimizer._grid_points == [
        {"temperature": 0.0},
        {"temperature": 0.5},
        {"temperature": 1.0},
    ]


def test_satisfiability_returns_unknown_for_large_stepped_range_without_values(
    monkeypatch: pytest.MonkeyPatch,
):
    """Large finite SAT spaces return UNKNOWN before value materialization."""

    def fail_stepped_float_values(
        _low: float, _high: float, _step: float
    ) -> tuple[float, ...]:
        raise AssertionError("SAT domain values should not be materialized before cap")

    monkeypatch.setattr(
        "traigent_validation.validators.stepped_float_values",
        fail_stepped_float_values,
    )
    temperature = Range(0.0, 20.0, step=0.001, name="temperature")
    space = ConfigSpace(
        tvars={"temperature": temperature},
        constraints=(require(temperature.gte(0.0)),),
    )

    result = space.check_satisfiability()

    assert result.status == SatStatus.UNKNOWN
    assert "more than 10000" in result.message


def test_random_search_stepped_float_cardinality_and_exhaustion():
    optimizer = RandomSearchOptimizer(
        {"temperature": Range(0.0, 1.0, step=0.5).to_config_value()},
        ["accuracy"],
        max_trials=10,
        random_seed=7,
    )

    values = {optimizer.suggest_next_trial([])["temperature"] for _ in range(3)}

    assert optimizer.config_space_cardinality == 3
    assert values == {0.0, 0.5, 1.0}
    with pytest.raises(OptimizationError, match="Config space exhausted"):
        optimizer.suggest_next_trial([])


def test_untyped_categorical_dicts_report_choice_cardinality():
    assert discrete_cardinality_for_config_param({"values": ["a", "b"]}) == 2
    assert discrete_cardinality_for_config_param({"choices": ["a", "b"]}) == 2


def test_random_search_falls_back_to_untried_config_after_collision_streak(
    monkeypatch,
):
    optimizer = RandomSearchOptimizer({"x": [0, 1]}, ["accuracy"], max_trials=2)
    optimizer.register_tried_config({"x": 0})

    monkeypatch.setattr(optimizer, "_sample_parameter", lambda *_args: 0)

    assert optimizer.suggest_next_trial([]) == {"x": 1}


def test_random_search_large_stepped_collision_fails_without_materializing_fallback(
    monkeypatch: pytest.MonkeyPatch,
):
    optimizer = RandomSearchOptimizer(
        {"x": {"type": "float", "low": 0.0, "high": 200.0, "step": 0.001}},
        ["accuracy"],
        max_trials=2,
    )

    def always_zero(_param_name: str, _param_def: object) -> float:
        return 0.0

    def two_attempts(_cardinality: int | None) -> int:
        return 2

    def fail_discrete_values(_definition: object) -> tuple[object, ...] | None:
        raise AssertionError("large range fallback should not materialize values")

    monkeypatch.setattr(optimizer, "_sample_parameter", always_zero)
    monkeypatch.setattr(optimizer, "_max_unique_sampling_attempts", two_attempts)
    monkeypatch.setattr(
        "traigent.optimizers.random.discrete_values_for_config_param",
        fail_discrete_values,
    )

    assert optimizer.suggest_next_trial([]) == {"x": 0.0}
    with pytest.raises(OptimizationError, match="Failed to find unique config"):
        optimizer.suggest_next_trial([])


@pytest.mark.asyncio
async def test_sequential_lifecycle_treats_optimizer_exhaustion_as_clean_break():
    class ExhaustedOptimizer:
        _trial_count = 0
        _tried_config_hashes: set[str] = set()

        def suggest_next_trial(self, history):
            raise OptimizationError("Config space exhausted")

    class DummyOrchestrator:
        max_trials = None
        _trials = []
        _stop_reason = None
        optimizer = ExhaustedOptimizer()

        def _consume_default_config(self):
            return None

    orchestrator = DummyOrchestrator()
    lifecycle = TrialLifecycle(orchestrator)  # type: ignore[arg-type]

    trial_count, action = await lifecycle.run_sequential_trial(
        func=lambda: None,
        dataset=object(),  # type: ignore[arg-type]
        session_id=None,
        function_name="target",
        trial_count=0,
    )

    assert (trial_count, action) == (0, "break")
    assert orchestrator._stop_reason == "space_exhausted"


@pytest.mark.asyncio
async def test_sequential_lifecycle_propagates_non_terminal_optimizer_errors():
    class InvalidOptimizer:
        _trial_count = 0
        _tried_config_hashes: set[str] = set()

        def suggest_next_trial(self, _history: list[object]) -> dict[str, object]:
            raise OptimizationError("Invalid step for float parameter 'x': 0")

    class DummyOrchestrator:
        max_trials = None
        _trials: list[object] = []
        _stop_reason = None
        optimizer = InvalidOptimizer()

        def _consume_default_config(self) -> None:
            return None

    orchestrator = DummyOrchestrator()
    lifecycle = TrialLifecycle(orchestrator)  # type: ignore[arg-type]

    with pytest.raises(OptimizationError, match="Invalid step"):
        await lifecycle.run_sequential_trial(
            func=lambda: None,
            dataset=object(),  # type: ignore[arg-type]
            session_id=None,
            function_name="target",
            trial_count=0,
        )
    assert orchestrator._stop_reason is None


@pytest.mark.asyncio
async def test_random_search_invalid_float_step_raises_through_lifecycle():
    class DummyOrchestrator:
        max_trials = None
        _trials: list[object] = []
        _stop_reason = None
        optimizer = RandomSearchOptimizer(
            {"x": {"type": "float", "low": 0.0, "high": 1.0, "step": 0.0}},
            ["accuracy"],
            max_trials=2,
        )

        def _consume_default_config(self) -> None:
            return None

    orchestrator = DummyOrchestrator()
    lifecycle = TrialLifecycle(orchestrator)  # type: ignore[arg-type]

    with pytest.raises(OptimizationError, match="Invalid step for float parameter 'x'"):
        await lifecycle.run_sequential_trial(
            func=lambda: None,
            dataset=object(),  # type: ignore[arg-type]
            session_id=None,
            function_name="target",
            trial_count=0,
        )

    assert orchestrator._stop_reason is None


def test_random_search_stepped_float_sampling_includes_high_for_uneven_step():
    with pytest.warns(UserWarning, match="does not evenly divide span"):
        range_def = Range(0.0, 1.0, step=0.4).to_config_value()
    optimizer = RandomSearchOptimizer({"x": range_def}, ["accuracy"], max_trials=4)

    values = {optimizer.suggest_next_trial([])["x"] for _ in range(4)}

    assert values == {0.0, 0.4, 0.8, 1.0}


def test_constraint_rejected_config_restore_removes_tried_hash():
    optimizer = RandomSearchOptimizer({"x": [0, 1]}, ["accuracy"], random_seed=1)

    class DummyOrchestrator:
        pass

    orchestrator = DummyOrchestrator()
    orchestrator.optimizer = optimizer
    lifecycle = TrialLifecycle(orchestrator)  # type: ignore[arg-type]

    state = lifecycle._snapshot_optimizer_state()
    optimizer.suggest_next_trial([])
    assert optimizer.trial_count == 1
    assert optimizer.unique_configs_tried == 1

    lifecycle._restore_optimizer_state(state)

    assert optimizer.trial_count == 0
    assert optimizer.unique_configs_tried == 0
