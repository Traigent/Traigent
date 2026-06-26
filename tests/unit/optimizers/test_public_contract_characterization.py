"""Characterize retained optimizer public contracts during Optuna eviction."""

from __future__ import annotations

import pytest

import traigent.optimizers as optimizer_exports
from traigent.config.types import (
    accepted_algorithm_values,
    resolve_execution_policy,
    validate_algorithm_name,
)
from traigent.core.stop_conditions import (
    CostLimitStopCondition,
    HypervolumeConvergenceStopCondition,
    MaxSamplesStopCondition,
    MaxTrialsStopCondition,
    MetricLimitStopCondition,
    PlateauAfterNStopCondition,
)
from traigent.optimizers import get_optimizer, list_optimizers
from traigent.utils.exceptions import ConfigurationError, OptimizationError

EXPECTED_OPTIMIZER_ALL = [
    "BaseOptimizer",
    "GridSearchOptimizer",
    "RandomSearchOptimizer",
    "BatchOptimizationConfig",
    "ParallelBatchOptimizer",
    "MultiObjectiveBatchOptimizer",
    "AdaptiveBatchOptimizer",
    "_is_smart_algorithm",
    "get_optimizer",
    "register_optimizer",
    "list_optimizers",
    # Intentional public-surface change: local Optuna pruner exports were removed.
    "RemoteOptimizer",
    "CloudOptimizer",
]

EXPECTED_REGISTERED_OPTIMIZERS = [
    "grid",
    "random",
    "parallel_batch",
    "multi_objective_batch",
    "adaptive_batch",
    "remote",
]

SMART_OPTIMIZER_NAMES = [
    "bayesian",
    "optuna",
    "tpe",
    "optuna_tpe",
    "optuna_random",
    "optuna_grid",
    "optuna_cmaes",
    "optuna_nsga2",
    "nsga2",
    "cmaes",
]


def test_optimizer_public_exports_match_baseline() -> None:
    assert optimizer_exports.__all__ == EXPECTED_OPTIMIZER_ALL


def test_list_optimizers_matches_registered_names() -> None:
    assert list_optimizers() == EXPECTED_REGISTERED_OPTIMIZERS


@pytest.mark.parametrize("name", SMART_OPTIMIZER_NAMES)
def test_smart_optimizer_names_fail_loud_to_cloud(name: str) -> None:
    with pytest.raises(OptimizationError) as exc_info:
        get_optimizer(name, {}, [])

    message = str(exc_info.value).lower()
    assert "cloud" in message
    assert "grid" in message
    assert "random" in message


@pytest.mark.parametrize("name", ["auto", "grid", "random", *SMART_OPTIMIZER_NAMES])
def test_validate_algorithm_name_accepts_public_names(name: str) -> None:
    assert validate_algorithm_name(name) == name


def test_validate_algorithm_name_rejects_unknown_optuna_variant() -> None:
    with pytest.raises(ValueError) as exc_info:
        validate_algorithm_name("optuna_foo")

    message = str(exc_info.value)
    for algorithm_name in accepted_algorithm_values():
        assert algorithm_name in message
    assert "optuna_foo" in message


@pytest.mark.parametrize("name", SMART_OPTIMIZER_NAMES)
def test_offline_smart_algorithm_raises_configuration_error(name: str) -> None:
    with pytest.raises(ConfigurationError, match="offline=True"):
        resolve_execution_policy(algorithm=name, offline=True)


class _ObjectiveSchemaStub:
    def compute_weighted_score(self, metrics: dict[str, float]) -> float:
        return float(metrics.get("score", 0.0))


class _CostEnforcerStub:
    is_limit_reached = False

    def get_status(self):
        return type(
            "CostStatus",
            (),
            {
                "unknown_cost_mode": False,
                "trial_count": 0,
                "accumulated_cost_usd": 0.0,
                "limit_usd": 1.0,
            },
        )()


def test_local_stop_conditions_import_and_instantiate() -> None:
    assert MaxTrialsStopCondition(max_trials=3)
    assert MaxSamplesStopCondition(max_samples=10)
    assert MetricLimitStopCondition(limit=1.0, metric_name="score")
    assert PlateauAfterNStopCondition(
        window_size=2,
        epsilon=0.01,
        objective_schema=_ObjectiveSchemaStub(),
    )
    assert CostLimitStopCondition(cost_enforcer=_CostEnforcerStub())
    assert HypervolumeConvergenceStopCondition(
        window=2,
        threshold=0.01,
        objective_names=["quality", "cost"],
        directions=["maximize", "minimize"],
    )
