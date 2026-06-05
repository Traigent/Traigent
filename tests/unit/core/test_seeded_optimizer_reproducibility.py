"""Regression tests for canonical optimizer seeding and retired mock keys."""

from __future__ import annotations

import logging
from typing import Any

import pytest

from traigent.api.decorators import optimize
from traigent.api.types import OptimizationResult
from traigent.optimizers.registry import register_optuna_optimizers

CONFIG_SPACE = {
    "model": ["gpt-3.5-turbo", "gpt-4"],
    "temperature": [0.3, 0.7],
}
EVAL_DATASET = [{"input": {"text": "hello"}, "expected": "hello"}]


def _build_decorated_target(*, mock: dict[str, Any] | None = None) -> Any:
    @optimize(
        configuration_space=CONFIG_SPACE,
        objectives=["accuracy"],
        injection={"injection_mode": "parameter", "config_param": "traigent_config"},
        evaluation={"eval_dataset": EVAL_DATASET},
        mock=mock,
    )
    def target(text: str, traigent_config: dict[str, Any] | None = None) -> str:
        return text

    return target


def _trial_configs(result: OptimizationResult) -> list[dict[str, Any]]:
    return [dict(trial.config) for trial in result.trials]


@pytest.mark.asyncio
@pytest.mark.parametrize("algorithm", ["grid", "optuna_tpe", "optuna_random"])
async def test_seeded_canonical_optimizer_runs_produce_identical_trial_configs(
    algorithm: str,
) -> None:
    if algorithm.startswith("optuna"):
        register_optuna_optimizers(force=True)

    first = _build_decorated_target()
    second = _build_decorated_target()

    result1 = await first.optimize(
        algorithm=algorithm,
        random_seed=42,
        max_trials=4,
    )
    result2 = await second.optimize(
        algorithm=algorithm,
        random_seed=42,
        max_trials=4,
    )

    assert _trial_configs(result1) == _trial_configs(result2)


@pytest.mark.asyncio
async def test_legacy_mock_optimizer_keys_warn_without_changing_algorithm(
    caplog: pytest.LogCaptureFixture,
) -> None:
    decorated = _build_decorated_target(mock={"optimizer": "grid", "random_seed": 42})

    caplog.set_level(logging.WARNING, logger="traigent.core.optimized_function")
    result = await decorated.optimize(max_trials=2)

    assert result.algorithm == "RandomSearchOptimizer"
    assert any(
        "inert post-F5" in record.getMessage()
        and "decorated.optimize(algorithm=..., random_seed=...)" in record.getMessage()
        for record in caplog.records
    )
