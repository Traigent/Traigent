"""Regression tests for canonical optimizer seeding and removed mock parameters."""

from __future__ import annotations

from typing import Any

import pytest

from traigent.api.decorators import optimize
from traigent.api.types import OptimizationResult

CONFIG_SPACE = {
    "model": ["gpt-3.5-turbo", "gpt-4"],
    "temperature": [0.3, 0.7],
}
EVAL_DATASET = [{"input": {"text": "hello"}, "expected": "hello"}]


def _build_decorated_target(**extra: Any) -> Any:
    @optimize(
        configuration_space=CONFIG_SPACE,
        objectives=["accuracy"],
        injection={"injection_mode": "parameter", "config_param": "traigent_config"},
        evaluation={"eval_dataset": EVAL_DATASET},
        **extra,
    )
    def target(text: str, traigent_config: dict[str, Any] | None = None) -> str:
        return text

    return target


def _trial_configs(result: OptimizationResult) -> list[dict[str, Any]]:
    return [dict(trial.config) for trial in result.trials]


@pytest.mark.asyncio
@pytest.mark.parametrize("algorithm", ["grid", "random"])
async def test_seeded_canonical_optimizer_runs_produce_identical_trial_configs(
    algorithm: str,
) -> None:
    # Smart algorithms (optuna_*) are no longer available locally; the local
    # SDK supports only grid and random, both of which must be reproducible.
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


@pytest.mark.parametrize("name", ["mock", "mock_mode_config"])
def test_removed_mock_optimizer_keys_are_rejected(name: str) -> None:
    """The former inert-warning case: the parameter now fails loudly."""
    with pytest.raises(TypeError, match=name):
        _build_decorated_target(**{name: {"optimizer": "grid", "random_seed": 42}})
