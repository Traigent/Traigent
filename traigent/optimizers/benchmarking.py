"""Benchmark utilities comparing local optimizers against baselines.

.. note::
    The Optuna-based benchmark arm (``_run_optuna``) is cloud-only.  Calling
    ``run_optuna_random_parity`` raises :class:`~traigent.utils.exceptions.OptimizationError`
    because ``OptunaTPEOptimizer`` runs in the Traigent cloud and is not
    available in the local SDK.  Connect to the Traigent backend to run the
    full Optuna-vs-random parity benchmark.
"""

# Traceability: CONC-Layer-Tooling CONC-Quality-Performance CONC-Quality-Compatibility FUNC-OPT-ALGORITHMS REQ-OPT-ALG-004 SYNC-OptimizationFlow

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from statistics import mean
from typing import Any

from traigent.api.types import TrialResult, TrialStatus
from traigent.optimizers.random import RandomSearchOptimizer
from traigent.utils.exceptions import OptimizationError


@dataclass
class BenchmarkRun:
    """Summary of a single optimizer run."""

    best_value: float
    best_config: dict[str, Any]
    values: list[float]


def _objective(config: dict[str, Any]) -> float:
    """Synthetic objective with categorical + continuous interactions."""

    model_bonus = {"alpha": 0.35, "beta": 0.2, "gamma": 0.1}
    temperature: float = config["temperature"]
    base_score = 1.0 - abs(temperature - 0.25)
    return base_score + model_bonus.get(config["model"], 0.0)


def _create_trial_result(
    trial_id: int, config: dict[str, Any], score: float
) -> TrialResult:
    return TrialResult(
        trial_id=str(trial_id),
        config=config,
        metrics={"score": score},
        status=TrialStatus.COMPLETED,
        duration=0.0,
        timestamp=datetime.now(UTC),
    )


def _run_optuna(n_trials: int, seed: int) -> BenchmarkRun:
    """Run the Optuna TPE arm.

    Raises:
        OptimizationError: Always — Optuna-based optimizers are cloud-only and
            not available in the local SDK.
    """
    raise OptimizationError(
        "Smart optimization ('optuna_tpe') runs in the Traigent cloud and is not "
        "available in the local SDK (which supports 'grid' and 'random'). "
        "Connect to the Traigent backend to use smart algorithms."
    )


def _run_random(n_trials: int, seed: int) -> BenchmarkRun:
    optimizer = RandomSearchOptimizer(
        {
            "model": ["alpha", "beta", "gamma"],
            "temperature": (0.0, 1.0),
        },
        ["score"],
        max_trials=n_trials,
        random_seed=seed,
    )

    best_score = float("-inf")
    best_config: dict[str, Any] = {}
    values: list[float] = []

    for _ in range(n_trials):
        config = optimizer.suggest_next_trial([])
        score = _objective(config)
        values.append(score)
        if score > best_score:
            best_score = score
            best_config = config

    return BenchmarkRun(best_value=best_score, best_config=best_config, values=values)


def run_optuna_random_parity(
    *,
    n_trials: int = 20,
    runs: int = 5,
    seed_offset: int = 0,
) -> dict[str, Any]:
    """Run multiple parity comparisons between Optuna and random search.

    Raises:
        OptimizationError: Optuna-based optimizers are cloud-only and not
            available in the local SDK.  Connect to the Traigent backend to
            run this benchmark.
    """
    raise OptimizationError(
        "run_optuna_random_parity requires the Optuna TPE optimizer, which runs in "
        "the Traigent cloud and is not available in the local SDK.  "
        "Connect to the Traigent backend to use smart algorithms."
    )


def run_random_parity(
    *,
    n_trials: int = 20,
    runs: int = 5,
    seed_offset: int = 0,
) -> dict[str, Any]:
    """Run multiple random-search baseline runs (local-only benchmark arm).

    Args:
        n_trials: Number of trials per run.
        runs: Number of independent runs.
        seed_offset: Seed offset for reproducibility.

    Returns:
        Dictionary with ``random`` key containing aggregated metrics.
    """
    random_runs: list[BenchmarkRun] = []

    for idx in range(runs):
        seed = seed_offset + idx
        random_runs.append(_run_random(n_trials, seed))

    def _aggregate(run_list: Iterable[BenchmarkRun]) -> dict[str, Any]:
        best_values = [run.best_value for run in run_list]
        return {
            "average_best_value": mean(best_values),
            "max_best_value": max(best_values),
            "min_best_value": min(best_values),
        }

    return {
        "random": _aggregate(random_runs),
    }
