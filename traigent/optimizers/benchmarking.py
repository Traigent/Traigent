"""Benchmark utilities comparing Optuna optimizers against baselines."""

# Traceability: CONC-Layer-Tooling CONC-Quality-Performance CONC-Quality-Compatibility FUNC-OPT-ALGORITHMS REQ-OPT-ALG-004 SYNC-OptimizationFlow

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from statistics import mean
from typing import Any, Iterable

import optuna

from traigent.api.types import TrialResult, TrialStatus
from traigent.optimizers.optuna_optimizer import OptunaTPEOptimizer
from traigent.optimizers.random import RandomSearchOptimizer


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
        timestamp=datetime.now(timezone.utc),
    )


def _run_optuna(n_trials: int, seed: int) -> BenchmarkRun:
    startup_trials = max(1, n_trials // 3)
    sampler = optuna.samplers.TPESampler(
        seed=seed,
        n_startup_trials=min(5, startup_trials),
        consider_endpoints=True,
    )
    optimizer = OptunaTPEOptimizer(
        {
            "model": ["alpha", "beta", "gamma"],
            "temperature": (0.0, 1.0),
        },
        ["score"],
        max_trials=n_trials,
        sampler=sampler,
    )

    history: list[TrialResult] = []
    best_score = float("-inf")
    best_config: dict[str, Any] = {}
    values: list[float] = []

    for _ in range(n_trials):
        config = optimizer.suggest_next_trial(history)
        score = _objective(config)
        values.append(score)
        optimizer.report_trial_result(config["_optuna_trial_id"], score)
        trial_result = _create_trial_result(config["_optuna_trial_id"], config, score)
        history.append(trial_result)
        if score > best_score:
            best_score = score
            best_config = config

    return BenchmarkRun(best_value=best_score, best_config=best_config, values=values)


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
    """Run multiple parity comparisons between Optuna and random search."""

    optuna_runs: list[BenchmarkRun] = []
    random_runs: list[BenchmarkRun] = []

    for idx in range(runs):
        seed = seed_offset + idx
        optuna_runs.append(_run_optuna(n_trials, seed))
        random_runs.append(_run_random(n_trials, seed))

    def _aggregate(runs: Iterable[BenchmarkRun]) -> dict[str, Any]:
        best_values = [run.best_value for run in runs]
        return {
            "average_best_value": mean(best_values),
            "max_best_value": max(best_values),
            "min_best_value": min(best_values),
        }

    return {
        "optuna": _aggregate(optuna_runs),
        "random": _aggregate(random_runs),
    }
