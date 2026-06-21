"""Capture the pre-eviction oracle for the retired local Optuna pruners.

Run this script before removing ``traigent.optimizers.pruners``.  The emitted
JSON is intentionally small and descriptive so the future cloud implementation
can be checked against the decisions the local SDK used to make.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import optuna

from traigent.optimizers.pruners import CeilingPruner, StatisticalInferiorityPruner

optuna.logging.set_verbosity(optuna.logging.WARNING)


OUTPUT_PATH = Path(__file__).with_name("pruner_decisions.json")


def _distribution_for(value: float) -> optuna.distributions.FloatDistribution:
    padding = max(abs(value), 1.0)
    return optuna.distributions.FloatDistribution(value - padding, value + padding)


def _completed_trial(params: dict[str, float], value: float) -> optuna.trial.FrozenTrial:
    distributions = {name: _distribution_for(raw) for name, raw in params.items()}
    return optuna.trial.create_trial(
        params=params,
        distributions=distributions,
        value=value,
        state=optuna.trial.TrialState.COMPLETE,
    )


def _running_trial(
    *,
    number: int,
    params: dict[str, float] | None = None,
    intermediate_values: dict[int, float] | None = None,
) -> optuna.trial.FrozenTrial:
    params = params or {}
    distributions = {name: _distribution_for(raw) for name, raw in params.items()}
    return optuna.trial.FrozenTrial(
        number=number,
        state=optuna.trial.TrialState.RUNNING,
        value=None,
        datetime_start=None,
        datetime_complete=None,
        params=params,
        distributions=distributions,
        user_attrs={},
        system_attrs={},
        intermediate_values=intermediate_values or {},
        trial_id=number,
    )


def _study(
    *,
    direction: str,
    completed: list[tuple[dict[str, float], float]],
) -> optuna.Study:
    study = optuna.create_study(direction=direction)
    for params, value in completed:
        study.add_trial(_completed_trial(params, value))
    return study


def _decision(value: bool) -> str:
    return "prune" if value else "keep"


def _ceiling_scenarios() -> dict[str, dict[str, Any]]:
    cases: dict[str, dict[str, Any]] = {}

    completed_baseline = [
        ({}, 0.80),
        ({}, 0.76),
        ({}, 0.72),
    ]

    warmup_study = _study(direction="maximize", completed=completed_baseline)
    warmup_pruner = CeilingPruner(
        min_completed_trials=3,
        warmup_steps=3,
        epsilon=0.05,
        cost_threshold=None,
    )
    warmup_trial = _running_trial(number=100, intermediate_values={2: 0.10})
    warmup_decision = warmup_pruner.prune(warmup_study, warmup_trial)
    cases["ceiling/warmup-not-met -> keep"] = {
        "description": "Latest reported step is below warmup_steps, so no ceiling comparison is allowed.",
        "pruner": "CeilingPruner",
        "config": {
            "min_completed_trials": 3,
            "warmup_steps": 3,
            "epsilon": 0.05,
            "cost_threshold": None,
        },
        "study": {
            "direction": "maximize",
            "completed_values": [value for _, value in completed_baseline],
        },
        "trial": {"intermediate_values": {"2": 0.10}, "last_step": 2},
        "decision": _decision(warmup_decision),
        "pruned": warmup_decision,
    }

    below_best_study = _study(direction="maximize", completed=completed_baseline)
    below_best_pruner = CeilingPruner(
        min_completed_trials=3,
        warmup_steps=1,
        epsilon=0.05,
        cost_threshold=None,
    )
    below_best_trial = _running_trial(number=101, intermediate_values={1: 0.74})
    below_best_decision = below_best_pruner.prune(below_best_study, below_best_trial)
    cases["ceiling/optimistic-ceiling-below-best -> prune"] = {
        "description": "For maximize, latest <= best_completed - epsilon (0.74 <= 0.80 - 0.05).",
        "pruner": "CeilingPruner",
        "config": {
            "min_completed_trials": 3,
            "warmup_steps": 1,
            "epsilon": 0.05,
            "cost_threshold": None,
        },
        "study": {
            "direction": "maximize",
            "completed_values": [value for _, value in completed_baseline],
        },
        "trial": {"intermediate_values": {"1": 0.74}, "last_step": 1},
        "decision": _decision(below_best_decision),
        "pruned": below_best_decision,
    }

    competitive_study = _study(direction="maximize", completed=completed_baseline)
    competitive_pruner = CeilingPruner(
        min_completed_trials=3,
        warmup_steps=1,
        epsilon=0.05,
        cost_threshold=None,
    )
    competitive_trial = _running_trial(number=102, intermediate_values={1: 0.77})
    competitive_decision = competitive_pruner.prune(
        competitive_study, competitive_trial
    )
    cases["ceiling/competitive -> keep"] = {
        "description": "For maximize, latest remains above best_completed - epsilon (0.77 > 0.75).",
        "pruner": "CeilingPruner",
        "config": {
            "min_completed_trials": 3,
            "warmup_steps": 1,
            "epsilon": 0.05,
            "cost_threshold": None,
        },
        "study": {
            "direction": "maximize",
            "completed_values": [value for _, value in completed_baseline],
        },
        "trial": {"intermediate_values": {"1": 0.77}, "last_step": 1},
        "decision": _decision(competitive_decision),
        "pruned": competitive_decision,
    }

    cost_study = _study(direction="minimize", completed=[])
    cost_pruner = CeilingPruner(
        min_completed_trials=3,
        warmup_steps=1,
        epsilon=0.05,
        cost_threshold=1.0,
    )
    cost_trial = _running_trial(number=103, intermediate_values={1: 1.25})
    cost_decision = cost_pruner.prune(cost_study, cost_trial)
    cases["ceiling/cost-threshold-exceeded -> prune"] = {
        "description": "Scalar latest value exceeds absolute cost_threshold before completed-trial gating.",
        "pruner": "CeilingPruner",
        "config": {
            "min_completed_trials": 3,
            "warmup_steps": 1,
            "epsilon": 0.05,
            "cost_threshold": 1.0,
        },
        "study": {"direction": "minimize", "completed_values": []},
        "trial": {"intermediate_values": {"1": 1.25}, "last_step": 1},
        "decision": _decision(cost_decision),
        "pruned": cost_decision,
    }

    return cases


def _statistical_scenarios() -> dict[str, dict[str, Any]]:
    cases: dict[str, dict[str, Any]] = {}

    sufficient_best = [
        ({"learning_rate": 0.01}, 0.90),
        ({"learning_rate": 0.01}, 0.91),
        ({"learning_rate": 0.01}, 0.89),
    ]
    clearly_inferior = [
        ({"learning_rate": 0.20}, 0.50),
        ({"learning_rate": 0.20}, 0.51),
        ({"learning_rate": 0.20}, 0.49),
    ]
    overlapping_best = [
        ({"learning_rate": 0.01}, 0.80),
        ({"learning_rate": 0.01}, 0.82),
        ({"learning_rate": 0.01}, 0.78),
    ]
    overlapping_candidate = [
        ({"learning_rate": 0.02}, 0.79),
        ({"learning_rate": 0.02}, 0.81),
        ({"learning_rate": 0.02}, 0.77),
    ]

    insufficient_completed = sufficient_best + [
        ({"learning_rate": 0.20}, 0.50),
        ({"learning_rate": 0.20}, 0.51),
    ]
    insufficient_study = _study(direction="maximize", completed=insufficient_completed)
    insufficient_pruner = StatisticalInferiorityPruner(
        confidence=0.95,
        min_samples_per_config=3,
        warmup_trials=5,
    )
    insufficient_trial = _running_trial(number=200, params={"learning_rate": 0.20})
    insufficient_decision = insufficient_pruner.prune(
        insufficient_study, insufficient_trial
    )
    cases["statistical/insufficient-samples -> keep"] = {
        "description": "The candidate config has only two completed samples, below min_samples_per_config.",
        "pruner": "StatisticalInferiorityPruner",
        "config": {
            "confidence": 0.95,
            "min_samples_per_config": 3,
            "warmup_trials": 5,
        },
        "study": {
            "direction": "maximize",
            "completed_trials": [
                {"params": params, "value": value}
                for params, value in insufficient_completed
            ],
        },
        "trial": {"params": {"learning_rate": 0.20}, "last_step": None},
        "decision": _decision(insufficient_decision),
        "pruned": insufficient_decision,
    }

    inferior_completed = sufficient_best + clearly_inferior
    inferior_study = _study(direction="maximize", completed=inferior_completed)
    inferior_pruner = StatisticalInferiorityPruner(
        confidence=0.95,
        min_samples_per_config=3,
        warmup_trials=5,
    )
    inferior_trial = _running_trial(number=201, params={"learning_rate": 0.20})
    inferior_decision = inferior_pruner.prune(inferior_study, inferior_trial)
    cases["statistical/clearly-inferior-UCB-below-best-LCB -> prune"] = {
        "description": "The candidate config's upper confidence bound is below the best config's lower confidence bound.",
        "pruner": "StatisticalInferiorityPruner",
        "config": {
            "confidence": 0.95,
            "min_samples_per_config": 3,
            "warmup_trials": 5,
        },
        "study": {
            "direction": "maximize",
            "completed_trials": [
                {"params": params, "value": value}
                for params, value in inferior_completed
            ],
        },
        "trial": {"params": {"learning_rate": 0.20}, "last_step": None},
        "decision": _decision(inferior_decision),
        "pruned": inferior_decision,
    }

    overlapping_completed = overlapping_best + overlapping_candidate
    overlapping_study = _study(direction="maximize", completed=overlapping_completed)
    overlapping_pruner = StatisticalInferiorityPruner(
        confidence=0.95,
        min_samples_per_config=3,
        warmup_trials=5,
    )
    overlapping_trial = _running_trial(number=202, params={"learning_rate": 0.02})
    overlapping_decision = overlapping_pruner.prune(
        overlapping_study, overlapping_trial
    )
    cases["statistical/overlapping-CI -> keep"] = {
        "description": "The candidate config's confidence interval overlaps the leading config, so it is not inferior.",
        "pruner": "StatisticalInferiorityPruner",
        "config": {
            "confidence": 0.95,
            "min_samples_per_config": 3,
            "warmup_trials": 5,
        },
        "study": {
            "direction": "maximize",
            "completed_trials": [
                {"params": params, "value": value}
                for params, value in overlapping_completed
            ],
        },
        "trial": {"params": {"learning_rate": 0.02}, "last_step": None},
        "decision": _decision(overlapping_decision),
        "pruned": overlapping_decision,
    }

    warmup_completed = [
        ({"learning_rate": 0.01}, 0.90),
        ({"learning_rate": 0.20}, 0.50),
        ({"learning_rate": 0.20}, 0.51),
    ]
    warmup_study = _study(direction="maximize", completed=warmup_completed)
    warmup_pruner = StatisticalInferiorityPruner(
        confidence=0.95,
        min_samples_per_config=2,
        warmup_trials=5,
    )
    warmup_trial = _running_trial(number=203, params={"learning_rate": 0.20})
    warmup_decision = warmup_pruner.prune(warmup_study, warmup_trial)
    cases["statistical/warmup-not-met -> keep"] = {
        "description": "Total completed trials are below warmup_trials, so no statistical comparison is allowed.",
        "pruner": "StatisticalInferiorityPruner",
        "config": {
            "confidence": 0.95,
            "min_samples_per_config": 2,
            "warmup_trials": 5,
        },
        "study": {
            "direction": "maximize",
            "completed_trials": [
                {"params": params, "value": value}
                for params, value in warmup_completed
            ],
        },
        "trial": {"params": {"learning_rate": 0.20}, "last_step": None},
        "decision": _decision(warmup_decision),
        "pruned": warmup_decision,
    }

    return cases


def main() -> int:
    scenarios = {
        "schema_version": 1,
        "purpose": (
            "Immutable before-oracle for local Optuna pruner decisions captured "
            "before moving smart pruning to the cloud."
        ),
        "scenarios": {
            **_ceiling_scenarios(),
            **_statistical_scenarios(),
        },
    }
    OUTPUT_PATH.write_text(json.dumps(scenarios, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {OUTPUT_PATH}")
    for name, payload in scenarios["scenarios"].items():
        print(f"{name}: {payload['decision']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
