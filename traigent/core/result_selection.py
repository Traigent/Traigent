"""Utilities for selecting the best configuration from completed trials."""

# Traceability: CONC-Layer-Core CONC-Quality-Performance FUNC-ORCH-LIFECYCLE REQ-ORCH-003 SYNC-OptimizationFlow

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from traigent.api.types import TrialResult

__all__ = [
    "SelectionResult",
    "select_best_configuration",
]


@dataclass(slots=True)
class SelectionResult:
    """Best-configuration selection output for orchestrator result building."""

    best_config: dict[str, Any]
    best_score: float
    session_summary: dict[str, Any] | None


def _is_minimization_objective(objective: str) -> bool:
    minimize_patterns = ("cost", "latency", "error", "loss", "time", "duration")
    objective_lower = objective.lower()
    return any(pattern in objective_lower for pattern in minimize_patterns)


def select_best_configuration(
    trials: Iterable[TrialResult],
    primary_objective: str,
    *,
    config_space_keys: Iterable[str],
    aggregate_configs: bool,
) -> SelectionResult:
    """Return the configuration that best satisfies the primary objective."""
    successful_trials = [t for t in trials if t.is_successful]
    if not successful_trials:
        return SelectionResult(best_config={}, best_score=0.0, session_summary=None)

    minimization = _is_minimization_objective(primary_objective)

    if not aggregate_configs:
        chooser = min if minimization else max
        best_trial = chooser(
            successful_trials,
            key=lambda t: t.get_metric(primary_objective, 0.0) or 0.0,
        )
        return SelectionResult(
            best_config=best_trial.config or {},
            best_score=best_trial.get_metric(primary_objective, 0.0) or 0.0,
            session_summary=None,
        )

    from traigent.utils.hashing import generate_config_hash

    keys = set(config_space_keys)
    replicate_counters: dict[str, int] = {}
    aggregated: dict[str, dict[str, Any]] = {}

    for trial in successful_trials:
        cfg = trial.config or {}
        filtered_cfg = {k: cfg.get(k) for k in keys} if keys else cfg
        config_hash = generate_config_hash(filtered_cfg)

        replicate_counters[config_hash] = replicate_counters.get(config_hash, 0) + 1
        trial.metadata = getattr(trial, "metadata", {})
        trial.metadata["replicate_index"] = replicate_counters[config_hash]

        entry = aggregated.setdefault(
            config_hash,
            {"config": trial.config, "metrics_sum": {}, "count": 0},
        )

        for metric_name, metric_value in (trial.metrics or {}).items():
            if metric_name == "examples_attempted":
                continue
            if isinstance(metric_value, (int, float)):
                entry["metrics_sum"][metric_name] = entry["metrics_sum"].get(
                    metric_name, 0.0
                ) + float(metric_value)

        entry["count"] += 1

    if not aggregated:
        return SelectionResult(best_config={}, best_score=0.0, session_summary=None)

    def mean_metrics(entry: dict[str, Any]) -> dict[str, float]:
        count = entry["count"] or 1
        return {
            metric: value / count
            for metric, value in entry.get("metrics_sum", {}).items()
        }

    def score(entry: dict[str, Any]) -> float:
        return mean_metrics(entry).get(primary_objective, 0.0)

    chooser = min if minimization else max
    best_entry = chooser(aggregated.values(), key=score)
    best_metrics = mean_metrics(best_entry)

    allowed_unprefixed = {"accuracy", "latency", "cost", "custom_metric"}
    sanitized_metrics: dict[str, float] = {}
    for metric_name, metric_value in best_metrics.items():
        target_key = (
            metric_name if metric_name in allowed_unprefixed else f"run_{metric_name}"
        )
        sanitized_metrics[target_key] = metric_value

    session_summary = {
        "selection_mode": "aggregated_mean",
        "primary_objective": primary_objective,
        "samples_per_config": {
            key: entry["count"] for key, entry in aggregated.items()
        },
        "metrics": sanitized_metrics,
        "sanitized": True,
    }

    return SelectionResult(
        best_config=best_entry["config"] or {},
        best_score=best_metrics.get(primary_objective, 0.0),
        session_summary=session_summary,
    )
