"""Integration tests ensuring Optuna optimizers emit telemetry events."""

from __future__ import annotations

import optuna

from traigent.optimizers.optuna_optimizer import OptunaTPEOptimizer
from traigent.security.enterprise import MetricsCollector
from traigent.telemetry.optuna_metrics import OptunaMetricsEmitter


def test_optuna_optimizer_emits_telemetry_events():
    collector = MetricsCollector()
    emitter = OptunaMetricsEmitter(metrics_collector=collector)

    optimizer = OptunaTPEOptimizer(
        {"temperature": (0.0, 1.0)},
        ["score"],
        max_trials=1,
        sampler=optuna.samplers.RandomSampler(seed=1),
        metrics_emitter=emitter,
    )

    config = optimizer.suggest_next_trial([])
    optimizer.report_trial_result(config["_optuna_trial_id"], 0.5)

    events = collector.get_optuna_events()
    event_names = {event["event"] for event in events}
    assert {"trial_suggested", "trial_completed"}.issubset(event_names)
