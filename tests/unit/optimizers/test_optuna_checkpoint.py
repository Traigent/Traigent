"""Tests for Optuna checkpoint persistence."""

from __future__ import annotations

import optuna

from traigent.optimizers.optuna_checkpoint import OptunaCheckpointManager
from traigent.optimizers.optuna_optimizer import OptunaTPEOptimizer
from traigent.telemetry.optuna_metrics import sanitize_config

CONFIG_SPACE = {"model": ["m1", "m2"], "temperature": (0.0, 1.0)}


def test_checkpoint_roundtrip(tmp_path):
    checkpoint_path = tmp_path / "optuna" / "checkpoint.json"
    manager = OptunaCheckpointManager(checkpoint_path)

    sampler = optuna.samplers.RandomSampler(seed=0)
    optimizer = OptunaTPEOptimizer(
        CONFIG_SPACE,
        ["score"],
        max_trials=2,
        sampler=sampler,
        checkpoint_manager=manager,
    )

    history = []
    config = optimizer.suggest_next_trial(history)
    assert checkpoint_path.exists()

    sanitized = sanitize_config(config)

    # Simulate crash by not reporting result and creating a fresh optimizer
    sampler2 = optuna.samplers.RandomSampler(seed=0)
    restored = OptunaTPEOptimizer(
        CONFIG_SPACE,
        ["score"],
        max_trials=2,
        sampler=sampler2,
        checkpoint_manager=manager,
    )

    restored_config = restored.suggest_next_trial([])
    assert sanitize_config(restored_config) == sanitized

    trial_id = restored_config["_optuna_trial_id"]
    restored.report_trial_result(trial_id, 0.9)
    assert not checkpoint_path.exists()
