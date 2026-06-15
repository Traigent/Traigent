"""Unit tests for benchmarking helpers."""

from __future__ import annotations

import pytest

from traigent.optimizers.benchmarking import run_optuna_random_parity, run_random_parity
from traigent.utils.exceptions import OptimizationError


def test_run_optuna_random_parity_raises_cloud_error():
    """run_optuna_random_parity is cloud-only and must raise OptimizationError."""
    with pytest.raises(OptimizationError) as exc_info:
        run_optuna_random_parity(n_trials=5, runs=2, seed_offset=3)
    msg = str(exc_info.value)
    assert "cloud" in msg.lower()
    assert "Traigent" in msg


def test_run_random_parity_shape():
    """Local random-only benchmark arm returns expected shape."""
    results = run_random_parity(n_trials=5, runs=2, seed_offset=3)
    assert "random" in results
    assert "average_best_value" in results["random"]
    assert "max_best_value" in results["random"]
    assert "min_best_value" in results["random"]
