"""Unit tests for benchmarking helpers."""

from __future__ import annotations

from traigent.optimizers.benchmarking import run_optuna_random_parity


def test_benchmark_results_shape():
    results = run_optuna_random_parity(n_trials=5, runs=2, seed_offset=3)
    assert "optuna" in results and "random" in results
    assert "average_best_value" in results["optuna"]
    assert "average_best_value" in results["random"]
