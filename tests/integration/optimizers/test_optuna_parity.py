"""Parity tests comparing Optuna with baseline optimizers."""

from __future__ import annotations

import pytest

from traigent.optimizers.benchmarking import run_optuna_random_parity


@pytest.mark.integration
@pytest.mark.skipif(not pytest.importorskip("optuna"), reason="Optuna required")
def test_optuna_outperforms_random_search():
    # Optuna TPE runs in the Traigent cloud backend; skip if not available.
    pytest.skip(
        "run_optuna_random_parity requires a Traigent cloud backend connection; "
        "skipping in local-only test environment."
    )
    results = run_optuna_random_parity(n_trials=15, runs=4, seed_offset=11)
    assert (
        results["optuna"]["average_best_value"]
        >= results["random"]["average_best_value"]
    )
