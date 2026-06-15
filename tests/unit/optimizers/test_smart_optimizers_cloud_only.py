"""Regression tests asserting smart optimizers are cloud-only.

These tests enforce the product decision that Bayesian and Optuna-family
algorithms run in the Traigent cloud and are NOT available in the local SDK.
The local SDK exposes 'grid' and 'random' only.
"""

from __future__ import annotations

import pytest

from traigent.optimizers.registry import get_optimizer, list_optimizers
from traigent.utils.exceptions import OptimizationError

_SMART_NAMES = [
    "bayesian",
    "optuna",
    "tpe",
    "optuna_tpe",
    "optuna_random",
    "optuna_grid",
    "optuna_cmaes",
    "optuna_nsga2",
    "nsga2",
    "cmaes",
]


class TestLocalOptimizerList:
    """list_optimizers() must contain grid/random and no smart algorithm names."""

    def test_grid_and_random_are_registered(self):
        available = list_optimizers()
        assert "grid" in available, f"'grid' missing from {available}"
        assert "random" in available, f"'random' missing from {available}"

    @pytest.mark.parametrize("name", _SMART_NAMES)
    def test_smart_name_not_registered(self, name: str):
        available = list_optimizers()
        assert name not in available, (
            f"Smart optimizer '{name}' must not be registered locally; "
            f"found in {available}"
        )


class TestSmartOptimizerCloudError:
    """get_optimizer() must raise OptimizationError with cloud-redirect message."""

    @pytest.mark.parametrize("name", _SMART_NAMES)
    def test_get_optimizer_raises_cloud_error(self, name: str):
        with pytest.raises(OptimizationError) as exc_info:
            get_optimizer(name, {"x": [1, 2]}, ["score"])

        msg = str(exc_info.value)
        assert "cloud" in msg.lower(), (
            f"Error for '{name}' should mention 'cloud'; got: {msg!r}"
        )
        # Confirm it doesn't look like a plain "Unknown optimizer" error
        assert "Unknown optimizer" not in msg, (
            f"Error for '{name}' should be a cloud redirect, not unknown-optimizer: {msg!r}"
        )

    @pytest.mark.parametrize("name", _SMART_NAMES)
    def test_cloud_error_message_mentions_grid_and_random(self, name: str):
        with pytest.raises(OptimizationError) as exc_info:
            get_optimizer(name, {}, [])

        msg = str(exc_info.value)
        assert "grid" in msg and "random" in msg, (
            f"Cloud redirect for '{name}' should list 'grid' and 'random'; got: {msg!r}"
        )


class TestBayesianModuleDeleted:
    """traigent.optimizers.bayesian must not be importable (IP relocated)."""

    def test_bayesian_module_raises_module_not_found(self):
        with pytest.raises(ModuleNotFoundError):
            __import__("traigent.optimizers.bayesian")
