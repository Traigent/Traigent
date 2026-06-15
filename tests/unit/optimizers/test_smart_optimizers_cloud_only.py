"""Regression tests asserting smart optimizers are cloud-only.

These tests enforce the product decision that Bayesian and Optuna-family
algorithms run in the Traigent cloud and are NOT available in the local SDK.
The local SDK exposes 'grid' and 'random' only.
"""

from __future__ import annotations

import sys

import pytest

from traigent.api.functions import set_strategy
from traigent.optimizers.registry import (
    _is_smart_algorithm,
    get_optimizer,
    list_optimizers,
)
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


class TestSetStrategySmartError:
    """set_strategy() must raise OptimizationError for smart algorithm names."""

    def test_set_strategy_bayesian_raises_cloud_error(self):
        with pytest.raises(OptimizationError) as exc_info:
            set_strategy("bayesian")
        msg = str(exc_info.value)
        assert "cloud" in msg.lower(), (
            f"set_strategy('bayesian') error should mention 'cloud'; got: {msg!r}"
        )

    def test_set_strategy_tpe_raises_cloud_error(self):
        with pytest.raises(OptimizationError) as exc_info:
            set_strategy("tpe")
        msg = str(exc_info.value)
        assert "cloud" in msg.lower(), (
            f"set_strategy('tpe') error should mention 'cloud'; got: {msg!r}"
        )


class TestNormalizedSmartAlgorithmDetection:
    """_is_smart_algorithm() must correctly normalize and detect smart names."""

    @pytest.mark.parametrize(
        "name",
        [
            "OPTUNA",
            " optuna_tpe ",
            "nsga-ii",
            "cma-es",
            "NSGA2",
            "CmaEs",
            "Bayesian",
            "CMAES",
        ],
    )
    def test_normalized_names_detected_as_smart(self, name: str):
        assert _is_smart_algorithm(name), (
            f"_is_smart_algorithm({name!r}) should return True"
        )

    @pytest.mark.parametrize("name", ["OPTUNA", " optuna_tpe ", "nsga-ii", "cma-es"])
    def test_get_optimizer_raises_for_normalized_names(self, name: str):
        with pytest.raises(OptimizationError) as exc_info:
            get_optimizer(name, {"x": [1, 2]}, ["score"])
        msg = str(exc_info.value)
        assert "cloud" in msg.lower(), (
            f"get_optimizer({name!r}) should raise cloud error; got: {msg!r}"
        )

    @pytest.mark.parametrize("name", ["grid", "random"])
    def test_local_names_not_detected_as_smart(self, name: str):
        assert not _is_smart_algorithm(name), (
            f"_is_smart_algorithm({name!r}) should return False"
        )


class TestOptunaModulesDeleted:
    """Deleted optuna execution modules must not be importable."""

    def test_optuna_adapter_import_raises_module_not_found(self):
        # Ensure the module isn't cached from a previous partial import
        for key in list(sys.modules.keys()):
            if "optuna_adapter" in key and "seamless" not in key:
                del sys.modules[key]
        with pytest.raises(ModuleNotFoundError):
            import traigent.optimizers.optuna_adapter  # noqa: F401

    def test_optuna_adapter_from_import_raises_import_error(self):
        # traigent.optimizers no longer exports OptunaAdapter
        with pytest.raises((ImportError, AttributeError)):
            from traigent.optimizers import OptunaAdapter  # noqa: F401

    def test_optuna_optimizer_module_not_found(self):
        for key in list(sys.modules.keys()):
            if "optuna_optimizer" in key:
                del sys.modules[key]
        with pytest.raises(ModuleNotFoundError):
            import traigent.optimizers.optuna_optimizer  # noqa: F401

    def test_optuna_coordinator_module_not_found(self):
        for key in list(sys.modules.keys()):
            if "optuna_coordinator" in key:
                del sys.modules[key]
        with pytest.raises(ModuleNotFoundError):
            import traigent.optimizers.optuna_coordinator  # noqa: F401
