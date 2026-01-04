"""Registry for optimization algorithms."""

# Traceability: CONC-Layer-Core CONC-Quality-Maintainability CONC-Quality-Compatibility FUNC-OPT-ALGORITHMS REQ-OPT-ALG-004 SYNC-OptimizationFlow

from __future__ import annotations

from typing import Any

from traigent.config.feature_flags import FlagNames, flag_registry
from traigent.optimizers.base import BaseOptimizer
from traigent.utils.exceptions import OptimizationError, PluginError
from traigent.utils.logging import get_logger

logger = get_logger(__name__)

# Global registry of optimization algorithms
_OPTIMIZER_REGISTRY: dict[str, type[BaseOptimizer]] = {}


def register_optimizer(name: str, optimizer_class: type[BaseOptimizer]) -> None:
    """Register an optimization algorithm.

    Args:
        name: Name to register the optimizer under
        optimizer_class: Optimizer class that inherits from BaseOptimizer

    Raises:
        PluginError: If optimizer is invalid or name already registered
    """
    if not isinstance(name, str) or not name.strip():
        raise PluginError("Optimizer name must be a non-empty string")

    # Allow mocks in tests, but validate real classes
    is_mock = hasattr(optimizer_class, "_mock_name") or str(
        type(optimizer_class)
    ).startswith("<Mock")

    if not is_mock and (
        not isinstance(optimizer_class, type)
        or not issubclass(optimizer_class, BaseOptimizer)
    ):
        raise PluginError(
            f"Optimizer class {optimizer_class} must inherit from BaseOptimizer"
        )

    if name in _OPTIMIZER_REGISTRY:
        logger.warning(f"Overriding existing optimizer registration for '{name}'")

    _OPTIMIZER_REGISTRY[name] = optimizer_class
    logger.debug(f"Registered optimizer '{name}': {optimizer_class}")


def get_optimizer(
    name: str, config_space: dict[str, Any], objectives: list[str], **kwargs: Any
) -> BaseOptimizer:
    """Create an optimizer instance by name.

    Args:
        name: Name of the registered optimizer
        config_space: Configuration space for optimization
        objectives: List of objectives to optimize
        **kwargs: Additional arguments passed to optimizer constructor

    Returns:
        Initialized optimizer instance

    Raises:
        OptimizationError: If optimizer name not found
    """
    if name not in _OPTIMIZER_REGISTRY:
        available = list(_OPTIMIZER_REGISTRY.keys())
        raise OptimizationError(
            f"Unknown optimizer '{name}'. Available optimizers: {available}"
        )

    optimizer_class = _OPTIMIZER_REGISTRY[name]

    try:
        return optimizer_class(config_space, objectives, **kwargs)
    except Exception as e:
        raise OptimizationError(f"Failed to create optimizer '{name}': {e}") from e


def list_optimizers() -> list[str]:
    """Get list of registered optimizer names.

    Returns:
        List of optimizer names
    """
    return list(_OPTIMIZER_REGISTRY.keys())


def get_optimizer_info(name: str) -> dict[str, Any]:
    """Get information about a registered optimizer.

    Args:
        name: Name of the optimizer

    Returns:
        Dictionary with optimizer information

    Raises:
        OptimizationError: If optimizer name not found
    """
    if name not in _OPTIMIZER_REGISTRY:
        available = list(_OPTIMIZER_REGISTRY.keys())
        raise OptimizationError(
            f"Unknown optimizer '{name}'. Available optimizers: {available}"
        )

    optimizer_class = _OPTIMIZER_REGISTRY[name]

    return {
        "name": name,
        "class": optimizer_class.__name__,
        "module": optimizer_class.__module__,
        "description": optimizer_class.__doc__ or "No description available",
    }


def clear_registry() -> None:
    """Clear the optimizer registry (mainly for testing)."""
    _OPTIMIZER_REGISTRY.clear()
    logger.debug("Cleared optimizer registry")


def _register_builtin_optimizers() -> None:
    """Register built-in optimization algorithms."""
    from traigent.optimizers.grid import GridSearchOptimizer
    from traigent.optimizers.random import RandomSearchOptimizer

    register_optimizer("grid", GridSearchOptimizer)
    register_optimizer("random", RandomSearchOptimizer)

    # Try to register Bayesian optimizer (requires scikit-learn)
    try:
        from traigent.optimizers.bayesian import BayesianOptimizer

        register_optimizer("bayesian", BayesianOptimizer)
        logger.debug("Registered Bayesian optimizer")
    except ImportError:
        logger.debug("Bayesian optimizer not available (requires scikit-learn)")

    # Optuna optimizers will be registered after the function is defined
    logger.debug("Registered built-in optimizers")


def register_optuna_optimizers(force: bool = False) -> None:
    """Register Optuna-backed optimizers, honoring rollout feature flags."""

    if not force and not flag_registry.is_enabled(FlagNames.OPTUNA_ROLLOUT):
        logger.debug(
            "Optuna optimizers disabled via feature flag '%s'",
            FlagNames.OPTUNA_ROLLOUT,
        )
        return

    try:
        from traigent.optimizers.optuna_optimizer import (
            OptunaCMAESOptimizer,
            OptunaGridOptimizer,
            OptunaNSGAIIOptimizer,
            OptunaRandomOptimizer,
            OptunaTPEOptimizer,
        )
    except OptimizationError as exc:
        logger.debug("Optuna optimizers not registered: %s", exc)
        return
    except ImportError:
        logger.debug("Optuna optimizers not available (requires Optuna)")
        return

    # Base package optimizers (always available with Optuna)
    register_optimizer("optuna_tpe", OptunaTPEOptimizer)
    register_optimizer("optuna_random", OptunaRandomOptimizer)
    register_optimizer("optuna_grid", OptunaGridOptimizer)

    # Advanced optimizers (future: traigent-advanced-algorithms plugin)
    # TODO: Gate these with plugin feature flag when extracted
    register_optimizer("optuna_cmaes", OptunaCMAESOptimizer)
    register_optimizer("optuna_nsga2", OptunaNSGAIIOptimizer)

    # Backwards-compatible aliases (used by TVL strategy mapping and legacy configs)
    register_optimizer("optuna", OptunaTPEOptimizer)
    register_optimizer("tpe", OptunaTPEOptimizer)
    register_optimizer(
        "nsga2", OptunaNSGAIIOptimizer
    )  # TODO: Gate with multi-obj plugin
    logger.debug("Registered Optuna optimizers (force=%s)", force)


# Register built-in optimizers on module import
_register_builtin_optimizers()
# Register Optuna optimizers after builtin ones
register_optuna_optimizers()
