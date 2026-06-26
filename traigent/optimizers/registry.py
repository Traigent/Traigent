"""Registry for optimization algorithms."""

# Traceability: CONC-Layer-Core CONC-Quality-Maintainability CONC-Quality-Compatibility FUNC-OPT-ALGORITHMS REQ-OPT-ALG-004 SYNC-OptimizationFlow

from __future__ import annotations

from typing import Any

from traigent.config.types import is_smart_algorithm as _is_smart_algorithm_canonical
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


def _is_smart_algorithm(name: str) -> bool:
    """Return True if *name* refers to a cloud-only smart optimizer.

    Delegates to :func:`traigent.config.types.is_smart_algorithm` —
    the single canonical source of truth for smart-algorithm classification
    (``config/types.py:_SMART_ALGORITHMS``).  This thin wrapper preserves the
    existing ``registry`` import surface so callers in ``api/functions.py``,
    ``optimizers/remote_services.py``, and the public ``optimizers.__all__``
    re-export do not need to change.

    See also: GitHub issue #1402 (de-dup ``_SMART_OPTIMIZER_NAMES`` →
    ``config.types._SMART_ALGORITHMS`` single source of truth).

    Args:
        name: Algorithm name to test (raw, un-normalized).

    Returns:
        True if the name identifies a cloud-only smart algorithm.
    """
    return _is_smart_algorithm_canonical(name)


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
        OptimizationError: If optimizer name not found or is a cloud-only smart algorithm
    """
    if _is_smart_algorithm(name):
        raise OptimizationError(
            f"Smart optimization ('{name}') runs in the Traigent cloud and is not "
            f"available in the local SDK (which supports 'grid' and 'random'). "
            f"Connect to the Traigent backend to use smart algorithms."
        )

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
    """Register built-in optimization algorithms (grid and random only).

    Smart algorithms (Bayesian, Optuna family) run in the Traigent cloud
    and are not registered locally.
    """
    from traigent.optimizers.grid import GridSearchOptimizer
    from traigent.optimizers.random import RandomSearchOptimizer

    register_optimizer("grid", GridSearchOptimizer)
    register_optimizer("random", RandomSearchOptimizer)

    _register_batch_optimizers()
    _register_remote_optimizer()

    logger.debug("Registered built-in optimizers (grid and random)")


def _register_batch_optimizers() -> None:
    """Register batch optimization algorithms under documented public names."""
    from traigent.optimizers.batch_optimizers import (
        AdaptiveBatchOptimizer,
        MultiObjectiveBatchOptimizer,
        ParallelBatchOptimizer,
    )

    register_optimizer("parallel_batch", ParallelBatchOptimizer)
    register_optimizer("multi_objective_batch", MultiObjectiveBatchOptimizer)
    register_optimizer("adaptive_batch", AdaptiveBatchOptimizer)
    logger.debug("Registered batch optimizers")


def _register_remote_optimizer() -> None:
    """Register the remote optimizer compatibility entry.

    The remote optimizer intentionally fails loud at construction unless a
    remote client is injected, but the registry entry must remain present so
    algorithm="remote" returns the migration error instead of KeyError.
    """
    try:
        from traigent.optimizers.remote import RemoteOptimizer
    except ImportError:
        logger.debug("Remote optimizer not available")
        return

    if "remote" not in _OPTIMIZER_REGISTRY:
        register_optimizer("remote", RemoteOptimizer)


# Register built-in optimizers on module import
_register_builtin_optimizers()
