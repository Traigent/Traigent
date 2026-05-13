"""Optimization algorithms for Traigent SDK."""

# Traceability: CONC-Layer-Core CONC-Quality-Maintainability CONC-Quality-Performance FUNC-OPT-ALGORITHMS REQ-OPT-ALG-004 SYNC-OptimizationFlow

from __future__ import annotations

from traigent.optimizers.base import BaseOptimizer
from traigent.optimizers.batch_optimizers import (
    AdaptiveBatchOptimizer,
    BatchOptimizationConfig,
    MultiObjectiveBatchOptimizer,
    ParallelBatchOptimizer,
)
from traigent.optimizers.grid import GridSearchOptimizer
from traigent.optimizers.optuna_adapter import OptunaAdapter
from traigent.optimizers.optuna_coordinator import (
    BatchOptimizer,
    EdgeExecutor,
    OptunaCoordinator,
    RateLimitedOptimizer,
)
from traigent.optimizers.optuna_optimizer import (
    OptunaBaseOptimizer,
    OptunaCMAESOptimizer,
    OptunaGridOptimizer,
    OptunaNSGAIIOptimizer,
    OptunaRandomOptimizer,
    OptunaTPEOptimizer,
)
from traigent.optimizers.pruners import (
    CeilingPruner,
    CeilingPrunerConfig,
    StatisticalInferiorityPruner,
    StatisticalInferiorityPrunerConfig,
)
from traigent.optimizers.random import RandomSearchOptimizer
from traigent.optimizers.registry import (
    get_optimizer,
    list_optimizers,
    register_optimizer,
)


def _is_missing_optional_module(
    exc: ModuleNotFoundError, module_prefixes: tuple[str, ...]
) -> bool:
    missing_module = getattr(exc, "name", "")
    return any(
        missing_module == prefix or missing_module.startswith(f"{prefix}.")
        for prefix in module_prefixes
    )


_OPTIONAL_EXPORT_ERRORS: dict[str, str] = {}

try:
    from traigent.optimizers.cloud_optimizer import CloudOptimizer  # noqa: F401
except ModuleNotFoundError as exc:
    if not _is_missing_optional_module(
        exc,
        (
            "traigent.optimizers.cloud_optimizer",
            "traigent.optimizers.remote_services",
        ),
    ):
        raise

    _OPTIONAL_EXPORT_ERRORS["CloudOptimizer"] = (
        "module 'traigent.optimizers' has no attribute 'CloudOptimizer'. "
        "Cloud optimizer exports are unavailable in this build."
    )

try:
    from traigent.optimizers.remote import RemoteOptimizer  # noqa: F401
except ModuleNotFoundError as exc:
    if not _is_missing_optional_module(exc, ("traigent.optimizers.remote",)):
        raise

    _OPTIONAL_EXPORT_ERRORS["RemoteOptimizer"] = (
        "module 'traigent.optimizers' has no attribute 'RemoteOptimizer'. "
        "Remote optimizer exports are unavailable in this build."
    )

__all__ = [
    "BaseOptimizer",
    "GridSearchOptimizer",
    "RandomSearchOptimizer",
    "BatchOptimizationConfig",
    "ParallelBatchOptimizer",
    "MultiObjectiveBatchOptimizer",
    "AdaptiveBatchOptimizer",
    "OptunaBaseOptimizer",
    "OptunaTPEOptimizer",
    "OptunaRandomOptimizer",
    "OptunaCMAESOptimizer",
    "OptunaNSGAIIOptimizer",
    "OptunaGridOptimizer",
    "OptunaCoordinator",
    "BatchOptimizer",
    "RateLimitedOptimizer",
    "EdgeExecutor",
    "OptunaAdapter",
    "get_optimizer",
    "register_optimizer",
    "list_optimizers",
    # Pruners for early stopping
    "CeilingPruner",
    "CeilingPrunerConfig",
    "StatisticalInferiorityPruner",
    "StatisticalInferiorityPrunerConfig",
]

if "RemoteOptimizer" in globals():
    __all__.append("RemoteOptimizer")

if "CloudOptimizer" in globals():
    __all__.append("CloudOptimizer")


def __getattr__(name: str):
    if name in _OPTIONAL_EXPORT_ERRORS:
        raise AttributeError(_OPTIONAL_EXPORT_ERRORS[name])
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
