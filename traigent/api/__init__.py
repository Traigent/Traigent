"""Public API module for Traigent SDK."""

# Traceability: CONC-Layer-API CONC-Quality-Usability FUNC-API-ENTRY REQ-API-001 SYNC-OptimizationFlow

from __future__ import annotations

from traigent.api.decorators import optimize
from traigent.api.functions import (
    configure,
    get_available_strategies,
    get_version_info,
    override_config,
    set_strategy,
)
from traigent.api.types import (
    ConfigurationComparison,
    OptimizationResult,
    OptimizationStatus,
    ParetoFront,
    SensitivityAnalysis,
    StopReason,
    StrategyConfig,
    TrialResult,
)

__all__ = [
    "optimize",
    "configure",
    "override_config",
    "set_strategy",
    "get_available_strategies",
    "get_version_info",
    "OptimizationResult",
    "TrialResult",
    "SensitivityAnalysis",
    "ConfigurationComparison",
    "ParetoFront",
    "OptimizationStatus",
    "StopReason",
    "StrategyConfig",
]
