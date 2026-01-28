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
from traigent.api.parameter_ranges import Choices, IntRange, Range
from traigent.api.safety import (  # RAGAS metric presets (require ragas package); Non-RAGAS metric factories; Core classes; Utilities
    CompoundSafetyConstraint,
    SafetyConstraint,
    SafetyThreshold,
    SafetyValidator,
    answer_relevancy,
    answer_similarity,
    bias_score,
    context_precision,
    context_recall,
    custom_safety,
    faithfulness,
    get_available_safety_presets,
    hallucination_rate,
    safety_score,
    toxicity_score,
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
    # Main decorator
    "optimize",
    # Configuration functions
    "configure",
    "override_config",
    "set_strategy",
    "get_available_strategies",
    "get_version_info",
    # Result types
    "OptimizationResult",
    "TrialResult",
    "SensitivityAnalysis",
    "ConfigurationComparison",
    "ParetoFront",
    "OptimizationStatus",
    "StopReason",
    "StrategyConfig",
    # Parameter ranges for tuned variables
    "Range",
    "IntRange",
    "Choices",
    # Safety constraint presets (RAGAS)
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
    "answer_similarity",
    # Safety constraint factories
    "hallucination_rate",
    "toxicity_score",
    "bias_score",
    "safety_score",
    "custom_safety",
    # Safety constraint classes
    "SafetyConstraint",
    "CompoundSafetyConstraint",
    "SafetyThreshold",
    "SafetyValidator",
    "get_available_safety_presets",
]
