"""Public API module for Traigent SDK.

This package intentionally uses lazy exports so submodule imports such as
``traigent.api.types`` do not trigger the full decorator/core import graph.
"""

from __future__ import annotations

import warnings
from importlib import import_module
from typing import Any

__all__ = [
    "optimize",
    "configure",
    "configure_for_budget",
    "override_config",
    "set_strategy",
    "get_available_strategies",
    "get_version_info",
    "OptimizationResult",
    "PresetSelection",
    "TrialError",
    "TrialResult",
    "serialize_trials",
    "SensitivityAnalysis",
    "ConfigurationComparison",
    "ParetoFront",
    "OptimizationStatus",
    "StopReason",
    "StrategyConfig",
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
    "answer_similarity",
    "hallucination_rate",
    "toxicity_score",
    "bias_score",
    "safety_score",
    "custom_safety",
    "SafetyConstraint",
    "CompoundSafetyConstraint",
    "SafetyThreshold",
    "SafetyValidator",
    "get_available_safety_presets",
    "TextDocument",
    "ExternalServiceEvaluator",
]

_TRAIGENT_API_DECORATORS = "traigent.api.decorators"
_TRAIGENT_API_FUNCTIONS = "traigent.api.functions"
_TRAIGENT_API_TYPES = "traigent.api.types"
_TRAIGENT_API_SAFETY = "traigent.api.safety"

_EXPORT_MAP = {
    "optimize": (_TRAIGENT_API_DECORATORS, "optimize"),
    "configure": (_TRAIGENT_API_FUNCTIONS, "configure"),
    "configure_for_budget": (_TRAIGENT_API_FUNCTIONS, "configure_for_budget"),
    "override_config": (_TRAIGENT_API_FUNCTIONS, "override_config"),
    "set_strategy": (_TRAIGENT_API_FUNCTIONS, "set_strategy"),
    "get_available_strategies": (_TRAIGENT_API_FUNCTIONS, "get_available_strategies"),
    "get_version_info": (_TRAIGENT_API_FUNCTIONS, "get_version_info"),
    "OptimizationResult": (_TRAIGENT_API_TYPES, "OptimizationResult"),
    "PresetSelection": (_TRAIGENT_API_TYPES, "PresetSelection"),
    "TrialError": (_TRAIGENT_API_TYPES, "TrialError"),
    "TrialResult": (_TRAIGENT_API_TYPES, "TrialResult"),
    "serialize_trials": (_TRAIGENT_API_TYPES, "serialize_trials"),
    "SensitivityAnalysis": (_TRAIGENT_API_TYPES, "SensitivityAnalysis"),
    "ConfigurationComparison": (_TRAIGENT_API_TYPES, "ConfigurationComparison"),
    "ParetoFront": (_TRAIGENT_API_TYPES, "ParetoFront"),
    "OptimizationStatus": (_TRAIGENT_API_TYPES, "OptimizationStatus"),
    "StopReason": (_TRAIGENT_API_TYPES, "StopReason"),
    "StrategyConfig": (_TRAIGENT_API_TYPES, "StrategyConfig"),
    "faithfulness": (_TRAIGENT_API_SAFETY, "faithfulness"),
    "answer_relevancy": (_TRAIGENT_API_SAFETY, "answer_relevancy"),
    "context_precision": (_TRAIGENT_API_SAFETY, "context_precision"),
    "context_recall": (_TRAIGENT_API_SAFETY, "context_recall"),
    "answer_similarity": (_TRAIGENT_API_SAFETY, "answer_similarity"),
    "hallucination_rate": (_TRAIGENT_API_SAFETY, "hallucination_rate"),
    "toxicity_score": (_TRAIGENT_API_SAFETY, "toxicity_score"),
    "bias_score": (_TRAIGENT_API_SAFETY, "bias_score"),
    "safety_score": (_TRAIGENT_API_SAFETY, "safety_score"),
    "custom_safety": (_TRAIGENT_API_SAFETY, "custom_safety"),
    "SafetyConstraint": (_TRAIGENT_API_SAFETY, "SafetyConstraint"),
    "CompoundSafetyConstraint": (_TRAIGENT_API_SAFETY, "CompoundSafetyConstraint"),
    "SafetyThreshold": (_TRAIGENT_API_SAFETY, "SafetyThreshold"),
    "SafetyValidator": (_TRAIGENT_API_SAFETY, "SafetyValidator"),
    "get_available_safety_presets": (
        _TRAIGENT_API_SAFETY,
        "get_available_safety_presets",
    ),
    "TextDocument": ("traigent.api.parameter_ranges", "TextDocument"),
    "ExternalServiceEvaluator": (
        _TRAIGENT_API_DECORATORS,
        "ExternalServiceEvaluator",
    ),
    "HybridAPIOptions": (_TRAIGENT_API_DECORATORS, "HybridAPIOptions"),
}

_DEPRECATED_EXPORTS = {"HybridAPIOptions"}


def __getattr__(name: str) -> Any:
    if name not in _EXPORT_MAP:
        raise AttributeError(f"module 'traigent.api' has no attribute {name!r}")
    module_name, attr_name = _EXPORT_MAP[name]
    module = import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    if name in _DEPRECATED_EXPORTS:
        warnings.warn(
            f"traigent.api.{name} is a deprecated compatibility alias and is "
            "no longer part of the public API export surface.",
            DeprecationWarning,
            stacklevel=2,
        )
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
