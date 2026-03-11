"""Public API module for Traigent SDK.

This package intentionally uses lazy exports so submodule imports such as
``traigent.api.types`` do not trigger the full decorator/core import graph.
"""

from __future__ import annotations

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
    "TrialResult",
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
]

_EXPORT_MAP = {
    "optimize": ("traigent.api.decorators", "optimize"),
    "configure": ("traigent.api.functions", "configure"),
    "configure_for_budget": ("traigent.api.functions", "configure_for_budget"),
    "override_config": ("traigent.api.functions", "override_config"),
    "set_strategy": ("traigent.api.functions", "set_strategy"),
    "get_available_strategies": ("traigent.api.functions", "get_available_strategies"),
    "get_version_info": ("traigent.api.functions", "get_version_info"),
    "OptimizationResult": ("traigent.api.types", "OptimizationResult"),
    "TrialResult": ("traigent.api.types", "TrialResult"),
    "SensitivityAnalysis": ("traigent.api.types", "SensitivityAnalysis"),
    "ConfigurationComparison": ("traigent.api.types", "ConfigurationComparison"),
    "ParetoFront": ("traigent.api.types", "ParetoFront"),
    "OptimizationStatus": ("traigent.api.types", "OptimizationStatus"),
    "StopReason": ("traigent.api.types", "StopReason"),
    "StrategyConfig": ("traigent.api.types", "StrategyConfig"),
    "faithfulness": ("traigent.api.safety", "faithfulness"),
    "answer_relevancy": ("traigent.api.safety", "answer_relevancy"),
    "context_precision": ("traigent.api.safety", "context_precision"),
    "context_recall": ("traigent.api.safety", "context_recall"),
    "answer_similarity": ("traigent.api.safety", "answer_similarity"),
    "hallucination_rate": ("traigent.api.safety", "hallucination_rate"),
    "toxicity_score": ("traigent.api.safety", "toxicity_score"),
    "bias_score": ("traigent.api.safety", "bias_score"),
    "safety_score": ("traigent.api.safety", "safety_score"),
    "custom_safety": ("traigent.api.safety", "custom_safety"),
    "SafetyConstraint": ("traigent.api.safety", "SafetyConstraint"),
    "CompoundSafetyConstraint": ("traigent.api.safety", "CompoundSafetyConstraint"),
    "SafetyThreshold": ("traigent.api.safety", "SafetyThreshold"),
    "SafetyValidator": ("traigent.api.safety", "SafetyValidator"),
    "get_available_safety_presets": (
        "traigent.api.safety",
        "get_available_safety_presets",
    ),
}


def __getattr__(name: str) -> Any:
    if name not in _EXPORT_MAP:
        raise AttributeError(f"module 'traigent.api' has no attribute {name!r}")
    module_name, attr_name = _EXPORT_MAP[name]
    module = import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
