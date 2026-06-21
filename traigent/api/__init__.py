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
    "list_recommendation_agent_types",
    "recommend_configuration_space",
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
    "ADVISORY_SELECTION_NOTICE",
    "NormalizedStrategyPreset",
    "StrategyPresetError",
    "StrategyPresetValidationError",
    "UnknownStrategyPresetError",
    "VALID_PRESET_NAMES",
    "normalize_strategy_preset",
    "select_strategy_preset",
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
    "SmartPruningConfig",
]

_EXPORT_MAP = {
    "optimize": ("traigent.api.decorators", "optimize"),
    "configure": ("traigent.api.functions", "configure"),
    "configure_for_budget": ("traigent.api.functions", "configure_for_budget"),
    "override_config": ("traigent.api.functions", "override_config"),
    "set_strategy": ("traigent.api.functions", "set_strategy"),
    "get_available_strategies": ("traigent.api.functions", "get_available_strategies"),
    "list_recommendation_agent_types": (
        "traigent.api.functions",
        "list_recommendation_agent_types",
    ),
    "recommend_configuration_space": (
        "traigent.api.functions",
        "recommend_configuration_space",
    ),
    "get_version_info": ("traigent.api.functions", "get_version_info"),
    "OptimizationResult": ("traigent.api.types", "OptimizationResult"),
    "PresetSelection": ("traigent.api.types", "PresetSelection"),
    "TrialError": ("traigent.api.types", "TrialError"),
    "TrialResult": ("traigent.api.types", "TrialResult"),
    "serialize_trials": ("traigent.api.types", "serialize_trials"),
    "SensitivityAnalysis": ("traigent.api.types", "SensitivityAnalysis"),
    "ConfigurationComparison": ("traigent.api.types", "ConfigurationComparison"),
    "ParetoFront": ("traigent.api.types", "ParetoFront"),
    "OptimizationStatus": ("traigent.api.types", "OptimizationStatus"),
    "StopReason": ("traigent.api.types", "StopReason"),
    "StrategyConfig": ("traigent.api.types", "StrategyConfig"),
    "ADVISORY_SELECTION_NOTICE": (
        "traigent.api.strategy_presets",
        "ADVISORY_SELECTION_NOTICE",
    ),
    "NormalizedStrategyPreset": (
        "traigent.api.strategy_presets",
        "NormalizedStrategyPreset",
    ),
    "StrategyPresetError": ("traigent.api.strategy_presets", "StrategyPresetError"),
    "StrategyPresetValidationError": (
        "traigent.api.strategy_presets",
        "StrategyPresetValidationError",
    ),
    "UnknownStrategyPresetError": (
        "traigent.api.strategy_presets",
        "UnknownStrategyPresetError",
    ),
    "VALID_PRESET_NAMES": ("traigent.api.strategy_presets", "VALID_PRESET_NAMES"),
    "normalize_strategy_preset": (
        "traigent.api.strategy_presets",
        "normalize_strategy_preset",
    ),
    "select_strategy_preset": (
        "traigent.api.strategy_presets",
        "select_strategy_preset",
    ),
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
    "TextDocument": ("traigent.api.parameter_ranges", "TextDocument"),
    "ExternalServiceEvaluator": (
        "traigent.api.decorators",
        "ExternalServiceEvaluator",
    ),
    "SmartPruningConfig": ("traigent.api.decorators", "SmartPruningConfig"),
    "HybridAPIOptions": ("traigent.api.decorators", "HybridAPIOptions"),
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
