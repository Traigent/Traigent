"""Variable analysis module.

Provides centralized analysis of optimization results with elimination suggestions.
"""

from .variable_analyzer import (
    EliminationAction,
    EliminationSuggestion,
    OptimizationAnalysis,
    ValueRanking,
    VariableAnalysis,
    VariableAnalyzer,
)

__all__ = [
    "VariableAnalyzer",
    "OptimizationAnalysis",
    "VariableAnalysis",
    "EliminationSuggestion",
    "EliminationAction",
    "ValueRanking",
]
