"""Domain-aware tuned variables and variable analysis plugin for Traigent SDK.

This plugin provides:
- Domain presets for LLM, RAG, and prompting parameters
- Centralized variable analysis with elimination suggestions
- TunedCallable composition pattern for function-valued variables
- Optional DSPy integration for prompt optimization
"""

# Traceability: CONC-Layer-Core CONC-Quality-Observability CONC-Quality-Maintainability

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

# Analysis
from .analysis import (
    EliminationSuggestion,
    OptimizationAnalysis,
    ValueRanking,
    VariableAnalysis,
    VariableAnalyzer,
)

# Callables
from .callables import (
    ContextFormatters,
    Retrievers,
    TunedCallable,
)

# Presets
from .presets import (
    LLMPresets,
    PromptingPresets,
    RAGPresets,
)

__all__ = [
    # Presets
    "LLMPresets",
    "RAGPresets",
    "PromptingPresets",
    # Analysis
    "VariableAnalyzer",
    "OptimizationAnalysis",
    "VariableAnalysis",
    "EliminationSuggestion",
    "ValueRanking",
    # Callables
    "TunedCallable",
    "Retrievers",
    "ContextFormatters",
    # Plugin
    "TunedVariablesPlugin",
]


class TunedVariablesPlugin:
    """Traigent Tuned Variables Plugin.

    Provides domain-aware tuned variables and variable analysis capabilities.
    """

    name = "tuned_variables"
    version = "0.1.0"
    min_traigent_version = "0.9.0"
    features = [
        "domain_presets",
        "variable_analysis",
        "tuned_callables",
        "dspy_integration",
    ]

    @classmethod
    def initialize(cls) -> None:
        """Initialize the tuned variables plugin."""
        # Plugin is standalone, no initialization needed
        pass

    @classmethod
    def cleanup(cls) -> None:
        """Cleanup plugin resources."""
        pass

    @classmethod
    def get_capabilities(cls) -> dict:
        """Return plugin capabilities."""
        # Check if DSPy is available
        try:
            import dspy  # noqa: F401

            dspy_available = True
        except ImportError:
            dspy_available = False

        return {
            "domain_presets": True,
            "variable_analysis": True,
            "tuned_callables": True,
            "dspy_integration": dspy_available,
        }
