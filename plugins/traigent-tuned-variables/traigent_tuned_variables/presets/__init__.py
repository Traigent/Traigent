"""Domain-aware presets for tuned variables.

Provides pre-configured parameter ranges for common optimization scenarios.
"""

from .llm import LLMPresets
from .prompting import PromptingPresets
from .rag import RAGPresets

__all__ = [
    "LLMPresets",
    "RAGPresets",
    "PromptingPresets",
]
