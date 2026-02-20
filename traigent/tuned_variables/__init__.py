"""TunedVariable abstractions for Traigent optimization.

This module provides utilities for working with tunable variables including:
- Callable auto-discovery from modules (P-5)
- Domain-specific variable presets
- Variable analysis utilities
- Static analysis + LLM-based tuned variable detection
"""

from __future__ import annotations

from .detection_strategies import (
    ASTDetectionStrategy,
    DetectionStrategy,
    LLMDetectionStrategy,
)
from .detection_types import (
    CandidateType,
    DetectionConfidence,
    DetectionResult,
    SourceLocation,
    SuggestedRange,
    TunedVariableCandidate,
)
from .detector import TunedVariableDetector
from .discovery import (
    CallableInfo,
    discover_callables,
    discover_callables_by_decorator,
    filter_by_signature,
)

__all__ = [
    # Callable discovery (existing)
    "CallableInfo",
    "discover_callables",
    "discover_callables_by_decorator",
    "filter_by_signature",
    # Tuned variable detection (new)
    "ASTDetectionStrategy",
    "CandidateType",
    "DetectionConfidence",
    "DetectionResult",
    "DetectionStrategy",
    "LLMDetectionStrategy",
    "SourceLocation",
    "SuggestedRange",
    "TunedVariableCandidate",
    "TunedVariableDetector",
]
