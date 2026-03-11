"""Data model for tuned variable detection results.

Defines the output types for the TunedVariableDetector system, which combines
AST-based static analysis with optional LLM agent analysis to identify tuned
variable candidates in user code.

All result dataclasses are frozen (immutable) and use slots for thread safety
and memory efficiency, following the CallableInfo pattern from discovery.py.
"""

from __future__ import annotations

from collections.abc import Collection
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Literal


class DetectionConfidence(StrEnum):
    """Confidence level of a tuned variable detection."""

    HIGH = "high"  # Exact name match to known LLM params
    MEDIUM = "medium"  # Fuzzy name match or structural pattern
    LOW = "low"  # LLM-suggested or weak heuristic


class CandidateType(StrEnum):
    """Type of tuned variable candidate."""

    NUMERIC_CONTINUOUS = "numeric_continuous"  # float ranges (temperature, top_p)
    NUMERIC_INTEGER = "numeric_integer"  # int ranges (max_tokens, top_k)
    CATEGORICAL = "categorical"  # string/enum choices (model, strategy)
    BOOLEAN = "boolean"  # True/False toggles


@dataclass(frozen=True, slots=True)
class SourceLocation:
    """Location of a detected variable in source code."""

    line: int
    col_offset: int
    end_line: int | None = None
    end_col_offset: int | None = None


@dataclass(frozen=True, slots=True)
class SuggestedRange:
    """Suggested ParameterRange for a detected variable.

    Provides enough information to construct a Range, IntRange,
    Choices, or LogRange.

    Args:
        range_type: Name of the ParameterRange class ("Range", "IntRange",
            "Choices", "LogRange").
        kwargs: Constructor keyword arguments for the range class.
    """

    range_type: str
    kwargs: dict[str, Any] = field(default_factory=dict)

    def to_parameter_range_code(self) -> str:
        """Generate Python code to construct this ParameterRange.

        Returns:
            String like ``Range(low=0.0, high=2.0)`` or
            ``Choices(values=['gpt-4', 'gpt-3.5-turbo'])``.
        """
        parts = [f"{k}={v!r}" for k, v in self.kwargs.items()]
        return f"{self.range_type}({', '.join(parts)})"

    def to_parameter_range(self) -> Any:
        """Construct a ParameterRange object from this suggestion.

        Returns:
            A ``Range``, ``IntRange``, ``Choices``, or ``LogRange`` instance.

        Raises:
            ValueError: If ``range_type`` is unsupported.
        """
        from traigent.api.parameter_ranges import Choices, IntRange, LogRange, Range

        mapping = {
            "Range": Range,
            "IntRange": IntRange,
            "Choices": Choices,
            "LogRange": LogRange,
        }
        cls = mapping.get(self.range_type)
        if cls is None:
            raise ValueError(
                f"Unsupported range_type {self.range_type!r}. "
                f"Expected one of: {sorted(mapping)}"
            )
        return cls(**self.kwargs)


@dataclass(frozen=True, slots=True)
class TunedVariableCandidate:
    """A detected candidate for a tuned variable.

    Immutable result object following the CallableInfo conventions.

    Attributes:
        name: Variable name in source code.
        candidate_type: Inferred parameter type.
        confidence: Detection confidence level.
        location: Source code location.
        current_value: Current assigned value (if detectable from AST).
        suggested_range: Suggested ParameterRange for optimization.
        detection_source: Which strategy produced this ("ast", "llm", "combined").
        reasoning: Human-readable explanation of why this was flagged.
        canonical_name: Mapped canonical param name (e.g., "temperature").
    """

    name: str
    candidate_type: CandidateType
    confidence: DetectionConfidence
    location: SourceLocation
    current_value: Any | None = None
    suggested_range: SuggestedRange | None = None
    detection_source: str = "ast"
    reasoning: str = ""
    canonical_name: str | None = None


@dataclass(frozen=True, slots=True)
class DetectionResult:
    """Complete result from tuned variable detection.

    Follows the ValidationResult pattern from hooks/validator.py.

    Attributes:
        function_name: Name of the analyzed function.
        candidates: Detected tuned variable candidates.
        warnings: Non-blocking warnings from detection.
        source_hash: Hash of analyzed source for caching.
        detection_strategies_used: Names of strategies that ran.
    """

    function_name: str
    candidates: tuple[TunedVariableCandidate, ...] = ()
    warnings: tuple[str, ...] = ()
    source_hash: str = ""
    detection_strategies_used: tuple[str, ...] = ()

    @property
    def count(self) -> int:
        """Total number of candidates detected."""
        return len(self.candidates)

    @property
    def high_confidence(self) -> tuple[TunedVariableCandidate, ...]:
        """Candidates with HIGH confidence only."""
        return tuple(
            c for c in self.candidates if c.confidence == DetectionConfidence.HIGH
        )

    def to_configuration_space(
        self,
        *,
        format: Literal["normalized", "ranges"] = "normalized",
        min_confidence: DetectionConfidence | str = DetectionConfidence.MEDIUM,
        include: Collection[str] | None = None,
        exclude: Collection[str] | None = None,
    ) -> dict[str, Any]:
        """Convert selected candidates to a config space dict.

        Returns:
            Dictionary suitable for passing to ``@traigent.optimize(
            configuration_space=...)``.
        """
        confidence_order = {
            DetectionConfidence.HIGH: 2,
            DetectionConfidence.MEDIUM: 1,
            DetectionConfidence.LOW: 0,
        }
        threshold = (
            DetectionConfidence(min_confidence)
            if isinstance(min_confidence, str)
            else min_confidence
        )
        min_order = confidence_order[threshold]
        include_set = set(include) if include is not None else None
        exclude_set = set(exclude) if exclude is not None else set()

        config: dict[str, Any] = {}
        for c in self.candidates:
            if c.suggested_range is None:
                continue
            if confidence_order[c.confidence] < min_order:
                continue
            if include_set is not None and c.name not in include_set:
                continue
            if c.name in exclude_set:
                continue

            if format == "normalized":
                config[c.name] = c.suggested_range.kwargs
            elif format == "ranges":
                config[c.name] = c.suggested_range.to_parameter_range()
            else:
                raise ValueError(
                    f"format must be 'normalized' or 'ranges', got {format!r}"
                )
        return config
