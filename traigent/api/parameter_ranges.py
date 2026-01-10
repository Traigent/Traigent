"""SE-friendly parameter range definitions for Traigent configuration spaces.

This module provides intuitive constructors for defining parameter search spaces
that "speak the language" of software engineers while maintaining full backward
compatibility with tuple/list syntax.

These classes are TVL (Tuned Variable Language) first-class citizens, supporting:
- Domain specification (range, choices, log-scale)
- Optional naming and units for TVL integration
- Constraint builder methods for expressing structural constraints

Example::

    from traigent import Range, IntRange, Choices, LogRange, implies

    # Basic usage
    @traigent.optimize(
        temperature=Range(0.0, 2.0),
        max_tokens=IntRange(100, 4096),
        model=Choices(["gpt-4", "gpt-3.5-turbo"]),
        learning_rate=LogRange(1e-5, 1e-1),
    )
    def my_func(query): ...

    # With TVL features (constraints, units)
    temp = Range(0.0, 2.0, unit="ratio")
    model = Choices(["gpt-4", "gpt-3.5"])
    constraints = [
        implies(model.equals("gpt-4"), temp.lte(0.7)),
    ]
"""

from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Generic, Literal, TypeVar, cast

if TYPE_CHECKING:
    from traigent.api.config_space import ConfigSpace

from traigent.api.constraint_builders import (
    CategoricalConstraintBuilderMixin,
    NumericConstraintBuilderMixin,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")


class ParameterRange(ABC):
    """Base class for all parameter range types.

    All range classes inherit from this to enable isinstance() checks
    and provide a common interface for normalization.

    Default Value Precedence:
        When multiple defaults are specified, they are resolved in this order
        (highest precedence first):

        1. Explicit ``default_config`` dict values in ``@traigent.optimize()``
        2. ``ParameterRange.default`` values (e.g., ``Range(0.0, 1.0, default=0.7)``)
        3. Optimizer-suggested defaults (e.g., Optuna's suggest_* midpoint)

    Auto-Naming:
        When a ParameterRange is used as a decorator kwarg without an explicit
        ``name`` attribute, the kwarg key is automatically assigned as the name.
        This enables constraint building and TVL spec generation.

        Example::

            @traigent.optimize(
                temperature=Range(0.0, 1.0),  # name auto-assigned as "temperature"
            )
    """

    # Name attribute - set by subclasses or assigned from decorator kwarg
    name: str | None

    @abstractmethod
    def to_config_value(self) -> tuple[Any, ...] | list[Any] | dict[str, Any]:
        """Convert to the internal configuration space format.

        Returns:
            - tuple (low, high) for simple numeric ranges
            - list of values for categorical choices
            - dict with type info for ranges with log/step options
        """
        ...

    @abstractmethod
    def get_default(self) -> Any | None:
        """Return the default value if set, otherwise None."""
        ...


@dataclass(frozen=True, slots=True)
class Range(NumericConstraintBuilderMixin, ParameterRange):
    """Continuous float range for optimization.

    Args:
        low: Lower bound (inclusive)
        high: Upper bound (inclusive)
        step: Optional step size for discretization
        log: Whether to use log-scale sampling (default: False).
             Note: Cannot be combined with step (Optuna limitation).
        default: Optional default value (populates default_config)
        name: Optional TVAR name (auto-assigned from decorator kwarg if not set)
        unit: Optional unit of measurement (e.g., "ratio", "seconds", "USD")
        agent: Optional agent identifier for multi-agent experiments.
            When set, this parameter will be grouped with other parameters
            belonging to the same agent in the UI.

    Example:
        >>> temperature = Range(0.0, 2.0)
        >>> temperature_fine = Range(0.0, 1.0, step=0.1)
        >>> learning_rate = Range(1e-5, 1e-1, log=True)
        >>> # With TVL features
        >>> temp = Range(0.0, 2.0, unit="ratio")
        >>> # Multi-agent: assign to specific agent
        >>> financial_temp = Range(0.0, 1.0, agent="financial")

    Raises:
        ValueError: If low >= high, step <= 0, log with non-positive low,
                    or log combined with step
    """

    low: float
    high: float
    step: float | None = None
    log: bool = False
    default: float | None = None
    # TVL fields
    name: str | None = None
    unit: str | None = None
    # Multi-agent support
    agent: str | None = None

    def __post_init__(self) -> None:
        if self.low >= self.high:
            raise ValueError(
                f"Range low ({self.low}) must be less than high ({self.high})"
            )
        if self.step is not None and self.step <= 0:
            raise ValueError(f"Range step must be positive, got {self.step}")
        if self.log and self.low <= 0:
            raise ValueError(f"log=True requires positive bounds, got low={self.low}")
        if self.log and self.step is not None:
            raise ValueError(
                "Cannot use log=True with step (Optuna limitation). "
                "Use either log-scale OR discrete steps, not both."
            )
        if self.default is not None and not (self.low <= self.default <= self.high):
            raise ValueError(
                f"default {self.default} is outside range [{self.low}, {self.high}]"
            )

    def to_config_value(self) -> tuple[float, float] | dict[str, Any]:
        """Convert to internal format.

        Returns tuple for simple ranges, dict when log/step is set.
        """
        if self.step is None and not self.log:
            return (self.low, self.high)
        # Use dict format for advanced options
        result: dict[str, Any] = {
            "type": "float",
            "low": self.low,
            "high": self.high,
        }
        if self.step is not None:
            result["step"] = self.step
        if self.log:
            result["log"] = True
        return result

    def get_default(self) -> float | None:
        """Return the default value if set."""
        return self.default

    def to_tuple(self) -> tuple[float, float]:
        """Return as simple (low, high) tuple for backward compatibility."""
        return (self.low, self.high)

    # Constraint builder methods (equals, not_equals, gt, gte, lt, lte,
    # in_range, is_in, not_in) are provided by NumericConstraintBuilderMixin

    # =========================================================================
    # Factory Methods - Domain Presets
    # =========================================================================

    @classmethod
    def temperature(
        cls,
        *,
        conservative: bool = False,
        creative: bool = False,
    ) -> Range:
        """Pre-configured temperature for LLM optimization.

        Args:
            conservative: Use narrow range [0.0, 0.5] for factual tasks
            creative: Use higher range [0.7, 1.5] for creative tasks

        Returns:
            Range configured for the specified temperature profile

        Example:
            >>> temp = Range.temperature()  # Default: [0.0, 1.0]
            >>> temp_factual = Range.temperature(conservative=True)  # [0.0, 0.5]
            >>> temp_creative = Range.temperature(creative=True)  # [0.7, 1.5]
        """
        if conservative and creative:
            raise ValueError("Cannot specify both conservative=True and creative=True")
        if conservative:
            return cls(0.0, 0.5, default=0.2, name="temperature")
        elif creative:
            return cls(0.7, 1.5, default=1.0, name="temperature")
        return cls(0.0, 1.0, default=0.7, name="temperature")

    @classmethod
    def top_p(cls) -> Range:
        """Pre-configured top_p (nucleus sampling) parameter.

        Returns:
            Range [0.1, 1.0] with default 0.9

        Example:
            >>> top_p = Range.top_p()
        """
        return cls(0.1, 1.0, default=0.9, name="top_p")

    @classmethod
    def frequency_penalty(cls) -> Range:
        """Pre-configured frequency penalty parameter.

        Returns:
            Range [0.0, 2.0] with default 0.0

        Example:
            >>> freq_pen = Range.frequency_penalty()
        """
        return cls(0.0, 2.0, default=0.0, name="frequency_penalty")

    @classmethod
    def presence_penalty(cls) -> Range:
        """Pre-configured presence penalty parameter.

        Returns:
            Range [0.0, 2.0] with default 0.0

        Example:
            >>> pres_pen = Range.presence_penalty()
        """
        return cls(0.0, 2.0, default=0.0, name="presence_penalty")

    @classmethod
    def similarity_threshold(cls) -> Range:
        """Pre-configured similarity threshold for RAG retrieval.

        Returns:
            Range [0.0, 1.0] with default 0.5

        Example:
            >>> threshold = Range.similarity_threshold()
        """
        return cls(0.0, 1.0, default=0.5, name="similarity_threshold")

    @classmethod
    def mmr_lambda(cls) -> Range:
        """Pre-configured MMR (Maximal Marginal Relevance) lambda parameter.

        Lambda controls the trade-off between relevance and diversity.
        Higher values favor relevance, lower values favor diversity.

        Returns:
            Range [0.0, 1.0] with default 0.5

        Example:
            >>> mmr = Range.mmr_lambda()
        """
        return cls(0.0, 1.0, default=0.5, name="mmr_lambda")

    @classmethod
    def chunk_overlap_ratio(cls) -> Range:
        """Pre-configured chunk overlap ratio for RAG document splitting.

        Returns:
            Range [0.0, 0.5] with default 0.1

        Example:
            >>> overlap = Range.chunk_overlap_ratio()
        """
        return cls(0.0, 0.5, default=0.1, name="chunk_overlap_ratio")


@dataclass(frozen=True, slots=True)
class IntRange(NumericConstraintBuilderMixin, ParameterRange):
    """Integer range for optimization.

    Args:
        low: Lower bound (inclusive)
        high: Upper bound (inclusive)
        step: Optional step size (default: 1)
        log: Whether to use log-scale sampling (default: False).
             Note: Cannot be combined with step (Optuna limitation).
        default: Optional default value (populates default_config)
        name: Optional TVAR name (auto-assigned from decorator kwarg if not set)
        unit: Optional unit of measurement (e.g., "count", "tokens")
        agent: Optional agent identifier for multi-agent experiments.
            When set, this parameter will be grouped with other parameters
            belonging to the same agent in the UI.

    Example:
        >>> max_tokens = IntRange(100, 4096)
        >>> batch_size = IntRange(16, 256, step=16)
        >>> # With TVL features
        >>> tokens = IntRange(100, 4096, unit="tokens")
        >>> # Multi-agent: assign to specific agent
        >>> financial_tokens = IntRange(100, 4096, agent="financial")

    Raises:
        TypeError: If low/high are not integers
        ValueError: If low >= high, step <= 0, or invalid log/step combination
    """

    low: int
    high: int
    step: int | None = None
    log: bool = False
    default: int | None = None
    # TVL fields
    name: str | None = None
    unit: str | None = None
    # Multi-agent support
    agent: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.low, int) or not isinstance(self.high, int):
            raise TypeError(
                f"IntRange bounds must be integers, got low={type(self.low).__name__}, "
                f"high={type(self.high).__name__}"
            )
        if self.low >= self.high:
            raise ValueError(
                f"IntRange low ({self.low}) must be less than high ({self.high})"
            )
        if self.step is not None and self.step <= 0:
            raise ValueError(f"IntRange step must be positive, got {self.step}")
        if self.log and self.low <= 0:
            raise ValueError(f"log=True requires positive bounds, got low={self.low}")
        if self.log and self.step is not None:
            raise ValueError(
                "Cannot use log=True with step (Optuna limitation). "
                "Use either log-scale OR discrete steps, not both."
            )
        if self.default is not None and not (self.low <= self.default <= self.high):
            raise ValueError(
                f"default {self.default} is outside range [{self.low}, {self.high}]"
            )

    def to_config_value(self) -> tuple[int, int] | dict[str, Any]:
        """Convert to internal format.

        Returns tuple for simple ranges, dict when log/step is set.
        """
        if self.step is None and not self.log:
            return (self.low, self.high)
        result: dict[str, Any] = {
            "type": "int",
            "low": self.low,
            "high": self.high,
        }
        if self.step is not None:
            result["step"] = self.step
        if self.log:
            result["log"] = True
        return result

    def get_default(self) -> int | None:
        """Return the default value if set."""
        return self.default

    def to_tuple(self) -> tuple[int, int]:
        """Return as simple (low, high) tuple."""
        return (self.low, self.high)

    # Constraint builder methods (equals, not_equals, gt, gte, lt, lte,
    # in_range, is_in, not_in) are provided by NumericConstraintBuilderMixin

    # =========================================================================
    # Factory Methods - Domain Presets
    # =========================================================================

    @classmethod
    def max_tokens(
        cls,
        *,
        task: Literal["short", "medium", "long"] = "medium",
    ) -> IntRange:
        """Pre-configured max_tokens by task type.

        Args:
            task: Type of task - "short" (50-256), "medium" (256-1024),
                  or "long" (1024-4096)

        Returns:
            IntRange configured for the specified task type

        Example:
            >>> tokens = IntRange.max_tokens()  # medium: [256, 1024]
            >>> tokens_short = IntRange.max_tokens(task="short")  # [50, 256]
            >>> tokens_long = IntRange.max_tokens(task="long")  # [1024, 4096]
        """
        ranges: dict[str, tuple[int, int, int]] = {
            "short": (50, 256, 128),
            "medium": (256, 1024, 512),
            "long": (1024, 4096, 2048),
        }
        if task not in ranges:
            raise ValueError(f"task must be 'short', 'medium', or 'long', got '{task}'")
        low, high, default = ranges[task]
        return cls(low, high, step=64, default=default, name="max_tokens")

    @classmethod
    def k_retrieval(cls, *, max_k: int = 10) -> IntRange:
        """Number of documents to retrieve in RAG.

        Args:
            max_k: Maximum k value (default: 10)

        Returns:
            IntRange [1, max_k] with default 3

        Example:
            >>> k = IntRange.k_retrieval()  # [1, 10]
            >>> k_large = IntRange.k_retrieval(max_k=20)  # [1, 20]
        """
        return cls(1, max_k, default=3, name="k")

    @classmethod
    def chunk_size(cls) -> IntRange:
        """Document chunk size for RAG document splitting.

        Returns:
            IntRange [100, 1000] with step 100 and default 500

        Example:
            >>> chunk = IntRange.chunk_size()
        """
        return cls(100, 1000, step=100, default=500, name="chunk_size")

    @classmethod
    def chunk_overlap(cls) -> IntRange:
        """Document chunk overlap for RAG document splitting.

        Returns:
            IntRange [0, 200] with step 25 and default 50

        Example:
            >>> overlap = IntRange.chunk_overlap()
        """
        return cls(0, 200, step=25, default=50, name="chunk_overlap")

    @classmethod
    def few_shot_count(cls, *, max_examples: int = 10) -> IntRange:
        """Number of few-shot examples to include in prompt.

        Args:
            max_examples: Maximum number of examples (default: 10)

        Returns:
            IntRange [0, max_examples] with default 3

        Example:
            >>> few_shot = IntRange.few_shot_count()  # [0, 10]
            >>> few_shot_limited = IntRange.few_shot_count(max_examples=5)  # [0, 5]
        """
        return cls(0, max_examples, default=3, name="few_shot_count")

    @classmethod
    def batch_size(
        cls,
        *,
        min_size: int = 1,
        max_size: int = 64,
        default: int = 16,
    ) -> IntRange:
        """Batch size for parallel processing.

        Args:
            min_size: Minimum batch size (default: 1)
            max_size: Maximum batch size (default: 64)
            default: Default batch size (default: 16)

        Returns:
            IntRange configured for batch processing

        Example:
            >>> batch = IntRange.batch_size()  # [1, 64]
            >>> batch_large = IntRange.batch_size(max_size=128)  # [1, 128]
        """
        return cls(min_size, max_size, default=default, name="batch_size")


@dataclass(frozen=True, slots=True)
class LogRange(NumericConstraintBuilderMixin, ParameterRange):
    """Log-scale float range for optimization.

    Convenience class for Range(low, high, log=True). Useful for parameters
    that vary over orders of magnitude like learning rates and regularization.

    Args:
        low: Lower bound (must be positive)
        high: Upper bound (must be positive)
        default: Optional default value (populates default_config)
        name: Optional TVAR name (auto-assigned from decorator kwarg if not set)
        unit: Optional unit of measurement
        agent: Optional agent identifier for multi-agent experiments.
            When set, this parameter will be grouped with other parameters
            belonging to the same agent in the UI.

    Example:
        >>> learning_rate = LogRange(1e-5, 1e-1)
        >>> regularization = LogRange(0.001, 10.0)
        >>> # Multi-agent: assign to specific agent
        >>> financial_lr = LogRange(1e-5, 1e-1, agent="financial")

    Raises:
        ValueError: If bounds are not positive or low >= high
    """

    low: float
    high: float
    default: float | None = None
    # TVL fields
    name: str | None = None
    unit: str | None = None
    # Multi-agent support
    agent: str | None = None

    def __post_init__(self) -> None:
        if self.low <= 0 or self.high <= 0:
            raise ValueError(
                f"LogRange requires positive bounds, got ({self.low}, {self.high})"
            )
        if self.low >= self.high:
            raise ValueError(
                f"LogRange low ({self.low}) must be less than high ({self.high})"
            )
        if self.default is not None and not (self.low <= self.default <= self.high):
            raise ValueError(
                f"default {self.default} is outside range [{self.low}, {self.high}]"
            )

    def to_config_value(self) -> dict[str, Any]:
        """Convert to internal format with log=True."""
        return {
            "type": "float",
            "low": self.low,
            "high": self.high,
            "log": True,
        }

    def get_default(self) -> float | None:
        """Return the default value if set."""
        return self.default

    def to_tuple(self) -> tuple[float, float]:
        """Return as simple (low, high) tuple (loses log information)."""
        return (self.low, self.high)

    # Constraint builder methods (equals, not_equals, gt, gte, lt, lte,
    # in_range, is_in, not_in) are provided by NumericConstraintBuilderMixin


@dataclass(frozen=True, slots=True)
class Choices(CategoricalConstraintBuilderMixin, ParameterRange, Generic[T]):
    """Categorical choices for optimization.

    Args:
        values: Sequence of allowed values (list or tuple, NOT str/bytes)
        default: Optional default value (must be in values, populates default_config)
        name: Optional TVAR name (auto-assigned from decorator kwarg if not set)
        unit: Optional unit of measurement (rarely needed for categorical)
        agent: Optional agent identifier for multi-agent experiments.
            When set, this parameter will be grouped with other parameters
            belonging to the same agent in the UI.
        enforce_type: If True (default), validates all values have the same type.
            Set to False to allow mixed types (e.g., [None, "default", 1]).

    Example:
        >>> model = Choices(["gpt-4", "gpt-3.5-turbo", "claude-2"])
        >>> use_cache = Choices([True, False], default=True)
        >>> temperature = Choices([0.0, 0.3, 0.7, 1.0])
        >>> # Multi-agent: assign to specific agent
        >>> financial_model = Choices(["gpt-4", "gpt-3.5"], agent="financial")

    Raises:
        TypeError: If values is a string or bytes, or if enforce_type=True and
            values contain mixed types
        ValueError: If values is empty or default is not in values
    """

    values: Sequence[T]
    default: T | None = None
    # TVL fields
    name: str | None = None
    unit: str | None = None
    # Multi-agent support
    agent: str | None = None
    # Type enforcement (D-2 fix)
    enforce_type: bool = True

    def __post_init__(self) -> None:
        # Reject str/bytes which are technically sequences but not valid choices
        if isinstance(self.values, (str, bytes)):
            raise TypeError(
                "Choices values must be a list or tuple, not str/bytes. "
                f"Did you mean Choices([{self.values!r}])?"
            )
        # Convert to tuple for immutability (frozen dataclass)
        object.__setattr__(self, "values", tuple(self.values))
        if not self.values:
            raise ValueError("Choices must have at least one value")
        if self.default is not None and self.default not in self.values:
            raise ValueError(
                f"default {self.default!r} is not in choices {list(self.values)}"
            )
        # Type enforcement: validate all values have the same type
        if self.enforce_type and len(self.values) > 1:
            self._validate_type_consistency()

    def _validate_type_consistency(self) -> None:
        """Validate all values have consistent types.

        Handles special cases:
        - bool and int are treated as distinct types (bool is subclass of int)
        - None values are allowed alongside any type
        - Numbers (int, float) are allowed together
        """
        # Filter out None values for type checking
        non_none_values = [v for v in self.values if v is not None]
        if len(non_none_values) <= 1:
            return  # Single type or all None

        # Get types, treating bool specially (before int check since bool is subclass)
        def _get_type_category(v: object) -> str:
            if isinstance(v, bool):
                return "bool"
            if isinstance(v, int):
                return "int"
            if isinstance(v, float):
                return "float"
            return type(v).__name__

        type_categories = {_get_type_category(v) for v in non_none_values}

        # Allow int and float to coexist (both are numbers)
        numeric_types = {"int", "float"}
        if type_categories <= numeric_types:
            return  # All numeric, OK

        # Check for type consistency
        if len(type_categories) > 1:
            # Get actual types for error message
            types_found = sorted(type_categories)
            raise TypeError(
                f"Choices values must have consistent types. "
                f"Found mixed types: {types_found}. "
                f"Values: {list(self.values)[:5]}{'...' if len(self.values) > 5 else ''}. "
                f"Set enforce_type=False to allow mixed types."
            )

    def to_config_value(self) -> list[T]:
        """Convert to internal list format."""
        return list(self.values)

    def get_default(self) -> T | None:
        """Return the default value if set."""
        return self.default

    def to_list(self) -> list[T]:
        """Return as list for backward compatibility."""
        return list(self.values)

    def __iter__(self):
        """Allow iteration over choices."""
        return iter(self.values)

    def __len__(self) -> int:
        """Return number of choices."""
        return len(self.values)

    def __contains__(self, item: T) -> bool:
        """Check if item is in choices."""
        return item in self.values

    # Constraint builder methods (equals, not_equals, is_in, not_in)
    # are provided by CategoricalConstraintBuilderMixin

    # =========================================================================
    # Factory Methods - Domain Presets
    # =========================================================================

    @classmethod
    def model(
        cls,
        *,
        provider: str | None = None,
        tier: Literal["fast", "balanced", "quality"] = "balanced",
    ) -> Choices[str]:
        """Pre-configured model selection using provider registry.

        Uses environment variable TRAIGENT_MODELS_{PROVIDER}_{TIER} if set,
        otherwise falls back to sensible defaults.

        Args:
            provider: Provider name (e.g., "openai", "anthropic"). If None,
                     returns models from all providers.
            tier: Performance tier - "fast", "balanced", or "quality"

        Returns:
            Choices instance with model names

        Example:
            >>> model = Choices.model()  # Default balanced models
            >>> model_fast = Choices.model(provider="openai", tier="fast")
            >>> model_quality = Choices.model(provider="anthropic", tier="quality")
        """
        # Check for environment-based model list
        env_key = f"TRAIGENT_MODELS_{(provider or 'DEFAULT').upper()}_{tier.upper()}"
        env_models = os.environ.get(env_key)
        if env_models:
            models = [m.strip() for m in env_models.split(",") if m.strip()]
            if models:
                return Choices(models, name="model")

        # Fallback defaults
        fallbacks: dict[tuple[str | None, str], list[str]] = {
            ("openai", "fast"): ["gpt-4o-mini"],
            ("openai", "balanced"): ["gpt-4o-mini", "gpt-4o"],
            ("openai", "quality"): ["gpt-4o", "o1-preview"],
            ("anthropic", "fast"): ["claude-3-haiku-20240307"],
            ("anthropic", "balanced"): ["claude-3-5-sonnet-20241022"],
            ("anthropic", "quality"): ["claude-3-opus-20240229"],
            (None, "fast"): ["gpt-4o-mini", "claude-3-haiku-20240307"],
            (None, "balanced"): ["gpt-4o-mini", "gpt-4o", "claude-3-5-sonnet-20241022"],
            (None, "quality"): ["gpt-4o", "claude-3-opus-20240229"],
        }

        models_found = fallbacks.get((provider, tier))
        if not models_found:
            logger.warning(
                f"No model list found for provider={provider}, tier={tier}. "
                f"Using default models. Set {env_key} for explicit control."
            )
            models_found = ["gpt-4o-mini"]

        return Choices(models_found, name="model")

    @classmethod
    def prompting_strategy(cls) -> Choices[str]:
        """Pre-configured prompting strategies.

        Returns:
            Choices with common prompting strategy options

        Example:
            >>> strategy = Choices.prompting_strategy()
        """
        return Choices(
            ["direct", "chain_of_thought", "react", "self_consistency"],
            default="direct",
            name="prompting_strategy",
        )

    @classmethod
    def context_format(cls) -> Choices[str]:
        """Pre-configured context formatting options for RAG.

        Returns:
            Choices with common context format options

        Example:
            >>> fmt = Choices.context_format()
        """
        return Choices(
            ["bullet", "numbered", "xml", "markdown", "json"],
            default="bullet",
            name="context_format",
        )

    @classmethod
    def retriever_type(cls) -> Choices[str]:
        """Pre-configured retriever types for RAG.

        Returns:
            Choices with common retriever type options

        Example:
            >>> retriever = Choices.retriever_type()
        """
        return Choices(
            ["similarity", "mmr", "bm25", "hybrid"],
            default="similarity",
            name="retriever",
        )

    @classmethod
    def embedding_model(
        cls,
        *,
        provider: str | None = None,
    ) -> Choices[str]:
        """Pre-configured embedding model selection.

        Args:
            provider: Provider name (e.g., "openai"). If None,
                     returns models from common providers.

        Returns:
            Choices with embedding model names

        Example:
            >>> embedding = Choices.embedding_model()
            >>> embedding_openai = Choices.embedding_model(provider="openai")
        """
        fallbacks: dict[str | None, list[str]] = {
            "openai": [
                "text-embedding-3-small",
                "text-embedding-3-large",
                "text-embedding-ada-002",
            ],
            None: [
                "text-embedding-3-small",
                "text-embedding-3-large",
            ],
        }

        models = fallbacks.get(provider, fallbacks[None])
        return Choices(models, name="embedding_model")

    @classmethod
    def reranker_model(cls) -> Choices[str]:
        """Pre-configured reranker models for RAG.

        Returns:
            Choices with common reranker model options

        Example:
            >>> reranker = Choices.reranker_model()
        """
        return Choices(
            [
                "none",
                "cohere-rerank-v3",
                "cross-encoder/ms-marco-MiniLM-L-6-v2",
                "llm-rerank",
            ],
            default="none",
            name="reranker",
        )


# =============================================================================
# Utility Functions
# =============================================================================


def is_parameter_range(value: Any) -> bool:
    """Check if a value is a ParameterRange instance.

    Useful for validation code that needs to detect the new range types.
    """
    return isinstance(value, ParameterRange)


def is_inline_param_definition(value: Any) -> bool:
    """Check if a value looks like an inline parameter definition.

    Recognizes:
    - ParameterRange instances (Range, IntRange, LogRange, Choices)
    - Tuples with exactly 2 numeric elements (legacy range syntax)

    Note: Lists are NOT recognized as inline param definitions to catch typos.
    Users who want inline list syntax should use Choices() instead.

    Args:
        value: The value to check

    Returns:
        True if value appears to be a parameter definition
    """
    if is_parameter_range(value):
        return True
    if isinstance(value, tuple) and len(value) == 2:
        return all(isinstance(v, (int, float)) for v in value)
    # Lists are NOT treated as inline params to catch typos like `objectivs=[...]`
    # Users should use Choices([...]) for categorical parameters
    return False


def normalize_config_value(
    value: ParameterRange | tuple[Any, ...] | list[Any] | dict[str, Any],
) -> tuple[Any, ...] | list[Any] | dict[str, Any]:
    """Convert a ParameterRange to its primitive format.

    If value is already a primitive (tuple/list/dict), returns it unchanged.

    Args:
        value: A ParameterRange instance or primitive value

    Returns:
        The primitive configuration format
    """
    if isinstance(value, ParameterRange):
        return value.to_config_value()
    return value  # type: ignore[return-value]


def normalize_parameter_value(
    value: Any,
) -> tuple[Any, ...] | list[Any] | dict[str, Any]:
    """Normalize a ParameterRange or primitive configuration value.

    Alias for normalize_config_value to match API naming expectations.
    """
    return normalize_config_value(value)


def _process_param_entry(
    key: str,
    value: Any,
    result: dict[str, Any],
    defaults: dict[str, Any],
) -> ParameterRange | None:
    """Process a single parameter entry, updating result and defaults dicts.

    Args:
        key: The parameter name from the decorator kwarg
        value: The parameter definition (ParameterRange or primitive)
        result: Dict to update with normalized config values
        defaults: Dict to update with default values

    Returns:
        The ParameterRange with name auto-assigned if applicable, None otherwise.
        This allows callers to use the named parameter for constraint building.
    """
    if isinstance(value, ParameterRange):
        # D-1: Auto-assign name from decorator kwarg if not already set
        param = value
        if param.name is None:
            try:
                # Cast to Any since all ParameterRange subclasses are dataclasses
                param = replace(cast(Any, param), name=key)
            except TypeError:
                # Fallback if replace fails (shouldn't happen with our dataclasses)
                logger.debug(
                    f"Could not auto-assign name '{key}' to {type(value).__name__}"
                )
        result[key] = param.to_config_value()
        default_val = param.get_default()
        if default_val is not None:
            defaults[key] = default_val
        return param
    else:
        result[key] = normalize_config_value(value)
        return None


def normalize_configuration_space(
    config_space: dict[str, Any] | ConfigSpace | None,
    inline_params: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Normalize a configuration space and extract defaults.

    Merges inline decorator kwargs with an explicit configuration_space dict,
    normalizing all ParameterRange instances to their primitive formats.

    Precedence: inline_params override config_space entries.

    Args:
        config_space: Explicit configuration_space dict, or a ConfigSpace object
            (may contain Range/Choices)
        inline_params: Inline kwargs from decorator that are param definitions

    Returns:
        Tuple of (normalized_config_space, defaults_dict)
        - normalized_config_space: All values converted to primitives
        - defaults_dict: Extracted default values from Range/Choices
    """
    result: dict[str, Any] = {}
    defaults: dict[str, Any] = {}

    # Start with explicit config_space (lower precedence)
    if config_space:
        # Handle ConfigSpace objects by extracting their tvars
        if hasattr(config_space, "tvars") and hasattr(config_space, "constraints"):
            # It's a ConfigSpace object - extract the tvars dict
            config_space = config_space.tvars
        elif not isinstance(config_space, dict):
            from traigent.utils.exceptions import ValidationError

            raise ValidationError(
                f"Expected dictionary for configuration_space, got {type(config_space).__name__}"
            )
        for key, value in config_space.items():
            _process_param_entry(key, value, result, defaults)

    # Add/override with inline parameters (higher precedence)
    if inline_params:
        for key, value in inline_params.items():
            _process_param_entry(key, value, result, defaults)

    return result, defaults


__all__ = [
    "ParameterRange",
    "Range",
    "IntRange",
    "LogRange",
    "Choices",
    "is_parameter_range",
    "is_inline_param_definition",
    "normalize_config_value",
    "normalize_parameter_value",
    "normalize_configuration_space",
]
