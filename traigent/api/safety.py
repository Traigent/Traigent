"""Safety Constraint Presets for Traigent Optimization.

This module provides factory presets for common LLM safety metrics that integrate
with the @optimize decorator's constraint system. Safety constraints enforce
minimum quality thresholds on metrics like faithfulness, relevancy, and toxicity.

Example:
    >>> from traigent.api.safety import faithfulness, hallucination_rate
    >>>
    >>> @traigent.optimize(
    ...     safety_constraints=[
    ...         faithfulness.above(0.9),           # Faithfulness >= 90%
    ...         hallucination_rate().below(0.1),   # Hallucination rate < 10%
    ...     ]
    ... )
    ... def my_rag_function(query: str) -> str:
    ...     ...

RAGAS Integration:
    This module wraps RAGAS metrics (https://docs.ragas.io/) for RAG evaluation.
    RAGAS is distributed under Apache 2.0 license. See RAGAS_ATTRIBUTION for
    full attribution notice.

Traceability: CONC-Layer-API CONC-Quality-Safety FUNC-SAFETY-CONSTRAINTS
"""

from __future__ import annotations

import math
import threading
import warnings
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from traigent.tvl.models import ChanceConstraint

# =============================================================================
# RAGAS Attribution (Apache 2.0 License)
# =============================================================================

RAGAS_ATTRIBUTION = """
RAGAS (Retrieval Augmented Generation Assessment) Metrics
=========================================================
Copyright 2023 Vibrant Labs

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Source: https://github.com/explodinggradients/ragas
Documentation: https://docs.ragas.io/
"""

# =============================================================================
# Exceptions
# =============================================================================


class RAGASMetricNotAvailableError(ImportError):
    """Raised when RAGAS metrics are used but ragas is not installed."""

    def __init__(self, metric_name: str) -> None:
        super().__init__(
            f"RAGAS metric '{metric_name}' requires the 'ragas' package. "
            f"Install with: pip install 'traigent[ragas]' or pip install ragas"
        )
        self.metric_name = metric_name


class SafetyDependencyWarning(UserWarning):
    """Warning for missing optional safety dependencies."""

    pass


# =============================================================================
# Core Data Classes
# =============================================================================


@dataclass(frozen=True)
class SafetyThreshold:
    """Immutable threshold configuration for safety constraints.

    Attributes:
        metric_name: Name of the metric being constrained.
        operator: Comparison operator (">=", "<=", ">", "<", "==").
        value: Threshold value in [0, 1] range.
        confidence: Optional statistical confidence level for Clopper-Pearson
            validation. If None, uses simple threshold checking.
        min_samples: Minimum samples required for statistical validity.
    """

    metric_name: str
    operator: str
    value: float
    confidence: float | None = None
    min_samples: int = 30

    def __post_init__(self) -> None:
        """Validate threshold configuration."""
        if self.confidence is not None and not 0 < self.confidence < 1:
            raise ValueError(f"confidence must be in (0, 1), got {self.confidence}")
        if self.min_samples < 1:
            raise ValueError(f"min_samples must be >= 1, got {self.min_samples}")
        if self.operator not in (">=", "<=", ">", "<", "=="):
            raise ValueError(f"Unknown operator: {self.operator}")


@dataclass(frozen=True)
class SafetyValidationResult:
    """Result of statistical safety validation.

    Attributes:
        metric_name: Name of the validated metric.
        satisfied: Whether the constraint is satisfied.
        observed_rate: Observed success rate across samples.
        lower_bound: Clopper-Pearson lower bound (equals observed_rate if no CI).
        threshold: Required threshold value.
        confidence: Confidence level used for validation.
        sample_count: Number of samples evaluated.
        message: Human-readable explanation of the result.
    """

    metric_name: str
    satisfied: bool
    observed_rate: float
    lower_bound: float
    threshold: float
    confidence: float
    sample_count: int
    message: str

    @property
    def is_statistically_valid(self) -> bool:
        """Whether enough samples exist for statistical validity."""
        return self.sample_count >= 30


# =============================================================================
# Base Classes
# =============================================================================


class SafetyMetric(ABC):
    """Abstract base class for safety metrics.

    Provides factory methods (.above(), .below(), .between()) for creating
    threshold-based constraints that integrate with the @optimize decorator.

    Thread Safety:
        SafetyMetric instances are immutable and thread-safe.
        The evaluate() method must be implemented as thread-safe.

    Example:
        >>> # Using a built-in preset
        >>> constraint = faithfulness.above(0.9)
        >>>
        >>> # With statistical validation
        >>> constraint = faithfulness.above(0.85, confidence=0.95)
    """

    def __init__(self, name: str, description: str = "") -> None:
        """Initialize safety metric.

        Args:
            name: Unique identifier for this metric.
            description: Human-readable description.
        """
        self._name = name
        self._description = description

    @property
    def name(self) -> str:
        """Metric name."""
        return self._name

    @property
    def description(self) -> str:
        """Human-readable description."""
        return self._description

    @abstractmethod
    def evaluate(self, config: dict[str, Any], metrics: dict[str, Any]) -> float:
        """Evaluate the safety metric for a given trial.

        Args:
            config: Trial configuration parameters.
            metrics: Computed metrics from trial evaluation.

        Returns:
            Float score, typically in [0, 1] range where higher is safer.

        Thread Safety:
            This method MUST be thread-safe as it may be called
            from parallel trial executors.
        """
        pass

    def above(
        self,
        threshold: float,
        *,
        confidence: float | None = None,
        min_samples: int = 30,
    ) -> SafetyConstraint:
        """Create constraint requiring metric >= threshold.

        Args:
            threshold: Minimum acceptable value.
            confidence: Optional statistical confidence level (e.g., 0.95).
                If provided, uses Clopper-Pearson lower bound validation.
            min_samples: Minimum samples for statistical validity.

        Returns:
            SafetyConstraint usable in @optimize(safety_constraints=[...]).

        Example:
            >>> faithfulness.above(0.9)  # Simple threshold
            >>> faithfulness.above(0.85, confidence=0.95)  # With CI
        """
        return SafetyConstraint(
            metric=self,
            threshold=SafetyThreshold(
                metric_name=self._name,
                operator=">=",
                value=threshold,
                confidence=confidence,
                min_samples=min_samples,
            ),
        )

    def below(
        self,
        threshold: float,
        *,
        confidence: float | None = None,
        min_samples: int = 30,
    ) -> SafetyConstraint:
        """Create constraint requiring metric <= threshold.

        Useful for metrics where lower is better (e.g., hallucination_rate).

        Args:
            threshold: Maximum acceptable value.
            confidence: Optional statistical confidence level.
            min_samples: Minimum samples for statistical validity.

        Returns:
            SafetyConstraint usable in @optimize(safety_constraints=[...]).

        Example:
            >>> hallucination_rate().below(0.1)  # Max 10% hallucination
        """
        return SafetyConstraint(
            metric=self,
            threshold=SafetyThreshold(
                metric_name=self._name,
                operator="<=",
                value=threshold,
                confidence=confidence,
                min_samples=min_samples,
            ),
        )

    def between(
        self,
        lower: float,
        upper: float,
        *,
        confidence: float | None = None,
        min_samples: int = 30,
    ) -> CompoundSafetyConstraint:
        """Create constraint requiring lower <= metric <= upper.

        Args:
            lower: Minimum acceptable value.
            upper: Maximum acceptable value.
            confidence: Optional statistical confidence level.
            min_samples: Minimum samples for statistical validity.

        Returns:
            CompoundSafetyConstraint combining lower and upper bounds.
        """
        return CompoundSafetyConstraint(
            constraints=[
                self.above(lower, confidence=confidence, min_samples=min_samples),
                self.below(upper, confidence=confidence, min_samples=min_samples),
            ],
            combinator="and",
        )


@dataclass(frozen=True)
class SafetyConstraint:
    """A safety constraint combining a metric with a threshold.

    Implements __call__ to integrate with the existing constraint system,
    allowing use as a post-eval constraint callable.

    Attributes:
        metric: The safety metric to evaluate.
        threshold: Threshold configuration for the constraint.

    Thread Safety:
        SafetyConstraint is immutable and thread-safe.
    """

    metric: SafetyMetric
    threshold: SafetyThreshold

    def __call__(self, config: dict[str, Any], metrics: dict[str, Any]) -> bool:
        """Evaluate constraint as a callable.

        This signature matches the post-eval constraint callable pattern:
        Callable[[dict, dict], bool]

        Args:
            config: Trial configuration parameters.
            metrics: Computed metrics from trial evaluation.

        Returns:
            True if constraint is satisfied, False otherwise.
        """
        score = self.metric.evaluate(config, metrics)

        # Handle NaN scores (metric not computed)
        if math.isnan(score):
            return False

        return self._check_threshold(score)

    def _check_threshold(self, score: float) -> bool:
        """Check if score satisfies the threshold condition."""
        op = self.threshold.operator
        val = self.threshold.value

        if op == ">=":
            return score >= val
        elif op == "<=":
            return score <= val
        elif op == ">":
            return score > val
        elif op == "<":
            return score < val
        elif op == "==":
            return abs(score - val) < 1e-9
        else:
            raise ValueError(f"Unknown operator: {op}")

    def to_chance_constraint(self) -> ChanceConstraint:
        """Convert to ChanceConstraint for TVL statistical validation.

        Only valid if confidence is specified.

        Returns:
            ChanceConstraint instance for promotion gate validation.

        Raises:
            ValueError: If confidence is not specified.
        """
        if self.threshold.confidence is None:
            raise ValueError(
                "Cannot create ChanceConstraint without confidence level. "
                f"Use {self.metric.name}.above(threshold, confidence=0.95)"
            )

        from traigent.tvl.models import ChanceConstraint

        return ChanceConstraint(
            name=f"safety_{self.threshold.metric_name}",
            threshold=self.threshold.value,
            confidence=self.threshold.confidence,
        )

    @property
    def has_statistical_validation(self) -> bool:
        """Whether this constraint uses statistical validation."""
        return self.threshold.confidence is not None

    @property
    def requires_metrics(self) -> bool:
        """Safety constraints always require metrics (post-eval)."""
        return True

    def __and__(self, other: SafetyConstraint) -> CompoundSafetyConstraint:
        """Combine constraints with AND logic.

        Example:
            >>> combined = faithfulness.above(0.9) & toxicity_score().below(0.1)
        """
        return CompoundSafetyConstraint(constraints=[self, other], combinator="and")

    def __or__(self, other: SafetyConstraint) -> CompoundSafetyConstraint:
        """Combine constraints with OR logic.

        Example:
            >>> combined = accuracy.above(0.9) | accuracy.above(0.85)
        """
        return CompoundSafetyConstraint(constraints=[self, other], combinator="or")


class CompoundSafetyConstraint:
    """Compound constraint combining multiple safety constraints.

    Attributes:
        constraints: Tuple of safety constraints to combine.
        combinator: How to combine results ("and" or "or").
    """

    def __init__(
        self,
        constraints: list[SafetyConstraint | CompoundSafetyConstraint],
        combinator: str = "and",
    ) -> None:
        """Initialize compound constraint.

        Args:
            constraints: List of safety constraints to combine.
            combinator: Combination logic ("and" or "or").
        """
        self._constraints = tuple(constraints)
        self._combinator = combinator

    @property
    def constraints(self) -> tuple[SafetyConstraint | CompoundSafetyConstraint, ...]:
        """Constituent safety constraints."""
        return self._constraints

    @property
    def combinator(self) -> str:
        """Combination logic."""
        return self._combinator

    def __call__(self, config: dict[str, Any], metrics: dict[str, Any]) -> bool:
        """Evaluate compound constraint.

        Args:
            config: Trial configuration parameters.
            metrics: Computed metrics from trial evaluation.

        Returns:
            Combined result based on combinator logic.
        """
        results = [c(config, metrics) for c in self._constraints]
        if self._combinator == "and":
            return all(results)
        elif self._combinator == "or":
            return any(results)
        else:
            raise ValueError(f"Unknown combinator: {self._combinator}")

    @property
    def requires_metrics(self) -> bool:
        """Compound safety constraints always require metrics."""
        return True

    def __and__(
        self, other: SafetyConstraint | CompoundSafetyConstraint
    ) -> CompoundSafetyConstraint:
        """Combine with another constraint using AND logic.

        Preserves tree structure: (c1 | c2) & c3 evaluates as "(c1 OR c2) AND c3",
        not "c1 AND c2 AND c3".
        """
        # Treat self as a single unit to preserve boolean expression semantics
        return CompoundSafetyConstraint(constraints=[self, other], combinator="and")

    def __or__(
        self, other: SafetyConstraint | CompoundSafetyConstraint
    ) -> CompoundSafetyConstraint:
        """Combine with another constraint using OR logic.

        Preserves tree structure: (c1 & c2) | c3 evaluates as "(c1 AND c2) OR c3",
        not "c1 OR c2 OR c3".
        """
        # Treat self as a single unit to preserve boolean expression semantics
        return CompoundSafetyConstraint(constraints=[self, other], combinator="or")


# =============================================================================
# RAGAS Metric Implementation
# =============================================================================

_RAGAS_AVAILABLE: bool | None = None
_RAGAS_CACHE: dict[str, Any] = {}
_RAGAS_CACHE_LOCK = threading.Lock()


def _check_ragas_installed() -> bool:
    """Check if RAGAS is installed (cached result)."""
    global _RAGAS_AVAILABLE
    if _RAGAS_AVAILABLE is None:
        try:
            import ragas  # noqa: F401

            _RAGAS_AVAILABLE = True
        except ImportError:
            _RAGAS_AVAILABLE = False
    return _RAGAS_AVAILABLE


class RAGASMetric(SafetyMetric):
    """Safety metric backed by RAGAS evaluation framework.

    Wraps RAGAS metrics (faithfulness, answer_relevancy, etc.) with proper
    attribution and lazy loading to avoid import errors when ragas is not
    installed.

    Attribution:
        RAGAS is distributed under Apache 2.0 license.
        See RAGAS_ATTRIBUTION constant for full attribution notice.

    Thread Safety:
        Thread-safe through class-level caching with locks.
    """

    def __init__(
        self,
        name: str,
        ragas_metric_name: str,
        description: str = "",
        requires_context: bool = True,
    ) -> None:
        """Initialize RAGAS metric wrapper.

        Args:
            name: Display name for this metric.
            ragas_metric_name: Internal RAGAS metric identifier.
            description: Human-readable description.
            requires_context: Whether this metric needs context data.
        """
        super().__init__(name, description)
        self._ragas_metric_name = ragas_metric_name
        self._requires_context = requires_context

    @property
    def requires_context(self) -> bool:
        """Whether this metric requires context data."""
        return self._requires_context

    def _ensure_ragas_available(self) -> None:
        """Raise helpful error if RAGAS is not installed."""
        if not _check_ragas_installed():
            raise RAGASMetricNotAvailableError(self._name)

    def evaluate(self, config: dict[str, Any], metrics: dict[str, Any]) -> float:
        """Evaluate RAGAS metric.

        Looks for pre-computed RAGAS scores in the metrics dict under
        keys like "faithfulness" or "ragas_faithfulness".

        Args:
            config: Trial configuration (unused for RAGAS metrics).
            metrics: Computed metrics, should contain RAGAS scores.

        Returns:
            Float score in [0, 1], or NaN if not available.
        """
        # Check if pre-computed RAGAS score exists
        ragas_key = f"ragas_{self._ragas_metric_name}"
        if ragas_key in metrics:
            score = metrics[ragas_key]
            if isinstance(score, (int, float)) and not math.isnan(score):
                return float(score)

        # Alternative: check for metric under standard name
        if self._ragas_metric_name in metrics:
            score = metrics[self._ragas_metric_name]
            if isinstance(score, (int, float)) and not math.isnan(score):
                return float(score)

        # If no pre-computed score, return NaN
        return math.nan


# =============================================================================
# Non-RAGAS Metric Implementations
# =============================================================================


class MetricKeyMetric(SafetyMetric):
    """Safety metric that reads from a key in the metrics dict.

    Useful when metrics are pre-computed by evaluators and stored
    in the metrics dictionary.

    Thread Safety:
        Thread-safe (stateless evaluation).
    """

    def __init__(
        self,
        name: str,
        metric_key: str,
        description: str = "",
        default: float = 0.0,
        invert: bool = False,
    ) -> None:
        """Initialize metric key reader.

        Args:
            name: Display name for this metric.
            metric_key: Key to read from metrics dict.
            description: Human-readable description.
            default: Default value if key not found.
            invert: If True, return (1 - value) for metrics where lower is better.
        """
        super().__init__(name, description)
        self._metric_key = metric_key
        self._default = default
        self._invert = invert

    @property
    def metric_key(self) -> str:
        """Key used to read metric from metrics dict."""
        return self._metric_key

    def evaluate(self, config: dict[str, Any], metrics: dict[str, Any]) -> float:
        """Read metric value from metrics dict.

        Args:
            config: Trial configuration (unused).
            metrics: Computed metrics dictionary.

        Returns:
            Metric value, inverted if configured.
        """
        value = metrics.get(self._metric_key, self._default)
        if isinstance(value, (int, float)) and not math.isnan(value):
            result = float(value)
            return (1.0 - result) if self._invert else result
        return self._default


class CallableMetric(SafetyMetric):
    """Safety metric defined by a user-provided callable.

    Thread Safety:
        The provided callable MUST be thread-safe.
    """

    def __init__(
        self,
        name: str,
        evaluator: Callable[[dict[str, Any], dict[str, Any]], float],
        description: str = "",
    ) -> None:
        """Initialize callable metric.

        Args:
            name: Display name for this metric.
            evaluator: Callable(config, metrics) -> float in [0, 1].
            description: Human-readable description.
        """
        super().__init__(name, description)
        self._evaluator = evaluator

    def evaluate(self, config: dict[str, Any], metrics: dict[str, Any]) -> float:
        """Evaluate using the provided callable.

        Args:
            config: Trial configuration.
            metrics: Computed metrics.

        Returns:
            Result from the callable.
        """
        return self._evaluator(config, metrics)


# =============================================================================
# Statistical Validation
# =============================================================================


class SafetyValidator:
    """Validates safety constraints with optional statistical rigor.

    For constraints with confidence levels, uses Clopper-Pearson
    confidence intervals to ensure the lower bound of the success
    rate meets the threshold.

    Thread Safety:
        This class is thread-safe. Internal state is protected by locks.

    Example:
        >>> validator = SafetyValidator()
        >>> for trial in trials:
        ...     validator.record_result(constraint, config, metrics)
        >>> result = validator.validate(constraint)
        >>> print(result.message)
    """

    def __init__(self) -> None:
        """Initialize safety validator."""
        self._results_lock = threading.Lock()
        self._sample_results: dict[str, list[bool]] = {}

    def record_result(
        self,
        constraint: SafetyConstraint,
        config: dict[str, Any],
        metrics: dict[str, Any],
    ) -> bool:
        """Record a single trial result for a safety constraint.

        Args:
            constraint: The safety constraint being validated.
            config: Trial configuration.
            metrics: Trial metrics.

        Returns:
            Whether the constraint passed for this trial.
        """
        passed = constraint(config, metrics)

        with self._results_lock:
            key = constraint.threshold.metric_name
            if key not in self._sample_results:
                self._sample_results[key] = []
            self._sample_results[key].append(passed)

        return passed

    def validate(self, constraint: SafetyConstraint) -> SafetyValidationResult:
        """Validate a safety constraint with statistical analysis.

        For constraints without confidence level, checks if all recorded
        trials passed (or uses observed rate).

        For constraints with confidence level, uses Clopper-Pearson
        lower bound to validate statistical significance.

        Args:
            constraint: The safety constraint to validate.

        Returns:
            SafetyValidationResult with validation outcome.
        """
        key = constraint.threshold.metric_name

        with self._results_lock:
            results = list(self._sample_results.get(key, []))

        if not results:
            return SafetyValidationResult(
                metric_name=key,
                satisfied=False,
                observed_rate=0.0,
                lower_bound=0.0,
                threshold=constraint.threshold.value,
                confidence=constraint.threshold.confidence or 0.95,
                sample_count=0,
                message=f"No samples recorded for {key}",
            )

        n_samples = len(results)
        n_passed = sum(results)
        observed_rate = n_passed / n_samples

        # Simple threshold check (no confidence)
        if constraint.threshold.confidence is None:
            op = constraint.threshold.operator
            threshold = constraint.threshold.value

            if op == ">=":
                satisfied = observed_rate >= threshold
            elif op == "<=":
                satisfied = observed_rate <= threshold
            else:
                satisfied = observed_rate >= threshold

            return SafetyValidationResult(
                metric_name=key,
                satisfied=satisfied,
                observed_rate=observed_rate,
                lower_bound=observed_rate,
                threshold=threshold,
                confidence=1.0,
                sample_count=n_samples,
                message=(
                    f"{key}: {observed_rate:.2%} "
                    f"{'meets' if satisfied else 'fails'} "
                    f"{threshold:.2%} threshold"
                ),
            )

        # Statistical validation with Clopper-Pearson
        from traigent.tvl.statistics import clopper_pearson_lower_bound

        lower_bound = clopper_pearson_lower_bound(
            successes=n_passed,
            trials=n_samples,
            confidence=constraint.threshold.confidence,
        )

        satisfied = lower_bound >= constraint.threshold.value

        return SafetyValidationResult(
            metric_name=key,
            satisfied=satisfied,
            observed_rate=observed_rate,
            lower_bound=lower_bound,
            threshold=constraint.threshold.value,
            confidence=constraint.threshold.confidence,
            sample_count=n_samples,
            message=(
                f"{key}: observed {observed_rate:.2%}, "
                f"{constraint.threshold.confidence:.0%} CI lower bound {lower_bound:.2%} "
                f"{'meets' if satisfied else 'fails'} "
                f"{constraint.threshold.value:.2%} threshold"
            ),
        )

    def validate_all(
        self, constraints: list[SafetyConstraint]
    ) -> list[SafetyValidationResult]:
        """Validate all safety constraints.

        Args:
            constraints: List of safety constraints to validate.

        Returns:
            List of validation results.
        """
        return [self.validate(c) for c in constraints]

    def all_satisfied(self, constraints: list[SafetyConstraint]) -> bool:
        """Check if all safety constraints are satisfied.

        Args:
            constraints: List of safety constraints to check.

        Returns:
            True if all constraints are satisfied.
        """
        return all(r.satisfied for r in self.validate_all(constraints))

    def reset(self) -> None:
        """Clear all recorded results."""
        with self._results_lock:
            self._sample_results.clear()


# =============================================================================
# Built-in RAGAS Safety Presets
# =============================================================================

faithfulness = RAGASMetric(
    name="faithfulness",
    ragas_metric_name="faithfulness",
    description=(
        "Measures factual consistency of generated answer with given context. "
        "Score in [0, 1] where 1 = fully faithful (no hallucinations)."
    ),
    requires_context=True,
)

answer_relevancy = RAGASMetric(
    name="answer_relevancy",
    ragas_metric_name="answer_relevancy",
    description=(
        "Measures how relevant the answer is to the question. "
        "Score in [0, 1] where 1 = fully relevant."
    ),
    requires_context=False,
)

context_precision = RAGASMetric(
    name="context_precision",
    ragas_metric_name="context_precision",
    description=(
        "Measures signal-to-noise ratio of retrieved context. "
        "Score in [0, 1] where 1 = all context is relevant."
    ),
    requires_context=True,
)

context_recall = RAGASMetric(
    name="context_recall",
    ragas_metric_name="context_recall",
    description=(
        "Measures how much of ground truth is captured by context. "
        "Score in [0, 1] where 1 = full recall."
    ),
    requires_context=True,
)

answer_similarity = RAGASMetric(
    name="answer_similarity",
    ragas_metric_name="answer_similarity",
    description=(
        "Semantic similarity between generated and ground truth answer. "
        "Score in [0, 1] where 1 = identical meaning."
    ),
    requires_context=False,
)


# =============================================================================
# Built-in Non-RAGAS Safety Presets (Factory Functions)
# =============================================================================


def hallucination_rate(metric_key: str = "hallucination_rate") -> MetricKeyMetric:
    """Factory for hallucination rate metric.

    Expects a pre-computed hallucination_rate in the metrics dict.
    Lower is better (0 = no hallucinations).

    Args:
        metric_key: Key to read from metrics dict.

    Returns:
        MetricKeyMetric configured for hallucination rate.

    Example:
        >>> hallucination_rate().below(0.1)  # Max 10% hallucination rate
    """
    return MetricKeyMetric(
        name="hallucination_rate",
        metric_key=metric_key,
        description="Rate of hallucinated content (lower is better, 0 = no hallucinations)",
        default=1.0,  # Fail-safe: assume worst case if not computed
        invert=False,
    )


def toxicity_score(metric_key: str = "toxicity") -> MetricKeyMetric:
    """Factory for toxicity score metric.

    Expects a pre-computed toxicity score in the metrics dict.
    Lower is better (0 = no toxicity).

    Args:
        metric_key: Key to read from metrics dict.

    Returns:
        MetricKeyMetric configured for toxicity scoring.

    Example:
        >>> toxicity_score().below(0.05)  # Max 5% toxicity
    """
    return MetricKeyMetric(
        name="toxicity_score",
        metric_key=metric_key,
        description="Toxicity score (lower is better, 0 = no toxic content)",
        default=1.0,
        invert=False,
    )


def bias_score(metric_key: str = "bias") -> MetricKeyMetric:
    """Factory for bias detection metric.

    Expects a pre-computed bias score in the metrics dict.
    Lower is better (0 = no bias detected).

    Args:
        metric_key: Key to read from metrics dict.

    Returns:
        MetricKeyMetric configured for bias detection.

    Example:
        >>> bias_score().below(0.1)  # Max 10% bias
    """
    return MetricKeyMetric(
        name="bias_score",
        metric_key=metric_key,
        description="Bias detection score (lower is better, 0 = no bias)",
        default=1.0,
        invert=False,
    )


def safety_score(metric_key: str = "safety_score") -> MetricKeyMetric:
    """Factory for general safety score metric.

    Expects a pre-computed safety score in the metrics dict.
    Higher is better (1 = fully safe).

    Args:
        metric_key: Key to read from metrics dict.

    Returns:
        MetricKeyMetric configured for general safety.

    Example:
        >>> safety_score().above(0.9)  # Require 90%+ safety
    """
    return MetricKeyMetric(
        name="safety_score",
        metric_key=metric_key,
        description="General safety score (higher is better, 1 = fully safe)",
        default=0.0,
        invert=False,
    )


def custom_safety(
    name: str,
    evaluator: Callable[[dict[str, Any], dict[str, Any]], float],
    description: str = "",
) -> CallableMetric:
    """Create a custom safety metric from a callable.

    Args:
        name: Metric name for reporting.
        evaluator: Callable(config, metrics) -> float.
            Should return value in [0, 1] range.
        description: Human-readable description.

    Returns:
        CallableMetric usable with .above(), .below(), etc.

    Thread Safety:
        The evaluator callable MUST be thread-safe.

    Example:
        >>> my_metric = custom_safety(
        ...     name="my_check",
        ...     evaluator=lambda cfg, m: m.get("my_score", 0),
        ...     description="My custom safety check",
        ... )
        >>> constraint = my_metric.above(0.8)
    """
    return CallableMetric(name, evaluator, description)


# =============================================================================
# Utility Functions
# =============================================================================


def get_available_safety_presets() -> dict[str, SafetyMetric]:
    """Get all available safety presets.

    Returns dict of preset name -> SafetyMetric instance.
    Only includes RAGAS presets if ragas is installed.

    Returns:
        Dictionary of available safety metric presets.
    """
    presets: dict[str, SafetyMetric] = {
        # Always available (read from metrics dict)
        "hallucination_rate": hallucination_rate(),
        "toxicity_score": toxicity_score(),
        "bias_score": bias_score(),
        "safety_score": safety_score(),
    }

    if _check_ragas_installed():
        presets.update(
            {
                "faithfulness": faithfulness,
                "answer_relevancy": answer_relevancy,
                "context_precision": context_precision,
                "context_recall": context_recall,
                "answer_similarity": answer_similarity,
            }
        )
    else:
        warnings.warn(
            "RAGAS metrics not available. Install with: pip install 'traigent[ragas]'",
            SafetyDependencyWarning,
            stacklevel=2,
        )

    return presets


# =============================================================================
# Type alias for constraint lists
# =============================================================================

SafetyConstraintType = SafetyConstraint | CompoundSafetyConstraint

# =============================================================================
# Public API
# =============================================================================

__all__ = [
    # Core classes
    "SafetyMetric",
    "SafetyConstraint",
    "SafetyThreshold",
    "CompoundSafetyConstraint",
    "SafetyValidator",
    "SafetyValidationResult",
    # Metric implementations
    "RAGASMetric",
    "MetricKeyMetric",
    "CallableMetric",
    # RAGAS presets (may raise if not installed when evaluated)
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
    "answer_similarity",
    # Non-RAGAS preset factories (always available)
    "hallucination_rate",
    "toxicity_score",
    "bias_score",
    "safety_score",
    "custom_safety",
    # Utilities
    "get_available_safety_presets",
    "RAGAS_ATTRIBUTION",
    # Exceptions
    "RAGASMetricNotAvailableError",
    "SafetyDependencyWarning",
    # Type aliases
    "SafetyConstraintType",
]
