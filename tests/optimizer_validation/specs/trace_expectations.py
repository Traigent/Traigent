"""Per-test trace expectations.

Defines TraceExpectations dataclass for specifying
what a test's trace should look like.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from tests.optimizer_validation.tracing.capture import CapturedTrace


@dataclass
class TraceExpectations:
    """Per-test trace expectations.

    Specifies what a test's trace should contain:
    - Span count bounds
    - Required/forbidden spans
    - Attribute checks
    - Custom validators

    Example:
        expectations = TraceExpectations(
            min_trial_spans=2,
            max_trial_spans=5,
            required_spans=["optimization_session", "trial_execution"],
            root_attributes={"injection_mode": "context"},
        )
    """

    # Span count expectations
    min_trial_spans: int | None = None
    max_trial_spans: int | None = None

    # Required span names (must exist)
    required_spans: list[str] = field(default_factory=list)

    # Forbidden span names (must not exist)
    forbidden_spans: list[str] = field(default_factory=list)

    # Root span attribute checks
    root_attributes: dict[str, Any] = field(default_factory=dict)

    # Expected span sequence (order matters, partial match)
    expected_sequence: list[str] | None = None

    # Custom validators: callable(CapturedTrace) -> str | None
    # Returns error message if validation fails, None if passed
    custom_validators: list[Callable[[CapturedTrace], str | None]] = field(
        default_factory=list
    )

    # Skip specific global invariants for this test
    skip_invariants: list[str] = field(default_factory=list)

    def with_trial_bounds(
        self,
        min_trials: int | None = None,
        max_trials: int | None = None,
    ) -> TraceExpectations:
        """Create copy with trial bounds set.

        Args:
            min_trials: Minimum expected trial spans
            max_trials: Maximum expected trial spans

        Returns:
            New TraceExpectations with bounds set
        """
        return TraceExpectations(
            min_trial_spans=min_trials or self.min_trial_spans,
            max_trial_spans=max_trials or self.max_trial_spans,
            required_spans=self.required_spans.copy(),
            forbidden_spans=self.forbidden_spans.copy(),
            root_attributes=self.root_attributes.copy(),
            expected_sequence=self.expected_sequence,
            custom_validators=self.custom_validators.copy(),
            skip_invariants=self.skip_invariants.copy(),
        )

    def with_required_spans(self, *spans: str) -> TraceExpectations:
        """Create copy with additional required spans.

        Args:
            *spans: Span names that must exist

        Returns:
            New TraceExpectations with spans added
        """
        return TraceExpectations(
            min_trial_spans=self.min_trial_spans,
            max_trial_spans=self.max_trial_spans,
            required_spans=self.required_spans + list(spans),
            forbidden_spans=self.forbidden_spans.copy(),
            root_attributes=self.root_attributes.copy(),
            expected_sequence=self.expected_sequence,
            custom_validators=self.custom_validators.copy(),
            skip_invariants=self.skip_invariants.copy(),
        )

    def with_custom_validator(
        self,
        validator: Callable[[CapturedTrace], str | None],
    ) -> TraceExpectations:
        """Create copy with additional custom validator.

        Args:
            validator: Validation function

        Returns:
            New TraceExpectations with validator added
        """
        return TraceExpectations(
            min_trial_spans=self.min_trial_spans,
            max_trial_spans=self.max_trial_spans,
            required_spans=self.required_spans.copy(),
            forbidden_spans=self.forbidden_spans.copy(),
            root_attributes=self.root_attributes.copy(),
            expected_sequence=self.expected_sequence,
            custom_validators=self.custom_validators + [validator],
            skip_invariants=self.skip_invariants.copy(),
        )


# Common expectation presets
def basic_expectations(min_trials: int = 1) -> TraceExpectations:
    """Create basic expectations for standard optimization tests.

    Args:
        min_trials: Minimum expected trial count

    Returns:
        TraceExpectations with basic checks
    """
    return TraceExpectations(
        min_trial_spans=min_trials,
        required_spans=["optimization_session", "trial_execution"],
    )


def multi_objective_expectations(
    objective_count: int,
    min_trials: int = 1,
) -> TraceExpectations:
    """Create expectations for multi-objective optimization tests.

    Args:
        objective_count: Number of objectives
        min_trials: Minimum expected trial count

    Returns:
        TraceExpectations with multi-objective checks
    """

    def check_objectives(trace: CapturedTrace) -> str | None:
        """Check that root span has expected objective count."""
        if not trace.root_span:
            return None  # Will be caught by invariant

        objectives_attr = trace.root_span.attributes.get("objectives")
        if objectives_attr:
            # Parse objectives from string or count directly
            if isinstance(objectives_attr, str):
                # Count comma-separated objectives
                actual_count = len(objectives_attr.split(","))
            elif isinstance(objectives_attr, (list, tuple)):
                actual_count = len(objectives_attr)
            else:
                return f"Unexpected objectives attribute type: {type(objectives_attr)}"

            if actual_count != objective_count:
                return f"Expected {objective_count} objectives, got {actual_count}"

        return None

    return TraceExpectations(
        min_trial_spans=min_trials,
        required_spans=["optimization_session", "trial_execution"],
        custom_validators=[check_objectives],
    )


def failure_expectations(
    allow_zero_trials: bool = False,
) -> TraceExpectations:
    """Create expectations for failure scenario tests.

    Args:
        allow_zero_trials: Whether zero trials is acceptable

    Returns:
        TraceExpectations for failure scenarios
    """
    return TraceExpectations(
        min_trial_spans=0 if allow_zero_trials else 1,
        required_spans=["optimization_session"],
        # Trials may fail but should still be traced
        skip_invariants=["completed_trials_have_metrics"],
    )
