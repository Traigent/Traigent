"""Trace analyzer for validating captured traces.

Provides TraceAnalyzer class that validates traces against
global invariants and per-test expectations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from .capture import CapturedTrace
from .invariants import GLOBAL_INVARIANTS, check_invariant

if TYPE_CHECKING:
    from tests.optimizer_validation.specs.trace_expectations import TraceExpectations
    from traigent.api.types import OptimizationResult


@dataclass
class TraceValidationResult:
    """Result of trace validation."""

    passed: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def summary(self) -> str:
        """Generate summary of validation result."""
        if self.passed:
            return "PASSED"

        lines = ["FAILED:"]
        for error in self.errors:
            lines.append(f"  - {error}")

        if self.warnings:
            lines.append("WARNINGS:")
            for warning in self.warnings:
                lines.append(f"  - {warning}")

        return "\n".join(lines)


class TraceAnalyzer:
    """Analyzes captured traces against expectations.

    Checks both global invariants (apply to all tests) and
    per-test expectations (custom assertions per scenario).
    """

    def __init__(
        self,
        trace: CapturedTrace,
        result: OptimizationResult | None = None,
    ) -> None:
        """Initialize analyzer.

        Args:
            trace: Captured trace to analyze
            result: Optimization result for cross-validation
        """
        self.trace = trace
        self.result = result

    def validate(
        self,
        expectations: TraceExpectations | None = None,
        skip_invariants: list[str] | None = None,
    ) -> TraceValidationResult:
        """Run all validations.

        Args:
            expectations: Per-test trace expectations
            skip_invariants: List of invariant names to skip

        Returns:
            TraceValidationResult with errors and warnings
        """
        errors: list[str] = []
        warnings: list[str] = []

        # Check global invariants
        invariant_errors = self._check_global_invariants(skip=skip_invariants or [])
        errors.extend(invariant_errors)

        # Check per-test expectations if provided
        if expectations:
            expectation_errors = self._check_expectations(expectations)
            errors.extend(expectation_errors)

        # Check consistency with result if provided
        if self.result:
            consistency_errors = self._check_result_consistency()
            errors.extend(consistency_errors)

        return TraceValidationResult(
            passed=len(errors) == 0,
            errors=errors,
            warnings=warnings,
        )

    def _check_global_invariants(self, skip: list[str]) -> list[str]:
        """Check invariants that apply to all tests.

        Args:
            skip: Invariant names to skip

        Returns:
            List of error messages
        """
        errors = []

        for invariant in GLOBAL_INVARIANTS:
            if invariant in skip:
                continue

            error = check_invariant(invariant, self.trace, self.result)
            if error:
                errors.append(f"[invariant:{invariant}] {error}")

        return errors

    def _check_expectations(self, expectations: TraceExpectations) -> list[str]:
        """Check per-test expectations.

        Args:
            expectations: Expectations to check

        Returns:
            List of error messages
        """
        errors = []

        # Check span count expectations
        if expectations.min_trial_spans is not None:
            if self.trace.trial_count < expectations.min_trial_spans:
                errors.append(
                    f"[trial_count] Expected at least {expectations.min_trial_spans} "
                    f"trial spans, got {self.trace.trial_count}"
                )

        if expectations.max_trial_spans is not None:
            if self.trace.trial_count > expectations.max_trial_spans:
                errors.append(
                    f"[trial_count] Expected at most {expectations.max_trial_spans} "
                    f"trial spans, got {self.trace.trial_count}"
                )

        # Check required spans exist
        for span_name in expectations.required_spans:
            if not self.trace.get_span(span_name):
                errors.append(f"[required_span] Missing required span: {span_name}")

        # Check forbidden spans don't exist
        for span_name in expectations.forbidden_spans:
            if self.trace.get_span(span_name):
                errors.append(f"[forbidden_span] Found forbidden span: {span_name}")

        # Check root span attributes
        if expectations.root_attributes and self.trace.root_span:
            for key, expected_value in expectations.root_attributes.items():
                actual_value = self.trace.root_span.attributes.get(key)
                if actual_value != expected_value:
                    errors.append(
                        f"[root_attribute] Expected {key}={expected_value}, "
                        f"got {actual_value}"
                    )

        # Check expected sequence
        if expectations.expected_sequence:
            actual_sequence = [s.name for s in self.trace.spans]
            if not self._check_sequence(
                expectations.expected_sequence, actual_sequence
            ):
                errors.append(
                    f"[sequence] Expected sequence {expectations.expected_sequence} "
                    f"not found in {actual_sequence}"
                )

        # Run custom validators
        for validator in expectations.custom_validators:
            error = validator(self.trace)
            if error:
                errors.append(f"[custom] {error}")

        return errors

    def _check_result_consistency(self) -> list[str]:
        """Check trace is consistent with optimization result.

        Returns:
            List of error messages
        """
        errors = []

        if not self.result:
            return errors

        # Trial count should match
        result_trial_count = len(self.result.trials)
        if self.trace.trial_count != result_trial_count:
            errors.append(
                f"[consistency] Trial count mismatch: trace has {self.trace.trial_count}, "
                f"result has {result_trial_count}"
            )

        # Each result trial should have a corresponding span
        for trial in self.result.trials:
            trial_span = self.trace.get_trial_span(trial.trial_id)
            if not trial_span:
                errors.append(f"[consistency] Missing span for trial {trial.trial_id}")

        return errors

    def _check_sequence(
        self,
        expected: list[str],
        actual: list[str],
    ) -> bool:
        """Check if expected sequence appears in actual (in order).

        Args:
            expected: Expected sequence of span names
            actual: Actual sequence of span names

        Returns:
            True if expected sequence found in order
        """
        if not expected:
            return True

        expected_idx = 0
        for name in actual:
            if name == expected[expected_idx]:
                expected_idx += 1
                if expected_idx == len(expected):
                    return True

        return False

    def get_trial_metrics_summary(self) -> dict[str, Any]:
        """Get summary of metrics across all trial spans.

        Returns:
            Dict with metric statistics
        """
        metrics: dict[str, list[float]] = {}

        for trial in self.trace.trial_spans:
            for key, value in trial.attributes.items():
                if key.startswith("trial.metric."):
                    metric_name = key.replace("trial.metric.", "")
                    if metric_name not in metrics:
                        metrics[metric_name] = []
                    if isinstance(value, (int, float)):
                        metrics[metric_name].append(float(value))

        summary: dict[str, Any] = {}
        for name, values in metrics.items():
            if values:
                summary[name] = {
                    "count": len(values),
                    "min": min(values),
                    "max": max(values),
                    "mean": sum(values) / len(values),
                }

        return summary

    def get_timing_summary(self) -> dict[str, Any]:
        """Get timing summary of trace.

        Returns:
            Dict with timing statistics
        """
        trial_durations = [s.duration_ms for s in self.trace.trial_spans]

        summary = {
            "total_duration_ms": self.trace.duration_ms,
            "span_count": self.trace.span_count,
            "trial_count": self.trace.trial_count,
        }

        if trial_durations:
            summary["trial_timing"] = {
                "min_ms": min(trial_durations),
                "max_ms": max(trial_durations),
                "mean_ms": sum(trial_durations) / len(trial_durations),
                "total_ms": sum(trial_durations),
            }

        return summary
