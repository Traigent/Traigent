"""Result validators for optimizer test scenarios.

This module provides validation utilities for checking that optimization
results match expected outcomes defined in TestScenario specifications.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from .scenario import ExpectedOutcome, TestScenario

if TYPE_CHECKING:
    from traigent.api.types import OptimizationResult


@dataclass
class ValidationError:
    """A single validation error."""

    category: str
    message: str
    expected: str | None = None
    actual: str | None = None


@dataclass
class ValidationResult:
    """Result of validating an optimization result against expected outcomes."""

    passed: bool
    errors: list[ValidationError] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def add_error(
        self,
        category: str,
        message: str,
        expected: str | None = None,
        actual: str | None = None,
    ) -> None:
        """Add a validation error."""
        self.errors.append(
            ValidationError(
                category=category,
                message=message,
                expected=expected,
                actual=actual,
            )
        )
        self.passed = False

    def add_warning(self, message: str) -> None:
        """Add a validation warning."""
        self.warnings.append(message)

    def summary(self) -> str:
        """Get a summary of validation results."""
        if self.passed:
            return "PASSED"
        error_msgs = [f"  - [{e.category}] {e.message}" for e in self.errors]
        return "FAILED:\n" + "\n".join(error_msgs)


class ResultValidator:
    """Validates optimization results against expected outcomes.

    Usage:
        validator = ResultValidator(scenario, result)
        validation = validator.validate()
        assert validation.passed, validation.summary()
    """

    def __init__(
        self,
        scenario: TestScenario,
        result: OptimizationResult | Exception,
    ) -> None:
        """Initialize validator.

        Args:
            scenario: The test scenario specification
            result: The optimization result or exception
        """
        self.scenario = scenario
        self.result = result
        self.expected = scenario.expected

    def validate(self) -> ValidationResult:
        """Run all validations and return result.

        Returns:
            ValidationResult with pass/fail status and any errors
        """
        validation = ValidationResult(passed=True)

        # Handle exception case
        if isinstance(self.result, Exception):
            self._validate_exception(validation)
            return validation

        # If we expected failure but got success
        if self.expected.outcome == ExpectedOutcome.FAILURE:
            validation.add_error(
                category="outcome",
                message="Expected failure but optimization succeeded",
                expected="Exception",
                actual="OptimizationResult",
            )
            return validation

        # Validate successful result
        self._validate_trial_count(validation)
        self._validate_stop_reason(validation)
        self._validate_best_score(validation)
        self._validate_required_metrics(validation)
        self._validate_result_structure(validation)

        return validation

    def _validate_exception(self, validation: ValidationResult) -> None:
        """Validate exception matches expected failure."""
        if self.expected.outcome != ExpectedOutcome.FAILURE:
            validation.add_error(
                category="outcome",
                message=f"Got exception but expected {self.expected.outcome.value}",
                expected=self.expected.outcome.value,
                actual=type(self.result).__name__,
            )
            return

        # Check exception type
        if self.expected.error_type:
            if not isinstance(self.result, self.expected.error_type):
                validation.add_error(
                    category="error_type",
                    message="Wrong exception type",
                    expected=self.expected.error_type.__name__,
                    actual=type(self.result).__name__,
                )

        # Check error message
        if self.expected.error_message_contains:
            error_msg = str(self.result)
            if self.expected.error_message_contains not in error_msg:
                validation.add_error(
                    category="error_message",
                    message="Error message missing expected text",
                    expected=f"contains '{self.expected.error_message_contains}'",
                    actual=error_msg[:100],
                )

    def _validate_trial_count(self, validation: ValidationResult) -> None:
        """Validate number of trials."""
        # Import here to avoid circular imports
        trial_count = len(self.result.trials)  # type: ignore[union-attr]

        if trial_count < self.expected.min_trials:
            validation.add_error(
                category="trial_count",
                message="Too few trials",
                expected=f">= {self.expected.min_trials}",
                actual=str(trial_count),
            )

        if self.expected.max_trials and trial_count > self.expected.max_trials:
            validation.add_error(
                category="trial_count",
                message="Too many trials",
                expected=f"<= {self.expected.max_trials}",
                actual=str(trial_count),
            )

    def _validate_stop_reason(self, validation: ValidationResult) -> None:
        """Validate stop reason."""
        if self.expected.expected_stop_reason:
            actual_reason = self.result.stop_reason  # type: ignore[union-attr]
            if actual_reason != self.expected.expected_stop_reason:
                validation.add_error(
                    category="stop_reason",
                    message="Wrong stop reason",
                    expected=self.expected.expected_stop_reason,
                    actual=str(actual_reason),
                )

    def _validate_best_score(self, validation: ValidationResult) -> None:
        """Validate best score range."""
        if self.expected.best_score_range:
            min_score, max_score = self.expected.best_score_range
            best_score = self.result.best_score  # type: ignore[union-attr]

            if not (min_score <= best_score <= max_score):
                validation.add_error(
                    category="best_score",
                    message="Best score outside expected range",
                    expected=f"[{min_score}, {max_score}]",
                    actual=str(best_score),
                )

    def _validate_required_metrics(self, validation: ValidationResult) -> None:
        """Validate required metrics are present."""
        if not self.expected.required_metrics:
            return

        # Check metrics in successful trials
        successful_trials = self.result.successful_trials  # type: ignore[union-attr]

        if not successful_trials:
            validation.add_warning("No successful trials to check metrics")
            return

        for trial in successful_trials:
            for metric in self.expected.required_metrics:
                if metric not in trial.metrics:
                    validation.add_error(
                        category="required_metrics",
                        message=f"Missing metric '{metric}' in trial {trial.trial_id}",
                        expected=metric,
                        actual=str(list(trial.metrics.keys())),
                    )
                    break  # Only report once per trial

    def _validate_result_structure(self, validation: ValidationResult) -> None:
        """Validate overall result structure."""
        result = self.result  # type: ignore[union-attr]

        # Check required fields are present
        if result.optimization_id is None:
            validation.add_error(
                category="structure",
                message="optimization_id is None",
            )

        # Check best_config consistency
        successful_trials = result.successful_trials
        if successful_trials and result.best_config is None:
            validation.add_error(
                category="structure",
                message="best_config is None despite successful trials",
            )

        # Check trial consistency
        for trial in result.trials:
            if trial.status.value == "completed" and not trial.metrics:
                validation.add_error(
                    category="structure",
                    message=f"Completed trial {trial.trial_id} has no metrics",
                )


def validate_scenario_result(
    scenario: TestScenario,
    result: OptimizationResult | Exception,
) -> ValidationResult:
    """Convenience function to validate a scenario result.

    Args:
        scenario: The test scenario specification
        result: The optimization result or exception

    Returns:
        ValidationResult with pass/fail status and any errors
    """
    validator = ResultValidator(scenario, result)
    return validator.validate()
