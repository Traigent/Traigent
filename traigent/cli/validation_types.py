"""Data structures for Traigent optimization validation system."""

# Traceability: CONC-Layer-API CONC-Quality-Maintainability CONC-Quality-Usability FUNC-API-ENTRY REQ-API-001 SYNC-OptimizationFlow

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


@dataclass
class OptimizedFunction:
    """Information about a function decorated with @traigent.optimize."""

    name: str
    func: Any  # Should be an OptimizedFunction instance with .optimize() method
    decorator_config: dict[str, Any]  # Configuration from @traigent.optimize decorator
    default_params: dict[str, Any]  # Default parameters from function signature
    eval_dataset: str | None  # Evaluation dataset from decorator
    objectives: list[str]  # Objectives from decorator (e.g., ["accuracy", "cost"])

    @property
    def has_defaults(self) -> bool:
        """Check if function has any default parameters."""
        return bool(self.default_params)

    @property
    def baseline_config(self) -> dict[str, Any]:
        """Configuration the ``check`` baseline runs.

        The decorator's ``default_config`` wins; otherwise the function's
        signature defaults; otherwise an empty config.
        """
        declared = self.decorator_config.get("default_config")
        if isinstance(declared, dict) and declared:
            return dict(declared)
        return dict(self.default_params)

    @property
    def baseline_source(self) -> str:
        """Where ``baseline_config`` comes from: default_config|signature|none."""
        declared = self.decorator_config.get("default_config")
        if isinstance(declared, dict) and declared:
            return "default_config"
        if self.default_params:
            return "signature"
        return "none"

    @property
    def has_dataset(self) -> bool:
        """Check if function has an evaluation dataset configured."""
        return self.eval_dataset is not None

    def __str__(self) -> str:
        """String representation for logging and display."""
        return (
            f"OptimizedFunction(name='{self.name}', "
            f"defaults={len(self.default_params)}, "
            f"objectives={self.objectives}, "
            f"dataset={self.eval_dataset is not None})"
        )


@dataclass
class ValidationResult:
    """Result of optimization validation comparing optimized vs default parameters."""

    function_name: str
    baseline_metrics: dict[str, float]  # Metrics using default parameters
    optimized_metrics: dict[str, float]  # Metrics using optimized parameters
    is_superior: bool  # Whether optimization is superior to baseline
    improvement_details: dict[str, float]  # Per-metric improvement percentages
    blocking_issues: list[str]  # Issues that would block the validation

    # Additional metadata
    baseline_config: dict[str, Any] | None = None  # Default configuration used
    optimized_config: dict[str, Any] | None = None  # Best optimized configuration
    threshold_used: float = 0.1  # Threshold percentage used (0.1 = 10%)

    @property
    def has_improvement(self) -> bool:
        """Check if there's any improvement over baseline."""
        return any(improvement > 0 for improvement in self.improvement_details.values())

    @property
    def max_improvement(self) -> float:
        """Get maximum improvement percentage across all metrics.

        ``inf`` means at least one metric improved from a zero baseline, where
        a relative percentage is undefined.
        """
        if not self.improvement_details:
            return 0.0
        return max(self.improvement_details.values())

    @property
    def has_errors(self) -> bool:
        """True when validation could not complete (nothing was compared)."""
        return bool(self.blocking_issues)

    @property
    def should_block(self) -> bool:
        """Determine if this result should block a git push."""
        return not self.is_superior or bool(self.blocking_issues)

    def get_summary(self) -> str:
        """Get a human-readable summary of the validation result."""
        if self.should_block:
            status = "❌ BLOCKED"
            reason = "Optimization does not improve over defaults"
            if self.blocking_issues:
                reason = f"Issues: {', '.join(self.blocking_issues)}"
        else:
            status = "✅ PASSED"
            max_imp = self.max_improvement
            if math.isinf(max_imp):
                reason = "Optimization improves over a zero baseline"
            else:
                reason = f"Optimization improves by up to {max_imp:.1f}%"

        return f"{self.function_name}: {status} - {reason}"

    def _format_compared_metric(self, metric: str, value: float) -> str:
        improvement = self.improvement_details[metric]
        if math.isinf(improvement):
            # Baseline was exactly zero: a percentage is undefined, so show
            # the absolute change instead (#2803).
            baseline = self.baseline_metrics.get(metric, 0.0)
            direction = "↑" if improvement > 0 else "↓"
            label = "improved" if improvement > 0 else "regressed"
            return (
                f"  {metric}: {value:.3f} ({direction} {label} from zero, "
                f"{value - baseline:+.3f})"
            )
        if improvement > 0:
            direction = "↑"
        elif improvement < 0:
            direction = "↓"
        else:
            direction = "="
        return f"  {metric}: {value:.3f} ({direction} {abs(improvement):.1f}%)"

    def get_detailed_report(self) -> str:
        """Get a detailed report of the validation result."""
        lines = [
            f"Function: {self.function_name}",
            f"Status: {'✅ PASSED' if not self.should_block else '❌ BLOCKED'}",
            "",
            "Baseline Metrics (default parameters):",
        ]
        if self.baseline_config is not None:
            lines.insert(3, f"Baseline configuration: {self.baseline_config or '{}'}")

        for metric, value in self.baseline_metrics.items():
            lines.append(f"  {metric}: {value:.3f}")

        lines.extend(
            [
                "",
                "Optimized Metrics:",
            ]
        )

        for metric, value in self.optimized_metrics.items():
            if metric not in self.improvement_details:
                # Not compared (no baseline counterpart): show no percentage.
                lines.append(f"  {metric}: {value:.3f}")
                continue
            lines.append(self._format_compared_metric(metric, value))

        if self.blocking_issues:
            lines.extend(
                [
                    "",
                    "Blocking Issues:",
                ]
            )
            for issue in self.blocking_issues:
                lines.append(f"  • {issue}")

        return "\n".join(lines)
