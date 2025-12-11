"""Metrics computation for evaluation results."""

# Traceability: CONC-Layer-Core CONC-Quality-Reliability CONC-Quality-Performance FUNC-EVAL-METRICS REQ-EVAL-005 SYNC-OptimizationFlow

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from traigent.invokers.base import InvocationResult
from traigent.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class MetricsEvaluationResult:
    """Result of evaluating invocation results against expected outputs.

    This class contains computed metrics and analysis of how well
    the invoked function performed against expected results.

    Attributes:
        metrics: Dictionary of computed metrics
        total_invocations: Total number of invocations evaluated
        successful_invocations: Number of successful invocations
        duration: Time taken for evaluation (seconds)
        metadata: Additional evaluation metadata
    """

    metrics: dict[str, float] = field(default_factory=dict)
    total_invocations: int = 0
    successful_invocations: int = 0
    duration: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def success_rate(self) -> float:
        """Get success rate of invocations."""
        if self.total_invocations == 0:
            return 0.0
        return self.successful_invocations / self.total_invocations

    @property
    def error_rate(self) -> float:
        """Get error rate of invocations."""
        return 1.0 - self.success_rate


class MetricsComputer:
    """Computes evaluation metrics from invocation results.

    This class focuses solely on computing metrics from invocation
    results and expected outputs, without handling function invocation.
    """

    def __init__(self, metrics: list[str] | None = None) -> None:
        """Initialize metrics computer.

        Args:
            metrics: List of metric names to compute
        """
        self.metrics = metrics or ["accuracy", "success_rate"]
        logger.debug(f"MetricsComputer initialized with metrics: {self.metrics}")

    def compute_metrics(
        self, invocation_results: list[InvocationResult], expected_outputs: list[Any]
    ) -> MetricsEvaluationResult:
        """Compute metrics from invocation results and expected outputs.

        Args:
            invocation_results: Results from function invocations
            expected_outputs: Expected outputs for comparison

        Returns:
            EvaluationResult with computed metrics
        """
        start_time = time.time()

        if len(invocation_results) != len(expected_outputs):
            raise ValueError(
                f"Number of invocation results ({len(invocation_results)}) "
                f"must match number of expected outputs ({len(expected_outputs)})"
            )

        total_invocations = len(invocation_results)
        successful_invocations = sum(1 for r in invocation_results if r.is_successful)

        # Collect successful outputs and corresponding expected outputs
        successful_pairs = [
            (result.output, expected)
            for result, expected in zip(invocation_results, expected_outputs)
            if result.is_successful and expected is not None
        ]

        # Compute metrics
        computed_metrics = {}

        # Success rate
        if "success_rate" in self.metrics:
            computed_metrics["success_rate"] = (
                successful_invocations / total_invocations
                if total_invocations > 0
                else 0.0
            )

        # Error rate
        if "error_rate" in self.metrics:
            computed_metrics["error_rate"] = 1.0 - computed_metrics.get(
                "success_rate", 0.0
            )

        # Accuracy (exact match)
        if "accuracy" in self.metrics and successful_pairs:
            correct = sum(
                1 for output, expected in successful_pairs if output == expected
            )
            computed_metrics["accuracy"] = correct / len(successful_pairs)
        elif "accuracy" in self.metrics:
            computed_metrics["accuracy"] = 0.0

        # Average execution time
        if "avg_execution_time" in self.metrics:
            execution_times = [
                r.execution_time for r in invocation_results if r.execution_time > 0
            ]
            computed_metrics["avg_execution_time"] = (
                sum(execution_times) / len(execution_times) if execution_times else 0.0
            )

        # Average output length (for text outputs)
        if "avg_output_length" in self.metrics:
            valid_outputs = [
                r.output
                for r in invocation_results
                if r.is_successful and r.output is not None
            ]
            if valid_outputs:
                lengths = []
                for output in valid_outputs:
                    if isinstance(output, str):
                        lengths.append(len(output))
                    elif hasattr(output, "__len__"):
                        lengths.append(len(output))

                if lengths:
                    computed_metrics["avg_output_length"] = sum(lengths) / len(lengths)
                else:
                    computed_metrics["avg_output_length"] = 0.0
            else:
                computed_metrics["avg_output_length"] = 0.0

        # Custom metric computation can be added here

        end_time = time.time()
        duration = end_time - start_time

        # Create metadata
        metadata = {
            "metrics_requested": self.metrics,
            "evaluation_start_time": start_time,
            "evaluation_end_time": end_time,
            "successful_pairs": len(successful_pairs),
            "failed_invocations": total_invocations - successful_invocations,
        }

        logger.debug(
            f"Computed metrics for {total_invocations} invocations: "
            f"{successful_invocations} successful, metrics: {computed_metrics}"
        )

        return MetricsEvaluationResult(
            metrics=computed_metrics,
            total_invocations=total_invocations,
            successful_invocations=successful_invocations,
            duration=duration,
            metadata=metadata,
        )

    def add_custom_metric(
        self, name: str, compute_func: Callable[[dict[str, Any]], float]
    ) -> None:
        """Add custom metric computation function.

        Args:
            name: Metric name
            compute_func: Function that takes (outputs, expected_outputs) and returns float
        """
        if name not in self.metrics:
            self.metrics.append(name)

        # Store custom function for later use
        if not hasattr(self, "_custom_metrics"):
            self._custom_metrics: dict[str, Any] = {}
        self._custom_metrics[name] = compute_func

        logger.debug(f"Added custom metric: {name}")

    def compute_custom_metrics(
        self, invocation_results: list[InvocationResult], expected_outputs: list[Any]
    ) -> dict[str, float]:
        """Compute custom metrics if any are defined."""
        if not hasattr(self, "_custom_metrics"):
            return {}

        custom_metrics = {}

        # Extract successful outputs
        successful_outputs = [r.output for r in invocation_results if r.is_successful]

        # Only compute for successful invocations with expected outputs
        valid_expected = [
            expected
            for result, expected in zip(invocation_results, expected_outputs)
            if result.is_successful and expected is not None
        ]

        if len(successful_outputs) != len(valid_expected):
            # Align the arrays
            successful_outputs = [
                r.output
                for r, expected in zip(invocation_results, expected_outputs)
                if r.is_successful and expected is not None
            ]

        for metric_name, compute_func in self._custom_metrics.items():
            try:
                value = compute_func(successful_outputs, valid_expected)
                custom_metrics[metric_name] = float(value)
            except Exception as e:
                logger.warning(f"Custom metric {metric_name} computation failed: {e}")
                custom_metrics[metric_name] = 0.0

        return custom_metrics
