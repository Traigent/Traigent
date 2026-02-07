"""Cost estimation and trial cost extraction for optimization runs."""

# Traceability: CONC-Layer-Core CONC-Quality-Reliability FUNC-ORCH-LIFECYCLE

from __future__ import annotations

from traigent.api.types import TrialResult
from traigent.core.cost_enforcement import CostEnforcer
from traigent.evaluators.base import Dataset
from traigent.utils.logging import get_logger

logger = get_logger(__name__)


class CostEstimator:
    """Estimates optimization costs and extracts trial cost data.

    Provides pre-optimization cost approval checks and per-trial cost
    extraction from metrics/metadata for cost enforcement tracking.
    """

    def __init__(
        self,
        cost_enforcer: CostEnforcer,
        max_trials: int | None,
        max_total_examples: int | None,
    ) -> None:
        """Initialize cost estimator.

        Args:
            cost_enforcer: Cost enforcement instance for approval checks
            max_trials: Maximum number of trials
            max_total_examples: Global sample budget (None = unlimited)
        """
        self._cost_enforcer = cost_enforcer
        self._max_trials = max_trials
        self._max_total_examples = max_total_examples

    def check_cost_approval(self, dataset: Dataset) -> None:
        """Check cost approval before optimization.

        Estimates the total optimization cost and checks against the cost
        enforcer's configured limit. Raises OptimizationAborted if declined.

        Args:
            dataset: The evaluation dataset for cost estimation

        Raises:
            OptimizationAborted: If cost approval is declined
        """
        if self._cost_enforcer.is_mock_mode:
            return
        estimated_cost = self.estimate_optimization_cost(dataset)
        if not self._cost_enforcer.check_and_approve(estimated_cost):
            from traigent.core.cost_enforcement import OptimizationAborted

            raise OptimizationAborted(
                f"Cost approval declined. Estimated cost: ${estimated_cost:.2f}, "
                f"limit: ${self._cost_enforcer.config.limit:.2f}. "
                f"Set TRAIGENT_COST_APPROVED=true or increase TRAIGENT_RUN_COST_LIMIT."
            )

    def estimate_optimization_cost(self, dataset: Dataset) -> float:
        """Estimate total optimization cost for pre-approval check.

        Calculation:
        - Estimates total samples based on configuration and dataset size
        - Uses max_total_examples if configured (shared budget across trials)
        - Otherwise estimates samples_per_trial x max_trials
        - Includes retry factor (1.2x) for potential failures
        - Uses conservative estimates for unknown models

        Note: This is an ESTIMATE. Actual costs may vary significantly.

        Args:
            dataset: The evaluation dataset.

        Returns:
            Estimated cost in USD.
        """
        # Base cost per example (conservative estimate for GPT-4 class models)
        # Assumes ~2000 tokens input, ~500 tokens output per example
        base_cost_per_example = 0.01  # $0.01 per example (conservative)

        # Get dataset size
        dataset_size = len(dataset) if hasattr(dataset, "__len__") else 100

        # Get max trials (default to 10 if not set)
        max_trials = self._max_trials or 10

        # Determine total samples based on configuration
        if self._max_total_examples is not None:
            # Global sample budget is set - use it directly for cost estimation
            # Don't clip to dataset_size because:
            # 1. With multiple trials, samples can be re-evaluated with different configs
            # 2. The budget represents total API calls, not unique samples
            # 3. Clipping would underestimate cost when budget > dataset_size
            total_samples = self._max_total_examples
            estimation_mode = "total_examples_budget"
        else:
            # No global budget - each trial may evaluate the full dataset
            # This is a worst-case conservative estimate
            samples_per_trial = dataset_size
            total_samples = max_trials * samples_per_trial
            estimation_mode = "per_trial_full_dataset"

        # Total cost with retry factor for failures and potential re-evaluations
        retry_factor = 1.2
        estimated_total = total_samples * base_cost_per_example * retry_factor

        logger.debug(
            f"Cost estimate ({estimation_mode}): {total_samples} total samples "
            f"× ${base_cost_per_example}/sample × {retry_factor} retry = ${estimated_total:.2f}"
        )

        return estimated_total

    @staticmethod
    def extract_trial_cost(trial_result: TrialResult) -> float | None:
        """Extract cost from trial result for cost enforcement tracking.

        Attempts to find cost from multiple sources:
        1. trial_result.metrics["total_cost"] or ["cost"]
        2. trial_result.metadata["total_example_cost"]
        3. Returns None if cost cannot be determined (triggers fallback mode)

        Args:
            trial_result: The completed trial result.

        Returns:
            Cost in USD, or None if cost cannot be determined.
        """
        # Try metrics first
        metrics = trial_result.metrics or {}
        for key in ("total_cost", "cost"):
            if key in metrics:
                try:
                    return float(metrics[key])
                except (TypeError, ValueError):
                    pass

        # Try metadata
        metadata = trial_result.metadata or {}
        if "total_example_cost" in metadata:
            try:
                return float(metadata["total_example_cost"])
            except (TypeError, ValueError):
                pass

        # Cost cannot be determined
        return None
