"""Cost estimation and trial cost extraction for optimization runs."""

# Traceability: CONC-Layer-Core CONC-Quality-Reliability FUNC-ORCH-LIFECYCLE

from __future__ import annotations

from typing import Any

from traigent.api.types import TrialResult
from traigent.core.cost_enforcement import CostEnforcer
from traigent.evaluators.base import Dataset
from traigent.utils.logging import get_logger

logger = get_logger(__name__)

# Keys checked in config space for model names (order doesn't matter — all tried)
_MODEL_KEYS = ("model", "model_name", "llm_model", "llm")


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
        configuration_space: dict[str, Any] | None = None,
        invocations_per_example: int = 1,
    ) -> None:
        """Initialize cost estimator.

        Args:
            cost_enforcer: Cost enforcement instance for approval checks
            max_trials: Maximum number of trials
            max_total_examples: Global sample budget (None = unlimited)
            configuration_space: Optimizer config space for model-aware pricing
            invocations_per_example: LLM invocations per example (for multi-agent)
        """
        self._cost_enforcer = cost_enforcer
        self._max_trials = max_trials
        self._max_total_examples = max_total_examples
        self._configuration_space = configuration_space
        self._invocations_per_example = max(1, invocations_per_example)

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

        Uses model-aware pricing when a configuration space with model
        information is available. Falls back to the conservative hardcoded
        $0.01/example estimate when no model info is present.

        Note: This is an ESTIMATE. Actual costs may vary significantly.
        The EMA in CostEnforcer adapts after the first trial.

        Args:
            dataset: The evaluation dataset.

        Returns:
            Estimated cost in USD.
        """
        # Get dataset size
        dataset_size = len(dataset) if hasattr(dataset, "__len__") else 100

        # Get max trials (default to 10 if not set)
        max_trials = self._max_trials or 10

        # Determine total samples
        if self._max_total_examples is not None:
            total_samples = self._max_total_examples
            estimation_mode = "total_examples_budget"
        else:
            total_samples = max_trials * dataset_size
            estimation_mode = "per_trial_full_dataset"

        retry_factor = 1.2

        # Model-aware estimation when config space is available
        if self._configuration_space is not None:
            per_example = self._estimate_per_example_cost(dataset)
            estimated_total = total_samples * per_example * retry_factor

            logger.debug(
                "Cost estimate (%s, model-aware): %d samples "
                "× $%.6f/example × %.1f retry = $%.2f",
                estimation_mode,
                total_samples,
                per_example,
                retry_factor,
                estimated_total,
            )
            return estimated_total

        # Fallback: hardcoded $0.01/example (backward compatibility)
        base_cost_per_example = 0.01
        estimated_total = total_samples * base_cost_per_example * retry_factor

        logger.debug(
            "Cost estimate (%s, hardcoded): %d samples "
            "× $%.4f/example × %.1f retry = $%.2f",
            estimation_mode,
            total_samples,
            base_cost_per_example,
            retry_factor,
            estimated_total,
        )
        return estimated_total

    def _estimate_per_example_cost(self, dataset: Dataset) -> float:
        """Estimate per-example cost using model pricing and token estimation.

        Args:
            dataset: The evaluation dataset for token estimation.

        Returns:
            Estimated cost per example in USD.
        """
        from traigent.utils.cost_calculator import get_model_token_pricing

        # Extract model names and pick the most expensive
        models = self._extract_models_from_config_space()
        if models:
            best_input, best_output, best_method = 0.0, 0.0, ""
            for model in models:
                inp, out, method = get_model_token_pricing(model)
                if (inp + out) > (best_input + best_output):
                    best_input, best_output, best_method = inp, out, method
            logger.debug(
                "Selected most expensive model pricing: method=%s, "
                "input=%.2e, output=%.2e (from %d models)",
                best_method,
                best_input,
                best_output,
                len(models),
            )
        else:
            # No model key found — use mid-tier default
            from traigent.utils.cost_calculator import _TIER_MID

            best_input = _TIER_MID["input"]
            best_output = _TIER_MID["output"]
            logger.debug("No model key in config space, using mid-tier default pricing")

        # Estimate tokens per example
        input_tokens, output_tokens = self._estimate_tokens_per_example(dataset)

        # Per-example cost with invocation multiplier
        per_example = (
            input_tokens * best_input + output_tokens * best_output
        ) * self._invocations_per_example

        return per_example

    def _extract_models_from_config_space(self) -> list[str]:
        """Extract model names from config space, handling all value forms.

        Handles:
        - list: ["gpt-4o", "gpt-3.5-turbo"]
        - str: "gpt-4o"
        - tuple: ("gpt-4o", "gpt-3.5-turbo")
        - dict with "choices": {"choices": ["a", "b"]}
        - dict with "values": {"values": ["a", "b"]}
        - dict with "value": {"value": "gpt-4o"}

        Returns:
            List of model name strings (may be empty).
        """
        if not self._configuration_space:
            return []

        for key in _MODEL_KEYS:
            if key not in self._configuration_space:
                continue

            raw = self._configuration_space[key]
            candidates = self._extract_candidates(raw)
            # Filter to strings only (skip numeric, bool, etc.)
            models = [c for c in candidates if isinstance(c, str)]
            if models:
                return models

        return []

    @staticmethod
    def _extract_candidates(raw: Any) -> list[Any]:
        """Extract candidate values from a config-space entry."""
        if isinstance(raw, list):
            return raw
        if isinstance(raw, tuple):
            return list(raw)
        if isinstance(raw, str):
            return [raw]
        if isinstance(raw, dict):
            for dict_key in ("choices", "values"):
                val = raw.get(dict_key)
                if isinstance(val, (list, tuple)):
                    return list(val)
            val = raw.get("value")
            if val is not None:
                return [val]
        return []

    @staticmethod
    def _estimate_tokens_per_example(dataset: Dataset) -> tuple[int, int]:
        """Estimate input and output tokens per example.

        SAFE: Only samples indexable datasets (with both ``__getitem__`` and
        ``__len__``). Never consumes iterators or generators.

        Returns:
            ``(input_tokens, output_tokens)`` — always >= (500, 250).
        """
        default_input, default_output = 2000, 1000

        if not (hasattr(dataset, "__getitem__") and hasattr(dataset, "__len__")):
            return default_input, default_output

        try:
            n = len(dataset)
        except TypeError:
            return default_input, default_output

        if n == 0:
            return default_input, default_output

        # Sample up to 5 indices spread across the dataset
        indices = sorted({0, n // 4, n // 2, 3 * n // 4, n - 1})
        max_chars = 0

        for idx in indices:
            try:
                example = dataset[idx]
                text = _extract_text(example)
                if len(text) > max_chars:
                    max_chars = len(text)
            except Exception:
                continue

        if max_chars == 0:
            return default_input, default_output

        # ~4 chars per token, conservative (MAX of samples)
        input_tokens = max(500, max_chars // 4)
        output_tokens = max(250, input_tokens // 2)
        return input_tokens, output_tokens

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


def _extract_text(example: Any) -> str:
    """Extract text from a dataset example for token estimation.

    Fallback chain:
    1. EvaluationExample: input_data + expected_output
    2. str: use directly
    3. dict or any other: str(obj)
    """
    # Check for EvaluationExample (has input_data attribute)
    input_data = getattr(example, "input_data", None)
    if input_data is not None:
        parts = [str(input_data)]
        expected = getattr(example, "expected_output", None)
        if expected is not None:
            parts.append(str(expected))
        return " ".join(parts)

    if isinstance(example, str):
        return example

    # dict or any other object
    return str(example)
