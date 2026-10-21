"""Cost estimation and trial cost extraction for optimization runs."""

# Traceability: CONC-Layer-Core CONC-Quality-Reliability FUNC-ORCH-LIFECYCLE

from __future__ import annotations

import json
from collections.abc import Sequence

from traigent.api.types import TrialResult
from traigent.core.cost_enforcement import CostEnforcer
from traigent.evaluators.base import Dataset
from traigent.utils.cost_calculator import UnknownModelError, get_model_token_pricing
from traigent.utils.env_config import cost_preflight_mock_exempt
from traigent.utils.exceptions import CostLimitExceeded
from traigent.utils.logging import get_logger
from traigent.utils.trial_costs import is_cost_unmeasured, known_spend_lower_bound

logger = get_logger(__name__)


_ESTIMATED_INPUT_TOKENS_PER_EXAMPLE = 2000
_ESTIMATED_OUTPUT_TOKENS_PER_EXAMPLE = 500
_CONSERVATIVE_INPUT_COST_PER_TOKEN = 15.0e-6
_CONSERVATIVE_OUTPUT_COST_PER_TOKEN = 75.0e-6
# Input estimate is max(sampled estimate, floor); the floor is the long-standing
# default so sampling can only raise the estimate, never lower it (#2498).
_INPUT_TOKENS_FLOOR = _ESTIMATED_INPUT_TOKENS_PER_EXAMPLE
_DATASET_SAMPLE_SIZE = 20
_BUFFER_FACTOR = 1.2


def _estimate_text_tokens(text: str) -> int:
    """Rough token estimate: ~4 ASCII chars/token (as used elsewhere in the
    SDK), but every non-ASCII char counts as one token so CJK and other
    non-English text is not under-counted."""
    non_ascii = sum(1 for ch in text if ord(ch) > 127)
    return (len(text) - non_ascii + 3) // 4 + non_ascii


def _example_text(example: object) -> str:
    """Serialize an example's inputs for length measurement only."""
    data = getattr(example, "input_data", None)
    if data is None:
        return ""
    try:
        return json.dumps(data, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return str(data)


def sample_input_tokens_per_example(
    dataset: object, sample_size: int = _DATASET_SAMPLE_SIZE
) -> tuple[int, int] | None:
    """Estimate mean input tokens/example from the first ``sample_size`` examples.

    Deterministic and bounded: never reads beyond the sample. Returns
    ``(estimate, sample_count)`` or None when no example inputs are readable.
    """
    examples = getattr(dataset, "examples", None)
    if not isinstance(examples, Sequence) or not examples:
        return None
    sample = examples[:sample_size]
    counts = [_estimate_text_tokens(_example_text(ex)) for ex in sample]
    if not any(counts):
        return None
    return -(-sum(counts) // len(counts)), len(counts)


class CostEstimator:
    """Estimates optimization costs and extracts trial cost data.

    Provides pre-optimization cost approval checks and per-trial cost
    extraction from metrics/metadata for cost enforcement tracking.
    """

    # Class-level fallback for instances constructed via ``__new__`` that
    # skip ``__init__`` (e.g. test doubles that only stub ``_cost_enforcer``
    # and patch ``estimate_optimization_cost``). ``check_cost_approval``
    # reads this attribute directly when building its declined-approval
    # message, outside of the patched ``estimate_optimization_cost`` path, so
    # it must resolve to a sane default even when ``__init__`` never ran
    # (issue #1750 follow-up).
    _estimated_calls_per_example: int = 1
    _configured_max_output_tokens: int | None = None
    # (input, output, input_source, sample_count, example_count) from the last
    # estimate_optimization_cost call; None until then.
    _resolved_assumptions: tuple[int, int, str, int, int] | None = None

    def __init__(
        self,
        cost_enforcer: CostEnforcer,
        max_trials: int | None,
        max_total_examples: int | None,
        model_name: str | None = None,
        candidate_models: Sequence[str] | None = None,
        estimated_input_tokens_per_example: int | None = None,
        estimated_output_tokens_per_example: int | None = None,
        estimated_calls_per_example: int | None = None,
        configured_max_output_tokens: int | None = None,
    ) -> None:
        """Initialize cost estimator.

        Args:
            cost_enforcer: Cost enforcement instance for approval checks
            max_trials: Maximum number of trials
            max_total_examples: Global sample budget (None = unlimited)
            model_name: Primary model name for model-aware pricing estimation
            candidate_models: Candidate model names from config space for
                worst-case estimation when no fixed model is set for the run
            estimated_input_tokens_per_example: Optional service-provided
                per-example input token estimate
            estimated_output_tokens_per_example: Optional service-provided
                per-example output token estimate
            estimated_calls_per_example: Optional declared number of LLM calls
                the optimized function makes per evaluated example
                (self-consistency voting, repair passes, model cascades).
                Multiplies the per-example base cost. Defaults to 1 when
                unset or invalid (issue #1750).
        """
        self._configured_max_output_tokens = (
            configured_max_output_tokens
            if isinstance(configured_max_output_tokens, int)
            and not isinstance(configured_max_output_tokens, bool)
            and configured_max_output_tokens > 0
            else None
        )
        self._cost_enforcer = cost_enforcer
        self._max_trials = max_trials
        self._max_total_examples = max_total_examples
        self._model_name = model_name
        self._candidate_models = tuple(
            dict.fromkeys(
                model.strip()
                for model in candidate_models or ()
                if isinstance(model, str) and model.strip()
            )
        )
        self._estimated_input_tokens_per_example = (
            estimated_input_tokens_per_example
            if isinstance(estimated_input_tokens_per_example, int)
            and not isinstance(estimated_input_tokens_per_example, bool)
            and estimated_input_tokens_per_example > 0
            else None
        )
        self._estimated_output_tokens_per_example = (
            estimated_output_tokens_per_example
            if isinstance(estimated_output_tokens_per_example, int)
            and not isinstance(estimated_output_tokens_per_example, bool)
            and estimated_output_tokens_per_example > 0
            else None
        )
        self._estimated_calls_per_example = (
            estimated_calls_per_example
            if isinstance(estimated_calls_per_example, int)
            and not isinstance(estimated_calls_per_example, bool)
            and estimated_calls_per_example > 0
            else 1
        )

    def _get_estimated_tokens_per_example(self) -> tuple[int, int]:
        """Resolve per-example token estimate, preferring service metadata."""
        if self._resolved_assumptions is not None:
            return self._resolved_assumptions[0], self._resolved_assumptions[1]
        return self._resolve_tokens(None)[:2]  # type: ignore[return-value]

    def _resolve_tokens(self, dataset: object | None) -> tuple[int, int, str, int]:
        """Return (input, output, input_source, sample_count).

        Precedence: service metadata override > dataset sample (never below
        the floor) > default. Output: override > configured max_tokens > default.
        """
        sample_count = 0
        if self._estimated_input_tokens_per_example is not None:
            input_tokens = self._estimated_input_tokens_per_example
            source = "backend_override"
        else:
            sampled = (
                sample_input_tokens_per_example(dataset)
                if dataset is not None
                else None
            )
            if sampled is not None:
                sample_count = sampled[1]
                input_tokens = max(sampled[0], _INPUT_TOKENS_FLOOR)
                source = (
                    "dataset_sample"
                    if sampled[0] > _INPUT_TOKENS_FLOOR
                    else "floor_default"
                )
            else:
                input_tokens = _ESTIMATED_INPUT_TOKENS_PER_EXAMPLE
                source = "floor_default"
        if self._estimated_output_tokens_per_example is not None:
            output_tokens = self._estimated_output_tokens_per_example
        elif self._configured_max_output_tokens is not None:
            output_tokens = self._configured_max_output_tokens
        else:
            output_tokens = _ESTIMATED_OUTPUT_TOKENS_PER_EXAMPLE
        return input_tokens, output_tokens, source, sample_count

    def _assumptions_summary(self) -> str:
        """Value-free description of the estimate's assumptions."""
        models = (
            [self._model_name]
            if self._model_name
            else list(self._candidate_models) or ["unknown (conservative pricing)"]
        )
        resolved = self._resolved_assumptions
        if resolved is None:
            input_tokens, output_tokens = self._get_estimated_tokens_per_example()
            source, sampled, examples = "floor_default", 0, None
        else:
            input_tokens, output_tokens, source, sampled, examples = resolved
        examples_text = "unknown" if examples is None else str(examples)
        return (
            f"Assumptions: model(s)={', '.join(models)}; "
            f"examples in dataset={examples_text}; "
            f"input tokens/example={input_tokens} (estimate, source={source}, "
            f"{sampled} example(s) sampled, floor={_INPUT_TOKENS_FLOOR}); "
            f"output tokens/example={output_tokens} (estimate); "
            f"safety buffer={_BUFFER_FACTOR}x. "
            "Real billing may differ; the runtime cost limit still applies."
        )

    def _base_cost_for_rates(self, input_rate: float, output_rate: float) -> float:
        """Compute per-example base cost from token rates."""
        input_tokens, output_tokens = self._get_estimated_tokens_per_example()
        return float(input_tokens * input_rate + output_tokens * output_rate)

    def _estimate_from_resolved_model(
        self, model_name: str, source: str
    ) -> tuple[float, str]:
        """Compute a base estimate once model pricing is known."""
        input_rate, output_rate, pricing_source = get_model_token_pricing(model_name)
        base_cost = self._base_cost_for_rates(input_rate, output_rate)
        if source == "fixed":
            return base_cost, pricing_source
        return base_cost, f"{pricing_source}:config_space_max({model_name})"

    def _estimate_from_fixed_model(self) -> tuple[float, str] | None:
        """Estimate using the explicit run model when available."""
        if not self._model_name:
            return None
        try:
            return self._estimate_from_resolved_model(self._model_name, "fixed")
        except UnknownModelError:
            logger.warning(
                "Unknown model %r for pre-approval estimate; using conservative pricing.",
                self._model_name,
            )
        except Exception:
            logger.debug(
                "Model-aware estimate failed for %r; using conservative pricing.",
                self._model_name,
                exc_info=True,
            )
        return None

    def _collect_candidate_model_estimates(
        self,
    ) -> tuple[list[tuple[float, str]], list[str]]:
        """Resolve pricing for candidate models from config space."""
        candidate_costs: list[tuple[float, str]] = []
        unresolved_candidates: list[str] = []

        for candidate in self._candidate_models:
            try:
                base_cost, pricing_source = self._estimate_from_resolved_model(
                    candidate,
                    "candidate",
                )
                candidate_costs.append((base_cost, pricing_source))
            except UnknownModelError:
                unresolved_candidates.append(candidate)
            except Exception:
                logger.debug(
                    "Model-aware estimate failed for candidate %r; using conservative pricing.",
                    candidate,
                    exc_info=True,
                )
                unresolved_candidates.append(candidate)

        return candidate_costs, unresolved_candidates

    def _estimate_from_candidate_models(self) -> tuple[float, str] | None:
        """Estimate using the worst-case resolvable candidate model."""
        if not self._candidate_models:
            return None

        candidate_costs, unresolved_candidates = (
            self._collect_candidate_model_estimates()
        )
        if unresolved_candidates:
            logger.warning(
                "Unable to resolve pricing for candidate model(s) %s; using conservative pricing.",
                sorted(unresolved_candidates),
            )
            return None
        if not candidate_costs:
            return None
        return max(candidate_costs, key=lambda item: item[0])

    def _estimate_with_conservative_fallback(self) -> tuple[float, str]:
        """Estimate using conservative token pricing defaults."""
        input_tokens, output_tokens = self._get_estimated_tokens_per_example()
        conservative_cost = (
            input_tokens * _CONSERVATIVE_INPUT_COST_PER_TOKEN
            + output_tokens * _CONSERVATIVE_OUTPUT_COST_PER_TOKEN
        )
        return float(conservative_cost), "conservative_fallback"

    def _estimate_base_cost_per_example(self) -> tuple[float, str]:
        """Estimate per-example cost from model pricing with conservative fallback."""
        return (
            self._estimate_from_fixed_model()
            or self._estimate_from_candidate_models()
            or self._estimate_with_conservative_fallback()
        )

    def check_cost_approval(self, dataset: Dataset) -> None:
        """Check cost approval before optimization.

        Estimates the total optimization cost and checks against the cost
        enforcer's configured limit. If the pre-run approval is declined,
        raises ``CostLimitExceeded`` (an ``OptimizationError`` subclass) with
        the limit, spent amount, and estimate. Mid-run cost limits remain a
        graceful stop: inspect ``OptimizationResult.stop_reason == "cost_limit"``.

        Args:
            dataset: The evaluation dataset for cost estimation

        Raises:
            CostLimitExceeded: If pre-run cost approval is declined.
        """
        if cost_preflight_mock_exempt():
            logger.info(
                "Skipping optimized-function pricing preflight (explicit mock "
                "exemption, test-only); "
                "runtime CostEnforcer permits and accounting remain active."
            )
            return

        estimated_cost = self.estimate_optimization_cost(dataset)
        if not self._cost_enforcer.check_and_approve(
            estimated_cost,
            estimated_calls_per_example=self._estimated_calls_per_example,
        ):
            limit = self._cost_enforcer.config.limit
            raise CostLimitExceeded(
                accumulated=0.0,
                limit=limit,
                estimated=estimated_cost,
                message=(
                    "Cost approval declined. Rough conservative upper-bound "
                    f"estimate: ${estimated_cost:.2f}, limit: "
                    f"${limit:.2f}. "
                    f"{self._assumptions_summary()} "
                    "Pre-run estimates use conservative "
                    "fallback pricing when model pricing is unavailable, and assume "
                    f"{self._estimated_calls_per_example} LLM call(s) per example "
                    "(set estimated_calls_per_example on @traigent.optimize for "
                    "multi-call agents such as self-consistency voting, repair "
                    "passes, or model cascades). To proceed, "
                    "raise TRAIGENT_RUN_COST_LIMIT, approve after review with "
                    "TRAIGENT_COST_APPROVED=true or cost_approved=True, or calibrate "
                    "private/unpriced model rates with "
                    "TRAIGENT_CUSTOM_MODEL_PRICING_FILE or "
                    "TRAIGENT_CUSTOM_MODEL_PRICING_JSON."
                ),
            )

    def estimate_optimization_cost(self, dataset: Dataset) -> float:
        """Estimate total optimization cost for pre-approval check.

        Calculation:
        - Estimates total samples based on configuration and dataset size
        - Uses max_total_examples if configured (shared budget across trials)
        - Otherwise estimates samples_per_trial x max_trials
        - Scales the per-example cost by estimated_calls_per_example (default
          1 call per example; issue #1750)
        - Applies a flat 1.2x conservative buffer to the estimate
        - Uses conservative estimates for unknown models

        Note: This is an ESTIMATE. Actual costs may vary significantly.

        Args:
            dataset: The evaluation dataset.

        Returns:
            Estimated cost in USD.
        """
        dataset_size = len(dataset) if hasattr(dataset, "__len__") else 100
        input_tokens, output_tokens, source, sample_count = self._resolve_tokens(
            dataset
        )
        self._resolved_assumptions = (
            input_tokens,
            output_tokens,
            source,
            sample_count,
            dataset_size,
        )
        base_cost_per_example, pricing_source = self._estimate_base_cost_per_example()

        # Scale by the declared calls-per-example multiplier (issue #1750).
        # Without this, a multi-call agent (self-consistency voting, repair
        # passes, model cascades) is estimated as if it made exactly one LLM
        # call per example, structurally under-estimating cost at approval
        # time by roughly the call count.
        if self._estimated_calls_per_example > 1:
            base_cost_per_example *= self._estimated_calls_per_example
            pricing_source = (
                f"{pricing_source}:calls_per_example="
                f"{self._estimated_calls_per_example}"
            )

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

        # Apply a flat conservative buffer. The optimizer eval path invokes
        # invoker.invoke() directly today, so this does not model automatic
        # per-example retries in the default evaluation loop.
        retry_factor = _BUFFER_FACTOR
        estimated_total = total_samples * base_cost_per_example * retry_factor

        logger.debug(
            f"Cost estimate ({estimation_mode}): {total_samples} total samples "
            f"× ${base_cost_per_example}/sample × {retry_factor} buffer = ${estimated_total:.2f} "
            f"(pricing_source={pricing_source})"
        )

        return estimated_total

    @staticmethod
    def extract_trial_cost(trial_result: TrialResult) -> float | None:
        """Extract cost from trial result for cost enforcement tracking.

        Attempts to find cost from multiple sources:
        0. A trial flagged ``cost_unmeasured``: the known spend floor
           (``cost_lower_bound``/reported cost), or None when nothing was
           reported
        1. trial_result.metrics["total_cost"] or ["cost"]
        2. trial_result.metadata["total_example_cost"], ["total_cost"], or ["cost"]
        3. Returns None if cost cannot be determined (triggers fallback mode)

        Args:
            trial_result: The completed trial result.

        Returns:
            Cost in USD, or None if cost cannot be determined.
        """
        # Try metrics first
        metrics = trial_result.metrics or {}
        if is_cost_unmeasured(metrics):
            # Flagged unknown: a number elsewhere on the trial (metadata, a
            # telemetry zero) is not a measurement. Budgets and limits still
            # charge the spend that WAS reported (a floor, never less than
            # was seen); callers keep the "unmeasured" state visible via the
            # flag on the trial. Nothing reported at all stays unknown.
            floor = known_spend_lower_bound(metrics, trial_result.metadata)
            return floor if floor > 0 else None
        for key in ("total_cost", "cost"):
            if key in metrics:
                try:
                    return float(metrics[key])
                except (TypeError, ValueError):
                    pass

        # Try metadata
        metadata = trial_result.metadata or {}
        for key in ("total_example_cost", "total_cost", "cost"):
            if key not in metadata:
                continue
            try:
                return float(metadata[key])
            except (TypeError, ValueError):
                pass

        # Cost cannot be determined
        return None
