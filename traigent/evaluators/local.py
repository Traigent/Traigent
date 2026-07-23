"""Local evaluation strategy."""

# Traceability: CONC-Layer-Core CONC-Quality-Reliability CONC-Quality-Performance FUNC-EVAL-METRICS REQ-EVAL-005 SYNC-OptimizationFlow

from __future__ import annotations

import inspect
import math
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from traigent.core.execution_budget import ExecutionBudget
    from traigent.core.meta_types import TraigentMetadata

from traigent.config.types import resolve_execution_mode
from traigent.evaluators.base import (
    BaseEvaluator,
    Dataset,
    EvaluationResult,
    _accuracy_values_match,
    _example_correlation_key,
)
from traigent.evaluators.metrics_tracker import (
    EMPTY_OUTPUT_RATE_WARNING_THRESHOLD,
    ExampleMetrics,
    MetricsCalculator,
    MetricsTracker,
    compute_empty_output_rate,
    enforce_user_metric_ceiling,
    extract_llm_metrics,
)
from traigent.utils.exceptions import EvaluationError
from traigent.utils.langchain_interceptor import (
    clear_captured_responses,
    get_all_captured_responses,
    get_captured_response_by_key,
    patch_langchain_for_metadata_capture,
)
from traigent.utils.litellm_interceptor import patch_litellm_for_metadata_capture
from traigent.utils.logging import get_logger

logger = get_logger(__name__)
_METADATA_PATCHES_ATTEMPTED = False

_OUTPUT_METRIC_PARAM_NAMES = {
    "actual",
    "actual_output",
    "output",
    "prediction",
    "predicted",
    "result",
}
_EXPECTED_METRIC_PARAM_NAMES = {
    "expected",
    "expected_output",
    "ground_truth",
    "reference",
    "target",
}
_LLM_METRIC_PARAM_NAMES = {"llm_metrics", "metrics"}


@dataclass
class PromptInfo:
    """Extracted prompt information for metrics calculation."""

    original_prompt: list[dict[str, str]] | None
    prompt_length: int | None
    response_length: int | None


class _AggregatedResponses:
    """Wrapper that aggregates metrics from multiple LLM responses.

    Used when a single example makes multiple LLM calls (e.g., main response + judge).
    Provides usage_metadata and response_metadata that sum the individual responses.
    """

    def __init__(self, responses: list[Any]) -> None:
        self._responses = responses
        totals = self._sum_token_usage()
        self._build_metadata(totals)

    def _extract_tokens_from_response(self, resp: Any) -> tuple[int, int, int]:
        """Extract (input, output, total) tokens from a single response."""
        usage = getattr(resp, "usage_metadata", None)
        if usage:
            return (
                usage.get("input_tokens", 0),
                usage.get("output_tokens", 0),
                usage.get("total_tokens", 0),
            )

        usage_obj = getattr(resp, "usage", None)
        if usage_obj is not None:
            input_tokens = getattr(usage_obj, "prompt_tokens", 0) or 0
            output_tokens = getattr(usage_obj, "completion_tokens", 0) or 0
            total_tokens = getattr(usage_obj, "total_tokens", None)
            if isinstance(input_tokens, (int, float)) and isinstance(
                output_tokens, (int, float)
            ):
                if isinstance(total_tokens, (int, float)):
                    return (int(input_tokens), int(output_tokens), int(total_tokens))
                return (
                    int(input_tokens),
                    int(output_tokens),
                    int(input_tokens + output_tokens),
                )

        resp_meta = getattr(resp, "response_metadata", None)
        if not resp_meta or not isinstance(resp_meta, dict):
            return (0, 0, 0)

        token_usage = resp_meta.get("token_usage", {})
        if not isinstance(token_usage, dict):
            return (0, 0, 0)

        return (
            token_usage.get("prompt_tokens", 0),
            token_usage.get("completion_tokens", 0),
            token_usage.get("total_tokens", 0),
        )

    def _sum_token_usage(self) -> tuple[int, int, int]:
        """Sum up token usage from all responses."""
        total_input = 0
        total_output = 0
        total_tokens = 0

        for resp in self._responses:
            inp, out, tot = self._extract_tokens_from_response(resp)
            total_input += inp
            total_output += out
            total_tokens += tot

        # Ensure total is at least sum of parts
        if total_tokens == 0 and (total_input > 0 or total_output > 0):
            total_tokens = total_input + total_output

        return (total_input, total_output, total_tokens)

    def _build_metadata(self, totals: tuple[int, int, int]) -> None:
        """Build usage_metadata and response_metadata from totals."""
        total_input, total_output, total_tokens = totals

        self.usage_metadata = {
            "input_tokens": total_input,
            "output_tokens": total_output,
            "total_tokens": total_tokens,
        }

        self.response_metadata: dict[str, Any] = {
            "token_usage": {
                "prompt_tokens": total_input,
                "completion_tokens": total_output,
                "total_tokens": total_tokens,
            }
        }

        # Copy model name from first response if available
        if self._responses:
            first_meta = getattr(self._responses[0], "response_metadata", None)
            if first_meta and isinstance(first_meta, dict):
                for key in ("model", "model_name"):
                    if key in first_meta:
                        self.response_metadata[key] = first_meta[key]

        logger.debug(
            f"Aggregated {len(self._responses)} LLM responses: "
            f"input={total_input}, output={total_output}, total={total_tokens}"
        )


def _ensure_metadata_capture_patches() -> None:
    """Patch LangChain and LiteLLM metadata capture lazily.

    Importing the SDK should not emit optional-integration warnings just because
    LocalEvaluator is imported as part of broader package initialization.
    """
    global _METADATA_PATCHES_ATTEMPTED
    if _METADATA_PATCHES_ATTEMPTED:
        return
    patch_langchain_for_metadata_capture()
    patch_litellm_for_metadata_capture()
    _METADATA_PATCHES_ATTEMPTED = True


if TYPE_CHECKING:
    from traigent.core.sample_budget import SampleBudgetLease


class LocalEvaluator(BaseEvaluator):
    """Local evaluation strategy.

    Evaluates functions locally in the current process. Supports both
    synchronous and asynchronous functions with timeout handling.

    Example::

        evaluator = LocalEvaluator(["accuracy"], timeout=30.0)
        result = await evaluator.evaluate(my_function, config, dataset)
    """

    def __init__(
        self,
        metrics: list[str] | None = None,
        timeout: float = 60.0,
        max_workers: int = 1,
        detailed: bool = False,
        execution_mode: str | None = None,
        mock_mode_config: dict[str, Any] | None = None,
        metric_functions: dict[str, Callable[..., Any]] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize local evaluator.

        Args:
            metrics: List of metric names to compute
            timeout: Timeout for individual evaluations (seconds)
            max_workers: Maximum number of concurrent evaluations
            detailed: Whether to preserve detailed example results
            execution_mode: Execution mode ("local", "hybrid", or "hybrid_api") for determining submission format
            **kwargs: Additional configuration
        """
        if "privacy_enabled" in kwargs:
            import warnings

            kwargs.pop("privacy_enabled")
            warnings.warn(
                "LocalEvaluator privacy_enabled is deprecated and ignored; local "
                "evaluator traces and results are content-free by default.",
                DeprecationWarning,
                stacklevel=2,
            )

        _ensure_metadata_capture_patches()

        if metrics is None and metric_functions:
            metrics = list(metric_functions.keys())

        super().__init__(metrics, timeout, max_workers, **kwargs)
        self.detailed = detailed
        self.execution_mode_enum = (
            resolve_execution_mode(execution_mode)
            if execution_mode is not None
            else None
        )
        self.execution_mode = (
            self.execution_mode_enum.value if self.execution_mode_enum else None
        )
        # ``mock_mode_config`` is retained as an accepted parameter for
        # backward compatibility with public APIs that thread it through
        # (e.g. ``@traigent.optimize(mock_mode_config=...)``), but it no
        # longer drives any evaluator behaviour. The previous behaviour
        # (TRAIGENT_MOCK_LLM-gated fabricated accuracy via
        # ``_compute_mock_accuracy``) was removed because a stray env var
        # in production caused real evaluations to be silently replaced
        # with random.uniform()-based fake scores.
        self.mock_mode_config = mock_mode_config or {}
        self.metric_functions = metric_functions or {}
        # One-shot guard for the dual-scorer run notice (issue #1845). The
        # evaluator instance is shared across a run's trials, so a flag here
        # emits the notice once per run rather than once per example/trial.
        self._dual_scorer_notice_logged = False
        # One-shot guard for the high-empty-output-rate run warning (issue
        # #1851). Same rationale as the dual-scorer flag: the evaluator instance
        # is shared across the run's trials, so this emits ONE warning per run
        # (naming the first offending config) rather than one per trial/example.
        self._empty_output_warning_logged = False

    def _extract_prompt_info(
        self,
        example_input: Any,
        output: Any,
        config: dict[str, Any] | None = None,
    ) -> PromptInfo:
        """Extract prompt information for metrics calculation.

        Keeps prompt content out of metric extraction while retaining lengths
        for downstream cost estimation.

        Args:
            example_input: Input data from dataset example
            output: Output from function call
            config: Configuration dictionary (uses ``config["prompt"]`` for
                template-aware length calculation when available)

        Returns:
            PromptInfo with content-free prompt lengths.
        """
        prompt_length: int | None = None
        response_length: int | None = None
        original_prompt: list[dict[str, str]] | None = None

        prompt_length = self._calculate_input_length(example_input, config=config)
        response_length = self._extract_response_length(output)

        return PromptInfo(
            original_prompt=original_prompt,
            prompt_length=prompt_length,
            response_length=response_length,
        )

    def _calculate_prompt_template_length(
        self,
        example_input: Any,
        config: dict[str, Any] | None = None,
    ) -> int | None:
        """Calculate template-aware input length when prompt template is available.

        Uses ``config["prompt"]`` only. If formatting fails, falls back to additive
        approximation ``len(template) + len(str(example_input))``.
        """
        if not isinstance(config, dict):
            return None

        prompt_template = config.get("prompt")
        if not isinstance(prompt_template, str) or not prompt_template:
            return None

        if isinstance(example_input, dict):
            try:
                return len(prompt_template.format(**example_input))
            except Exception as exc:
                logger.debug(
                    "Prompt template format failed, using additive length fallback: %s",
                    exc,
                )
                return len(prompt_template) + len(str(example_input))

        return len(prompt_template) + len(str(example_input))

    def _calculate_input_length(
        self,
        example_input: Any,
        config: dict[str, Any] | None = None,
    ) -> int:
        """Calculate the character length of input data."""
        template_length = self._calculate_prompt_template_length(
            example_input, config=config
        )
        if template_length is not None:
            return template_length

        if isinstance(example_input, dict):
            if "text" in example_input:
                return len(example_input["text"])
            elif "messages" in example_input:
                # For messages, sum up all content lengths
                return sum(
                    len(msg.get("content", ""))
                    for msg in example_input["messages"]
                    if isinstance(msg, dict)
                )
            else:
                return len(str(example_input))
        return len(str(example_input))

    def _reconstruct_prompt(self, example_input: Any) -> list[dict[str, str]]:
        """Reconstruct prompt messages from input data."""
        if isinstance(example_input, dict):
            if "text" in example_input:
                return [{"role": "user", "content": example_input["text"]}]
            elif "messages" in example_input:
                return cast(list[dict[str, str]], example_input["messages"])
            else:
                return [{"role": "user", "content": str(example_input)}]
        return [{"role": "user", "content": str(example_input)}]

    @staticmethod
    def _infer_model_name_from_output(
        output: Any,
        index: int,
        captured_responses: list[Any],
    ) -> str | None:
        """Try to infer the model name from the output or captured LangChain responses.

        This is a fallback for when the optimization config does not contain a
        ``model`` key (e.g. the user is only tuning prompts, not models).
        Without a model name, cost calculation is skipped entirely, which
        causes the $0 cost bug reported in TraigentFrontend#325.
        """
        # 1. Check if output is a dict with a model field
        if isinstance(output, dict):
            for key in ("model", "model_name", "model_id"):
                val = output.get(key)
                if isinstance(val, str) and val:
                    return val

        # 2. Check output object attributes (e.g. OpenAI ChatCompletion)
        for attr in ("model", "model_name"):
            val = getattr(output, attr, None)
            if isinstance(val, str) and val:
                return val

        # 3. Check captured LangChain response (llm_output.model_name)
        if index < len(captured_responses):
            resp = captured_responses[index]
            llm_output = getattr(resp, "llm_output", None)
            if isinstance(llm_output, dict):
                val = llm_output.get("model_name") or llm_output.get("model")
                if isinstance(val, str) and val:
                    return val

        return None

    def _extract_response_text(self, output: Any) -> str | None:
        """Extract text content from various output types.

        Args:
            output: Output from function call (dict, str, or LLM response object)

        Returns:
            Extracted text or None if not extractable
        """
        if isinstance(output, dict):
            return output.get("text")
        elif isinstance(output, str):
            return output
        elif hasattr(output, "content") and output.content:
            first = output.content[0]
            if hasattr(first, "text"):
                return cast(str, first.text)
        elif hasattr(output, "text"):
            return cast(str, output.text)
        return None

    def _extract_response_length(self, output: Any) -> int | None:
        """Extract response length for privacy mode."""
        response_text = self._extract_response_text(output)
        if response_text is None:
            return None
        return len(response_text)

    def _get_metrics_source(
        self,
        output: Any,
        example_index: int,
        dataset: Dataset,
        all_captured_responses: list[Any],
    ) -> Any:
        """Determine the best source for extracting LLM metrics.

        Priority:
        1. If output is a dict with 'raw_response', prefer it (SDK object)
        2. Else use captured LangChain response by correlation key
        3. Else use captured response by index (with multi-call handling)
        4. Else fall back to output itself

        When there are multiple LLM calls per example (e.g., main response + judge),
        this method handles the mismatch by computing aggregate metrics.

        Args:
            output: The function output
            example_index: Current example index
            dataset: Evaluation dataset
            all_captured_responses: List of captured LangChain responses

        Returns:
            The best source object for metrics extraction
        """
        # Prefer raw_response from dict output
        if isinstance(output, dict) and output.get("raw_response") is not None:
            return output["raw_response"]

        # Try correlation key (example_id)
        try:
            ex = dataset.examples[example_index]
            ex_key = _example_correlation_key(ex, example_index)
        except Exception:
            ex_key = f"example_{example_index}"

        by_key = get_captured_response_by_key(ex_key)
        if by_key is not None:
            return by_key

        # Handle case where there are more captured responses than outputs
        # (multiple LLM calls per example, e.g., main response + judge scoring)
        num_outputs = len(dataset.examples)
        num_responses = len(all_captured_responses)

        if num_responses > num_outputs and num_outputs > 0:
            # Calculate how many responses per output (e.g., 2 for main + judge)
            responses_per_output = num_responses // num_outputs
            if responses_per_output >= 1:
                # Get the slice of responses for this example
                start_idx = example_index * responses_per_output
                end_idx = start_idx + responses_per_output
                # Return aggregated metrics wrapper if we have multiple
                if end_idx <= num_responses and responses_per_output > 1:
                    return _AggregatedResponses(
                        all_captured_responses[start_idx:end_idx]
                    )
                elif start_idx < num_responses:
                    return all_captured_responses[start_idx]

        # Fall back to index-based or output itself
        if example_index < len(all_captured_responses):
            return all_captured_responses[example_index]

        return output

    def _estimate_string_tokens(
        self,
        example_metric: ExampleMetrics,
        output: str,
        example_input: Any,
        config: dict[str, Any] | None = None,
    ) -> None:
        """Estimate token counts for string outputs.

        Updates example_metric in place with estimated token counts.
        Uses approximation of 1 token per 4 characters.

        Args:
            example_metric: Metrics object to update
            output: String output from function
            example_input: Original input for estimating input tokens
            config: Configuration dictionary (uses ``config["prompt"]`` when
                available for template-aware input token estimation)
        """
        # Estimate output tokens
        example_metric.tokens.output_tokens = max(1, len(output) // 4)

        # Estimate input tokens from local lengths only. Privacy mode may not
        # retain raw prompts, but it still needs length-derived cost metrics.
        input_length = self._calculate_prompt_template_length(
            example_input, config=config
        )
        if input_length is None:
            input_length = self._calculate_input_length(example_input, config=config)
        example_metric.tokens.input_tokens = max(1, input_length // 4)

        # Update total
        example_metric.tokens.total_tokens = (
            example_metric.tokens.input_tokens + example_metric.tokens.output_tokens
        )

    def _calculate_example_accuracy(
        self,
        actual_value: Any,
        expected_output: Any,
    ) -> float | None:
        """Calculate accuracy for a single example.

        Args:
            actual_value: Actual output value
            expected_output: Expected output value

        Returns:
            Accuracy value (1.0 or 0.0) or None if not calculable
        """
        if expected_output is None or actual_value is None:
            return None

        if isinstance(actual_value, str) and isinstance(expected_output, str):
            return 1.0 if _accuracy_values_match(actual_value, expected_output) else 0.0
        return 1.0 if _accuracy_values_match(actual_value, expected_output) else 0.0

    def _transfer_token_metrics_to_example_result(
        self,
        example_result: Any,
        example_metric: ExampleMetrics,
    ) -> None:
        """Transfer token and cost metrics to example result.

        Args:
            example_result: ExampleResult object to update
            example_metric: Source metrics
        """
        if example_result is None:
            return

        example_result.metrics["input_tokens"] = example_metric.tokens.input_tokens
        example_result.metrics["output_tokens"] = example_metric.tokens.output_tokens
        example_result.metrics["total_tokens"] = example_metric.tokens.total_tokens
        example_result.metrics["input_cost"] = example_metric.cost.input_cost
        example_result.metrics["output_cost"] = example_metric.cost.output_cost
        example_result.metrics["total_cost"] = example_metric.cost.total_cost

    def _objective_returned_none_error(
        self,
        metric_name: str,
        example_id: str,
        example_index: int,
        config: dict[str, Any],
    ) -> EvaluationError:
        """Build the fail-closed error for an objective metric returning None.

        Companion to the exception-path guard in ``_invoke_metric_function``
        (which raises when an objective metric *raises*). A ``None`` return is
        not an exception, so without this an optimization objective's ``None``
        would be substituted with a fabricated ``0.0`` and silently corrupt the
        search. Callers raise the returned error only for objective metrics
        (``metric_name in self.metrics``); non-objective metrics retain the
        legacy ``0.0``/skip behaviour.
        """
        return EvaluationError(
            f"Objective metric '{metric_name}' returned None for example "
            f"{example_id}. Refusing to substitute a fabricated 0.0 score for "
            "an optimization objective.",
            config=config,
            details={
                "metric_name": metric_name,
                "example_id": example_id,
                "example_index": example_index,
                "is_objective": True,
                # Mirror the exception-path record shape (error_type) so
                # downstream consumers parse both failure modes uniformly.
                "error_type": "NoneReturn",
                "failure_mode": "returned_none",
            },
        )

    def _handle_metric_function_exception(
        self,
        exc: Exception,
        metric_name: str,
        example_obj: Any,
        config: dict[str, Any],
        example_index: int,
        *,
        metric_errors: list[dict[str, Any]] | None = None,
    ) -> float:
        example_id = (
            example_obj.metadata.get("example_id", f"example_{example_index}")
            if example_obj.metadata
            else f"example_{example_index}"
        )
        logger.warning(
            "Metric function %s failed for example %s: %s",
            metric_name,
            example_id,
            exc,
        )
        # ``self.metrics`` is this evaluator's configured objective/metric
        # list (every real construction path -- optimization_pipeline's
        # _create_local_evaluator, cloud/client.py, cli/optimization_validator
        # -- builds it directly from the optimization's ``objectives``).
        # A metric named here is an optimization objective; a fabricated
        # 0.0 for it would silently pin/corrupt the search, so fail the
        # trial closed instead of scoring it.
        is_objective = metric_name in self.metrics
        error_record: dict[str, Any] = {
            "metric_name": metric_name,
            "example_id": example_id,
            "example_index": example_index,
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "is_objective": is_objective,
        }
        if is_objective:
            raise EvaluationError(
                f"Objective metric '{metric_name}' raised "
                f"{type(exc).__name__} for example {example_id}: {exc}. "
                "Refusing to substitute a fabricated 0.0 score for an "
                "optimization objective.",
                config=config,
                original_error=exc,
                details=error_record,
            ) from exc
        if metric_errors is not None:
            metric_errors.append(error_record)
        return 0.0

    async def _resolve_metric_function_value(
        self,
        value: Any,
        metric_name: str,
        example_obj: Any,
        config: dict[str, Any],
        example_index: int,
        *,
        metric_errors: list[dict[str, Any]] | None = None,
    ) -> Any:
        if not inspect.isawaitable(value):
            return value
        try:
            return await value
        except Exception as exc:
            return self._handle_metric_function_exception(
                exc,
                metric_name,
                example_obj,
                config,
                example_index,
                metric_errors=metric_errors,
            )

    async def _apply_custom_metric_functions(
        self,
        example_metric: ExampleMetrics,
        output: Any,
        example_obj: Any,
        config: dict[str, Any],
        example_index: int,
        *,
        metric_errors: list[dict[str, Any]] | None = None,
    ) -> list[str]:
        """Apply user-defined metric functions for a single example.

        Updates example_metric.custom_metrics in place.

        Args:
            example_metric: Metrics object to update
            output: Function output
            example_obj: Dataset example object
            config: Configuration parameters
            example_index: Current example index
            metric_errors: Optional accumulator for structured degradation
                records from non-objective metric failures; forwarded to
                ``_invoke_metric_function``. An objective failure instead
                raises ``EvaluationError`` and propagates out of this call.

        Returns:
            The custom-metric keys actually written to
            ``example_metric.custom_metrics`` by this call. For a scalar-
            returning function this is the function's own name; for a
            mapping-returning function it is the produced sub-keys (never the
            function name). Callers use this to transfer the produced metrics
            downstream without assuming a key exists for every function name
            (a mapping-returning function never populates ``custom_metrics``
            under its own name).
        """
        llm_payload = {
            "total_tokens": example_metric.tokens.total_tokens,
            "prompt_tokens": example_metric.tokens.input_tokens,
            "completion_tokens": example_metric.tokens.output_tokens,
            "total_cost": example_metric.cost.total_cost,
            "input_cost": example_metric.cost.input_cost,
            "output_cost": example_metric.cost.output_cost,
            "response_time_ms": example_metric.response.response_time_ms,
        }

        example_id = (
            example_obj.metadata.get("example_id", f"example_{example_index}")
            if example_obj.metadata
            else f"example_{example_index}"
        )
        produced_keys: list[str] = []
        for metric_name, metric_func in self.metric_functions.items():
            raw_value = self._invoke_metric_function(
                metric_func,
                metric_name,
                output,
                example_obj,
                config,
                llm_payload,
                example_index,
                metric_errors=metric_errors,
            )
            value = await self._resolve_metric_function_value(
                raw_value,
                metric_name,
                example_obj,
                config,
                example_index,
                metric_errors=metric_errors,
            )
            # A ``None`` return is NOT an exception, so it bypasses the
            # objective fail-closed guard in ``_invoke_metric_function``.
            # Mirror that guard here: an objective metric that returns
            # ``None`` (scalar) or omits its value inside a returned mapping
            # would otherwise be coerced to a fabricated ``0.0`` / silently
            # dropped, pinning and corrupting the search. Non-objective
            # metrics keep the legacy ``0.0``/skip behaviour.
            if isinstance(value, Mapping):
                for result_name, result_value in value.items():
                    # A mapping sub-value can itself be awaitable (e.g.
                    # ``{"quality": async_score(...)}``); resolve it the same way
                    # as a top-level async metric so it never reaches float() as
                    # a raw coroutine (and its exceptions get the objective /
                    # degradation-record handling, keyed by the sub-metric name).
                    result_value = await self._resolve_metric_function_value(
                        result_value,
                        str(result_name),
                        example_obj,
                        config,
                        example_index,
                        metric_errors=metric_errors,
                    )
                    if result_value is None:
                        if str(result_name) in self.metrics:
                            raise self._objective_returned_none_error(
                                str(result_name),
                                example_id,
                                example_index,
                                config,
                            )
                        if metric_errors is not None:
                            metric_errors.append(
                                {
                                    "metric_name": str(result_name),
                                    "example_id": example_id,
                                    "example_index": example_index,
                                    "error_type": "NoneReturn",
                                    "failure_mode": "returned_none",
                                    "is_objective": False,
                                }
                            )
                        continue
                    key = str(result_name)
                    example_metric.custom_metrics[key] = float(result_value)
                    produced_keys.append(key)
                continue
            if value is None:
                if metric_name in self.metrics:
                    raise self._objective_returned_none_error(
                        metric_name,
                        example_id,
                        example_index,
                        config,
                    )
                if metric_errors is not None:
                    metric_errors.append(
                        {
                            "metric_name": metric_name,
                            "example_id": example_id,
                            "example_index": example_index,
                            "error_type": "NoneReturn",
                            "failure_mode": "returned_none",
                            "is_objective": False,
                        }
                    )
                example_metric.custom_metrics[metric_name] = 0.0
            else:
                example_metric.custom_metrics[metric_name] = float(value)
            produced_keys.append(metric_name)
        return produced_keys

    def _build_metric_keyword_arguments(
        self,
        parameters: list[inspect.Parameter],
        output: Any,
        example_obj: Any,
        config: dict[str, Any],
        llm_payload: dict[str, Any],
        example_index: int,
    ) -> dict[str, Any]:
        keyword_values: dict[str, Any] = {
            "example": example_obj,
            "input_data": example_obj.input_data,
            "metadata": example_obj.metadata or {},
            "config": config,
            "example_index": example_index,
        }
        for name in _OUTPUT_METRIC_PARAM_NAMES:
            keyword_values[name] = output
        for name in _EXPECTED_METRIC_PARAM_NAMES:
            keyword_values[name] = example_obj.expected_output
        for name in _LLM_METRIC_PARAM_NAMES:
            keyword_values[name] = llm_payload

        kwargs: dict[str, Any] = {}
        accepts_arbitrary_kwargs = False
        for parameter in parameters:
            if parameter.kind is inspect.Parameter.VAR_KEYWORD:
                accepts_arbitrary_kwargs = True
                continue
            if parameter.kind in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.VAR_POSITIONAL,
            ):
                continue
            if parameter.name in keyword_values:
                kwargs[parameter.name] = keyword_values[parameter.name]

        if accepts_arbitrary_kwargs:
            kwargs.setdefault("output", output)
            kwargs.setdefault("expected", example_obj.expected_output)
            kwargs.setdefault("llm_metrics", llm_payload)

        return kwargs

    def _invoke_metric_function(
        self,
        metric_func: Callable[..., Any],
        metric_name: str,
        output: Any,
        example_obj: Any,
        config: dict[str, Any],
        llm_payload: dict[str, Any],
        example_index: int,
        *,
        metric_errors: list[dict[str, Any]] | None = None,
    ) -> Any:
        """Invoke a single metric function with appropriate arguments.

        Args:
            metric_func: The metric function to call
            metric_name: Name of the metric (for logging)
            output: Function output
            example_obj: Dataset example
            config: Configuration parameters
            llm_payload: LLM metrics payload
            example_index: Current example index
            metric_errors: Optional accumulator for structured degradation
                records (``{metric_name, example_id, example_index,
                error_type, error_message, is_objective}``) for
                NON-objective metric failures. Callers that don't need to
                surface these (e.g. the best-effort progress-preview lane
                in ``BaseEvaluator``) may omit it; the failure is still
                logged either way.

        Returns:
            Metric value on success, an awaitable metric value for async metric
            functions, or ``0.0`` for a failed *informational* metric -- never
            a bare, indistinguishable 0.0. A structured error record is
            appended to ``metric_errors`` (when provided) so a caller can tell
            a real 0.0 apart from a degraded one.

        Raises:
            EvaluationError: If ``metric_name`` is one of this evaluator's
                configured objectives (``self.metrics``) and the metric
                function raises. Substituting a fabricated 0.0 for an
                optimization objective would silently corrupt the search,
                so the trial fails closed instead.
        """
        try:
            signature = inspect.signature(metric_func)
            parameters = list(signature.parameters.values())
            positional_args = (output, example_obj.expected_output, llm_payload)
            call_candidates: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

            if any(
                parameter.kind is inspect.Parameter.VAR_POSITIONAL
                for parameter in parameters
            ):
                call_candidates.append((positional_args, {}))

            kwargs = self._build_metric_keyword_arguments(
                parameters, output, example_obj, config, llm_payload, example_index
            )
            if kwargs:
                call_candidates.append(((), kwargs))

            call_candidates.extend(
                [
                    (positional_args, {}),
                    (positional_args[:2], {}),
                    (positional_args[:1], {}),
                    ((), {}),
                ]
            )

            bind_error: TypeError | None = None
            for args, candidate_kwargs in call_candidates:
                try:
                    signature.bind(*args, **candidate_kwargs)
                except TypeError as exc:
                    bind_error = exc
                    continue
                return metric_func(*args, **candidate_kwargs)

            if bind_error is not None:
                raise bind_error
            return metric_func(output, example_obj.expected_output)
        except Exception as exc:
            return self._handle_metric_function_exception(
                exc,
                metric_name,
                example_obj,
                config,
                example_index,
                metric_errors=metric_errors,
            )

    def _build_local_progress_payload(
        self,
        example_metric: ExampleMetrics,
        example_result: Any,
        output: Any,
    ) -> dict[str, Any]:
        """Build payload for progress callback in local evaluation mode.

        Note: This method has a different signature from the base class
        _build_progress_payload to support LocalEvaluator's specific needs.

        Args:
            example_metric: Current example metrics
            example_result: Example result (if in detailed mode)
            output: Function output

        Returns:
            Payload dictionary for progress callback
        """
        payload_metrics = dict(example_metric.custom_metrics)
        total_cost_value = example_metric.cost.total_cost

        if total_cost_value is not None:
            payload_metrics.setdefault("total_cost", float(total_cost_value))
            payload_metrics.setdefault("cost", float(total_cost_value))

        if self.detailed and example_result is not None:
            for key, value in example_result.metrics.items():
                if key in {"cost", "total_cost"} and key in payload_metrics:
                    continue
                payload_metrics.setdefault(key, value)

        return {
            "success": example_metric.success,
            "error": example_metric.error,
            "metrics": payload_metrics,
            "output": output,
        }

    def _extract_llm_metrics_for_output(
        self,
        output: Any,
        index: int,
        config: dict[str, Any],
        dataset: Dataset,
        all_captured_responses: list[Any],
    ) -> ExampleMetrics:
        """Extract LLM metrics for a single output.

        Args:
            output: Function output
            index: Example index
            config: Configuration parameters
            dataset: Evaluation dataset
            all_captured_responses: Captured LangChain responses

        Returns:
            ExampleMetrics with extracted or estimated metrics
        """
        example_metric = ExampleMetrics()

        if output is None:
            return example_metric

        model_name = config.get("model") or config.get("model_name")

        # Fallback: try to extract model name from the output or captured responses
        # when the optimization config does not include "model" as a parameter.
        if not model_name:
            model_name = self._infer_model_name_from_output(
                output, index, all_captured_responses
            )

        logger.debug(
            f"EVALUATOR DEBUG: i={index}, output type={type(output).__name__}, "
            f"model_name='{model_name}'"
        )

        original_prompt = None
        prompt_length = None
        response_length = None

        # Extract prompt info using helper method
        if index < len(dataset.examples):
            example_input = dataset.examples[index].input_data
            prompt_info = self._extract_prompt_info(
                example_input, output, config=config
            )
            original_prompt = prompt_info.original_prompt
            prompt_length = prompt_info.prompt_length
            response_length = prompt_info.response_length

        # Determine response length without passing content to metric extraction.
        response_text = self._extract_response_text(output)
        if response_text is not None and response_length is None:
            response_length = len(response_text)

        # Get best metrics source using helper
        metrics_source = self._get_metrics_source(
            output, index, dataset, all_captured_responses
        )

        # Pass lengths to extract_llm_metrics for privacy mode
        extracted_metrics = extract_llm_metrics(
            response=metrics_source,
            model_name=model_name,
            original_prompt=original_prompt,
            response_text=None,
            prompt_length=prompt_length,
            response_length=response_length,
        )

        logger.debug(
            f"Extracted metrics for model {model_name}: "
            f"tokens={extracted_metrics.tokens.total_tokens}, "
            f"cost=${extracted_metrics.cost.total_cost:.8f}"
        )

        example_metric = extracted_metrics

        # If we didn't get token metrics but output is a string, estimate tokens
        if extracted_metrics.tokens.total_tokens == 0 and isinstance(output, str):
            example_input_for_tokens = (
                dataset.examples[index].input_data
                if index < len(dataset.examples)
                else None
            )
            self._estimate_string_tokens(
                example_metric, output, example_input_for_tokens, config=config
            )

        return example_metric

    def _update_example_metric_from_result(
        self,
        example_metric: ExampleMetrics,
        example_result: Any,
        index: int,
    ) -> None:
        """Update example metric with data from example result.

        Args:
            example_metric: Metrics to update (modified in place)
            example_result: Source example result
            index: Example index
        """
        if example_result is None:
            return

        # Transfer execution time to response_time_ms (convert seconds to milliseconds)
        if hasattr(example_result, "execution_time"):
            example_metric.response.response_time_ms = (
                example_result.execution_time * 1000
            )

        # Transfer computed metrics (like accuracy) to custom_metrics
        if hasattr(example_result, "metrics") and example_result.metrics:
            example_metric.custom_metrics.update(example_result.metrics)

        # Transfer token/cost metrics to example_results if in detailed mode
        if self.detailed:
            self._transfer_token_metrics_to_example_result(
                example_result, example_metric
            )

    def _add_standard_metrics_to_custom(self, example_metric: ExampleMetrics) -> None:
        """Add standard token/cost metrics to custom_metrics.

        Args:
            example_metric: Metrics object to update (modified in place)
        """
        example_metric.custom_metrics["input_tokens"] = (
            example_metric.tokens.input_tokens
        )
        example_metric.custom_metrics["output_tokens"] = (
            example_metric.tokens.output_tokens
        )
        example_metric.custom_metrics["total_tokens"] = (
            example_metric.tokens.total_tokens
        )
        example_metric.custom_metrics["input_cost"] = example_metric.cost.input_cost
        example_metric.custom_metrics["output_cost"] = example_metric.cost.output_cost
        example_metric.custom_metrics["total_cost"] = example_metric.cost.total_cost

    def _inject_usage_from_meta(
        self, usage: dict[str, Any], metrics: ExampleMetrics
    ) -> None:
        """Inject usage data (tokens, response time) from __traigent_meta__.

        Args:
            usage: Usage dict from __traigent_meta__ with optional keys:
                input_tokens, output_tokens, response_time_ms.
            metrics: ExampleMetrics to update.
        """
        # Inject token counts (key-presence checks allow explicit zeros)
        for key, attr in [
            ("input_tokens", "input_tokens"),
            ("output_tokens", "output_tokens"),
        ]:
            if key not in usage:
                continue
            try:
                val = usage[key]
                if val < 0:
                    logger.warning(f"Negative {key} clamped: {val} → 0")
                setattr(metrics.tokens, attr, max(0, val))
            except Exception as e:
                logger.error(f"Failed to inject {key}: {e}", extra={"usage": usage})

        # Recompute total_tokens if any token counts were injected
        if "input_tokens" in usage or "output_tokens" in usage:
            try:
                metrics.tokens.total_tokens = (
                    metrics.tokens.input_tokens + metrics.tokens.output_tokens
                )
            except Exception as e:
                logger.error(f"Failed to compute total_tokens: {e}")

        # Inject response_time_ms if provided
        if "response_time_ms" not in usage:
            return
        try:
            response_time = usage["response_time_ms"]
            if response_time < 0:
                logger.warning(
                    f"Negative response_time_ms clamped: {response_time} → 0.0"
                )
            metrics.response.response_time_ms = max(0.0, float(response_time))
        except Exception as e:
            logger.error(f"Failed to inject response_time_ms: {e}")

    def _extract_and_inject_traigent_meta(
        self,
        output: Any,
        metrics: ExampleMetrics,
        model_name: str | None = None,
    ) -> TraigentMetadata | None:
        """Extract __traigent_meta__ from output and inject into metrics.

        This method runs AFTER cost calculation to allow user-provided costs
        to override SDK-calculated values when plausible.

        Args:
            output: Raw function output (may be dict with __traigent_meta__).
            metrics: ExampleMetrics to update with user-provided values.
            model_name: Optional model name for best-effort token-cost plausibility
                checks. Without a model or known pricing, reported BYOK costs are
                accepted as-is.

        Returns:
            The meta dict if found, None otherwise.
        """
        from traigent.core.meta_types import TraigentMetadata, is_traigent_metadata

        if not isinstance(output, dict):
            return None

        meta = output.get("__traigent_meta__")
        if meta is None:
            return None

        if not is_traigent_metadata(meta):
            logger.error(
                "Invalid __traigent_meta__ structure",
                extra={
                    "meta": meta,
                    "expected_keys": ["total_cost (required)", "usage (optional)"],
                    "validation": "Type guard failed",
                },
            )
            return None

        meta = cast(TraigentMetadata, meta)
        logger.debug(f"Validated __traigent_meta__ with keys: {meta.keys()}")

        # Inject usage data (tokens, response time)
        if "usage" in meta:
            self._inject_usage_from_meta(cast(dict, meta["usage"]), metrics)

        # Inject cost
        try:
            total_cost = meta["total_cost"]
            if total_cost < 0:
                logger.warning(f"Negative total_cost clamped: {total_cost} → 0.0")
            metrics.cost.total_cost = max(0.0, float(total_cost))
            if model_name:
                from traigent.evaluators.metrics_tracker import (
                    _reconcile_reported_cost_with_tokens,
                )

                _reconcile_reported_cost_with_tokens(metrics, model_name)
        except Exception as e:
            logger.error(f"Failed to inject total_cost: {e}")

        # Recompute derived metrics
        try:
            if metrics.tokens.total_tokens == 0:
                metrics.response.tokens_per_second = 0.0
            else:
                MetricsCalculator.calculate_tokens_per_second(metrics)
        except Exception as e:
            logger.error(f"Failed to recompute tokens_per_second: {e}")

        logger.debug(
            f"Injected cost=${metrics.cost.total_cost:.4f}, "
            f"tokens={metrics.tokens.total_tokens}"
        )

        return meta

    async def _process_single_output(
        self,
        output: Any,
        index: int,
        config: dict[str, Any],
        dataset: Dataset,
        errors: list[str | None],
        example_results: list[Any] | None,
        all_captured_responses: list[Any],
        progress_callback: Callable[[int, dict[str, Any]], Any] | None,
        *,
        metric_errors: list[dict[str, Any]] | None = None,
    ) -> ExampleMetrics:
        """Process metrics for a single output.

        Args:
            output: Function output
            index: Example index
            config: Configuration parameters
            dataset: Evaluation dataset
            errors: List of error messages
            example_results: List of example results (may be None)
            all_captured_responses: Captured responses
            progress_callback: Optional progress callback
            metric_errors: Optional accumulator for structured degradation
                records from non-objective custom metric failures for this
                evaluation run; forwarded to ``_apply_custom_metric_functions``.

        Returns:
            ExampleMetrics for this output
        """
        # Per-example user metrics: prefer the carrier the boundary already
        # attached to the ExampleResult (detailed mode). When the carrier is
        # present the output was ALREADY unpacked upstream, so re-unpacking here
        # would double-unpack a nested matching tuple — e.g. an output of
        # ``("inner", {"x": 1.0})`` would be split again into ``"inner"`` and a
        # spurious ``{"x": 1.0}``. We therefore unpack HERE only when the
        # carrier is absent (the genuinely non-detailed lane, where ``outputs``
        # still holds the raw tuple). Any non-matching shape is left untouched.
        carrier = None
        if (
            example_results
            and index < len(example_results)
            and example_results[index] is not None
            and getattr(example_results[index], "user_metrics", None) is not None
        ):
            carrier = example_results[index]

        if carrier is not None:
            user_metrics = carrier.user_metrics
        else:
            output, user_metrics = self._unpack_user_metrics(output)

        # Extract LLM metrics
        example_metric = self._extract_llm_metrics_for_output(
            output, index, config, dataset, all_captured_responses
        )

        model_name = config.get("model") or config.get("model_name")
        if not model_name:
            model_name = self._infer_model_name_from_output(
                output, index, all_captured_responses
            )

        # META-EXTRACTION-POINT: Extract and inject __traigent_meta__ if present.
        # This happens after extract_llm_metrics so honest reported costs can
        # override SDK calculation while implausible token-priced under-reports
        # are clamped back to the canonical token-derived estimate.
        self._extract_and_inject_traigent_meta(
            output, example_metric, model_name=model_name
        )

        # Set success status
        example_metric.success = errors[index] is None
        example_metric.error = errors[index]

        # Calculate accuracy
        expected_output = (
            dataset.examples[index].expected_output
            if index < len(dataset.examples)
            else None
        )
        actual_value = output.get("text") if isinstance(output, dict) else output
        accuracy_value = self._calculate_example_accuracy(actual_value, expected_output)
        if accuracy_value is not None:
            example_metric.custom_metrics.setdefault("accuracy", accuracy_value)

        # Merge per-example user metrics into custom_metrics, never overriding
        # an evaluator-computed key (e.g. accuracy stays the computed value).
        self._merge_user_metrics(
            example_metric.custom_metrics, user_metrics, context="local lane"
        )

        # Transfer metrics from example_results if available
        if example_results and index < len(example_results):
            self._update_example_metric_from_result(
                example_metric, example_results[index], index
            )

        # Add standard metrics to custom_metrics
        self._add_standard_metrics_to_custom(example_metric)

        # Apply custom metric functions
        if self.metric_functions and index < len(dataset.examples):
            example_obj = dataset.examples[index]
            produced_metric_keys = await self._apply_custom_metric_functions(
                example_metric,
                output,
                example_obj,
                config,
                index,
                metric_errors=metric_errors,
            )

            # Transfer custom metrics to example_results if in detailed mode.
            # Iterate the keys actually produced (which include mapping
            # sub-keys and exclude any function name a mapping-returning
            # function never populated) rather than the function names -- the
            # latter KeyErrors for mapping-returning metric functions.
            if (
                self.detailed
                and example_results
                and index < len(example_results)
                and example_results[index] is not None
            ):
                # A metric function is authoritative for the key(s) it
                # produces and overrides a built-in computed value of the same
                # name -- this is the contract the custom ``scoring_function``
                # path relies on (it defines the objective, e.g. ``accuracy``).
                # Mapping sub-keys inherit the same override semantics as
                # scalar metric-function values, for consistency.
                target_metrics = example_results[index].metrics
                for metric_key in produced_metric_keys:
                    target_metrics[metric_key] = example_metric.custom_metrics[
                        metric_key
                    ]

        # Send progress callback
        if progress_callback:
            example_result_for_progress = (
                example_results[index]
                if example_results and index < len(example_results)
                else None
            )
            payload = self._build_local_progress_payload(
                example_metric, example_result_for_progress, output
            )
            progress_callback(index, payload)

        return example_metric

    def _log_dual_scorer_notice_once(self) -> None:
        """Emit a single run-level notice when both scorers appear (issue #1845).

        Fires at most once per evaluator instance (== once per run), never per
        example/trial, so a customer sees one clear line rather than N.
        """
        if self._dual_scorer_notice_logged:
            return
        self._dual_scorer_notice_logged = True
        logger.info(
            "A custom scoring_function defines the 'accuracy' objective; "
            "metrics['score'] carries the optimization signal (the objective "
            "value, == metrics['accuracy']), and the SDK's built-in "
            "exact-match scorer is recorded separately as "
            "metrics['exact_match_default'] for diagnostics."
        )

    def _maybe_warn_high_empty_output_rate(
        self, rate: float, config: dict[str, Any]
    ) -> None:
        """Emit a single run-level warning for a high empty-output rate (#1851).

        Fires at most once per evaluator instance (== once per run), never per
        example/trial, mirroring the dual-scorer notice (#1845). An empty or
        whitespace-only output at any meaningful rate signals truncation,
        output-parsing failure, or refusals, so the accuracy comparison is
        measuring an artifact, not the config knobs. Every trial still records
        its own ``empty_output_rate`` metric; this warning names the first
        offending config so the run does not finish silently.
        """
        if rate <= EMPTY_OUTPUT_RATE_WARNING_THRESHOLD:
            return
        if self._empty_output_warning_logged:
            return
        self._empty_output_warning_logged = True
        logger.warning(
            "%.1f%% of outputs for config %r are empty or whitespace-only "
            "(threshold %.0f%%) — likely token truncation, output-parsing "
            "failure, or refusals; accuracy comparisons are unreliable until "
            "resolved. Each trial's empty_output_rate metric records the "
            "per-config rate.",
            rate * 100.0,
            config,
            EMPTY_OUTPUT_RATE_WARNING_THRESHOLD * 100.0,
        )

    def _compute_accuracy_aggregated(
        self,
        outputs: list[Any],
        dataset: Dataset,
    ) -> tuple[float | None, int]:
        """Compute aggregated accuracy across outputs.

        Args:
            outputs: List of outputs
            dataset: Evaluation dataset

        Returns:
            Tuple of (accuracy value or None, total count)
        """
        total = 0
        correct = 0

        for raw_output, example in zip(outputs, dataset.examples, strict=False):
            expected = example.expected_output
            if expected is None:
                continue

            value = (
                raw_output.get("text") if isinstance(raw_output, dict) else raw_output
            )
            if value is None:
                continue

            total += 1
            if _accuracy_values_match(value, expected):
                correct += 1

        if total > 0:
            return correct / total, total
        return None, total

    def _merge_comprehensive_metrics(
        self,
        aggregated_metrics: dict[str, float],
        comprehensive_metrics: dict[str, Any],
    ) -> None:
        """Merge comprehensive metrics into aggregated metrics.

        Args:
            aggregated_metrics: Target metrics dict (modified in place)
            comprehensive_metrics: Source comprehensive metrics
        """
        logger.debug(
            f"LOCAL EVALUATOR DEBUG: comprehensive_metrics['cost'] = "
            f"{comprehensive_metrics.get('cost', 'MISSING')}"
        )

        # If cost is in objectives but was computed as 0, use comprehensive value
        if "cost" in self.metrics and "cost" in comprehensive_metrics:
            logger.debug(
                f"LOCAL EVALUATOR DEBUG: aggregated cost="
                f"{aggregated_metrics.get('cost', 'MISSING')}, "
                f"comprehensive cost={comprehensive_metrics['cost']}"
            )
            aggregated_cost = float(aggregated_metrics.get("cost", 0.0) or 0.0)
            comprehensive_cost = float(comprehensive_metrics["cost"])
            if math.isclose(aggregated_cost, 0.0, abs_tol=1e-9) and not math.isclose(
                comprehensive_cost, 0.0, abs_tol=1e-9
            ):
                logger.info(
                    f"🔍 LOCAL EVALUATOR: Overriding cost metric: "
                    f"{aggregated_cost} -> {comprehensive_cost}"
                )

        for key, value in comprehensive_metrics.items():
            if value is None:
                continue
            if (
                key in {"accuracy", "score"}
                and key in aggregated_metrics
                and aggregated_metrics[key] not in (None, 0.0)
            ):
                continue
            aggregated_metrics[key] = value

    @staticmethod
    def _compute_percentile(values: list[float], percentile: float) -> float:
        """Compute percentile for a list of floats using linear interpolation."""
        if not values:
            return 0.0
        if len(values) == 1:
            return float(values[0])

        ordered = sorted(values)
        index = (len(ordered) - 1) * percentile
        lower_idx = math.floor(index)
        upper_idx = math.ceil(index)

        if lower_idx == upper_idx:
            return float(ordered[int(index)])

        lower_val = ordered[lower_idx]
        upper_val = ordered[upper_idx]
        fraction = index - lower_idx
        return float(lower_val + (upper_val - lower_val) * fraction)

    def _compute_aggregated_custom_metrics(
        self,
        example_results: list[Any],
    ) -> dict[str, float]:
        """Compute aggregated metrics from custom metric functions.

        Args:
            example_results: List of example results with metrics

        Returns:
            Dictionary of aggregated metric values
        """
        aggregated: dict[str, float] = {}

        for metric_name in self.metric_functions:
            values: list[float] = []
            # A mapping-returning metric function populates its SUB-keys, never
            # its own name, so ``metric_name`` may be absent from every result.
            # Track whether it was seen so we don't fabricate a bogus aggregate
            # ``0.0`` for a function name that produced no such key.
            saw_metric_name = False
            for result in example_results:
                if result is None:
                    continue
                if metric_name not in result.metrics:
                    continue
                saw_metric_name = True
                raw_value = result.metrics[metric_name]
                if raw_value is None:
                    continue
                try:
                    values.append(float(raw_value))
                except (TypeError, ValueError):
                    continue

            if values:
                if "p95" in metric_name.lower():
                    aggregated[metric_name] = self._compute_percentile(values, 0.95)
                else:
                    aggregated[metric_name] = sum(values) / len(values)
            elif saw_metric_name or not self.detailed:
                # Keep the legacy 0.0 sentinel when the name was present but
                # produced no usable numeric value, OR in non-detailed mode
                # (no per-example rows to inspect, so scalar metric functions
                # must retain their legacy 0.0 aggregate). In detailed mode a
                # name never seen at all -- e.g. a mapping function's own name,
                # whose sub-keys are aggregated elsewhere -- is NOT fabricated
                # as a bogus 0.0.
                aggregated[metric_name] = 0.0

        return aggregated

    async def evaluate(
        self,
        func: Callable[..., Any],
        config: dict[str, Any],
        dataset: Dataset,
        *,
        sample_lease: SampleBudgetLease | None = None,
        progress_callback: Callable[[int, dict[str, Any]], Any] | None = None,
        budget: ExecutionBudget | None = None,
    ) -> EvaluationResult:
        """Evaluate function with given configuration on dataset.

        Args:
            func: Function to evaluate
            config: Configuration parameters
            dataset: Evaluation dataset

        Returns:
            EvaluationResult with metrics and outputs

        Raises:
            EvaluationError: If evaluation fails
        """
        self.validate_function(func)
        self.validate_config(config)

        execution_budget_lease, blocked_result = (
            self._prepare_execution_budget_evaluation(budget, config)
        )
        if blocked_result is not None:
            return blocked_result
        if sample_lease is None:
            sample_lease = execution_budget_lease

        if not dataset.examples:
            raise EvaluationError("Dataset cannot be empty")

        self._warn_context_mode_metadata_defaulted_params(func, dataset)

        logger.info(
            f"Starting {'detailed ' if self.detailed else ''}evaluation with {len(dataset.examples)} examples, "
            f"config: {config}"
        )

        # Initialize metrics tracker
        metrics_tracker = MetricsTracker()
        metrics_tracker.start_tracking()

        start_time = time.time()

        # Use common batch evaluation method from BaseEvaluator
        (
            outputs,
            errors,
            example_results,
            consumed_examples,
            budget_exhausted,
        ) = await self._evaluate_batch(
            func,
            config,
            dataset,
            sample_lease=sample_lease,
            detailed=self.detailed,
            progress_callback=progress_callback,
        )

        # Collect expected outputs
        if consumed_examples:
            capped = min(consumed_examples, len(dataset.examples))
            expected_outputs = [
                dataset.examples[i].expected_output for i in range(capped)
            ]
        else:
            expected_outputs = []

        duration = time.time() - start_time
        metrics_tracker.end_tracking()

        # If detailed mode, compute individual example metrics first
        if self.detailed and example_results:
            for _i, ex_result in enumerate(example_results):
                if (
                    ex_result is not None
                    and ex_result.success
                    and ex_result.error_message is None
                ):
                    # Compute metrics for this specific example
                    example_metrics = self._compute_example_metrics(
                        ex_result.actual_output, ex_result.expected_output
                    )
                    # Update the ExampleResult with computed metrics
                    ex_result.metrics.update(example_metrics)

        # Get all captured LangChain responses before clearing
        all_captured_responses = get_all_captured_responses()
        logger.debug(
            f"Got {len(all_captured_responses)} captured LangChain responses for {len(outputs)} outputs"
        )

        # Clear captured responses now that we have them
        clear_captured_responses()

        # Structured degradation records for non-objective metric failures,
        # accumulated across this evaluate() call and surfaced on the
        # returned EvaluationResult (never silently coalesced into 0.0).
        # Kept as a local rather than instance state: LocalEvaluator instances
        # are shared across concurrent trials (parallel_trials), so instance
        # state here would race between trials.
        metric_errors: list[dict[str, Any]] = []

        # Track metrics for each example output using helper method
        for i, output in enumerate(outputs):
            example_metric = await self._process_single_output(
                output=output,
                index=i,
                config=config,
                dataset=dataset,
                errors=errors,
                example_results=example_results,
                all_captured_responses=all_captured_responses,
                progress_callback=progress_callback,
                metric_errors=metric_errors,
            )
            metrics_tracker.add_example_metrics(example_metric)

        # Compute aggregated metrics (include built-ins even with custom functions)
        if self.metric_functions:
            # Filter to registry/RAGAS metrics, but exclude those already provided
            # by custom metric_functions (e.g., skip built-in accuracy if
            # scoring_function provides it - avoids redundant computation).
            # Note: scoring_function maps to "accuracy" key only; cost/latency
            # built-ins remain unless explicitly overridden via metric_functions.
            custom_metric_keys = set(self.metric_functions.keys())
            base_metric_names = [
                name
                for name in self.metrics
                if (name in self._metric_registry or name in self._ragas_metric_names)
                and name not in custom_metric_keys
            ]
            if base_metric_names:
                aggregated_metrics = self.compute_metrics(
                    outputs,
                    expected_outputs,
                    errors,
                    dataset=dataset,
                    example_results=example_results,
                    example_metrics=metrics_tracker.example_metrics,
                    metrics_override=base_metric_names,
                    metric_errors=metric_errors,
                )
            else:
                aggregated_metrics = {}
        else:
            aggregated_metrics = self.compute_metrics(
                outputs,
                expected_outputs,
                errors,
                dataset=dataset,
                example_results=example_results,
                example_metrics=metrics_tracker.example_metrics,
                metric_errors=metric_errors,
            )

        # A custom scoring_function / metric_function OWNS the "accuracy"
        # objective when "accuracy" is both an objective and a custom metric
        # function. In that case the SDK's built-in exact-match scorer is a
        # DIAGNOSTIC, not the optimization signal, so it must not occupy
        # metrics["score"] (issue #1845).
        accuracy_is_custom_objective = (
            "accuracy" in self.metrics and "accuracy" in self.metric_functions
        )

        if "accuracy" in self.metrics:
            accuracy_value, _ = self._compute_accuracy_aggregated(outputs, dataset)
            if accuracy_value is not None:
                if accuracy_is_custom_objective:
                    # Keep the default exact-match value as a labelled
                    # diagnostic; the custom aggregate below owns "accuracy"
                    # and "score" (set to the objective value).
                    aggregated_metrics["exact_match_default"] = accuracy_value
                    self._log_dual_scorer_notice_once()
                else:
                    # No custom scorer: the default exact-match IS the objective,
                    # so it is correctly both "accuracy" and "score".
                    aggregated_metrics["accuracy"] = accuracy_value
                    aggregated_metrics.setdefault("score", accuracy_value)

        # Aggregate custom metric functions using helper
        if self.metric_functions:
            custom_aggregated = self._compute_aggregated_custom_metrics(example_results)
            aggregated_metrics.update(custom_aggregated)
            # metrics["score"] is the optimization signal. When a custom scorer
            # owns the objective, that signal is the objective value (the custom
            # scorer mean == metrics["accuracy"]), NOT the default exact-match
            # scorer (which now lives under "exact_match_default"). Issue #1845.
            if accuracy_is_custom_objective and "accuracy" in aggregated_metrics:
                aggregated_metrics["score"] = aggregated_metrics["accuracy"]

        # Add comprehensive metrics from tracker using helper. Thread the
        # evaluator's runtime-only computable names (registry + RAGAS) so a
        # user tuple key cannot overwrite an evaluator-computed value during the
        # tracker's user-metric aggregation pass.
        comprehensive_metrics = metrics_tracker.format_for_backend(
            extra_reserved=self._evaluator_computable_metric_names()
        )
        self._merge_comprehensive_metrics(aggregated_metrics, comprehensive_metrics)

        aggregated_metrics.setdefault("examples_attempted", len(outputs))

        # Empty/whitespace-output guard (issue #1851): compute the fraction of
        # this config's outputs that are empty on EVERY trial and expose it as a
        # reserved metric so portals/harnesses can gate on it, then surface ONE
        # run-level warning naming the offending config above the threshold. This
        # works for every calling pattern (decorator/wrapper/managed) because it
        # reads only the returned outputs — the metadata-free complement to the
        # finish_reason guard (#1809).
        empty_output_rate = compute_empty_output_rate(outputs)
        aggregated_metrics["empty_output_rate"] = empty_output_rate
        self._maybe_warn_high_empty_output_rate(empty_output_rate, config)

        # Authoritative cap on the FINAL trial metrics: user keys (arriving via
        # the per-example custom_metrics -> comprehensive_metrics merge above)
        # must not push the union past the MeasuresDict ceiling. Only user keys
        # are dropped; reserved evaluator keys are never sacrificed.
        enforce_user_metric_ceiling(
            aggregated_metrics,
            context="local trial aggregation",
            extra_reserved=self._evaluator_computable_metric_names(),
        )

        # Generate summary_stats (needed for insights)
        summary_stats = None
        if self.execution_mode_enum:
            summary_stats = metrics_tracker.format_as_summary_stats()
            logger.debug(f"Generated summary_stats for {self.execution_mode} mode")

        # Calculate success statistics
        if self.detailed and example_results:
            success_count = sum(
                1 for ex_result in example_results if ex_result and ex_result.success
            )
            total_count = len(example_results)
        else:
            success_count = sum(1 for error in errors if error is None)
            total_count = len(errors)

        # Log results
        logger.info(
            f"{'Detailed ' if self.detailed else ''}evaluation completed: {success_count}/{total_count} successful, "
            f"duration: {duration:.2f}s, metrics: {aggregated_metrics}"
        )
        if budget_exhausted:
            logger.info(
                "Sample budget exhausted after %s examples (dataset size=%s)",
                consumed_examples,
                len(dataset.examples),
            )

        result = EvaluationResult(
            config=config,
            example_results=example_results if self.detailed else [],
            aggregated_metrics=aggregated_metrics,
            total_examples=total_count,
            successful_examples=success_count,
            duration=duration,
            # Legacy compatibility fields
            metrics=aggregated_metrics,
            outputs=outputs,
            errors=errors,
            metric_errors=metric_errors,
        )

        # Attach summary_stats if generated
        if summary_stats:
            summary_stats.setdefault("metadata", {})
            summary_stats["metadata"]["sample_budget_exhausted"] = budget_exhausted
            summary_stats["metadata"]["examples_consumed"] = total_count
            result.summary_stats = summary_stats

        result.sample_budget_exhausted = budget_exhausted
        result.examples_consumed = consumed_examples

        return self._finalize_execution_budget_evaluation(
            budget, result, execution_budget_lease
        )

    def _compute_real_accuracy(self, actual_output: Any, expected_output: Any) -> float:
        """Compute real accuracy by comparing actual vs expected output.

        Args:
            actual_output: Output from function
            expected_output: Expected output

        Returns:
            Accuracy value (1.0 for match, 0.0 otherwise)
        """
        # Check for missing expected output
        if expected_output is None:
            return 0.0
        if isinstance(expected_output, str) and not expected_output.strip():
            return 0.0

        # If actual_output is dict, use its 'text' for accuracy comparison
        actual_to_compare = (
            actual_output.get("text")
            if isinstance(actual_output, dict)
            else actual_output
        )

        if _accuracy_values_match(actual_to_compare, expected_output):
            return 1.0

        return 0.0

    def _compute_example_metrics(
        self, actual_output: Any, expected_output: Any
    ) -> dict[str, float]:
        """Compute metrics for a single example.

        Args:
            actual_output: Output from function
            expected_output: Expected output

        Returns:
            Dictionary of metric names to values for this example
        """
        metrics = {}

        # SECURITY: Always compute real accuracy. The previous
        # TRAIGENT_MOCK_LLM-gated branch fabricated accuracy via
        # ``random.uniform()`` plus output-string heuristics, which would
        # silently replace real evaluation results with fake scores
        # whenever the env var was set in a production environment.
        if "accuracy" in self.metrics:
            metrics["accuracy"] = self._compute_real_accuracy(
                actual_output, expected_output
            )

        # Success (whether function completed without error)
        if "success_rate" in self.metrics:
            metrics["success"] = 1.0 if actual_output is not None else 0.0

        return metrics

    def compute_metrics(
        self,
        outputs: list[Any],
        expected_outputs: list[Any],
        errors: list[str | None],
        **context: Any,
    ) -> dict[str, float]:
        """Compute evaluation metrics.

        Args:
            outputs: Actual outputs from function
            expected_outputs: Expected outputs
            errors: Error messages (None for successful evaluations)

        Returns:
            Dictionary of metric name to value
        """
        # Use the base class implementation which now includes all default metrics
        return cast(
            dict[str, float],
            super().compute_metrics(outputs, expected_outputs, errors, **context),
        )
