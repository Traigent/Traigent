"""Local evaluation strategy."""

# Traceability: CONC-Layer-Core CONC-Quality-Reliability CONC-Quality-Performance FUNC-EVAL-METRICS REQ-EVAL-005 SYNC-OptimizationFlow

from __future__ import annotations

import inspect
import math
import random
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

from traigent.config.types import ExecutionMode, resolve_execution_mode
from traigent.evaluators.base import BaseEvaluator, Dataset, EvaluationResult
from traigent.evaluators.metrics_tracker import (
    ExampleMetrics,
    MetricsTracker,
    extract_llm_metrics,
)
from traigent.utils.exceptions import EvaluationError
from traigent.utils.langchain_interceptor import (
    clear_captured_responses,
    get_all_captured_responses,
    get_captured_response_by_key,
    patch_langchain_for_metadata_capture,
)
from traigent.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class PromptInfo:
    """Extracted prompt information for metrics calculation."""

    original_prompt: list[dict[str, str]] | None
    prompt_length: int | None
    response_length: int | None


# Apply LangChain patch on module import
patch_langchain_for_metadata_capture()

if TYPE_CHECKING:
    from traigent.core.sample_budget import SampleBudgetLease


class LocalEvaluator(BaseEvaluator):
    """Local evaluation strategy.

    Evaluates functions locally in the current process. Supports both
    synchronous and asynchronous functions with timeout handling.

    Example:
        >>> evaluator = LocalEvaluator(["accuracy"], timeout=30.0)
        >>> result = await evaluator.evaluate(my_function, config, dataset)
    """

    def __init__(
        self,
        metrics: list[str] | None = None,
        timeout: float = 60.0,
        max_workers: int = 1,
        detailed: bool = False,
        execution_mode: str | None = None,
        privacy_enabled: bool = False,
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
            execution_mode: Execution mode (privacy, edge_analytics, cloud) for determining submission format
            **kwargs: Additional configuration
        """
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
        self.privacy_enabled = privacy_enabled
        self.mock_mode_config = mock_mode_config or {}
        self.metric_functions = metric_functions or {}
        self._mock_mode_warning_shown = (
            False  # Track if we've shown the mock mode warning
        )

        # Create seeded random instance for mock mode reproducibility
        self._mock_random = random.Random()
        if self.mock_mode_config.get("random_seed") is not None:
            self._mock_random.seed(self.mock_mode_config["random_seed"])

    def _extract_prompt_info(
        self,
        example_input: Any,
        output: Any,
    ) -> PromptInfo:
        """Extract prompt information for metrics calculation.

        Handles both privacy mode (storing lengths only) and normal mode
        (reconstructing full prompts).

        Args:
            example_input: Input data from dataset example
            output: Output from function call

        Returns:
            PromptInfo with prompt data or lengths based on privacy mode
        """
        prompt_length: int | None = None
        response_length: int | None = None
        original_prompt: list[dict[str, str]] | None = None

        if self.privacy_enabled:
            # In privacy mode, store lengths for cost calculation
            prompt_length = self._calculate_input_length(example_input)
            response_length = self._extract_response_length(output)
            original_prompt = None
        else:
            # Non-privacy mode: Reconstruct original prompt from dataset example
            original_prompt = self._reconstruct_prompt(example_input)

        return PromptInfo(
            original_prompt=original_prompt,
            prompt_length=prompt_length,
            response_length=response_length,
        )

    def _calculate_input_length(self, example_input: Any) -> int:
        """Calculate the character length of input data."""
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
        return len(response_text) if response_text else None

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
        3. Else use captured response by index
        4. Else fall back to output itself

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
            ex_key = (
                ex.metadata.get("example_id", f"example_{example_index}")
                if ex.metadata
                else f"example_{example_index}"
            )
        except Exception:
            ex_key = f"example_{example_index}"

        by_key = get_captured_response_by_key(ex_key)
        if by_key is not None:
            return by_key

        # Fall back to index-based or output itself
        if example_index < len(all_captured_responses):
            return all_captured_responses[example_index]

        return output

    def _estimate_string_tokens(
        self,
        example_metric: ExampleMetrics,
        output: str,
        example_input: Any,
    ) -> None:
        """Estimate token counts for string outputs.

        Updates example_metric in place with estimated token counts.
        Uses approximation of 1 token per 4 characters.

        Args:
            example_metric: Metrics object to update
            output: String output from function
            example_input: Original input for estimating input tokens
        """
        # Estimate output tokens
        example_metric.tokens.output_tokens = max(1, len(output) // 4)

        # Estimate input tokens if not in privacy mode
        if not self.privacy_enabled:
            input_text = str(example_input)
            example_metric.tokens.input_tokens = max(1, len(input_text) // 4)

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
            return (
                1.0
                if actual_value.strip().lower() == expected_output.strip().lower()
                else 0.0
            )
        return 1.0 if actual_value == expected_output else 0.0

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

    def _apply_custom_metric_functions(
        self,
        example_metric: ExampleMetrics,
        output: Any,
        example_obj: Any,
        config: dict[str, Any],
        example_index: int,
    ) -> None:
        """Apply user-defined metric functions for a single example.

        Updates example_metric.custom_metrics in place.

        Args:
            example_metric: Metrics object to update
            output: Function output
            example_obj: Dataset example object
            config: Configuration parameters
            example_index: Current example index
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

        for metric_name, metric_func in self.metric_functions.items():
            value = self._invoke_metric_function(
                metric_func,
                metric_name,
                output,
                example_obj,
                config,
                llm_payload,
                example_index,
            )
            example_metric.custom_metrics[metric_name] = (
                float(value) if value is not None else 0.0
            )

    def _invoke_metric_function(
        self,
        metric_func: Callable[..., Any],
        metric_name: str,
        output: Any,
        example_obj: Any,
        config: dict[str, Any],
        llm_payload: dict[str, Any],
        example_index: int,
    ) -> float:
        """Invoke a single metric function with appropriate arguments.

        Args:
            metric_func: The metric function to call
            metric_name: Name of the metric (for logging)
            output: Function output
            example_obj: Dataset example
            config: Configuration parameters
            llm_payload: LLM metrics payload
            example_index: Current example index

        Returns:
            Metric value or 0.0 on failure
        """
        try:
            params = inspect.signature(metric_func).parameters.keys()
            kwargs: dict[str, Any] = {}

            if "output" in params or "actual" in params:
                kwargs["output"] = output
            if "expected" in params:
                kwargs["expected"] = example_obj.expected_output
            if "example" in params:
                kwargs["example"] = example_obj
            if "input_data" in params:
                kwargs["input_data"] = example_obj.input_data
            if "metadata" in params:
                kwargs["metadata"] = example_obj.metadata or {}
            if "config" in params:
                kwargs["config"] = config
            if "llm_metrics" in params:
                kwargs["llm_metrics"] = llm_payload
            if "example_index" in params:
                kwargs["example_index"] = example_index

            return cast(float, metric_func(**kwargs))
        except Exception as exc:
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
            return 0.0

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

        model_name = config.get("model")
        logger.debug(
            f"EVALUATOR DEBUG: i={index}, output type={type(output).__name__}, "
            f"model_name from config='{model_name}'"
        )

        original_prompt = None
        prompt_length = None
        response_length = None

        # Extract prompt info using helper method
        if index < len(dataset.examples):
            example_input = dataset.examples[index].input_data
            prompt_info = self._extract_prompt_info(example_input, output)
            original_prompt = prompt_info.original_prompt
            prompt_length = prompt_info.prompt_length
            response_length = prompt_info.response_length

        # Determine response text for scoring logic
        response_text = self._extract_response_text(output)
        if self.privacy_enabled and response_text and response_length is None:
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
            response_text=response_text,
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
                example_metric, output, example_input_for_tokens
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

    def _process_single_output(
        self,
        output: Any,
        index: int,
        config: dict[str, Any],
        dataset: Dataset,
        errors: list[str | None],
        example_results: list[Any] | None,
        all_captured_responses: list[Any],
        progress_callback: Callable[[int, dict[str, Any]], Any] | None,
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

        Returns:
            ExampleMetrics for this output
        """
        # Extract LLM metrics
        example_metric = self._extract_llm_metrics_for_output(
            output, index, config, dataset, all_captured_responses
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
            self._apply_custom_metric_functions(
                example_metric, output, example_obj, config, index
            )

            # Transfer custom metrics to example_results if in detailed mode
            if (
                self.detailed
                and example_results
                and index < len(example_results)
                and example_results[index] is not None
            ):
                for metric_name in self.metric_functions:
                    example_results[index].metrics[metric_name] = (
                        example_metric.custom_metrics[metric_name]
                    )

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
            if isinstance(value, str) and isinstance(expected, str):
                if value.strip().lower() == expected.strip().lower():
                    correct += 1
            elif value == expected:
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
            if (
                aggregated_metrics.get("cost", 0.0) == 0.0
                and comprehensive_metrics["cost"] != 0.0
            ):
                logger.info(
                    f"🔍 LOCAL EVALUATOR: Overriding cost metric: "
                    f"{aggregated_metrics.get('cost', 0.0)} -> {comprehensive_metrics['cost']}"
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
            for result in example_results:
                if result is None:
                    continue
                raw_value = result.metrics.get(metric_name)
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
            else:
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

        if not dataset.examples:
            raise EvaluationError("Dataset cannot be empty")

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

        # Track metrics for each example output using helper method
        for i, output in enumerate(outputs):
            example_metric = self._process_single_output(
                output=output,
                index=i,
                config=config,
                dataset=dataset,
                errors=errors,
                example_results=example_results,
                all_captured_responses=all_captured_responses,
                progress_callback=progress_callback,
            )
            metrics_tracker.add_example_metrics(example_metric)

        # Compute aggregated metrics
        if self.metric_functions:
            aggregated_metrics: dict[str, float] = {}
        else:
            aggregated_metrics = self.compute_metrics(
                outputs,
                expected_outputs,
                errors,
                dataset=dataset,
                example_results=example_results,
                example_metrics=metrics_tracker.example_metrics,
            )

        if "accuracy" in self.metrics:
            accuracy_value, _ = self._compute_accuracy_aggregated(outputs, dataset)
            if accuracy_value is not None:
                aggregated_metrics["accuracy"] = accuracy_value
                aggregated_metrics.setdefault("score", accuracy_value)

        # Aggregate custom metric functions using helper
        if self.metric_functions:
            custom_aggregated = self._compute_aggregated_custom_metrics(example_results)
            aggregated_metrics.update(custom_aggregated)

        # Add comprehensive metrics from tracker using helper
        comprehensive_metrics = metrics_tracker.format_for_backend()
        self._merge_comprehensive_metrics(aggregated_metrics, comprehensive_metrics)

        aggregated_metrics.setdefault("examples_attempted", len(outputs))

        # Generate summary_stats for all modes except CLOUD (needed for insights)
        summary_stats = None
        if (
            self.execution_mode_enum
            and self.execution_mode_enum is not ExecutionMode.CLOUD
        ):
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
        )

        # Attach summary_stats if generated
        if summary_stats:
            summary_stats.setdefault("metadata", {})
            summary_stats["metadata"]["sample_budget_exhausted"] = budget_exhausted
            summary_stats["metadata"]["examples_consumed"] = total_count
            result.summary_stats = summary_stats

        result.sample_budget_exhausted = budget_exhausted
        result.examples_consumed = consumed_examples

        return result

    def _compute_mock_accuracy(
        self, actual_output: Any, base_accuracy: float, variance: float
    ) -> float:
        """Compute simulated accuracy for mock mode.

        Args:
            actual_output: Output from function
            base_accuracy: Base accuracy value (e.g., 0.75)
            variance: Variance range for randomization

        Returns:
            Simulated accuracy value between 0.0 and 1.0
        """
        if actual_output is None:
            return 0.0

        adjusted_accuracy = base_accuracy
        # Simulate better accuracy for certain outputs
        if isinstance(actual_output, str):
            if len(actual_output) > 20:  # Longer outputs generally better
                adjusted_accuracy += 0.05
            sentiment_words = ["positive", "negative", "neutral"]
            if any(word in actual_output.lower() for word in sentiment_words):
                adjusted_accuracy += 0.03  # Sentiment-like outputs

        # Add random variance (using seeded random for reproducibility)
        half_variance = variance / 2
        raw_accuracy = adjusted_accuracy + self._mock_random.uniform(
            -half_variance, half_variance
        )
        return min(1.0, max(0.0, raw_accuracy))

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

        # Exact match
        if actual_to_compare == expected_output:
            return 1.0

        # Try case-insensitive comparison for string outputs
        if isinstance(actual_to_compare, str) and isinstance(expected_output, str):
            if actual_to_compare.lower().strip() == expected_output.lower().strip():
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
        import os

        metrics = {}

        # Check if we're in mock mode and if it should be applied
        mock_mode_env = os.environ.get("TRAIGENT_MOCK_MODE", "").lower() == "true"

        # Check configuration for mock mode settings
        mock_enabled = self.mock_mode_config.get("enabled", True)
        override_evaluator = self.mock_mode_config.get("override_evaluator", True)
        base_accuracy_config = self.mock_mode_config.get("base_accuracy", 0.75)
        variance_config = self.mock_mode_config.get("variance", 0.25)

        # Determine if we should actually use mock mode
        use_mock = mock_mode_env and mock_enabled and override_evaluator

        # Accuracy (exact match or mock)
        if "accuracy" in self.metrics:
            if use_mock:
                self._log_mock_mode_warning(base_accuracy_config, variance_config)
                metrics["accuracy"] = self._compute_mock_accuracy(
                    actual_output, base_accuracy_config, variance_config
                )
            else:
                metrics["accuracy"] = self._compute_real_accuracy(
                    actual_output, expected_output
                )

        # Success (whether function completed without error)
        if "success_rate" in self.metrics:
            metrics["success"] = 1.0 if actual_output is not None else 0.0

        return metrics

    def _log_mock_mode_warning(self, base_accuracy: float, variance: float) -> None:
        """Log a one-time warning about mock mode accuracy."""
        if not self._mock_mode_warning_shown:
            logger.warning(
                "MOCK MODE ACTIVE: Accuracy metrics are SIMULATED (base=%.2f ± %.2f), "
                "not computed from actual vs expected outputs. These metrics do not "
                "reflect real model performance. Set TRAIGENT_MOCK_MODE=false for "
                "real evaluations.",
                base_accuracy,
                variance / 2,
            )
            self._mock_mode_warning_shown = True

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
        return super().compute_metrics(outputs, expected_outputs, errors, **context)
