"""Custom evaluator wrapper for TraiGent optimization system.

This module provides the CustomEvaluatorWrapper class that adapts user-provided
custom evaluation functions to conform to the BaseEvaluator interface.
"""

# Traceability: CONC-Layer-Core CONC-Quality-Reliability FUNC-EVAL-METRICS REQ-EVAL-005 SYNC-OptimizationFlow

from __future__ import annotations

import asyncio
import time
from typing import Any, Callable, cast

from traigent.api.types import ExampleResult
from traigent.core.utils import safe_get_nested_attr
from traigent.evaluators.base import (
    BaseEvaluator,
    Dataset,
    EvaluationResult,
    _maybe_restore_trial_context,
)
from traigent.utils.logging import get_logger

logger = get_logger(__name__)


class CustomEvaluatorWrapper(BaseEvaluator):
    """Wrapper for custom evaluation functions.

    This class wraps a user-provided custom evaluation function to conform
    to the BaseEvaluator interface. The custom evaluator should accept
    (func, config, example) and return an ExampleResult object.
    """

    def __init__(
        self,
        custom_evaluator: Callable[..., Any],
        metrics: list[str] | None = None,
        timeout: float = 60.0,
        capture_llm_metrics: bool = True,
        **kwargs: Any,
    ) -> None:
        """Initialize custom evaluator wrapper.

        Args:
            custom_evaluator: Custom evaluation function that takes (func, config, example)
                            and returns ExampleResult
            metrics: List of metric names to compute
            timeout: Timeout for individual evaluations (seconds)
            capture_llm_metrics: Whether to automatically capture LLM metrics (default: True)
            **kwargs: Additional configuration
        """
        super().__init__(metrics, timeout, **kwargs)
        self.custom_evaluator = custom_evaluator
        self.capture_llm_metrics = capture_llm_metrics

        # Import metrics tracking modules
        try:
            from traigent.evaluators.metrics_tracker import (
                MetricsTracker,
                extract_llm_metrics,
            )
            from traigent.utils.langchain_interceptor import (
                clear_captured_responses,
                get_all_captured_responses,
                patch_langchain_for_metadata_capture,
            )

            self._metrics_tracker = MetricsTracker()
            self._extract_llm_metrics = extract_llm_metrics
            self._clear_captured_responses = clear_captured_responses
            self._get_all_captured_responses = get_all_captured_responses
            # Ensure LangChain patch is applied
            patch_langchain_for_metadata_capture()
            self._metrics_available = True
        except ImportError:
            logger.warning(
                "Metrics tracking modules not available, LLM metrics capture disabled"
            )
            self._metrics_available = False
            self.capture_llm_metrics = False

    def _extract_model_from_response(
        self, response: Any, config: dict[str, Any]
    ) -> str | None:
        """Extract model name from response or config.

        Args:
            response: The captured response object
            config: Configuration dictionary

        Returns:
            Model name or None
        """
        model_name: str | None = config.get("model")
        if model_name:
            return model_name

        if not response:
            return None

        # Check for model in response metadata (LangChain pattern)
        if hasattr(response, "response_metadata") and isinstance(
            response.response_metadata, dict
        ):
            model_name = response.response_metadata.get("model")
            if not model_name:
                model_name = response.response_metadata.get("model_name")

        # Check for model attribute directly
        if not model_name and hasattr(response, "model"):
            model_name = cast(str | None, response.model)

        # Check for model_name attribute
        if not model_name and hasattr(response, "model_name"):
            model_name = cast(str | None, response.model_name)

        return model_name

    def _reconstruct_original_prompt(self, example: Any) -> list[dict[str, str]]:
        """Reconstruct original prompt from example input.

        Args:
            example: The evaluation example

        Returns:
            List of message dictionaries
        """
        if isinstance(example.input_data, dict):
            if "messages" in example.input_data:
                return cast(list[dict[str, str]], example.input_data.get("messages"))
            elif "text" in example.input_data:
                return [{"role": "user", "content": example.input_data["text"]}]
            else:
                return [{"role": "user", "content": str(example.input_data)}]
        return [{"role": "user", "content": str(example.input_data)}]

    def _extract_response_text(self, output: Any) -> str | None:
        """Extract text content from output.

        Args:
            output: The output from evaluation

        Returns:
            Extracted text string or None
        """
        if not output:
            return None
        if isinstance(output, str):
            return output
        if hasattr(output, "text"):
            return cast(str | None, output.text)
        if hasattr(output, "content"):
            if isinstance(output.content, str):
                return output.content
            if isinstance(output.content, list) and output.content:
                if hasattr(output.content[0], "text"):
                    return cast(str | None, output.content[0].text)
        return None

    def _capture_llm_metrics_for_example(
        self,
        config: dict[str, Any],
        example: Any,
        example_result: ExampleResult,
        example_index: int,
    ) -> Any | None:
        """Capture LLM metrics after evaluation.

        Args:
            config: Configuration dictionary
            example: The evaluation example
            example_result: The result from custom evaluator
            example_index: Index of the example

        Returns:
            LLM metrics object or None
        """
        if not (self.capture_llm_metrics and self._metrics_available):
            return None

        captured_responses = self._get_all_captured_responses()
        if not captured_responses:
            return None

        response = captured_responses[0]
        if not response:
            return None

        model_name = self._extract_model_from_response(response, config)
        original_prompt = self._reconstruct_original_prompt(example)
        response_text = self._extract_response_text(example_result.actual_output)

        llm_metrics = self._extract_llm_metrics(
            response=response,
            model_name=model_name,
            original_prompt=original_prompt,
            response_text=response_text,
        )

        if llm_metrics:
            logger.debug(
                f"Captured LLM metrics for example {example_index}: "
                f"tokens={llm_metrics.tokens.total_tokens}, "
                f"cost=${llm_metrics.cost.total_cost:.8f}"
            )

        return llm_metrics

    def _enhance_result_with_llm_metrics(
        self,
        example_result: ExampleResult,
        llm_metrics: Any,
        per_example_duration: float,
    ) -> None:
        """Enhance example result with captured LLM metrics.

        Args:
            example_result: The result to enhance (modified in place)
            llm_metrics: The captured LLM metrics object
            per_example_duration: Measured duration for this example
        """
        # Ensure metadata container exists
        if getattr(example_result, "metadata", None) is None:
            example_result.metadata = {}

        if llm_metrics:
            if example_result.metrics is None:
                example_result.metrics = {}

            # Add token and cost metrics
            example_result.metrics["input_tokens"] = safe_get_nested_attr(
                llm_metrics, "tokens.input_tokens", 0
            )
            example_result.metrics["output_tokens"] = safe_get_nested_attr(
                llm_metrics, "tokens.output_tokens", 0
            )
            example_result.metrics["total_tokens"] = safe_get_nested_attr(
                llm_metrics, "tokens.total_tokens", 0
            )
            example_result.metrics["input_cost"] = safe_get_nested_attr(
                llm_metrics, "cost.input_cost", 0.0
            )
            example_result.metrics["output_cost"] = safe_get_nested_attr(
                llm_metrics, "cost.output_cost", 0.0
            )
            example_result.metrics["total_cost"] = safe_get_nested_attr(
                llm_metrics, "cost.total_cost", 0.0
            )

            response_time_ms = safe_get_nested_attr(
                llm_metrics, "response.response_time_ms", 0
            )
            if response_time_ms > 0:
                model_response_time = response_time_ms / 1000.0
                example_result.metrics["model_response_time"] = model_response_time
                example_result.metadata["model_response_time"] = model_response_time

        # Track total function duration
        if per_example_duration is not None:
            example_result.metrics["function_duration"] = per_example_duration
            example_result.metadata["function_duration"] = per_example_duration

        # Set execution time if not already set
        current_execution_time = getattr(example_result, "execution_time", 0.0)
        if not current_execution_time:
            example_result.execution_time = per_example_duration
        else:
            example_result.metadata.setdefault(
                "function_duration", current_execution_time
            )

    def _create_failed_example_result(
        self, example_index: int, example: Any, error: Exception
    ) -> ExampleResult:
        """Create an ExampleResult for a failed evaluation.

        Args:
            example_index: Index of the example
            example: The example that failed
            error: The exception that occurred

        Returns:
            ExampleResult with failure information
        """
        return ExampleResult(
            example_id=f"example_{example_index}",
            input_data=example.input_data,
            expected_output=example.expected_output,
            actual_output=None,
            metrics=dict.fromkeys(self.metrics, 0.0),
            execution_time=0.0,
            success=False,
            error_message=str(error),
        )

    def _aggregate_custom_metrics(
        self, all_metrics: list[dict[str, Any]]
    ) -> dict[str, float]:
        """Aggregate custom metrics across all examples.

        Args:
            all_metrics: List of metric dictionaries from each example

        Returns:
            Dictionary of aggregated metrics
        """
        aggregated = {}
        for metric in self.metrics:
            metric_values = [m.get(metric, 0.0) for m in all_metrics if m]
            if metric_values:
                aggregated[metric] = sum(metric_values) / len(metric_values)
            else:
                aggregated[metric] = 0.0
        return aggregated

    def _aggregate_llm_metrics(
        self, all_metrics: list[dict[str, Any]], example_results: list[ExampleResult]
    ) -> dict[str, float]:
        """Aggregate LLM metrics across all examples.

        Args:
            all_metrics: List of metric dictionaries from each example
            example_results: List of ExampleResult objects

        Returns:
            Dictionary of aggregated LLM metrics
        """
        aggregated = {}

        # Aggregate token metrics (total, not average)
        token_metrics = ["input_tokens", "output_tokens", "total_tokens"]
        for metric in token_metrics:
            metric_values = [
                m.get(metric, 0.0) for m in all_metrics if m and metric in m
            ]
            aggregated[metric] = sum(metric_values) if metric_values else 0.0

        # Aggregate cost metrics (total, not average)
        cost_metrics = ["input_cost", "output_cost", "total_cost"]
        for metric in cost_metrics:
            metric_values = [
                m.get(metric, 0.0) for m in all_metrics if m and metric in m
            ]
            aggregated[metric] = sum(metric_values) if metric_values else 0.0

        # Calculate average response time
        response_times = [
            r.execution_time
            for r in example_results
            if hasattr(r, "execution_time") and r.execution_time > 0
        ]
        if response_times:
            aggregated["avg_response_time"] = sum(response_times) / len(response_times)

        return aggregated

    async def evaluate(
        self,
        func: Callable[..., Any],
        config: dict[str, Any],
        dataset: Dataset,
        *,
        progress_callback: Callable[[int, dict[str, Any]], Any] | None = None,
    ) -> EvaluationResult:
        """Evaluate function using custom evaluator.

        Args:
            func: Function to evaluate
            config: Configuration parameters
            dataset: Evaluation dataset

        Returns:
            EvaluationResult with metrics and outputs

        Raises:
            EvaluationError: If evaluation fails
        """
        logger.info(
            f"Starting custom evaluation with {len(dataset.examples)} examples, config: {config}"
        )

        start_time = time.time()

        # Initialize metrics tracker if capturing LLM metrics
        if self.capture_llm_metrics and self._metrics_available:
            self._metrics_tracker.start_tracking()

        from traigent.config.context import ConfigurationContext, get_trial_context

        trial_ctx = get_trial_context()

        # Evaluate function on all examples using custom evaluator
        example_results = []
        all_metrics = []
        outputs = []
        errors = []

        for i, example in enumerate(dataset.examples):
            try:
                # Clear any previous captured responses before evaluation
                if self.capture_llm_metrics and self._metrics_available:
                    self._clear_captured_responses()

                # Set configuration context and call custom evaluator
                with _maybe_restore_trial_context(trial_ctx):
                    with ConfigurationContext(config):
                        per_example_start = time.time()
                        if asyncio.iscoroutinefunction(self.custom_evaluator):
                            example_result = await self.custom_evaluator(
                                func, config, example
                            )
                        else:
                            example_result = self.custom_evaluator(
                                func, config, example
                            )
                        per_example_duration = time.time() - per_example_start

                # Ensure we got an ExampleResult
                if not isinstance(example_result, ExampleResult):
                    raise ValueError(
                        f"Custom evaluator must return ExampleResult, got {type(example_result)}"
                    )

                # Capture and enhance with LLM metrics
                llm_metrics = self._capture_llm_metrics_for_example(
                    config, example, example_result, i
                )
                self._enhance_result_with_llm_metrics(
                    example_result, llm_metrics, per_example_duration
                )

                example_results.append(example_result)
                if progress_callback:
                    progress_callback(
                        i,
                        {
                            "success": example_result.success,
                            "error": example_result.error_message,
                            "metrics": example_result.metrics,
                            "output": example_result.actual_output,
                        },
                    )
                if example_result.metrics:
                    logger.debug(
                        f"CustomEvaluatorWrapper: Example {i} returned metrics: {example_result.metrics}"
                    )
                all_metrics.append(example_result.metrics)
                outputs.append(example_result.actual_output)
                errors.append(example_result.error_message)

            except Exception as e:
                logger.warning(f"Custom evaluation failed for example {i}: {e}")
                failed_result = self._create_failed_example_result(i, example, e)
                example_results.append(failed_result)
                if progress_callback:
                    progress_callback(
                        i,
                        {
                            "success": False,
                            "error": str(e),
                            "metrics": failed_result.metrics,
                            "output": None,
                        },
                    )
                all_metrics.append(failed_result.metrics)
                outputs.append(None)
                errors.append(str(e))

        duration = time.time() - start_time

        # End metrics tracking if it was started
        if self.capture_llm_metrics and self._metrics_available:
            self._metrics_tracker.end_tracking()

        # Aggregate metrics across all examples
        aggregated_metrics = self._aggregate_custom_metrics(all_metrics)

        # Add LLM metrics aggregation if captured
        if self.capture_llm_metrics and self._metrics_available:
            llm_agg = self._aggregate_llm_metrics(all_metrics, example_results)
            aggregated_metrics.update(llm_agg)

        # Log results
        success_count = sum(1 for result in example_results if result.success)
        logger.info(
            f"Custom evaluation completed: {success_count}/{len(example_results)} successful, "
            f"duration: {duration:.2f}s, metrics: {aggregated_metrics}"
        )

        return EvaluationResult(
            config=config,
            example_results=example_results,
            aggregated_metrics=aggregated_metrics,
            total_examples=len(example_results),
            successful_examples=success_count,
            duration=duration,
            metrics=aggregated_metrics,
            outputs=outputs,
            errors=errors,
        )
