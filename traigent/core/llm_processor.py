"""LLM response processing utilities for Traigent optimization system.

This module provides utilities for processing LLM responses, extracting metrics,
and handling response parsing logic.
"""

# Traceability: CONC-Layer-Core CONC-Quality-Performance FUNC-EVAL-METRICS REQ-EVAL-005 SYNC-OptimizationFlow

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any, cast

from traigent.core.types_ext import LLMMetrics
from traigent.core.utils import safe_get_nested_attr
from traigent.utils.logging import get_logger

logger = get_logger(__name__)


class LLMResponseProcessor:
    """Processor for LLM responses and metrics extraction.

    This class handles the processing of LLM responses, extracting relevant
    metrics and parsing response content for evaluation purposes.
    """

    def __init__(self) -> None:
        """Initialize the LLM response processor."""
        self._metrics_extractors: list[Callable[..., Any]] = []
        self._response_parsers: list[Callable[..., Any]] = []

    def extract_model_name(
        self, response: Any, config_model: str | None = None
    ) -> str | None:
        """Extract model name from response or config.

        Args:
            response: LLM response object
            config_model: Model name from configuration

        Returns:
            Model name or None if not found
        """
        # First try to get model from config (for backward compatibility)
        model_name = config_model

        # Try to extract model from the response itself
        if not model_name and response:
            # Check for model in response metadata (LangChain pattern)
            if hasattr(response, "response_metadata") and isinstance(
                response.response_metadata, dict
            ):
                # OpenAI responses often have model in response_metadata
                model_name = response.response_metadata.get("model")
                if not model_name:
                    # Sometimes it's nested under model_name
                    model_name = response.response_metadata.get("model_name")

            # Check for model attribute directly
            if not model_name and hasattr(response, "model"):
                model_name = response.model

            # Check for model_name attribute
            if not model_name and hasattr(response, "model_name"):
                model_name = response.model_name

        return model_name

    def reconstruct_prompt(self, input_data: Any) -> list[dict[str, str]] | None:
        """Reconstruct original prompt from example input data.

        Args:
            input_data: Input data from evaluation example

        Returns:
            Reconstructed prompt as list of messages or None
        """
        if isinstance(input_data, dict):
            if "messages" in input_data:
                return input_data.get("messages")
            elif "text" in input_data:
                return [{"role": "user", "content": input_data["text"]}]
            else:
                return [{"role": "user", "content": str(input_data)}]
        else:
            return [{"role": "user", "content": str(input_data)}]

    def extract_response_text(self, output: Any) -> str | None:
        """Extract response text from various output formats.

        Args:
            output: Output from LLM response

        Returns:
            Extracted response text or None
        """
        if output is None:
            return None

        if isinstance(output, str):
            return output

        # Try different attribute patterns
        if hasattr(output, "text"):
            return cast(str | None, output.text)

        if hasattr(output, "content"):
            if isinstance(output.content, str):
                return output.content
            elif isinstance(output.content, list) and output.content:
                if hasattr(output.content[0], "text"):
                    return cast(str | None, output.content[0].text)

        return None

    def extract_llm_metrics(
        self,
        response: Any,
        model_name: str | None = None,
        original_prompt: list[dict[str, str]] | None = None,
        response_text: str | None = None,
    ) -> LLMMetrics | None:
        """Extract LLM metrics from response using available extractors.

        Args:
            response: LLM response object
            model_name: Name of the model used
            original_prompt: Original prompt sent to LLM
            response_text: Response text content

        Returns:
            LLMMetrics object or None if extraction fails
        """
        try:
            # Try to use the configured metrics extractor
            if hasattr(self, "_extract_llm_metrics_func"):
                return cast(
                    LLMMetrics | None,
                    self._extract_llm_metrics_func(
                        response=response,
                        model_name=model_name,
                        original_prompt=original_prompt,
                        response_text=response_text,
                    ),
                )

            # Fallback: try to extract metrics directly from response
            return self._extract_metrics_from_response(response, model_name)

        except Exception as e:
            logger.debug(f"Failed to extract LLM metrics: {e}")
            return None

    def _extract_metrics_from_response(
        self, response: Any, model_name: str | None
    ) -> LLMMetrics | None:
        """Extract metrics directly from response object.

        Args:
            response: LLM response object
            model_name: Name of the model used

        Returns:
            LLMMetrics object or None
        """
        try:
            # Extract token information
            input_tokens = safe_get_nested_attr(response, "usage.prompt_tokens", 0)
            output_tokens = safe_get_nested_attr(response, "usage.completion_tokens", 0)
            total_tokens = safe_get_nested_attr(
                response, "usage.total_tokens", input_tokens + output_tokens
            )

            # Extract cost information (if available)
            input_cost = safe_get_nested_attr(response, "cost.input_cost", 0.0)
            output_cost = safe_get_nested_attr(response, "cost.output_cost", 0.0)
            total_cost = safe_get_nested_attr(
                response, "cost.total_cost", input_cost + output_cost
            )

            # Extract response time
            response_time_ms = safe_get_nested_attr(response, "response_time_ms", 0)
            if response_time_ms == 0:
                response_time_ms = safe_get_nested_attr(
                    response, "metrics.response_time_ms", 0
                )

            # Create LLMMetrics object
            return LLMMetrics(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
                input_cost=input_cost,
                output_cost=output_cost,
                total_cost=total_cost,
                response_time_ms=response_time_ms,
                model_name=model_name or "unknown",
            )

        except Exception as e:
            logger.debug(f"Failed to extract metrics from response: {e}")
            return None

    def enhance_example_result(
        self, example_result: Any, llm_metrics: LLMMetrics | None, execution_time: float
    ) -> None:
        """Enhance example result with LLM metrics.

        Args:
            example_result: ExampleResult object to enhance
            llm_metrics: LLM metrics to add
            execution_time: Measured execution time
        """
        if llm_metrics:
            # Add LLM metrics to the metrics dictionary
            if example_result.metrics is None:
                example_result.metrics = {}

            # Add token and cost metrics
            # Use 'input_tokens' and 'output_tokens' to match backend expectations
            example_result.metrics["input_tokens"] = llm_metrics.get("input_tokens", 0)
            example_result.metrics["output_tokens"] = llm_metrics.get(
                "output_tokens", 0
            )
            example_result.metrics["total_tokens"] = llm_metrics.get("total_tokens", 0)
            example_result.metrics["input_cost"] = llm_metrics.get("input_cost", 0.0)
            example_result.metrics["output_cost"] = llm_metrics.get("output_cost", 0.0)
            example_result.metrics["total_cost"] = llm_metrics.get("total_cost", 0.0)

            # Always update execution time when we have actual metrics
            # This ensures we use real response times even if custom evaluator hardcoded a value
            response_time_ms = llm_metrics.get("response_time_ms", 0)
            if response_time_ms > 0:
                example_result.execution_time = response_time_ms / 1000.0

        # Only treat numeric zero as "unset"; preserve custom non-numeric sentinels.
        # If no response-time from metrics and evaluator didn't set it, use measured duration.
        if not hasattr(example_result, "execution_time") or (
            isinstance(example_result.execution_time, (int, float))
            and math.isclose(example_result.execution_time, 0.0, abs_tol=1e-9)
        ):
            example_result.execution_time = execution_time

    def validate_response_format(self, response: Any) -> bool:
        """Validate that response has expected format.

        Args:
            response: Response to validate

        Returns:
            True if response format is valid
        """
        if response is None:
            return False

        # Check for basic response attributes
        required_attrs = ["content", "usage"]
        for attr in required_attrs:
            if not hasattr(response, attr):
                return False

        return True

    def set_metrics_extractor(self, extractor_func: Callable[..., Any]) -> None:
        """Set the metrics extraction function.

        Args:
            extractor_func: Function to extract LLM metrics
        """
        self._extract_llm_metrics_func = extractor_func

    def add_response_parser(self, parser_func: Callable[..., Any]) -> None:
        """Add a response parser function.

        Args:
            parser_func: Function to parse response content
        """
        self._response_parsers.append(parser_func)

    def parse_response_content(self, response: Any) -> dict[str, Any]:
        """Parse response content using registered parsers.

        Args:
            response: Response to parse

        Returns:
            Parsed content dictionary
        """
        parsed_content = {}

        for parser in self._response_parsers:
            try:
                result = parser(response)
                if result:
                    parsed_content.update(result)
            except Exception as e:
                logger.debug(f"Response parser failed: {e}")

        return parsed_content


# Global instance for convenience
_default_processor = LLMResponseProcessor()


def get_default_processor() -> LLMResponseProcessor:
    """Get the default LLM response processor instance.

    Returns:
        Default processor instance
    """
    return _default_processor


def extract_model_name(response: Any, config_model: str | None = None) -> str | None:
    """Convenience function to extract model name.

    Args:
        response: LLM response object
        config_model: Model name from configuration

    Returns:
        Model name or None
    """
    return _default_processor.extract_model_name(response, config_model)


def extract_response_text(output: Any) -> str | None:
    """Convenience function to extract response text.

    Args:
        output: Output from LLM response

    Returns:
        Extracted response text
    """
    return _default_processor.extract_response_text(output)
