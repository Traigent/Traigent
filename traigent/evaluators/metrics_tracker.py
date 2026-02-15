"""Enhanced metrics tracking for comprehensive evaluation metrics."""

# Traceability: CONC-Layer-Core CONC-Quality-Reliability CONC-Quality-Observability FUNC-EVAL-METRICS REQ-EVAL-005 SYNC-OptimizationFlow

import os
import statistics
import time
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Optional, cast

from traigent._version import get_version
from traigent.utils.cost_calculator import calculate_llm_cost
from traigent.utils.logging import get_logger

# Initialize logger first
logger = get_logger(__name__)

# Expose TOKENCOST availability and backward compatibility functions for tests
try:
    from traigent.utils.cost_calculator import (
        TOKENCOST_AVAILABLE as _TOKENCOST_AVAILABLE,
    )
    from traigent.utils.cost_calculator import (
        calculate_completion_cost,
        calculate_prompt_cost,
    )
except Exception:
    _TOKENCOST_AVAILABLE = False
    calculate_prompt_cost = None  # type: ignore[assignment]
    calculate_completion_cost = None  # type: ignore[assignment]

TOKENCOST_AVAILABLE = _TOKENCOST_AVAILABLE


@dataclass
class TokenMetrics:
    """Token usage metrics for a single evaluation."""

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0

    def __post_init__(self) -> None:
        # Ensure non-negative values and handle None
        self.input_tokens = max(0, self.input_tokens or 0)
        self.output_tokens = max(0, self.output_tokens or 0)
        self.total_tokens = self.input_tokens + self.output_tokens


@dataclass
class ResponseMetrics:
    """Response time metrics for a single evaluation."""

    response_time_ms: float = 0.0
    first_token_ms: float | None = None
    tokens_per_second: float | None = None

    def __post_init__(self) -> None:
        # Ensure non-negative response time
        self.response_time_ms = max(0.0, self.response_time_ms or 0.0)
        if self.first_token_ms is not None:
            self.first_token_ms = max(0.0, self.first_token_ms)
        if self.tokens_per_second is not None:
            self.tokens_per_second = max(0.0, self.tokens_per_second)


@dataclass
class CostMetrics:
    """Cost metrics for a single evaluation."""

    input_cost: float = 0.0
    output_cost: float = 0.0
    total_cost: float = 0.0

    def __post_init__(self) -> None:
        # Ensure non-negative costs and handle None
        self.input_cost = max(0.0, self.input_cost or 0.0)
        self.output_cost = max(0.0, self.output_cost or 0.0)
        self.total_cost = self.input_cost + self.output_cost


@dataclass
class ExampleMetrics:
    """Complete metrics for a single example evaluation."""

    tokens: TokenMetrics = field(default_factory=TokenMetrics)
    response: ResponseMetrics = field(default_factory=ResponseMetrics)
    cost: CostMetrics = field(default_factory=CostMetrics)
    success: bool = True
    error: str | None = None
    custom_metrics: dict[str, Any] = field(default_factory=dict)


class MetricsTracker:
    """Tracks and aggregates metrics across multiple evaluations."""

    def __init__(self) -> None:
        self.example_metrics: list[ExampleMetrics] = []
        self.start_time: float | None = None
        self.end_time: float | None = None

    def start_tracking(self) -> None:
        """Start tracking metrics."""
        self.start_time = time.time()
        self.example_metrics = []

    def add_example_metrics(self, metrics: ExampleMetrics) -> None:
        """Add metrics for a single example."""
        if metrics is None:
            logger.warning(
                "Attempted to add None metrics, creating empty metrics instead"
            )
            metrics = ExampleMetrics()
        elif not isinstance(metrics, ExampleMetrics):
            logger.error(
                f"Invalid metrics type: {type(metrics)}. Expected ExampleMetrics"
            )
            return
        self.example_metrics.append(metrics)

    def end_tracking(self) -> None:
        """End tracking and calculate duration."""
        self.end_time = time.time()

    def get_duration(self) -> float:
        """Get total duration in seconds."""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return 0.0

    def calculate_statistics(self, values: Sequence[int | float]) -> dict[str, float]:
        """Calculate mean, median, and std for a list of values."""
        if not values:
            return {"mean": 0.0, "median": 0.0, "std": 0.0}

        # Filter out None values and ensure all are float
        clean_values = []
        for v in values:
            if v is not None:
                try:
                    clean_values.append(float(v))
                except (TypeError, ValueError) as e:
                    logger.warning(f"Skipping invalid value {v}: {e}")

        if not clean_values:
            return {"mean": 0.0, "median": 0.0, "std": 0.0}

        try:
            mean_val = statistics.mean(clean_values)
            median_val = statistics.median(clean_values)
            std_val = statistics.stdev(clean_values) if len(clean_values) > 1 else 0.0
        except Exception as e:
            logger.error(f"Error calculating statistics: {e}")
            return {"mean": 0.0, "median": 0.0, "std": 0.0}

        return {
            "mean": round(mean_val, 6),
            "median": round(median_val, 6),
            "std": round(std_val, 6),
        }

    def aggregate_metrics(self) -> dict[str, Any]:
        """Aggregate all tracked metrics into summary statistics."""
        if not self.example_metrics:
            return self._empty_aggregated_metrics()

        # Filter successful examples for metrics calculation
        successful_metrics = [m for m in self.example_metrics if m.success]

        if not successful_metrics:
            return self._empty_aggregated_metrics()

        # Extract values for each metric type
        input_tokens = [m.tokens.input_tokens for m in successful_metrics]
        output_tokens = [m.tokens.output_tokens for m in successful_metrics]
        total_tokens = [m.tokens.total_tokens for m in successful_metrics]

        response_times = [m.response.response_time_ms for m in successful_metrics]

        input_costs = [m.cost.input_cost for m in successful_metrics]
        output_costs = [m.cost.output_cost for m in successful_metrics]
        total_costs = [m.cost.total_cost for m in successful_metrics]

        # Calculate statistics for each metric
        aggregated = {
            # Token metrics
            "input_tokens": self.calculate_statistics(input_tokens),
            "output_tokens": self.calculate_statistics(output_tokens),
            "total_tokens": self.calculate_statistics(total_tokens),
            # Response time metrics
            "response_time_ms": self.calculate_statistics(response_times),
            # Cost metrics
            "input_cost": self.calculate_statistics(input_costs),
            "output_cost": self.calculate_statistics(output_costs),
            "total_cost": self.calculate_statistics(total_costs),
            # Summary metrics
            "total_examples": len(self.example_metrics),
            "successful_examples": len(successful_metrics),
            "success_rate": (
                len(successful_metrics) / len(self.example_metrics)
                if len(self.example_metrics) > 0
                else 0.0
            ),
            "duration": self.get_duration(),
        }

        # Add tokens per second if available
        tps_values = [
            m.response.tokens_per_second
            for m in successful_metrics
            if m.response.tokens_per_second is not None
        ]
        if tps_values:
            aggregated["tokens_per_second"] = self.calculate_statistics(tps_values)

        return aggregated

    def _empty_aggregated_metrics(self) -> dict[str, Any]:
        """Return empty aggregated metrics structure."""
        # Check for strict metrics nulls mode
        strict_nulls = os.environ.get("TRAIGENT_STRICT_METRICS_NULLS", "").lower() in (
            "true",
            "1",
            "yes",
        )
        missing_default = None if strict_nulls else 0.0

        empty_stats = {
            "mean": missing_default,
            "median": missing_default,
            "std": missing_default,
        }
        return {
            "input_tokens": empty_stats.copy(),
            "output_tokens": empty_stats.copy(),
            "total_tokens": empty_stats.copy(),
            "response_time_ms": empty_stats.copy(),
            "input_cost": empty_stats.copy(),
            "output_cost": empty_stats.copy(),
            "total_cost": empty_stats.copy(),
            "total_examples": 0,
            "successful_examples": 0,
            "success_rate": missing_default,
            "duration": missing_default,
        }

    def format_for_backend(self) -> dict[str, Any]:
        """Format aggregated metrics for backend submission."""
        try:
            aggregated = self.aggregate_metrics()
        except Exception as e:
            logger.error(f"Error aggregating metrics: {e}")
            return self._empty_backend_format()

        strict_nulls = os.environ.get("TRAIGENT_STRICT_METRICS_NULLS", "").lower() in (
            "true",
            "1",
            "yes",
        )

        # Safe getter with defaults
        def safe_get(
            d: dict[str, Any], key: str, subkey: str | None = None, default: float = 0.0
        ) -> float | None:
            """Safely get value from nested dict with default."""

            actual_default = None if strict_nulls else default

            try:
                if subkey:
                    return cast(
                        float | None, d.get(key, {}).get(subkey, actual_default)
                    )
                return cast(float | None, d.get(key, actual_default))
            except (AttributeError, TypeError):
                return actual_default

        # Format metrics in a cleaner format without _mean/_median/_std suffixes
        # Since measures now contain per-example data and summary_stats contain
        # the statistical aggregations, we only need the primary metrics here

        # Calculate accuracy from custom metrics if available
        accuracy_value: float | None = None
        if self.example_metrics:
            accuracy_scores = [
                m.custom_metrics["accuracy"]
                for m in self.example_metrics
                if "accuracy" in m.custom_metrics
                and m.custom_metrics["accuracy"] is not None
            ]
            if accuracy_scores:
                accuracy_value = sum(accuracy_scores) / len(accuracy_scores)
        else:
            # Fallback to success rate if no example metrics
            accuracy_value = safe_get(aggregated, "success_rate")

        if accuracy_value is None and not strict_nulls:
            accuracy_value = 0.0

        formatted = {
            # Core metrics (single values)
            "score": accuracy_value,  # Use actual accuracy for score
            "accuracy": accuracy_value,  # Use actual accuracy
            "duration": safe_get(aggregated, "duration"),
            # Use mean values as the primary metrics (without suffix)
            "input_tokens": safe_get(aggregated, "input_tokens", "mean"),
            "output_tokens": safe_get(aggregated, "output_tokens", "mean"),
            "total_tokens": safe_get(aggregated, "total_tokens", "mean"),
            "response_time_ms": safe_get(aggregated, "response_time_ms", "mean"),
            "cost": safe_get(aggregated, "total_cost", "mean"),
            # Additional useful metrics
            "total_examples": aggregated["total_examples"],
            "successful_examples": aggregated["successful_examples"],
        }

        # Add tokens per second if available
        if "tokens_per_second" in aggregated:
            formatted["tokens_per_second"] = safe_get(
                aggregated, "tokens_per_second", "mean"
            )

        return formatted

    def _empty_backend_format(self) -> dict[str, Any]:
        """Return empty backend format structure when error occurs."""
        # Check for strict metrics nulls mode
        strict_nulls = os.environ.get("TRAIGENT_STRICT_METRICS_NULLS", "").lower() in (
            "true",
            "1",
            "yes",
        )
        missing_default = None if strict_nulls else 0.0

        return {
            "score": missing_default,
            "accuracy": missing_default,
            "duration": missing_default,
            "input_tokens": missing_default,
            "output_tokens": missing_default,
            "total_tokens": missing_default,
            "response_time_ms": missing_default,
            "cost": missing_default,
            "total_examples": 0,
            "successful_examples": 0,
        }

    def format_as_summary_stats(self) -> dict[str, Any]:
        """Format metrics as pandas.describe()-compatible summary statistics.

        This format is used for privacy-preserving mode where individual
        results are not transmitted, only aggregated statistics.

        Returns:
            Dictionary with summary_stats structure matching pandas.describe()
        """
        if not self.example_metrics:
            return self._empty_summary_stats()

        # Filter successful examples for metrics calculation
        successful_metrics = [m for m in self.example_metrics if m.success]

        if not successful_metrics:
            return self._empty_summary_stats()

        # Extract values for each metric type
        # For accuracy, use custom metrics when available; otherwise derive from success flags
        accuracy_values = []
        for metric in self.example_metrics:
            if "accuracy" in metric.custom_metrics:
                accuracy_values.append(metric.custom_metrics["accuracy"] or 0.0)
            else:
                accuracy_values.append(1.0 if metric.success else 0.0)

        metrics_data: dict[str, list[int | float]] = {
            "accuracy": accuracy_values,
            "input_tokens": [m.tokens.input_tokens for m in successful_metrics],
            "output_tokens": [m.tokens.output_tokens for m in successful_metrics],
            "total_tokens": [m.tokens.total_tokens for m in successful_metrics],
            "response_time_ms": [
                m.response.response_time_ms for m in successful_metrics
            ],
            "input_cost": [m.cost.input_cost for m in successful_metrics],
            "output_cost": [m.cost.output_cost for m in successful_metrics],
            "total_cost": [m.cost.total_cost for m in successful_metrics],
        }

        # Add tokens per second if available
        tps_values = [
            m.response.tokens_per_second
            for m in successful_metrics
            if m.response.tokens_per_second is not None
        ]
        if tps_values:
            metrics_data["tokens_per_second"] = tps_values

        # Add any other custom metrics that appear in all examples
        if self.example_metrics:
            # Find common custom metrics across all examples
            all_custom_keys: set[str] = set()
            for m in self.example_metrics:
                all_custom_keys.update(m.custom_metrics.keys())

            # Add custom metrics that we haven't already handled
            for key in all_custom_keys:
                if (
                    key not in metrics_data and key != "accuracy"
                ):  # Skip accuracy as we've already handled it
                    custom_values = []
                    for m in self.example_metrics:
                        if key in m.custom_metrics:
                            custom_values.append(m.custom_metrics[key])
                        else:
                            custom_values.append(
                                0.0
                            )  # Default value if metric not present
                    if custom_values:
                        metrics_data[key] = custom_values

        # Generate pandas.describe()-compatible statistics for each metric
        summary_metrics = {}
        for metric_name, values in metrics_data.items():
            if values:
                summary_metrics[metric_name] = self._calculate_describe_stats(values)

        # Build the complete summary_stats structure
        summary_stats = {
            "metrics": summary_metrics,
            "execution_time": self.get_duration(),
            "total_examples": len(self.example_metrics),
            "metadata": {
                "sdk_version": get_version(),
                "aggregation_method": "pandas.describe",
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            },
        }

        return summary_stats

    def _empty_summary_stats(self) -> dict[str, Any]:
        """Return empty summary stats structure."""
        empty_describe = {
            "count": 0,
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "25%": 0.0,
            "50%": 0.0,
            "75%": 0.0,
            "max": 0.0,
        }

        return {
            "metrics": {
                "accuracy": empty_describe.copy(),
                "input_tokens": empty_describe.copy(),
                "output_tokens": empty_describe.copy(),
                "total_tokens": empty_describe.copy(),
                "response_time_ms": empty_describe.copy(),
                "total_cost": empty_describe.copy(),
            },
            "execution_time": 0.0,
            "total_examples": 0,
            "metadata": {
                "sdk_version": get_version(),
                "aggregation_method": "pandas.describe",
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            },
        }

    def _calculate_describe_stats(
        self, values: Sequence[int | float]
    ) -> dict[str, float]:
        """Calculate pandas.describe()-compatible statistics.

        Returns statistics in the exact format of pandas.DataFrame.describe():
        count, mean, std, min, 25%, 50%, 75%, max
        """
        if not values:
            return {
                "count": 0,
                "mean": 0.0,
                "std": 0.0,
                "min": 0.0,
                "25%": 0.0,
                "50%": 0.0,
                "75%": 0.0,
                "max": 0.0,
            }

        sorted_values = sorted(values)
        n = len(values)

        # Calculate percentiles
        def percentile(data, p):
            """Calculate percentile using linear interpolation (same as pandas)."""
            if not data:
                return 0.0
            try:
                k = (len(data) - 1) * p
                f = int(k)
                c = k - f
                if f + 1 < len(data):
                    return float(data[f] * (1 - c) + data[f + 1] * c)
                return float(data[f])
            except (IndexError, TypeError, ValueError) as e:
                logger.warning(f"Error calculating percentile: {e}")
                return 0.0

        return {
            "count": n,
            "mean": statistics.mean(values),
            "std": statistics.stdev(values) if n > 1 else 0.0,
            "min": min(values),
            "25%": percentile(sorted_values, 0.25),
            "50%": percentile(sorted_values, 0.50),  # median
            "75%": percentile(sorted_values, 0.75),
            "max": max(values),
        }


# Response Handler Hierarchy


class ResponseHandler(ABC):
    """Base class for handling different LLM provider response formats."""

    def __init__(self, next_handler: Optional["ResponseHandler"] = None) -> None:
        self._next_handler = next_handler

    @abstractmethod
    def can_handle(self, response: Any) -> bool:
        """Check if this handler can process the response."""
        pass

    @abstractmethod
    def extract_tokens(self, response: Any) -> TokenMetrics:
        """Extract token metrics from response."""
        pass

    @abstractmethod
    def extract_response_time(self, response: Any) -> float:
        """Extract response time in milliseconds."""
        pass

    def extract_metadata_cost(self, response: Any) -> CostMetrics:
        """Extract cost information from response metadata if available."""
        cost_metrics = CostMetrics()

        # Check for existing cost information in response
        if hasattr(response, "cost"):
            try:
                if isinstance(response.cost, dict):
                    cost_metrics.input_cost = response.cost.get("input", 0.0)
                    cost_metrics.output_cost = response.cost.get("output", 0.0)
                    cost_metrics.total_cost = response.cost.get("total", 0.0)
                else:
                    cost_metrics.total_cost = float(response.cost)
            except (TypeError, ValueError) as e:
                logger.debug(f"Failed to parse cost from response: {e}")

        return cost_metrics

    def extract_metadata_info(self, response: Any, metrics: ExampleMetrics) -> None:
        """Extract additional metrics from metadata."""
        if not hasattr(response, "metadata") or not isinstance(response.metadata, dict):
            return

        metadata = response.metadata

        # Extract token information
        if "tokens" in metadata and isinstance(metadata["tokens"], dict):
            tokens = metadata["tokens"]
            metrics.tokens.input_tokens = tokens.get(
                "input", metrics.tokens.input_tokens
            )
            metrics.tokens.output_tokens = tokens.get(
                "output", metrics.tokens.output_tokens
            )
            metrics.tokens.total_tokens = (
                metrics.tokens.input_tokens + metrics.tokens.output_tokens
            )

        # Extract cost information
        if "cost" in metadata:
            cost = metadata["cost"]
            if isinstance(cost, dict):
                metrics.cost.input_cost = cost.get("input", metrics.cost.input_cost)
                metrics.cost.output_cost = cost.get("output", metrics.cost.output_cost)
                metrics.cost.total_cost = cost.get("total", metrics.cost.total_cost)
            else:
                try:
                    metrics.cost.total_cost = float(cost)
                except (TypeError, ValueError):
                    pass

        # Extract response time
        if "response_time_ms" in metadata:
            try:
                metrics.response.response_time_ms = float(metadata["response_time_ms"])
            except (TypeError, ValueError):
                pass

    def handle(self, response: Any) -> ExampleMetrics | None:
        """Handle the response or pass to next handler."""
        if self.can_handle(response):
            metrics = ExampleMetrics()
            metrics.tokens = self.extract_tokens(response)
            metrics.response.response_time_ms = self.extract_response_time(response)

            # Extract cost from response if available
            response_cost = self.extract_metadata_cost(response)
            if response_cost.total_cost > 0:
                metrics.cost = response_cost

            # Extract additional metadata
            self.extract_metadata_info(response, metrics)

            return metrics
        elif self._next_handler:
            return self._next_handler.handle(response)
        else:
            return None


class OpenAIResponseHandler(ResponseHandler):
    """Handler for OpenAI ChatCompletion responses."""

    def can_handle(self, response: Any) -> bool:
        # More specific check for OpenAI - look for prompt_tokens and completion_tokens
        if not hasattr(response, "usage"):
            return False
        usage = response.usage
        return hasattr(usage, "prompt_tokens") and hasattr(usage, "completion_tokens")

    def extract_tokens(self, response: Any) -> TokenMetrics:
        tokens = TokenMetrics()
        if hasattr(response, "usage"):
            usage = response.usage
            if hasattr(usage, "prompt_tokens") and isinstance(
                usage.prompt_tokens, (int, float)
            ):
                tokens.input_tokens = int(usage.prompt_tokens)
            if hasattr(usage, "completion_tokens") and isinstance(
                usage.completion_tokens, (int, float)
            ):
                tokens.output_tokens = int(usage.completion_tokens)
            if hasattr(usage, "total_tokens") and isinstance(
                usage.total_tokens, (int, float)
            ):
                tokens.total_tokens = int(usage.total_tokens)
            else:
                tokens.total_tokens = tokens.input_tokens + tokens.output_tokens
        return tokens

    def extract_response_time(self, response: Any) -> float:
        if hasattr(response, "response_time_ms") and isinstance(
            response.response_time_ms, (int, float)
        ):
            return float(response.response_time_ms)
        elif hasattr(response, "_response_time"):
            return cast(float, response._response_time)
        return 0.0


class AnthropicResponseHandler(ResponseHandler):
    """Handler for Anthropic API responses."""

    def can_handle(self, response: Any) -> bool:
        # Check for Anthropic-specific attributes
        # 1. Check for explicit Anthropic Message class
        if type(response).__name__ == "Message" and hasattr(response, "content"):
            # Additional check for Anthropic-specific structure
            if (
                hasattr(response, "model")
                and "claude" in str(getattr(response, "model", "")).lower()
            ):
                return True

        # 2. Check for usage pattern specific to Anthropic
        if hasattr(response, "usage"):
            usage = response.usage
            # Anthropic uses input_tokens/output_tokens instead of prompt_tokens/completion_tokens
            has_anthropic_tokens = hasattr(usage, "input_tokens") and hasattr(
                usage, "output_tokens"
            )
            has_openai_tokens = hasattr(usage, "prompt_tokens") and hasattr(
                usage, "completion_tokens"
            )
            # Only return True if it has Anthropic tokens and NOT OpenAI tokens
            if has_anthropic_tokens and not has_openai_tokens:
                return True

        # 3. Check for model name containing claude
        if (
            hasattr(response, "model")
            and "claude" in str(getattr(response, "model", "")).lower()
        ):
            # Additional check to ensure it's not OpenAI
            if not hasattr(
                response, "choices"
            ):  # OpenAI has choices, Anthropic doesn't
                return True

        return False

    def extract_tokens(self, response: Any) -> TokenMetrics:
        tokens = TokenMetrics()
        if hasattr(response, "usage"):
            usage = response.usage

            # Support both object-style and dict-style usage
            def _get(u: Any, name: str) -> Any:
                if isinstance(u, dict):
                    return u.get(name)
                return getattr(u, name, None)

            input_val = (
                _get(usage, "input_tokens")
                or _get(usage, "num_input_tokens")
                or _get(usage, "prompt_tokens")
            )
            output_val = (
                _get(usage, "output_tokens")
                or _get(usage, "num_output_tokens")
                or _get(usage, "completion_tokens")
            )
            total_val = _get(usage, "total_tokens")

            if isinstance(input_val, (int, float)):
                tokens.input_tokens = int(input_val)
            if isinstance(output_val, (int, float)):
                tokens.output_tokens = int(output_val)
            if isinstance(total_val, (int, float)):
                tokens.total_tokens = int(total_val)
            else:
                tokens.total_tokens = tokens.input_tokens + tokens.output_tokens
        return tokens

    def extract_response_time(self, response: Any) -> float:
        if hasattr(response, "response_time_ms") and isinstance(
            response.response_time_ms, (int, float)
        ):
            return float(response.response_time_ms)
        elif hasattr(response, "latency") and isinstance(
            response.latency, (int, float)
        ):
            return float(response.latency) * 1000  # Convert to ms
        return 0.0


class LangChainResponseHandler(ResponseHandler):
    """Handler for LangChain response objects."""

    def can_handle(self, response: Any) -> bool:
        # Check for LangChain-specific patterns
        return (
            hasattr(response, "llm_output")
            or hasattr(response, "generations")
            or hasattr(response, "usage_metadata")  # LangChain AIMessage
            or hasattr(response, "response_metadata")  # LangChain ChatOpenAI responses
            or str(type(response).__module__).startswith("langchain")
        )

    def extract_tokens(self, response: Any) -> TokenMetrics:
        tokens = TokenMetrics()

        # Check usage_metadata for token usage (langchain_core AIMessage)
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            usage = response.usage_metadata
            tokens.input_tokens = usage.get("input_tokens", 0)
            tokens.output_tokens = usage.get("output_tokens", 0)
            tokens.total_tokens = usage.get(
                "total_tokens", tokens.input_tokens + tokens.output_tokens
            )
            if tokens.total_tokens > 0:
                return tokens

        # Check response_metadata for token usage (alternative format)
        if hasattr(response, "response_metadata") and response.response_metadata:
            metadata = response.response_metadata

            # Check for token_usage field (ChatOpenAI format)
            if "token_usage" in metadata and isinstance(metadata["token_usage"], dict):
                usage = metadata["token_usage"]
                tokens.input_tokens = usage.get("prompt_tokens", 0)
                tokens.output_tokens = usage.get("completion_tokens", 0)
                tokens.total_tokens = usage.get(
                    "total_tokens", tokens.input_tokens + tokens.output_tokens
                )
                if tokens.total_tokens > 0:
                    return tokens

            # Check for usage field (alternative format)
            if "usage" in metadata and isinstance(metadata["usage"], dict):
                usage = metadata["usage"]
                input_val = usage.get("input_tokens", usage.get("prompt_tokens", 0))
                output_val = usage.get(
                    "output_tokens", usage.get("completion_tokens", 0)
                )
                tokens.input_tokens = int(input_val) if input_val is not None else 0
                tokens.output_tokens = int(output_val) if output_val is not None else 0
                total_val = usage.get(
                    "total_tokens", tokens.input_tokens + tokens.output_tokens
                )
                tokens.total_tokens = (
                    int(total_val)
                    if total_val is not None
                    else tokens.input_tokens + tokens.output_tokens
                )
                if tokens.total_tokens > 0:
                    return tokens

        # Check llm_output for token usage (older format)
        if hasattr(response, "llm_output") and isinstance(response.llm_output, dict):
            llm_output = response.llm_output
            if "token_usage" in llm_output:
                usage = llm_output["token_usage"]
                tokens.input_tokens = usage.get("prompt_tokens", 0)
                tokens.output_tokens = usage.get("completion_tokens", 0)
                tokens.total_tokens = usage.get(
                    "total_tokens", tokens.input_tokens + tokens.output_tokens
                )

        return tokens

    def extract_response_time(self, response: Any) -> float:
        # First check response_metadata for injected timing
        if hasattr(response, "response_metadata") and isinstance(
            response.response_metadata, dict
        ):
            if "response_time_ms" in response.response_metadata:
                try:
                    return float(response.response_metadata["response_time_ms"])
                except (TypeError, ValueError):
                    pass

        # Fall back to direct attribute
        if hasattr(response, "response_time_ms"):
            return float(response.response_time_ms)

        return 0.0


class DictResponseHandler(ResponseHandler):
    """Handler for dictionary responses with token counts."""

    def can_handle(self, response: Any) -> bool:
        if not isinstance(response, dict):
            return False
        # Check if it has token-related keys
        return any(
            key in response
            for key in ["input_tokens", "output_tokens", "total_tokens", "usage"]
        )

    def extract_tokens(self, response: Any) -> TokenMetrics:
        """Extract tokens from dict response."""
        if not isinstance(response, dict):
            return TokenMetrics()

        # Direct token fields
        input_tokens = response.get("input_tokens", 0)
        output_tokens = response.get("output_tokens", 0)
        total_tokens = response.get("total_tokens", 0)

        # Check for usage field (OpenAI style)
        if "usage" in response and isinstance(response["usage"], dict):
            usage = response["usage"]
            input_tokens = usage.get(
                "prompt_tokens", usage.get("input_tokens", input_tokens)
            )
            output_tokens = usage.get(
                "completion_tokens", usage.get("output_tokens", output_tokens)
            )
            total_tokens = usage.get("total_tokens", total_tokens)

        # If total_tokens is not provided, calculate it
        if total_tokens == 0 and (input_tokens > 0 or output_tokens > 0):
            total_tokens = input_tokens + output_tokens

        return TokenMetrics(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
        )

    def extract_response_time(self, response: Any) -> float:
        """Extract response time from dict."""
        if not isinstance(response, dict):
            return 0.0
        # Check common keys for response time
        for key in ["response_time_ms", "response_time", "latency", "duration"]:
            if key in response:
                value = response[key]
                if isinstance(value, (int, float)):
                    # Convert to ms if needed
                    if key == "latency" or key == "duration":
                        return float(value * 1000)
                    return float(value)
        return 0.0


class GenericResponseHandler(ResponseHandler):
    """Fallback handler for any response format."""

    def can_handle(self, response: Any) -> bool:
        return True  # Always can handle as fallback

    def extract_tokens(self, response: Any) -> TokenMetrics:
        return TokenMetrics()  # Return empty metrics

    def extract_response_time(self, response: Any) -> float:
        # Try common response time attributes
        for attr in ["response_time_ms", "latency", "_response_time"]:
            if hasattr(response, attr):
                value = getattr(response, attr)
                if isinstance(value, (int, float)):
                    return float(value * 1000 if attr == "latency" else value)
        return 0.0


class ResponseHandlerFactory:
    """Factory for creating response handler chains."""

    @staticmethod
    def create_handler_chain() -> ResponseHandler:
        """Create a chain of response handlers in order of specificity."""
        generic = GenericResponseHandler()
        dict_handler = DictResponseHandler(generic)
        langchain = LangChainResponseHandler(dict_handler)
        anthropic = AnthropicResponseHandler(langchain)
        openai = OpenAIResponseHandler(anthropic)
        return openai


class CostCalculator:
    """Separate class for handling cost calculations."""

    def __init__(self) -> None:
        self.logger = logger

    def calculate_cost(
        self,
        metrics: ExampleMetrics,
        model_name: str | None,
        original_prompt: Any,
        response_text: str | None,
        prompt_length: int | None = None,
        response_length: int | None = None,
    ) -> None:
        """Calculate and update cost metrics.

        Args:
            metrics: ExampleMetrics to update with cost information
            model_name: Model name for cost lookup
            original_prompt: Original prompt (if not in privacy mode)
            response_text: Response text (if not in privacy mode)
            prompt_length: Length of prompt in privacy mode
            response_length: Length of response in privacy mode
        """
        # Check if we're in mock LLM mode
        import os

        from traigent.utils.env_config import is_mock_llm

        mock_mode = is_mock_llm()
        generate_mocks_env = os.environ.get("TRAIGENT_GENERATE_MOCKS", "").lower()

        # DEBUG: Log environment variables
        self.logger.debug(
            f"COST DEBUG: Checking mock mode - is_mock_llm={mock_mode}, "
            f"TRAIGENT_GENERATE_MOCKS='{generate_mocks_env}', model_name='{model_name}'"
        )

        if mock_mode or generate_mocks_env == "true":
            # In mock mode, set costs to 0 but estimate tokens if in privacy mode
            if prompt_length is not None and metrics.tokens.input_tokens == 0:
                # Estimate tokens from length (roughly 1 token per 4 characters)
                metrics.tokens.input_tokens = max(1, prompt_length // 4)
            if response_length is not None and metrics.tokens.output_tokens == 0:
                metrics.tokens.output_tokens = max(1, response_length // 4)
            if metrics.tokens.input_tokens > 0 or metrics.tokens.output_tokens > 0:
                metrics.tokens.total_tokens = (
                    metrics.tokens.input_tokens + metrics.tokens.output_tokens
                )

            metrics.cost.input_cost = 0.0
            metrics.cost.output_cost = 0.0
            metrics.cost.total_cost = 0.0
            self.logger.debug("COST DEBUG: Mock mode enabled - setting costs to 0")
            return

        if not model_name or metrics.cost.total_cost > 0.0:
            self.logger.debug(
                f"Cost calculation skipped - model_name: '{model_name}', current_total_cost: {metrics.cost.total_cost}"
            )
            return

        try:
            # Always use unified cost calculation which handles token counts properly
            self.logger.info(
                f"💰 COST DEBUG: Calling _try_unified_cost_calculation with model='{model_name}', "
                f"tokens: input={metrics.tokens.input_tokens}, output={metrics.tokens.output_tokens}"
            )
            self._try_unified_cost_calculation(
                metrics,
                model_name,
                original_prompt,
                response_text,
                prompt_length=prompt_length,
                response_length=response_length,
            )
        except Exception as e:
            # Make error more visible
            self.logger.error(
                f"Cost calculation failed for model {model_name}: {e}", exc_info=True
            )
            # Re-raise to make failures more visible during development
            if os.environ.get("TRAIGENT_DEBUG", "").lower() == "true":
                raise

    def _try_legacy_cost_calculation(
        self,
        metrics: ExampleMetrics,
        model_name: str,
        original_prompt: Any,
        response_text: str | None,
    ) -> bool:
        """Try backward compatible cost calculation."""
        backward_compatible = False

        if calculate_prompt_cost is not None and original_prompt is not None:
            metrics.cost.input_cost = calculate_prompt_cost(original_prompt, model_name)
            backward_compatible = True

        if calculate_completion_cost is not None and response_text is not None:
            metrics.cost.output_cost = calculate_completion_cost(
                response_text, model_name
            )
            backward_compatible = True

        if backward_compatible:
            metrics.cost.total_cost = metrics.cost.input_cost + metrics.cost.output_cost
            self.logger.debug(
                f"Cost calculated using legacy litellm functions for {model_name}: ${metrics.cost.total_cost:.6f}"
            )
            return True

        return False

    def _try_unified_cost_calculation(
        self,
        metrics: ExampleMetrics,
        model_name: str,
        original_prompt: Any,
        response_text: str | None,
        prompt_length: int | None = None,
        response_length: int | None = None,
    ) -> None:
        """Try unified cost calculation.

        Args:
            metrics: ExampleMetrics to update
            model_name: Model name for cost lookup
            original_prompt: Original prompt (if available)
            response_text: Response text (if available)
            prompt_length: Length of prompt in privacy mode
            response_length: Length of response in privacy mode
        """
        # If we have lengths but no tokens, estimate tokens first
        if prompt_length is not None and metrics.tokens.input_tokens == 0:
            # Estimate token count based on average token/character ratio.
            metrics.tokens.input_tokens = max(1, int(prompt_length * 0.25))

        if response_length is not None and metrics.tokens.output_tokens == 0:
            metrics.tokens.output_tokens = max(1, int(response_length * 0.25))

        # Update total tokens if we estimated
        if (
            prompt_length is not None or response_length is not None
        ) and metrics.tokens.total_tokens == 0:
            metrics.tokens.total_tokens = (
                metrics.tokens.input_tokens + metrics.tokens.output_tokens
            )

        # Use token counts if available, otherwise try to use prompt/response
        cost_breakdown = calculate_llm_cost(
            prompt=original_prompt if metrics.tokens.input_tokens == 0 else None,
            response=response_text if metrics.tokens.output_tokens == 0 else None,
            model_name=model_name,
            input_tokens=(
                metrics.tokens.input_tokens if metrics.tokens.input_tokens > 0 else None
            ),
            output_tokens=(
                metrics.tokens.output_tokens
                if metrics.tokens.output_tokens > 0
                else None
            ),
            logger=self.logger,
        )

        # Update metrics with cost calculation results
        self.logger.info(
            f"💰 COST DEBUG: cost_breakdown returned - input_cost=${cost_breakdown.input_cost:.8f}, "
            f"output_cost=${cost_breakdown.output_cost:.8f}, total=${cost_breakdown.total_cost:.8f}, "
            f"method={cost_breakdown.calculation_method}"
        )
        metrics.cost.input_cost = cost_breakdown.input_cost
        metrics.cost.output_cost = cost_breakdown.output_cost
        metrics.cost.total_cost = cost_breakdown.total_cost

        # Update token counts if they weren't available before
        if cost_breakdown.input_tokens > 0 and metrics.tokens.input_tokens == 0:
            metrics.tokens.input_tokens = cost_breakdown.input_tokens
        if cost_breakdown.output_tokens > 0 and metrics.tokens.output_tokens == 0:
            metrics.tokens.output_tokens = cost_breakdown.output_tokens
        if cost_breakdown.total_tokens > 0 and metrics.tokens.total_tokens == 0:
            metrics.tokens.total_tokens = cost_breakdown.total_tokens

        # Log successful cost calculation
        if cost_breakdown.total_cost > 0:
            self.logger.debug(
                f"Cost calculated for {model_name}: ${cost_breakdown.total_cost:.6f} "
                f"(method: {cost_breakdown.calculation_method})"
            )
            if cost_breakdown.mapped_model != model_name:
                self.logger.debug(
                    f"Model mapped: {model_name} -> {cost_breakdown.mapped_model}"
                )
        elif metrics.tokens.input_tokens > 0 or metrics.tokens.output_tokens > 0:
            self.logger.debug(
                f"No cost calculated for {model_name} despite having tokens: "
                f"input={metrics.tokens.input_tokens}, output={metrics.tokens.output_tokens}"
            )


class MetricsCalculator:
    """Helper for calculating derived metrics."""

    @staticmethod
    def calculate_tokens_per_second(metrics: ExampleMetrics) -> None:
        """Calculate and set tokens per second if possible."""
        try:
            if (
                isinstance(metrics.tokens.total_tokens, (int, float))
                and isinstance(metrics.response.response_time_ms, (int, float))
                and metrics.tokens.total_tokens > 0
                and metrics.response.response_time_ms
                > 0.001  # Minimum 1 microsecond to avoid division by zero
            ):
                metrics.response.tokens_per_second = metrics.tokens.total_tokens / (
                    metrics.response.response_time_ms / 1000
                )
        except (TypeError, ValueError, AttributeError) as e:
            logger.debug(f"Failed to calculate tokens per second: {e}")


def extract_llm_metrics(
    response: Any,
    model_name: str | None = None,
    original_prompt: Any | None = None,
    response_text: str | None = None,
    prompt_length: int | None = None,
    response_length: int | None = None,
) -> ExampleMetrics:
    """Extract metrics from LLM response objects with cost calculation.

    This function uses a handler pattern to extract metrics from various LLM response formats:
    - OpenAI ChatCompletion responses
    - Anthropic responses
    - LangChain responses
    - Custom response formats

    Args:
        response: The response object from an LLM call
        model_name: The model name used for the LLM call (for cost calculation)
        original_prompt: The original prompt sent to the LLM (for accurate cost calculation)
        response_text: The response text from the LLM (for accurate cost calculation)
        prompt_length: Length of prompt in privacy mode (alternative to original_prompt)
        response_length: Length of response in privacy mode (alternative to response_text)

    Returns:
        ExampleMetrics with extracted token, cost, and response metrics
    """
    # Create handler chain and extract basic metrics
    handler_chain = ResponseHandlerFactory.create_handler_chain()
    metrics = handler_chain.handle(response)

    if metrics is None:
        metrics = ExampleMetrics()
        logger.warning("No handler could process the response, using empty metrics")

    # Calculate cost using separate cost calculator
    cost_calculator = CostCalculator()
    cost_calculator.calculate_cost(
        metrics,
        model_name,
        original_prompt,
        response_text,
        prompt_length=prompt_length,
        response_length=response_length,
    )

    # Calculate derived metrics
    MetricsCalculator.calculate_tokens_per_second(metrics)

    return metrics
