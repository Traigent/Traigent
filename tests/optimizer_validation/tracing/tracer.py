"""OpenTelemetry tracer setup for validation tests.

Provides tracer configuration and span context managers for
instrumenting test execution.
"""

from __future__ import annotations

import json
import os
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Generator

try:
    from opentelemetry import trace
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, SpanExporter

    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False
    trace = None  # type: ignore
    TracerProvider = None  # type: ignore
    Resource = None  # type: ignore

if TYPE_CHECKING:
    from opentelemetry.trace import Span, Tracer

    from tests.optimizer_validation.specs.scenario import TestScenario


@dataclass
class TracingConfig:
    """Configuration for test tracing."""

    enabled: bool = True
    service_name: str = "traigent-validation"
    otlp_endpoint: str | None = None
    file_export_path: str | None = None

    @classmethod
    def from_env(cls) -> TracingConfig:
        """Create config from environment variables."""
        return cls(
            enabled=os.getenv("TRAIGENT_TRACING_ENABLED", "true").lower() == "true",
            service_name=os.getenv("OTEL_SERVICE_NAME", "traigent-validation"),
            otlp_endpoint=os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT"),
            file_export_path=os.getenv("TRAIGENT_TRACE_FILE"),
        )


def create_test_tracer(
    test_name: str,
    exporters: list[SpanExporter] | None = None,
    config: TracingConfig | None = None,
) -> Tracer | None:
    """Create a tracer for a specific test.

    Args:
        test_name: Name of the test for resource identification
        exporters: List of span exporters to use
        config: Tracing configuration

    Returns:
        Configured tracer or None if OTEL not available
    """
    if not OTEL_AVAILABLE:
        return None

    config = config or TracingConfig.from_env()
    if not config.enabled:
        return None

    resource = Resource.create(
        {
            "service.name": config.service_name,
            "test.name": test_name,
        }
    )

    provider = TracerProvider(resource=resource)

    # Add exporters
    if exporters:
        for exporter in exporters:
            provider.add_span_processor(BatchSpanProcessor(exporter))

    # Set as global provider for this test
    trace.set_tracer_provider(provider)

    return provider.get_tracer(test_name)


@contextmanager
def optimization_span(
    tracer: Tracer | None,
    scenario: TestScenario,
) -> Generator[Span | None, None, None]:
    """Create root span for optimization session.

    Args:
        tracer: OpenTelemetry tracer (can be None if tracing disabled)
        scenario: Test scenario being executed

    Yields:
        The span object or None if tracing disabled
    """
    if tracer is None:
        yield None
        return

    with tracer.start_as_current_span("optimization_session") as span:
        span.set_attribute("test.scenario.name", scenario.name)
        span.set_attribute("test.scenario.description", scenario.description)
        span.set_attribute("optimization.max_trials", scenario.max_trials)
        span.set_attribute("optimization.timeout", scenario.timeout)
        span.set_attribute("optimization.injection_mode", scenario.injection_mode)
        span.set_attribute("optimization.execution_mode", scenario.execution_mode)
        span.set_attribute(
            "optimization.objectives",
            json.dumps(_serialize_objectives(scenario.objectives)),
        )
        span.set_attribute(
            "optimization.config_space", json.dumps(scenario.config_space)
        )

        if scenario.constraints:
            span.set_attribute(
                "optimization.constraints_count", len(scenario.constraints)
            )

        yield span


@contextmanager
def trial_span(
    tracer: Tracer | None,
    trial_id: str,
    trial_number: int,
    config: dict[str, Any],
) -> Generator[Span | None, None, None]:
    """Create span for trial execution.

    Args:
        tracer: OpenTelemetry tracer (can be None if tracing disabled)
        trial_id: Unique identifier for the trial
        trial_number: Sequential trial number
        config: Configuration being tested

    Yields:
        The span object or None if tracing disabled
    """
    if tracer is None:
        yield None
        return

    with tracer.start_as_current_span("trial_execution") as span:
        span.set_attribute("trial.id", trial_id)
        span.set_attribute("trial.number", trial_number)
        span.set_attribute("trial.config", json.dumps(config))

        yield span


@contextmanager
def evaluation_span(
    tracer: Tracer | None,
    examples_count: int,
) -> Generator[Span | None, None, None]:
    """Create span for evaluation phase.

    Args:
        tracer: OpenTelemetry tracer
        examples_count: Number of examples being evaluated

    Yields:
        The span object or None if tracing disabled
    """
    if tracer is None:
        yield None
        return

    with tracer.start_as_current_span("evaluation") as span:
        span.set_attribute("evaluation.examples_count", examples_count)
        yield span


def add_trial_result(
    span: Span | None,
    status: str,
    metrics: dict[str, float],
    success_rate: float,
    error_message: str | None = None,
) -> None:
    """Add trial result attributes to span.

    Args:
        span: The trial span
        status: Trial status (completed, failed, etc.)
        metrics: Metrics from evaluation
        success_rate: Percentage of successful examples
        error_message: Error message if trial failed
    """
    if span is None:
        return

    span.set_attribute("trial.status", status)
    span.set_attribute("trial.success_rate", success_rate)
    span.set_attribute("trial.metrics", json.dumps(metrics))

    if error_message:
        span.set_attribute("trial.error_message", error_message)

    # Add key metrics as individual attributes for easier querying
    for key in ["accuracy", "cost", "latency", "score"]:
        if key in metrics:
            span.set_attribute(f"trial.metric.{key}", metrics[key])


def _serialize_objectives(objectives: list[Any]) -> list[dict[str, Any]]:
    """Serialize objectives to JSON-compatible format."""
    result = []
    for obj in objectives:
        if isinstance(obj, str):
            result.append({"name": obj, "type": "string"})
        elif hasattr(obj, "name"):
            result.append(
                {
                    "name": obj.name,
                    "orientation": getattr(obj, "orientation", "maximize"),
                    "weight": getattr(obj, "weight", 1.0),
                }
            )
        else:
            result.append({"value": str(obj), "type": "unknown"})
    return result
