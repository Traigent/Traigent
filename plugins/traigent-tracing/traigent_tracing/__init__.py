"""Traigent Tracing Plugin - OpenTelemetry distributed tracing for Traigent.

This plugin provides automatic tracing for Traigent optimization runs,
creating spans for sessions, trials, and evaluations.

Example:
    # Install and enable tracing
    pip install traigent-tracing

    # Configure via environment
    export TRAIGENT_TRACE_ENABLED=true
    export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317

    # Your optimization code runs with automatic tracing
    import traigent

    @traigent.optimize(...)
    def my_function(...):
        ...
"""

from __future__ import annotations

from typing import Any

from traigent.plugins import FEATURE_TRACING, FeaturePlugin
from traigent.utils.logging import get_logger

# Import tracing implementation
from .tracing import (
    OTEL_AVAILABLE,
    TRACING_ENABLED,
    SecureIdGenerator,
    clear_test_context,
    example_evaluation_span,
    get_test_context,
    get_tracer,
    optimization_session_span,
    record_example_result,
    record_optimization_complete,
    record_trial_result,
    set_test_context,
    trial_span,
)

logger = get_logger(__name__)

__version__ = "0.1.0"
__all__ = [
    # Plugin class
    "TracingPlugin",
    # Configuration
    "TRACING_ENABLED",
    "OTEL_AVAILABLE",
    "SecureIdGenerator",
    # Core functions
    "get_tracer",
    # Test context
    "set_test_context",
    "clear_test_context",
    "get_test_context",
    # Span context managers
    "optimization_session_span",
    "trial_span",
    "example_evaluation_span",
    # Result recording
    "record_trial_result",
    "record_optimization_complete",
    "record_example_result",
]


class TracingPlugin(FeaturePlugin):
    """Plugin that provides OpenTelemetry tracing for Traigent."""

    @property
    def name(self) -> str:
        return "traigent-tracing"

    @property
    def version(self) -> str:
        return __version__

    @property
    def description(self) -> str:
        return "OpenTelemetry distributed tracing for Traigent optimization runs"

    @property
    def author(self) -> str:
        return "Traigent Team"

    @property
    def dependencies(self) -> list[str]:
        return ["opentelemetry-api", "opentelemetry-sdk"]

    def provides_features(self) -> list[str]:
        """Declare that this plugin provides tracing capability."""
        return [FEATURE_TRACING]

    def initialize(self) -> None:
        """Initialize the tracing plugin.

        Auto-configures tracing if TRAIGENT_TRACE_ENABLED=true.
        """
        if TRACING_ENABLED:
            # Tracing will be initialized lazily on first get_tracer() call
            logger.info("Traigent tracing plugin initialized (tracing enabled)")
        else:
            logger.debug("Traigent tracing plugin initialized (tracing disabled)")

    def get_feature_impl(self, feature: str) -> Any | None:
        """Get tracing implementation.

        Args:
            feature: Feature name to get implementation for

        Returns:
            The tracer instance for FEATURE_TRACING, None otherwise
        """
        if feature == FEATURE_TRACING:
            return get_tracer()
        return None
