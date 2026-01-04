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

import os
from typing import Any

from traigent.plugins import FEATURE_TRACING, FeaturePlugin
from traigent.utils.logging import get_logger

logger = get_logger(__name__)

__version__ = "0.1.0"
__all__ = ["TracingPlugin", "configure_tracing", "get_tracer"]


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
        """Initialize the tracing plugin."""
        # Auto-configure if environment variable is set
        if os.environ.get("TRAIGENT_TRACE_ENABLED", "").lower() in ("true", "1", "yes"):
            configure_tracing(enabled=True)
            logger.info("Traigent tracing plugin initialized and enabled")
        else:
            logger.debug("Traigent tracing plugin initialized (disabled)")

    def get_feature_impl(self, feature: str) -> Any | None:
        """Get tracing implementation."""
        if feature == FEATURE_TRACING:
            return get_tracer()
        return None


# Tracing configuration
_tracer = None
_trace_enabled = False


def configure_tracing(
    enabled: bool = True,
    endpoint: str | None = None,
    service_name: str = "traigent-optimizer",
    exporter: str = "otlp",
) -> None:
    """Configure OpenTelemetry tracing.

    Args:
        enabled: Whether to enable tracing
        endpoint: OTLP endpoint URL (default from env OTEL_EXPORTER_OTLP_ENDPOINT)
        service_name: Service name for spans
        exporter: Exporter type ("otlp", "jaeger", "zipkin", "console")
    """
    global _tracer, _trace_enabled

    if not enabled:
        _trace_enabled = False
        _tracer = None
        return

    try:
        from opentelemetry import trace
        from opentelemetry.sdk.resources import SERVICE_NAME, Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        # Get endpoint from env if not specified
        if endpoint is None:
            endpoint = os.environ.get(
                "OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317"
            )

        # Create resource
        resource = Resource(attributes={SERVICE_NAME: service_name})

        # Create provider
        provider = TracerProvider(resource=resource)

        # Create exporter based on type
        if exporter == "otlp":
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
                OTLPSpanExporter,
            )

            span_exporter = OTLPSpanExporter(endpoint=endpoint)
        elif exporter == "console":
            from opentelemetry.sdk.trace.export import ConsoleSpanExporter

            span_exporter = ConsoleSpanExporter()
        else:
            raise ValueError(f"Unknown exporter type: {exporter}")

        # Add processor
        processor = BatchSpanProcessor(span_exporter)
        provider.add_span_processor(processor)

        # Set as global provider
        trace.set_tracer_provider(provider)

        # Get tracer
        _tracer = trace.get_tracer("traigent.tracing", __version__)
        _trace_enabled = True

        logger.info(f"Tracing configured with {exporter} exporter to {endpoint}")

    except ImportError as e:
        logger.warning(f"Failed to configure tracing: {e}")
        _trace_enabled = False


def get_tracer():
    """Get the configured tracer.

    Returns:
        OpenTelemetry Tracer or None if tracing is disabled
    """
    return _tracer if _trace_enabled else None
