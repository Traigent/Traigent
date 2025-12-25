"""Span exporters for trace capture and analysis.

Provides:
- InMemorySpanExporter: Captures spans in memory for test assertions
- FileSpanExporter: Exports spans to JSON files for offline analysis
- Factory function for creating exporter combinations
"""

from __future__ import annotations

import json
import os
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence

try:
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
        OTLPSpanExporter as OTLPHttpSpanExporter,
    )
    from opentelemetry.sdk.trace import ReadableSpan
    from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult

    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False
    ReadableSpan = object  # type: ignore
    SpanExporter = object  # type: ignore
    SpanExportResult = None  # type: ignore
    OTLPHttpSpanExporter = None  # type: ignore

if TYPE_CHECKING:
    from .tracer import TracingConfig


@dataclass
class SpanData:
    """Simplified span data for analysis."""

    name: str
    trace_id: str
    span_id: str
    parent_span_id: str | None
    start_time_ns: int
    end_time_ns: int
    attributes: dict[str, Any]
    status_code: str
    status_description: str | None
    events: list[dict[str, Any]]

    @property
    def duration_ms(self) -> float:
        """Get span duration in milliseconds."""
        return (self.end_time_ns - self.start_time_ns) / 1_000_000

    @property
    def start_time(self) -> datetime:
        """Get start time as datetime."""
        return datetime.fromtimestamp(self.start_time_ns / 1_000_000_000)

    @property
    def end_time(self) -> datetime:
        """Get end time as datetime."""
        return datetime.fromtimestamp(self.end_time_ns / 1_000_000_000)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "name": self.name,
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_span_id": self.parent_span_id,
            "start_time_ns": self.start_time_ns,
            "end_time_ns": self.end_time_ns,
            "duration_ms": self.duration_ms,
            "attributes": self.attributes,
            "status_code": self.status_code,
            "status_description": self.status_description,
            "events": self.events,
        }


class InMemorySpanExporter(SpanExporter if OTEL_AVAILABLE else object):
    """Captures spans in memory for analysis.

    Thread-safe exporter that stores spans for later retrieval
    and analysis in tests.
    """

    def __init__(self) -> None:
        self._spans: list[SpanData] = []
        self._lock = threading.Lock()
        self._shutdown = False

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        """Export spans to memory.

        Args:
            spans: Sequence of spans to export

        Returns:
            SUCCESS if export completed
        """
        if self._shutdown:
            return SpanExportResult.SUCCESS

        with self._lock:
            for span in spans:
                span_data = self._convert_span(span)
                self._spans.append(span_data)

        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:
        """Shut down the exporter."""
        self._shutdown = True

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        """Force flush any pending spans."""
        return True

    def get_spans(self) -> list[SpanData]:
        """Get all captured spans.

        Returns:
            Copy of captured spans list
        """
        with self._lock:
            return self._spans.copy()

    def get_spans_by_name(self, name: str) -> list[SpanData]:
        """Get spans filtered by name.

        Args:
            name: Span name to filter by

        Returns:
            List of matching spans
        """
        with self._lock:
            return [s for s in self._spans if s.name == name]

    def get_root_spans(self) -> list[SpanData]:
        """Get root spans (no parent).

        Returns:
            List of root spans
        """
        with self._lock:
            return [s for s in self._spans if s.parent_span_id is None]

    def clear(self) -> None:
        """Clear all captured spans."""
        with self._lock:
            self._spans.clear()

    def _convert_span(self, span: ReadableSpan) -> SpanData:
        """Convert OTEL span to SpanData."""
        context = span.get_span_context()
        parent = span.parent

        # Convert attributes to simple dict
        attributes = {}
        if span.attributes:
            for key, value in span.attributes.items():
                # Handle various attribute value types
                if hasattr(value, "__iter__") and not isinstance(value, (str, bytes)):
                    attributes[key] = list(value)
                else:
                    attributes[key] = value

        # Convert events
        events = []
        if span.events:
            for event in span.events:
                event_dict = {
                    "name": event.name,
                    "timestamp_ns": event.timestamp,
                }
                if event.attributes:
                    event_dict["attributes"] = dict(event.attributes)
                events.append(event_dict)

        return SpanData(
            name=span.name,
            trace_id=format(context.trace_id, "032x"),
            span_id=format(context.span_id, "016x"),
            parent_span_id=format(parent.span_id, "016x") if parent else None,
            start_time_ns=span.start_time or 0,
            end_time_ns=span.end_time or 0,
            attributes=attributes,
            status_code=span.status.status_code.name if span.status else "UNSET",
            status_description=span.status.description if span.status else None,
            events=events,
        )


class FileSpanExporter(SpanExporter if OTEL_AVAILABLE else object):
    """Exports spans to a JSON file for offline analysis.

    Writes spans as newline-delimited JSON for easy processing.
    """

    def __init__(self, file_path: str | Path) -> None:
        self._file_path = Path(file_path)
        self._file_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._shutdown = False

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        """Export spans to file.

        Args:
            spans: Sequence of spans to export

        Returns:
            SUCCESS if export completed
        """
        if self._shutdown:
            return SpanExportResult.SUCCESS

        with self._lock:
            with open(self._file_path, "a") as f:
                for span in spans:
                    span_data = self._convert_span(span)
                    f.write(json.dumps(span_data.to_dict()) + "\n")

        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:
        """Shut down the exporter."""
        self._shutdown = True

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        """Force flush any pending spans."""
        return True

    def _convert_span(self, span: ReadableSpan) -> SpanData:
        """Convert OTEL span to SpanData."""
        # Reuse InMemorySpanExporter's conversion logic
        temp_exporter = InMemorySpanExporter()
        return temp_exporter._convert_span(span)


def create_exporters(config: TracingConfig) -> list[SpanExporter]:
    """Create exporters based on configuration.

    Always creates an InMemorySpanExporter for test assertions.
    Optionally adds OTLP and file exporters based on config.

    Args:
        config: Tracing configuration

    Returns:
        List of configured exporters
    """
    if not OTEL_AVAILABLE:
        return []

    exporters: list[SpanExporter] = []

    # Always include in-memory exporter for test assertions
    memory_exporter = InMemorySpanExporter()
    exporters.append(memory_exporter)

    # Add OTLP exporter if endpoint configured
    if config.otlp_endpoint:
        otlp_exporter = OTLPHttpSpanExporter(
            endpoint=f"{config.otlp_endpoint}/v1/traces"
        )
        exporters.append(otlp_exporter)

    # Add file exporter if path configured
    if config.file_export_path:
        file_exporter = FileSpanExporter(config.file_export_path)
        exporters.append(file_exporter)

    return exporters


def get_memory_exporter(exporters: list[SpanExporter]) -> InMemorySpanExporter | None:
    """Get the InMemorySpanExporter from a list of exporters.

    Args:
        exporters: List of exporters to search

    Returns:
        The InMemorySpanExporter if found, None otherwise
    """
    for exporter in exporters:
        if isinstance(exporter, InMemorySpanExporter):
            return exporter
    return None
