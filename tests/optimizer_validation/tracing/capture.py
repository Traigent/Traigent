"""Trace capture fixture for test execution.

Provides TraceCapture class that wraps test execution and
captures all spans for later analysis.
"""

from __future__ import annotations

import time
from collections.abc import Generator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from .exporters import (
    InMemorySpanExporter,
    SpanData,
    create_exporters,
    get_memory_exporter,
)
from .tracer import TracingConfig, create_test_tracer

try:
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider

    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False
    trace = None  # type: ignore
    TracerProvider = None  # type: ignore

if TYPE_CHECKING:
    from opentelemetry.trace import Tracer

    from tests.optimizer_validation.specs.scenario import TestScenario


@dataclass
class CapturedTrace:
    """Captured trace data for a test.

    Contains all spans captured during test execution along with
    convenience methods for querying and analyzing the trace.
    """

    test_name: str
    scenario_name: str
    spans: list[SpanData] = field(default_factory=list)
    start_time_ns: int = 0
    end_time_ns: int = 0

    @property
    def root_span(self) -> SpanData | None:
        """Get the root span (optimization_session)."""
        root_spans = [s for s in self.spans if s.parent_span_id is None]
        if root_spans:
            # Prefer optimization_session if multiple roots
            for span in root_spans:
                if span.name == "optimization_session":
                    return span
            return root_spans[0]
        return None

    @property
    def trial_spans(self) -> list[SpanData]:
        """Get all trial execution spans."""
        return [s for s in self.spans if s.name == "trial_execution"]

    @property
    def evaluation_spans(self) -> list[SpanData]:
        """Get all evaluation spans."""
        return [s for s in self.spans if s.name == "evaluation"]

    @property
    def duration_ms(self) -> float:
        """Get total trace duration in milliseconds."""
        if self.end_time_ns and self.start_time_ns:
            return (self.end_time_ns - self.start_time_ns) / 1_000_000
        return 0.0

    @property
    def span_count(self) -> int:
        """Get total number of spans."""
        return len(self.spans)

    @property
    def trial_count(self) -> int:
        """Get number of trial spans."""
        return len(self.trial_spans)

    def get_span(self, name: str) -> SpanData | None:
        """Get first span by name.

        Args:
            name: Span name to search for

        Returns:
            First matching span or None
        """
        for span in self.spans:
            if span.name == name:
                return span
        return None

    def get_spans_by_name(self, name: str) -> list[SpanData]:
        """Get all spans with given name.

        Args:
            name: Span name to filter by

        Returns:
            List of matching spans
        """
        return [s for s in self.spans if s.name == name]

    def get_trial_span(self, trial_id: str) -> SpanData | None:
        """Get trial span by trial_id attribute.

        Args:
            trial_id: Trial ID to search for

        Returns:
            Matching trial span or None
        """
        for span in self.trial_spans:
            if span.attributes.get("trial.id") == trial_id:
                return span
        return None

    def get_child_spans(self, parent: SpanData) -> list[SpanData]:
        """Get all child spans of a given span.

        Args:
            parent: Parent span

        Returns:
            List of child spans
        """
        return [s for s in self.spans if s.parent_span_id == parent.span_id]

    def get_span_tree(self) -> dict[str, Any]:
        """Build hierarchical span tree.

        Returns:
            Dict representing span hierarchy
        """

        def build_tree(span: SpanData) -> dict[str, Any]:
            children = self.get_child_spans(span)
            return {
                "name": span.name,
                "span_id": span.span_id,
                "duration_ms": span.duration_ms,
                "attributes": span.attributes,
                "children": [build_tree(c) for c in children],
            }

        root = self.root_span
        if root:
            return build_tree(root)
        return {}

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "test_name": self.test_name,
            "scenario_name": self.scenario_name,
            "duration_ms": self.duration_ms,
            "span_count": self.span_count,
            "trial_count": self.trial_count,
            "spans": [s.to_dict() for s in self.spans],
        }


class TraceCapture:
    """Captures traces during test execution.

    Usage:
        capture = TraceCapture("test_name")
        capture.start()

        # Run test...
        capture.start_scenario(scenario)

        # After test...
        trace = capture.finish()
        # Analyze trace...
    """

    def __init__(self, test_name: str, config: TracingConfig | None = None) -> None:
        """Initialize trace capture.

        Args:
            test_name: Name of the test being captured
            config: Optional tracing configuration
        """
        self.test_name = test_name
        self.config = config or TracingConfig.from_env()
        self._exporters: list[Any] = []
        self._tracer: Tracer | None = None
        self._memory_exporter: InMemorySpanExporter | None = None
        self._scenario_name: str = ""
        self._start_time_ns: int = 0
        self._started = False

    @property
    def tracer(self) -> Tracer | None:
        """Get the configured tracer."""
        return self._tracer

    @property
    def is_enabled(self) -> bool:
        """Check if tracing is enabled."""
        return OTEL_AVAILABLE and self.config.enabled and self._tracer is not None

    def start(self) -> None:
        """Start trace capture.

        Sets up the tracer and exporters.
        """
        if not OTEL_AVAILABLE or not self.config.enabled:
            return

        self._exporters = create_exporters(self.config)
        self._memory_exporter = get_memory_exporter(self._exporters)
        self._tracer = create_test_tracer(
            self.test_name,
            exporters=self._exporters,
            config=self.config,
        )
        self._start_time_ns = time.time_ns()
        self._started = True

    def start_scenario(self, scenario: TestScenario) -> None:
        """Mark the start of a scenario execution.

        Args:
            scenario: The test scenario being executed
        """
        self._scenario_name = scenario.name

    def finish(self) -> CapturedTrace:
        """Finish capture and return captured trace.

        Forces flush of any pending spans and returns
        the captured trace data.

        Returns:
            CapturedTrace with all captured spans
        """
        end_time_ns = time.time_ns()

        # Force flush all exporters
        if OTEL_AVAILABLE:
            provider = trace.get_tracer_provider()
            if isinstance(provider, TracerProvider):
                provider.force_flush()

        # Get captured spans
        spans: list[SpanData] = []
        if self._memory_exporter:
            spans = self._memory_exporter.get_spans()

        return CapturedTrace(
            test_name=self.test_name,
            scenario_name=self._scenario_name,
            spans=spans,
            start_time_ns=self._start_time_ns,
            end_time_ns=end_time_ns,
        )

    def clear(self) -> None:
        """Clear captured spans."""
        if self._memory_exporter:
            self._memory_exporter.clear()

    def shutdown(self) -> None:
        """Shutdown the trace capture."""
        if OTEL_AVAILABLE:
            provider = trace.get_tracer_provider()
            if isinstance(provider, TracerProvider):
                provider.shutdown()


# Pytest fixture helper
def create_trace_capture_fixture():
    """Create a pytest fixture for trace capture.

    Usage in conftest.py:
        trace_capture = pytest.fixture(create_trace_capture_fixture())
    """
    import pytest

    @pytest.fixture
    def trace_capture(request) -> Generator[TraceCapture, None, None]:
        """Fixture for capturing and analyzing traces."""
        test_name = request.node.name
        capture = TraceCapture(test_name)
        capture.start()
        yield capture
        capture.shutdown()

    return trace_capture
