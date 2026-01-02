"""OpenTelemetry tracing infrastructure for optimizer validation tests.

This module provides:
- Tracer setup for test execution
- In-memory and OTLP exporters
- Trace capture fixture for test assertions
- Trace analysis and validation
"""

from .analyzer import TraceAnalyzer, TraceValidationResult
from .capture import CapturedTrace, TraceCapture
from .exporters import InMemorySpanExporter, create_exporters
from .invariants import GLOBAL_INVARIANTS, check_invariant
from .tracer import create_test_tracer, optimization_span, trial_span

__all__ = [
    # Tracer
    "create_test_tracer",
    "optimization_span",
    "trial_span",
    # Exporters
    "InMemorySpanExporter",
    "create_exporters",
    # Capture
    "TraceCapture",
    "CapturedTrace",
    # Analysis
    "TraceAnalyzer",
    "TraceValidationResult",
    # Invariants
    "GLOBAL_INVARIANTS",
    "check_invariant",
]
