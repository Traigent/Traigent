#!/usr/bin/env python
"""Analyze trace files from test runs.

This script provides analysis and reporting capabilities for
OpenTelemetry traces captured during test execution.

Usage:
    # Analyze traces from a directory
    python scripts/analyze_traces.py tests/optimizer_validation/traces

    # Analyze with JSON output
    python scripts/analyze_traces.py traces/ --output report.json

    # Show detailed span information
    python scripts/analyze_traces.py traces/ --verbose

    # Filter by test name
    python scripts/analyze_traces.py traces/ --test-filter "injection"
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from traigent.utils.secure_path import (
    PathTraversalError,
    safe_read_text,
    safe_write_text,
    validate_path,
)


@dataclass
class SpanSummary:
    """Summary of a single span."""

    name: str
    span_id: str
    parent_span_id: str | None
    start_time_ns: int
    end_time_ns: int
    duration_ms: float
    attributes: dict[str, Any]
    status: str

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SpanSummary:
        """Create from dictionary."""
        return cls(
            name=data.get("name", "unknown"),
            span_id=data.get("span_id", ""),
            parent_span_id=data.get("parent_span_id"),
            start_time_ns=data.get("start_time_ns", 0),
            end_time_ns=data.get("end_time_ns", 0),
            duration_ms=data.get("duration_ms", 0.0),
            attributes=data.get("attributes", {}),
            status=data.get("status", "ok"),
        )


@dataclass
class TraceSummary:
    """Summary of a complete trace."""

    file_path: str
    test_name: str
    scenario_name: str
    span_count: int
    trial_count: int
    duration_ms: float
    root_span: SpanSummary | None
    trial_spans: list[SpanSummary]
    errors: list[str]
    warnings: list[str]

    @property
    def has_errors(self) -> bool:
        """Check if trace has any errors."""
        return len(self.errors) > 0


@dataclass
class AnalysisReport:
    """Complete analysis report."""

    traces: list[TraceSummary] = field(default_factory=list)
    total_span_count: int = 0
    total_trial_count: int = 0
    total_duration_ms: float = 0.0
    error_count: int = 0
    warning_count: int = 0

    def add_trace(self, trace: TraceSummary) -> None:
        """Add a trace to the report."""
        self.traces.append(trace)
        self.total_span_count += trace.span_count
        self.total_trial_count += trace.trial_count
        self.total_duration_ms += trace.duration_ms
        self.error_count += len(trace.errors)
        self.warning_count += len(trace.warnings)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "summary": {
                "trace_count": len(self.traces),
                "total_span_count": self.total_span_count,
                "total_trial_count": self.total_trial_count,
                "total_duration_ms": self.total_duration_ms,
                "error_count": self.error_count,
                "warning_count": self.warning_count,
            },
            "traces": [
                {
                    "file": t.file_path,
                    "test_name": t.test_name,
                    "scenario_name": t.scenario_name,
                    "span_count": t.span_count,
                    "trial_count": t.trial_count,
                    "duration_ms": t.duration_ms,
                    "errors": t.errors,
                    "warnings": t.warnings,
                }
                for t in self.traces
            ],
        }


def load_trace_file(path: Path) -> list[dict[str, Any]]:
    """Load spans from a trace file.

    Args:
        path: Path to JSON trace file

    Returns:
        List of span dictionaries
    """
    data = json.loads(safe_read_text(path, path.parent))

    # Handle both single trace and list of spans
    if isinstance(data, list):
        return data
    elif isinstance(data, dict):
        if "spans" in data:
            return data["spans"]
        return [data]
    return []


def analyze_trace(spans: list[dict[str, Any]], file_path: str) -> TraceSummary:
    """Analyze a list of spans.

    Args:
        spans: List of span dictionaries
        file_path: Path to the trace file

    Returns:
        TraceSummary with analysis results
    """
    errors: list[str] = []
    warnings: list[str] = []

    # Parse spans
    span_summaries = [SpanSummary.from_dict(s) for s in spans]

    # Find root span
    root_spans = [s for s in span_summaries if s.parent_span_id is None]
    root_span = None
    if root_spans:
        # Prefer optimization_session if multiple roots
        for span in root_spans:
            if span.name == "optimization_session":
                root_span = span
                break
        if not root_span:
            root_span = root_spans[0]
    else:
        errors.append("No root span found")

    # Find trial spans
    trial_spans = [s for s in span_summaries if s.name == "trial_execution"]

    # Calculate total duration
    if span_summaries:
        min_start = min(s.start_time_ns for s in span_summaries)
        max_end = max(s.end_time_ns for s in span_summaries)
        duration_ms = (max_end - min_start) / 1_000_000
    else:
        duration_ms = 0.0

    # Extract test info from root span or file name
    test_name = ""
    scenario_name = ""
    if root_span:
        test_name = root_span.attributes.get("test_name", "")
        scenario_name = root_span.attributes.get("scenario_name", "")

    if not test_name:
        test_name = Path(file_path).stem

    # Check for error spans
    error_spans = [s for s in span_summaries if s.status == "error"]
    if error_spans:
        for span in error_spans:
            errors.append(f"Span '{span.name}' has error status")

    # Validate structure
    if root_span and len(root_spans) > 1:
        warnings.append(f"Multiple root spans found: {len(root_spans)}")

    # Check trial spans have required attributes
    for trial in trial_spans:
        if "trial.id" not in trial.attributes:
            warnings.append(f"Trial span {trial.span_id[:8]} missing trial.id")

    return TraceSummary(
        file_path=file_path,
        test_name=test_name,
        scenario_name=scenario_name,
        span_count=len(span_summaries),
        trial_count=len(trial_spans),
        duration_ms=duration_ms,
        root_span=root_span,
        trial_spans=trial_spans,
        errors=errors,
        warnings=warnings,
    )


def analyze_directory(
    traces_dir: Path,
    test_filter: str | None = None,
) -> AnalysisReport:
    """Analyze all trace files in a directory.

    Args:
        traces_dir: Directory containing trace files
        test_filter: Optional substring to filter test names

    Returns:
        AnalysisReport with all trace summaries
    """
    report = AnalysisReport()

    if not traces_dir.exists():
        print(f"Warning: Traces directory does not exist: {traces_dir}")
        return report

    trace_files = list(traces_dir.glob("*.json"))
    if not trace_files:
        print(f"Warning: No trace files found in: {traces_dir}")
        return report

    for trace_file in sorted(trace_files):
        try:
            spans = load_trace_file(trace_file)
            summary = analyze_trace(spans, str(trace_file))

            # Apply test filter
            if test_filter:
                if test_filter.lower() not in summary.test_name.lower():
                    continue

            report.add_trace(summary)
        except Exception as e:
            print(f"Error analyzing {trace_file}: {e}")

    return report


def print_report(report: AnalysisReport, verbose: bool = False) -> None:
    """Print analysis report to stdout.

    Args:
        report: Analysis report to print
        verbose: Whether to show detailed span information
    """
    print("=" * 60)
    print("TRACE ANALYSIS REPORT")
    print("=" * 60)
    print()

    print("Summary:")
    print(f"  Traces analyzed: {len(report.traces)}")
    print(f"  Total spans:     {report.total_span_count}")
    print(f"  Total trials:    {report.total_trial_count}")
    print(f"  Total duration:  {report.total_duration_ms:.2f}ms")
    print(f"  Errors:          {report.error_count}")
    print(f"  Warnings:        {report.warning_count}")
    print()

    if report.traces:
        print("-" * 60)
        print("Traces:")
        print("-" * 60)

        for trace in report.traces:
            status = "FAIL" if trace.has_errors else "OK"
            print(f"\n[{status}] {trace.test_name}")
            if trace.scenario_name:
                print(f"     Scenario: {trace.scenario_name}")
            print(f"     Spans: {trace.span_count}, Trials: {trace.trial_count}")
            print(f"     Duration: {trace.duration_ms:.2f}ms")

            if trace.errors:
                print("     Errors:")
                for error in trace.errors:
                    print(f"       - {error}")

            if trace.warnings:
                print("     Warnings:")
                for warning in trace.warnings:
                    print(f"       - {warning}")

            if verbose and trace.trial_spans:
                print("     Trial spans:")
                for trial in trace.trial_spans:
                    trial_id = trial.attributes.get("trial.id", "unknown")
                    config = trial.attributes.get("trial.config", "{}")
                    print(f"       - {trial_id}: {trial.duration_ms:.2f}ms")
                    if verbose:
                        print(f"         Config: {config}")

    print()
    print("=" * 60)


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Analyze trace files from test runs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "traces_dir",
        type=Path,
        help="Directory containing trace JSON files",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        help="Output JSON file for report",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show detailed span information",
    )
    parser.add_argument(
        "--test-filter",
        "-f",
        type=str,
        help="Filter traces by test name (substring match)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output report as JSON to stdout",
    )

    args = parser.parse_args()

    base_dir = Path.cwd()
    try:
        traces_dir = validate_path(args.traces_dir, base_dir, must_exist=True)
    except (PathTraversalError, FileNotFoundError) as exc:
        print(f"Error: {exc}")
        return 1

    # Analyze traces
    report = analyze_directory(traces_dir, test_filter=args.test_filter)

    # Output report
    if args.output:
        output_path = validate_path(args.output, base_dir)
        safe_write_text(
            output_path, json.dumps(report.to_dict(), indent=2), base_dir
        )
        print(f"Report written to: {output_path}")
    elif args.json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        print_report(report, verbose=args.verbose)

    # Return error code if any errors found
    return 1 if report.error_count > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
