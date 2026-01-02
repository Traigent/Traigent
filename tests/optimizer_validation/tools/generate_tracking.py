#!/usr/bin/env python3
"""Generate test_tracking.json for optimizer validation test review.

This script extends the existing TestParser from knowledge_graph.py to generate
a comprehensive tracking file for LLM-assisted test review and validation.

Usage:
    python -m tests.optimizer_validation.tools.generate_tracking
    python -m tests.optimizer_validation.tools.generate_tracking --output custom_path.json
"""

from __future__ import annotations

import ast
import hashlib
import json
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Import the existing TestParser
from tests.optimizer_validation.viewer.knowledge_graph import TestInfo, TestParser

SCHEMA_VERSION = "1.0.0"


@dataclass
class ReviewFindings:
    """Findings from a test review."""

    scenario_validity: dict[str, Any] = field(
        default_factory=lambda: {"verdict": None, "reasoning": None}
    )
    guard_trigger: dict[str, Any] = field(
        default_factory=lambda: {
            "verdict": None,
            "reasoning": None,
            "trace_evidence": None,
        }
    )
    failure_detection: dict[str, Any] = field(
        default_factory=lambda: {"verdict": None, "reasoning": None}
    )
    assertion_quality: dict[str, Any] = field(
        default_factory=lambda: {"verdict": None, "reasoning": None, "assertions": []}
    )


@dataclass
class ReviewStatus:
    """Review status for a test."""

    status: str = "not_started"  # not_started, in_progress, completed, flagged
    reviewer_id: str | None = None
    reviewed_at: str | None = None
    batch_id: str | None = None
    findings: ReviewFindings = field(default_factory=ReviewFindings)
    overall_verdict: str | None = None  # PASS, NEEDS_ATTENTION, FAIL
    confidence: float | None = None
    issues: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class ValidationStatus:
    """Validation status for a reviewed test."""

    status: str = "pending"  # pending, approved, rejected, needs_revision
    validator_id: str | None = None
    validated_at: str | None = None
    cross_references: list[dict[str, Any]] = field(default_factory=list)
    disagreements: list[dict[str, Any]] = field(default_factory=list)
    final_verdict: str | None = (
        None  # CONFIRMED_PASS, CONFIRMED_FAIL, OVERRIDE_*, ESCALATE
    )


@dataclass
class TestIdentity:
    """Identity information for a test."""

    file: str
    class_name: str
    function: str
    line_start: int
    line_end: int
    markers: list[str]
    parametrized: bool
    param_id: str | None
    param_values: dict[str, Any] | None


@dataclass
class ScenarioInfo:
    """Extracted scenario information."""

    name: str | None
    injection_mode: str | None
    execution_mode: str | None
    expected_outcome: str | None
    expected_stop_reasons: list[str] | None
    source_confidence: str  # extracted, inferred, unknown


@dataclass
class AuditInfo:
    """Audit information for tracking code changes."""

    code_hash: str
    code_excerpt: str


@dataclass
class TestTrackingEntry:
    """Complete tracking entry for a single test."""

    id: str
    identity: TestIdentity
    scenario: ScenarioInfo
    audit: AuditInfo
    review: ReviewStatus
    validation: ValidationStatus


class TrackingGenerator:
    """Generate test_tracking.json from test files."""

    def __init__(self, test_dir: Path):
        self.test_dir = test_dir
        self.parser = TestParser(test_dir)

    def get_git_sha(self) -> str:
        """Get current git commit SHA."""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                cwd=self.test_dir,
                check=False,
            )
            if result.returncode == 0:
                return result.stdout.strip()[:12]
        except Exception:
            pass
        return "unknown"

    def compute_code_hash(self, source: str) -> str:
        """Compute SHA256 hash of source code."""
        return f"sha256:{hashlib.sha256(source.encode()).hexdigest()[:16]}"

    def extract_code_excerpt(self, source: str, max_lines: int = 20) -> str:
        """Extract first N lines of source code."""
        lines = source.split("\n")[:max_lines]
        excerpt = "\n".join(lines)
        if len(source.split("\n")) > max_lines:
            excerpt += "\n..."
        return excerpt

    def get_function_source(
        self, file_path: Path, class_name: str, func_name: str
    ) -> tuple[str, int, int]:
        """Extract function source code and line numbers."""
        try:
            source = file_path.read_text()
            tree = ast.parse(source)

            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef) and node.name == class_name:
                    for item in node.body:
                        if isinstance(item, ast.FunctionDef | ast.AsyncFunctionDef):
                            if item.name == func_name:
                                func_source = ast.get_source_segment(source, item) or ""
                                line_start = item.lineno
                                line_end = item.end_lineno or item.lineno
                                return func_source, line_start, line_end
        except Exception:
            pass
        return "", 0, 0

    def infer_source_confidence(self, test_info: TestInfo) -> str:
        """Determine confidence level of extracted scenario info."""
        # Check if we have explicit scenario fields
        dims = test_info.dimensions
        if dims.get("InjectionMode") and dims.get("ExecutionMode"):
            return "extracted"
        if dims:
            return "inferred"
        return "unknown"

    def convert_test_info(self, test_info: TestInfo) -> TestTrackingEntry:
        """Convert TestInfo to TestTrackingEntry."""
        # Get full file path
        file_path = self.test_dir.parent.parent / test_info.file_path

        # Get function source and line numbers
        func_source, line_start, line_end = self.get_function_source(
            file_path, test_info.class_name, test_info.method_name
        )

        # Build param_id for parametrized tests
        param_id = None
        param_values = None
        if test_info.param_values:
            param_values = test_info.param_values
            param_parts = [
                f"{k}={v}" for k, v in sorted(test_info.param_values.items())
            ]
            param_id = ",".join(param_parts)

        # Extract relative file path
        rel_file = test_info.file_path.replace("tests/optimizer_validation/", "")

        # Build identity
        identity = TestIdentity(
            file=rel_file,
            class_name=test_info.class_name,
            function=test_info.method_name,
            line_start=line_start,
            line_end=line_end,
            markers=test_info.markers,
            parametrized=bool(test_info.param_values),
            param_id=param_id,
            param_values=param_values,
        )

        # Extract stop reasons as array
        stop_reasons = None
        stop_cond = test_info.dimensions.get("StopCondition")
        if stop_cond:
            stop_reasons = [stop_cond]

        # Build scenario info
        scenario = ScenarioInfo(
            name=test_info.display_name or test_info.method_name,
            injection_mode=test_info.dimensions.get("InjectionMode"),
            execution_mode=test_info.dimensions.get("ExecutionMode"),
            expected_outcome=test_info.expected_outcome,
            expected_stop_reasons=stop_reasons,
            source_confidence=self.infer_source_confidence(test_info),
        )

        # Build audit info
        audit = AuditInfo(
            code_hash=self.compute_code_hash(func_source) if func_source else "unknown",
            code_excerpt=self.extract_code_excerpt(func_source) if func_source else "",
        )

        # Build full test ID
        test_id = (
            test_info.test_id
            or f"{test_info.file_path}::{test_info.class_name}::{test_info.method_name}"
        )

        return TestTrackingEntry(
            id=test_id,
            identity=identity,
            scenario=scenario,
            audit=audit,
            review=ReviewStatus(),
            validation=ValidationStatus(),
        )

    def generate(self) -> dict[str, Any]:
        """Generate complete tracking data."""
        # Parse all tests using existing parser
        tests = self.parser.parse_all()

        # Convert to tracking entries
        entries = []
        for test_info in tests:
            try:
                entry = self.convert_test_info(test_info)
                entries.append(entry)
            except Exception as e:
                print(f"Warning: Failed to process {test_info.test_id}: {e}")

        # Build metadata
        now = datetime.now(timezone.utc).isoformat()
        total_tests = len(entries)

        tracking_data = {
            "metadata": {
                "schema_version": SCHEMA_VERSION,
                "git_sha": self.get_git_sha(),
                "created_at": now,
                "last_updated": now,
                "total_tests": total_tests,
                "review_progress": {
                    "not_started": total_tests,
                    "in_progress": 0,
                    "completed": 0,
                    "flagged": 0,
                },
                "validation_progress": {
                    "pending": total_tests,
                    "approved": 0,
                    "rejected": 0,
                    "needs_revision": 0,
                },
            },
            "tests": [self._entry_to_dict(e) for e in entries],
        }

        return tracking_data

    def _entry_to_dict(self, entry: TestTrackingEntry) -> dict[str, Any]:
        """Convert entry to dictionary, handling nested dataclasses."""
        return {
            "id": entry.id,
            "identity": {
                "file": entry.identity.file,
                "class": entry.identity.class_name,
                "function": entry.identity.function,
                "line_start": entry.identity.line_start,
                "line_end": entry.identity.line_end,
                "markers": entry.identity.markers,
                "parametrized": entry.identity.parametrized,
                "param_id": entry.identity.param_id,
                "param_values": entry.identity.param_values,
            },
            "scenario": {
                "name": entry.scenario.name,
                "injection_mode": entry.scenario.injection_mode,
                "execution_mode": entry.scenario.execution_mode,
                "expected_outcome": entry.scenario.expected_outcome,
                "expected_stop_reasons": entry.scenario.expected_stop_reasons,
                "source_confidence": entry.scenario.source_confidence,
            },
            "audit": {
                "code_hash": entry.audit.code_hash,
                "code_excerpt": entry.audit.code_excerpt,
            },
            "review": {
                "status": entry.review.status,
                "reviewer_id": entry.review.reviewer_id,
                "reviewed_at": entry.review.reviewed_at,
                "batch_id": entry.review.batch_id,
                "findings": {
                    "scenario_validity": entry.review.findings.scenario_validity,
                    "guard_trigger": entry.review.findings.guard_trigger,
                    "failure_detection": entry.review.findings.failure_detection,
                    "assertion_quality": entry.review.findings.assertion_quality,
                },
                "overall_verdict": entry.review.overall_verdict,
                "confidence": entry.review.confidence,
                "issues": entry.review.issues,
            },
            "validation": {
                "status": entry.validation.status,
                "validator_id": entry.validation.validator_id,
                "validated_at": entry.validation.validated_at,
                "cross_references": entry.validation.cross_references,
                "disagreements": entry.validation.disagreements,
                "final_verdict": entry.validation.final_verdict,
            },
        }


def main():
    """Generate test_tracking.json."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate test_tracking.json for optimizer validation tests"
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Output file path (default: test_tracking.json in optimizer_validation dir)",
    )
    parser.add_argument(
        "--test-dir",
        type=Path,
        default=None,
        help="Test directory (default: tests/optimizer_validation)",
    )

    args = parser.parse_args()

    # Determine paths
    script_dir = Path(__file__).parent
    test_dir = args.test_dir or script_dir.parent
    output_path = args.output or test_dir / "test_tracking.json"

    print(f"Parsing tests from: {test_dir}")

    # Generate tracking data
    generator = TrackingGenerator(test_dir)
    tracking_data = generator.generate()

    # Write output
    with open(output_path, "w") as f:
        json.dump(tracking_data, f, indent=2)

    total = tracking_data["metadata"]["total_tests"]
    print(f"Generated {output_path}")
    print(f"Total tests: {total}")

    # Print category breakdown
    categories = {}
    for test in tracking_data["tests"]:
        file_path = test["identity"]["file"]
        if file_path.startswith("dimensions/"):
            cat = "dimensions"
        elif file_path.startswith("failures/"):
            cat = "failures"
        elif file_path.startswith("interactions/"):
            cat = "interactions"
        elif file_path.startswith("viewer/"):
            cat = "viewer"
        else:
            cat = "other"
        categories[cat] = categories.get(cat, 0) + 1

    print("\nBy category:")
    for cat, count in sorted(categories.items()):
        print(f"  {cat}: {count}")


if __name__ == "__main__":
    main()
