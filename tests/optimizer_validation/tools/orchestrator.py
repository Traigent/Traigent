#!/usr/bin/env python3
"""Test Review Orchestrator - Coordinates multiple AI CLI tools for test reviews.

This orchestrator manages the test review workflow using:
- Claude CLI (claude) - Primary reviewer with Claude Sonnet 4
- Codex CLI (codex) - Secondary reviewer with code focus
- Internal validation logic - Cross-validates findings

Usage:
    python -m tests.optimizer_validation.tools.orchestrator status
    python -m tests.optimizer_validation.tools.orchestrator review --batch-size 10
    python -m tests.optimizer_validation.tools.orchestrator validate
    python -m tests.optimizer_validation.tools.orchestrator report
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

# Rich for beautiful terminal output
try:
    from rich.console import Console
    from rich.live import Live
    from rich.panel import Panel
    from rich.progress import (
        BarColumn,
        Progress,
        SpinnerColumn,
        TextColumn,
        TimeElapsedColumn,
    )
    from rich.table import Table
    from rich.text import Text

    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False


class CLITool(Enum):
    """Available CLI tools for review."""

    CLAUDE = "claude"
    CODEX = "codex"


class ReviewStatus(Enum):
    """Review status values."""

    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FLAGGED = "flagged"
    FAILED = "failed"


class ValidationStatus(Enum):
    """Validation status values."""

    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    NEEDS_REVISION = "needs_revision"


@dataclass
class BatchResult:
    """Result from a review batch."""

    batch_id: str
    tool: CLITool
    test_ids: list[str]
    success: bool
    reviews: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None
    duration_seconds: float = 0.0


@dataclass
class OrchestratorConfig:
    """Configuration for the orchestrator."""

    test_dir: Path
    tracking_file: Path
    review_protocol: Path
    validation_protocol: Path
    batch_size: int = 10
    max_retries: int = 3
    timeout_seconds: int = 300
    parallel_workers: int = 2


class ProgressDisplay:
    """Display progress in terminal."""

    def __init__(self, console: Console | None = None):
        self.console = console or (Console() if RICH_AVAILABLE else None)

    def show_status(self, tracking_data: dict[str, Any]) -> None:
        """Display current orchestration status."""
        metadata = tracking_data["metadata"]
        tests = tracking_data["tests"]

        review_progress = metadata["review_progress"]
        validation_progress = metadata["validation_progress"]
        total = metadata["total_tests"]

        completed = review_progress.get("completed", 0)
        in_progress = review_progress.get("in_progress", 0)
        not_started = review_progress.get("not_started", 0)
        flagged = review_progress.get("flagged", 0)

        validated = validation_progress.get("approved", 0)
        rejected = validation_progress.get("rejected", 0)
        pending = validation_progress.get("pending", 0)

        if RICH_AVAILABLE and self.console:
            self._show_rich_status(
                total,
                completed,
                in_progress,
                not_started,
                flagged,
                validated,
                rejected,
                pending,
            )
        else:
            self._show_plain_status(
                total,
                completed,
                in_progress,
                not_started,
                flagged,
                validated,
                rejected,
                pending,
            )

    def _show_rich_status(
        self,
        total: int,
        completed: int,
        in_progress: int,
        not_started: int,
        flagged: int,
        validated: int,
        rejected: int,
        pending: int,
    ) -> None:
        """Show status with rich formatting."""
        # Calculate percentages
        review_pct = (completed / total * 100) if total > 0 else 0
        validation_pct = (validated / total * 100) if total > 0 else 0

        # Build progress bar
        bar_width = 40
        filled = int(bar_width * review_pct / 100)
        bar = "█" * filled + "░" * (bar_width - filled)

        # Create header
        header = Text()
        header.append("═══ ", style="blue")
        header.append("Test Review Orchestrator", style="bold cyan")
        header.append(" ═══", style="blue")

        # Create progress section
        progress_text = Text()
        progress_text.append(
            f"\nProgress: [{bar}] {review_pct:.1f}% ({completed}/{total})\n\n"
        )

        # Review status table
        review_table = Table(title="Review Status", show_header=False, box=None)
        review_table.add_column("Icon", width=3)
        review_table.add_column("Status", width=15)
        review_table.add_column("Count", justify="right", width=8)

        review_table.add_row("✓", "Completed", str(completed), style="green")
        review_table.add_row("→", "In Progress", str(in_progress), style="yellow")
        review_table.add_row("○", "Not Started", str(not_started), style="dim")
        review_table.add_row("⚠", "Flagged", str(flagged), style="red")

        # Validation status table
        val_table = Table(title="Validation Status", show_header=False, box=None)
        val_table.add_column("Icon", width=3)
        val_table.add_column("Status", width=15)
        val_table.add_column("Count", justify="right", width=8)

        val_table.add_row("✓", "Approved", str(validated), style="green")
        val_table.add_row("✗", "Rejected", str(rejected), style="red")
        val_table.add_row("?", "Pending", str(pending), style="yellow")

        # Print everything
        self.console.print(header)
        self.console.print(progress_text)
        self.console.print(review_table)
        self.console.print()
        self.console.print(val_table)

    def _show_plain_status(
        self,
        total: int,
        completed: int,
        in_progress: int,
        not_started: int,
        flagged: int,
        validated: int,
        rejected: int,
        pending: int,
    ) -> None:
        """Show status with plain text."""
        review_pct = (completed / total * 100) if total > 0 else 0

        print("\n=== Test Review Orchestrator ===")
        print(f"\nProgress: {review_pct:.1f}% ({completed}/{total})")
        print("\nReview Status:")
        print(f"  ✓ Completed: {completed}")
        print(f"  → In Progress: {in_progress}")
        print(f"  ○ Not Started: {not_started}")
        print(f"  ⚠ Flagged: {flagged}")
        print("\nValidation Status:")
        print(f"  ✓ Approved: {validated}")
        print(f"  ✗ Rejected: {rejected}")
        print(f"  ? Pending: {pending}")
        print()


class CLIAdapter:
    """Adapter for AI CLI tools."""

    def __init__(self, tool: CLITool, timeout: int = 300, verbose: bool = True):
        self.tool = tool
        self.timeout = timeout
        self.verbose = verbose

    def _print_progress(self, message: str) -> None:
        """Print progress message if verbose."""
        if self.verbose:
            timestamp = datetime.now().strftime("%H:%M:%S")
            print(f"  [{timestamp}] {message}")

    async def _stream_claude_output(
        self, process: asyncio.subprocess.Process, timeout: int
    ) -> bytes:
        """Stream Claude CLI output with progress reporting."""
        stdout_data = b""
        last_progress_time = time.time()
        tool_calls_seen = 0

        while True:
            if time.time() - last_progress_time > timeout:
                process.kill()
                raise TimeoutError(f"Claude CLI timed out after {timeout}s")

            try:
                line = await asyncio.wait_for(process.stdout.readline(), timeout=10.0)
            except asyncio.TimeoutError:
                self._print_progress("Waiting for response...")
                continue

            if not line:
                break

            stdout_data += line
            last_progress_time = time.time()
            tool_calls_seen = self._parse_stream_event(line, tool_calls_seen)

        await process.wait()
        return stdout_data

    def _parse_stream_event(self, line: bytes, tool_calls_seen: int) -> int:
        """Parse a streaming JSON event and report progress."""
        try:
            event = json.loads(line.decode())
            event_type = event.get("type", "")

            if event_type == "assistant":
                self._print_progress("Claude is analyzing tests...")
            elif event_type == "content_block_start":
                block = event.get("content_block", {})
                if block.get("type") == "tool_use":
                    tool_calls_seen += 1
                    self._print_progress(
                        f"Tool call #{tool_calls_seen}: {block.get('name', '')}"
                    )
            elif event_type == "result":
                self._print_progress("Review complete, parsing results...")
        except json.JSONDecodeError:
            pass
        return tool_calls_seen

    def _extract_reviews_from_stream(self, output: str) -> list[dict[str, Any]]:
        """Extract reviews JSON from stream output."""
        result_text = ""

        # Parse each line looking for the result event
        for line in output.strip().split("\n"):
            if not line.strip():
                continue
            try:
                event = json.loads(line)
                if event.get("type") == "result":
                    result_text = event.get("result", "")
                    break
            except json.JSONDecodeError:
                continue

        if not result_text:
            result_text = output

        # Find the reviews JSON in the result text
        for pattern in ['{"batch_id"', '{"reviews"', "{"]:
            json_start = result_text.find(pattern)
            if json_start >= 0:
                break
        else:
            return []

        json_end = result_text.rfind("}") + 1
        if json_end <= json_start:
            return []

        reviews_data = json.loads(result_text[json_start:json_end])
        return reviews_data.get("reviews", [])

    async def review_batch(
        self,
        test_ids: list[str],
        protocol_path: Path,
        context_files: list[Path],
    ) -> BatchResult:
        """Run review on a batch of tests."""
        batch_id = (
            f"{self.tool.value}-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"
        )
        start_time = time.time()

        try:
            if self.tool == CLITool.CLAUDE:
                result = await self._run_claude_review(
                    test_ids, protocol_path, context_files
                )
            elif self.tool == CLITool.CODEX:
                result = await self._run_codex_review(
                    test_ids, protocol_path, context_files
                )
            else:
                raise ValueError(f"Unknown tool: {self.tool}")

            duration = time.time() - start_time

            return BatchResult(
                batch_id=batch_id,
                tool=self.tool,
                test_ids=test_ids,
                success=True,
                reviews=result,
                duration_seconds=duration,
            )

        except Exception as e:
            duration = time.time() - start_time
            return BatchResult(
                batch_id=batch_id,
                tool=self.tool,
                test_ids=test_ids,
                success=False,
                error=str(e),
                duration_seconds=duration,
            )

    async def _run_claude_review(
        self,
        test_ids: list[str],
        protocol_path: Path,
        context_files: list[Path],
    ) -> list[dict[str, Any]]:
        """Run Claude CLI for review."""
        # Build the prompt
        protocol_content = protocol_path.read_text()

        # Read context files
        context_parts = []
        for cf in context_files[:3]:  # Limit context files
            if cf.exists():
                content = cf.read_text()[:5000]  # Truncate
                context_parts.append(f"=== {cf.name} ===\n{content}\n")

        context_str = "\n".join(context_parts)

        prompt = f"""You are reviewing optimizer validation tests. Output ONLY valid JSON.

## Protocol Summary:
Answer four questions for each test with 50+ word reasoning:
1. Scenario Validity (VALID/QUESTIONABLE/INVALID)
2. Guard Trigger (TRIGGERED/BYPASSED/UNCLEAR)
3. Failure Detection (WOULD_FAIL/MIGHT_PASS/UNCLEAR)
4. Assertion Quality (CORRECT/PARTIAL/WRONG)

## Context Files:
{context_str}

## Tests to Review:
{json.dumps(test_ids, indent=2)}

## Output Format (JSON only):
{{
  "batch_id": "review-batch-001",
  "reviewer_id": "claude-cli",
  "reviews": [
    {{
      "test_id": "...",
      "findings": {{
        "scenario_validity": {{"verdict": "VALID", "reasoning": "...50+ words..."}},
        "guard_trigger": {{"verdict": "TRIGGERED", "reasoning": "...50+ words...", "trace_evidence": null}},
        "failure_detection": {{"verdict": "WOULD_FAIL", "reasoning": "...50+ words..."}},
        "assertion_quality": {{"verdict": "CORRECT", "reasoning": "...50+ words...", "assertions": []}}
      }},
      "overall_verdict": "PASS",
      "confidence": 0.9,
      "issues": []
    }}
  ]
}}

Read the test files and provide thorough reviews."""

        # Run claude CLI with streaming for progress
        # Note: stream-json requires --verbose when using --print
        cmd = [
            "claude",
            "-p",
            "--verbose",
            "--output-format",
            "stream-json",
            "--allowed-tools",
            "Read,Grep,Glob",
        ]

        self._print_progress(f"Starting Claude CLI for {len(test_ids)} tests...")

        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(protocol_path.parent.parent.parent),
            limit=10 * 1024 * 1024,  # 10MB buffer for large streaming JSON lines
        )

        # Send prompt to stdin
        process.stdin.write(prompt.encode())
        await process.stdin.drain()
        process.stdin.close()

        # Stream output with progress
        try:
            stdout_data = await self._stream_claude_output(process, self.timeout)
        except Exception:
            process.kill()
            raise

        stderr = await process.stderr.read()
        if process.returncode != 0:
            raise RuntimeError(f"Claude CLI failed: {stderr.decode()}")

        # Parse and return reviews
        try:
            return self._extract_reviews_from_stream(stdout_data.decode())
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse Claude response: {e}")

    async def _run_codex_review(
        self,
        test_ids: list[str],
        protocol_path: Path,
        context_files: list[Path],
    ) -> list[dict[str, Any]]:
        """Run Codex CLI for review."""
        protocol_content = protocol_path.read_text()[:3000]  # Truncate for context

        prompt = f"""Review these optimizer validation tests and output JSON.

Protocol: Answer four questions per test with 50+ words reasoning:
1. Scenario Validity (VALID/QUESTIONABLE/INVALID)
2. Guard Trigger (TRIGGERED/BYPASSED/UNCLEAR)
3. Failure Detection (WOULD_FAIL/MIGHT_PASS/UNCLEAR)
4. Assertion Quality (CORRECT/PARTIAL/WRONG)

Tests to review: {json.dumps(test_ids)}

Output JSON with structure:
{{"reviews": [{{"test_id": "...", "findings": {{...}}, "overall_verdict": "PASS|NEEDS_ATTENTION|FAIL", "confidence": 0.9}}]}}"""

        # Codex exec: use "-" to read prompt from stdin for large prompts
        cmd = [
            "codex",
            "exec",
            "-",  # Read prompt from stdin
        ]

        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(protocol_path.parent.parent.parent),
        )

        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(input=prompt.encode()), timeout=self.timeout
            )
        except asyncio.TimeoutError:
            process.kill()
            raise TimeoutError(f"Codex CLI timed out after {self.timeout}s")

        if process.returncode != 0:
            raise RuntimeError(f"Codex CLI failed: {stderr.decode()}")

        output = stdout.decode()
        try:
            response = json.loads(output)
            return response.get("reviews", [])
        except json.JSONDecodeError:
            # Try to extract JSON from output
            json_start = output.find("{")
            json_end = output.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                response = json.loads(output[json_start:json_end])
                return response.get("reviews", [])
            raise ValueError(f"Failed to parse Codex response: {output[:500]}")


class Orchestrator:
    """Main orchestrator for test reviews."""

    def __init__(self, config: OrchestratorConfig):
        self.config = config
        self.display = ProgressDisplay()
        self.adapters = {
            CLITool.CLAUDE: CLIAdapter(CLITool.CLAUDE, config.timeout_seconds),
            CLITool.CODEX: CLIAdapter(CLITool.CODEX, config.timeout_seconds),
        }

    def load_tracking(self) -> dict[str, Any]:
        """Load the tracking file."""
        with open(self.config.tracking_file) as f:
            return json.load(f)

    def save_tracking(self, data: dict[str, Any]) -> None:
        """Save the tracking file."""
        data["metadata"]["last_updated"] = datetime.now(timezone.utc).isoformat()
        with open(self.config.tracking_file, "w") as f:
            json.dump(data, f, indent=2)

    def status(self) -> None:
        """Show current status."""
        tracking = self.load_tracking()
        self.display.show_status(tracking)

        # Show by category
        categories: dict[str, dict[str, int]] = {}
        for test in tracking["tests"]:
            file_path = test["identity"]["file"]
            if file_path.startswith("dimensions/"):
                cat = "dimensions"
            elif file_path.startswith("failures/"):
                cat = "failures"
            elif file_path.startswith("interactions/"):
                cat = "interactions"
            else:
                cat = "other"

            status = test["review"]["status"]
            if cat not in categories:
                categories[cat] = {
                    "not_started": 0,
                    "in_progress": 0,
                    "completed": 0,
                    "flagged": 0,
                }
            categories[cat][status] = categories[cat].get(status, 0) + 1

        print("\nBy Category:")
        for cat, counts in sorted(categories.items()):
            completed = counts.get("completed", 0)
            total = sum(counts.values())
            pct = (completed / total * 100) if total > 0 else 0
            print(f"  {cat}: {completed}/{total} ({pct:.0f}%)")

    def get_pending_tests(
        self, category: str | None = None, limit: int = 10
    ) -> list[dict[str, Any]]:
        """Get tests pending review."""
        tracking = self.load_tracking()
        pending = []

        for test in tracking["tests"]:
            if test["review"]["status"] != ReviewStatus.NOT_STARTED.value:
                continue

            if category:
                file_path = test["identity"]["file"]
                if category == "dimensions" and not file_path.startswith("dimensions/"):
                    continue
                if category == "failures" and not file_path.startswith("failures/"):
                    continue
                if category == "interactions" and not file_path.startswith(
                    "interactions/"
                ):
                    continue
                if category == "other" and (
                    file_path.startswith("dimensions/")
                    or file_path.startswith("failures/")
                    or file_path.startswith("interactions/")
                ):
                    continue

            pending.append(test)
            if len(pending) >= limit:
                break

        return pending

    def claim_batch(self, test_ids: list[str], tool: CLITool) -> str:
        """Claim a batch of tests for review."""
        tracking = self.load_tracking()
        batch_id = f"{tool.value}-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"

        for test in tracking["tests"]:
            if test["id"] in test_ids:
                test["review"]["status"] = ReviewStatus.IN_PROGRESS.value
                test["review"]["batch_id"] = batch_id
                test["review"]["reviewer_id"] = f"{tool.value}-cli"

        # Update progress counts
        self._update_progress_counts(tracking)
        self.save_tracking(tracking)

        return batch_id

    def update_reviews(self, batch_result: BatchResult) -> None:
        """Update tracking with review results."""
        tracking = self.load_tracking()

        for review in batch_result.reviews:
            test_id = review.get("test_id")
            for test in tracking["tests"]:
                if test["id"] == test_id:
                    test["review"]["status"] = ReviewStatus.COMPLETED.value
                    test["review"]["reviewed_at"] = datetime.now(
                        timezone.utc
                    ).isoformat()
                    test["review"]["findings"] = review.get("findings", {})
                    test["review"]["overall_verdict"] = review.get("overall_verdict")
                    test["review"]["confidence"] = review.get("confidence")
                    test["review"]["issues"] = review.get("issues", [])
                    break

        # Mark failed tests
        if not batch_result.success:
            for test_id in batch_result.test_ids:
                for test in tracking["tests"]:
                    if (
                        test["id"] == test_id
                        and test["review"]["status"] == ReviewStatus.IN_PROGRESS.value
                    ):
                        test["review"]["status"] = ReviewStatus.NOT_STARTED.value
                        test["review"]["batch_id"] = None
                        test["review"]["reviewer_id"] = None

        self._update_progress_counts(tracking)
        self.save_tracking(tracking)

    def _update_progress_counts(self, tracking: dict[str, Any]) -> None:
        """Update progress count in metadata."""
        counts = {"not_started": 0, "in_progress": 0, "completed": 0, "flagged": 0}
        val_counts = {"pending": 0, "approved": 0, "rejected": 0, "needs_revision": 0}

        for test in tracking["tests"]:
            status = test["review"]["status"]
            counts[status] = counts.get(status, 0) + 1

            val_status = test["validation"]["status"]
            val_counts[val_status] = val_counts.get(val_status, 0) + 1

        tracking["metadata"]["review_progress"] = counts
        tracking["metadata"]["validation_progress"] = val_counts

    async def review_batch(
        self,
        batch_size: int = 10,
        category: str | None = None,
        tool: CLITool = CLITool.CLAUDE,
    ) -> BatchResult:
        """Review a batch of tests."""
        # Get pending tests
        pending = self.get_pending_tests(category, batch_size)
        if not pending:
            print("No pending tests to review")
            return BatchResult(
                batch_id="empty",
                tool=tool,
                test_ids=[],
                success=True,
                reviews=[],
            )

        test_ids = [t["id"] for t in pending]
        print(f"\nClaiming {len(test_ids)} tests for review with {tool.value}...")

        # Claim the batch
        batch_id = self.claim_batch(test_ids, tool)
        print(f"Batch ID: {batch_id}")

        # Get context files
        context_files = [
            self.config.test_dir / "specs" / "scenario.py",
            self.config.test_dir / "specs" / "validators.py",
            self.config.test_dir / "TESTING_SCENARIOS.md",
        ]

        # Run review
        adapter = self.adapters[tool]
        print(f"Running {tool.value} review...")

        result = await adapter.review_batch(
            test_ids,
            self.config.review_protocol,
            context_files,
        )

        # Update tracking
        self.update_reviews(result)

        if result.success:
            print(
                f"✓ Completed {len(result.reviews)} reviews in {result.duration_seconds:.1f}s"
            )
        else:
            print(f"✗ Review failed: {result.error}")

        return result

    async def run_parallel_reviews(
        self,
        batch_size: int = 10,
        category: str | None = None,
    ) -> list[BatchResult]:
        """Run reviews in parallel with multiple tools."""
        # Split tests between tools
        pending = self.get_pending_tests(category, batch_size * 2)
        if not pending:
            print("No pending tests to review")
            return []

        mid = len(pending) // 2
        claude_tests = pending[:mid]
        codex_tests = pending[mid:]

        results = []

        # Run Claude review
        if claude_tests:
            result = await self.review_batch(
                len(claude_tests), category, CLITool.CLAUDE
            )
            results.append(result)

        # Run Codex review
        if codex_tests:
            result = await self.review_batch(len(codex_tests), category, CLITool.CODEX)
            results.append(result)

        return results

    def validate_completed(self) -> None:
        """Validate completed reviews."""
        tracking = self.load_tracking()

        to_validate = [
            t
            for t in tracking["tests"]
            if t["review"]["status"] == ReviewStatus.COMPLETED.value
            and t["validation"]["status"] == ValidationStatus.PENDING.value
        ]

        if not to_validate:
            print("No reviews pending validation")
            return

        print(f"\nValidating {len(to_validate)} completed reviews...")

        for test in to_validate:
            # Simple validation: check if findings are complete
            findings = test["review"].get("findings", {})
            complete = all(
                findings.get(q, {}).get("verdict") is not None
                for q in [
                    "scenario_validity",
                    "guard_trigger",
                    "failure_detection",
                    "assertion_quality",
                ]
            )

            if complete:
                test["validation"]["status"] = ValidationStatus.APPROVED.value
                test["validation"]["validated_at"] = datetime.now(
                    timezone.utc
                ).isoformat()

                # Set final verdict based on review
                review_verdict = test["review"].get("overall_verdict", "").upper()
                if review_verdict == "PASS":
                    test["validation"]["final_verdict"] = "CONFIRMED_PASS"
                elif review_verdict == "FAIL":
                    test["validation"]["final_verdict"] = "CONFIRMED_FAIL"
                else:
                    test["validation"]["final_verdict"] = "CONFIRMED_NEEDS_ATTENTION"
            else:
                test["validation"]["status"] = ValidationStatus.NEEDS_REVISION.value
                test["validation"]["final_verdict"] = "NEEDS_REVISION"

        self._update_progress_counts(tracking)
        self.save_tracking(tracking)
        print(f"✓ Validated {len(to_validate)} reviews")

    def _collect_issues(self, tests: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Collect issues from test reviews."""
        issues = []
        for test in tests:
            test_issues = test["review"].get("issues", [])
            for issue in test_issues:
                if isinstance(issue, dict):
                    issues.append(
                        {
                            "test_id": test["id"],
                            "severity": issue.get("severity", "unknown"),
                            "description": issue.get("description", "No description"),
                        }
                    )
                else:
                    issues.append(
                        {
                            "test_id": test["id"],
                            "severity": "unknown",
                            "description": str(issue),
                        }
                    )
        return issues

    def generate_report(self) -> str:
        """Generate a summary report."""
        tracking = self.load_tracking()
        metadata = tracking["metadata"]

        report_lines = [
            "# Test Review Orchestration Report",
            f"\nGenerated: {datetime.now(timezone.utc).isoformat()}",
            f"Git SHA: {metadata.get('git_sha', 'unknown')}",
            "",
            "## Progress Summary",
            "",
            f"Total Tests: {metadata['total_tests']}",
            "",
            "### Review Status",
        ]

        review_progress = metadata["review_progress"]
        for status, count in review_progress.items():
            report_lines.append(f"- {status}: {count}")

        report_lines.extend(
            [
                "",
                "### Validation Status",
            ]
        )

        validation_progress = metadata["validation_progress"]
        for status, count in validation_progress.items():
            report_lines.append(f"- {status}: {count}")

        # Find issues
        issues = self._collect_issues(tracking["tests"])

        if issues:
            report_lines.extend(
                [
                    "",
                    "## Issues Found",
                    "",
                ]
            )
            for issue in issues[:20]:  # Limit to 20
                report_lines.append(
                    f"- [{issue['severity']}] {issue['test_id']}: {issue['description']}"
                )

        report = "\n".join(report_lines)

        # Save report
        report_path = self.config.test_dir / "review_report.md"
        report_path.write_text(report)
        print(f"Report saved to: {report_path}")

        return report


def get_default_config() -> OrchestratorConfig:
    """Get default configuration."""
    script_dir = Path(__file__).parent
    test_dir = script_dir.parent

    return OrchestratorConfig(
        test_dir=test_dir,
        tracking_file=test_dir / "test_tracking.json",
        review_protocol=test_dir / "REVIEW_PROTOCOL.md",
        validation_protocol=test_dir / "VALIDATION_PROTOCOL.md",
    )


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Test Review Orchestrator - Coordinate AI tools for test reviews"
    )
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Status command
    subparsers.add_parser("status", help="Show current status")

    # Review command
    review_parser = subparsers.add_parser("review", help="Run review batch")
    review_parser.add_argument(
        "--batch-size", "-b", type=int, default=10, help="Batch size"
    )
    review_parser.add_argument(
        "--category",
        "-c",
        choices=["dimensions", "failures", "interactions", "other"],
        help="Category filter",
    )
    review_parser.add_argument(
        "--tool",
        "-t",
        choices=["claude", "codex"],
        default="claude",
        help="CLI tool to use",
    )
    review_parser.add_argument(
        "--parallel", "-p", action="store_true", help="Run parallel reviews"
    )

    # Validate command
    subparsers.add_parser("validate", help="Validate completed reviews")

    # Report command
    subparsers.add_parser("report", help="Generate summary report")

    args = parser.parse_args()

    config = get_default_config()
    orchestrator = Orchestrator(config)

    if args.command == "status":
        orchestrator.status()

    elif args.command == "review":
        tool = CLITool.CLAUDE if args.tool == "claude" else CLITool.CODEX
        if args.parallel:
            await orchestrator.run_parallel_reviews(args.batch_size, args.category)
        else:
            await orchestrator.review_batch(args.batch_size, args.category, tool)
        orchestrator.status()

    elif args.command == "validate":
        orchestrator.validate_completed()
        orchestrator.status()

    elif args.command == "report":
        report = orchestrator.generate_report()
        print("\n" + report)

    else:
        parser.print_help()


if __name__ == "__main__":
    asyncio.run(main())
