#!/usr/bin/env python3
"""Test Description Reviewer - Reviews and improves test docstrings.

This tool helps AI agents review test descriptions and suggest improvements
following the DESCRIPTION_REVIEW_PROTOCOL.md guidelines.

Usage:
    python -m tests.optimizer_validation.tools.description_reviewer status
    python -m tests.optimizer_validation.tools.description_reviewer review --category dimensions
    python -m tests.optimizer_validation.tools.description_reviewer apply --batch-id desc-001
"""

from __future__ import annotations

import argparse
import ast
import asyncio
import json
import re
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table

    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False


@dataclass
class TestDescription:
    """Information about a test's description."""

    test_id: str
    file_path: str
    class_name: str
    function_name: str
    line_start: int
    line_end: int
    current_docstring: str
    quality_score: int | None = None
    issues: list[str] = field(default_factory=list)
    improved_docstring: str | None = None
    review_status: str = "pending"  # pending, reviewed, approved, applied


class DescriptionExtractor:
    """Extract test descriptions from Python files."""

    def __init__(self, test_dir: Path):
        self.test_dir = test_dir

    def extract_all(self, category: str | None = None) -> list[TestDescription]:
        """Extract all test descriptions."""
        descriptions = []

        # Find test files
        if category:
            pattern = f"{category}/test_*.py"
        else:
            pattern = "**/test_*.py"

        for test_file in self.test_dir.glob(pattern):
            if "tools" in str(test_file) or "viewer" in str(test_file):
                continue
            descriptions.extend(self._extract_from_file(test_file))

        return descriptions

    def _extract_from_file(self, file_path: Path) -> list[TestDescription]:
        """Extract test descriptions from a single file."""
        descriptions = []

        try:
            source = file_path.read_text()
            tree = ast.parse(source)

            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef) and node.name.startswith("Test"):
                    class_name = node.name
                    for item in node.body:
                        if isinstance(item, ast.FunctionDef | ast.AsyncFunctionDef):
                            if item.name.startswith("test_"):
                                docstring = ast.get_docstring(item) or ""
                                rel_path = file_path.relative_to(
                                    self.test_dir.parent.parent
                                )

                                descriptions.append(
                                    TestDescription(
                                        test_id=f"{rel_path}::{class_name}::{item.name}",
                                        file_path=str(file_path),
                                        class_name=class_name,
                                        function_name=item.name,
                                        line_start=item.lineno,
                                        line_end=item.end_lineno or item.lineno,
                                        current_docstring=docstring,
                                    )
                                )
        except Exception as e:
            print(f"Warning: Failed to parse {file_path}: {e}")

        return descriptions


class DescriptionAnalyzer:
    """Analyze test description quality."""

    def analyze(self, desc: TestDescription) -> TestDescription:
        """Analyze a test description and score it."""
        docstring = desc.current_docstring
        issues = []
        score = 0

        # Check specificity (0-2)
        if len(docstring) > 50 and any(
            word in docstring.lower()
            for word in ["validates", "verifies", "ensures", "checks"]
        ):
            score += 2
        elif len(docstring) > 20:
            score += 1
        else:
            issues.append("too_short")

        # Check for scenario description (0-2)
        if "scenario:" in docstring.lower() or "when" in docstring.lower():
            score += 2
        elif any(word in docstring.lower() for word in ["with", "using", "given"]):
            score += 1
        else:
            issues.append("missing_scenario")

        # Check for assertions/validations listed (0-2)
        if "validates:" in docstring.lower() or "- " in docstring:
            score += 2
        elif any(word in docstring.lower() for word in ["assert", "check", "verify"]):
            score += 1
        else:
            issues.append("no_assertions_listed")

        # Check for impact/why it matters (0-2)
        if "why it matters:" in docstring.lower() or "impact" in docstring.lower():
            score += 2
        elif any(
            word in docstring.lower() for word in ["important", "critical", "users"]
        ):
            score += 1
        else:
            issues.append("no_impact_explained")

        # Check for failure meaning (0-2)
        if "failure" in docstring.lower() or "breaks" in docstring.lower():
            score += 2
        elif "error" in docstring.lower() or "exception" in docstring.lower():
            score += 1
        else:
            issues.append("unclear_failure_meaning")

        # Generic/vague check
        vague_phrases = [
            "test that",
            "test the",
            "should work",
            "works correctly",
            "basic test",
        ]
        if any(phrase in docstring.lower() for phrase in vague_phrases):
            issues.append("vague_language")
            score = max(0, score - 1)

        desc.quality_score = score
        desc.issues = issues
        return desc


class DescriptionReviewer:
    """Main reviewer that coordinates AI tools for description improvement."""

    def __init__(self, test_dir: Path):
        self.test_dir = test_dir
        self.extractor = DescriptionExtractor(test_dir)
        self.analyzer = DescriptionAnalyzer()
        self.tracking_file = test_dir / "description_tracking.json"
        self.console = Console() if RICH_AVAILABLE else None

    def load_tracking(self) -> dict[str, Any]:
        """Load tracking data."""
        if self.tracking_file.exists():
            with open(self.tracking_file) as f:
                return json.load(f)
        return {
            "metadata": {
                "created_at": datetime.now(timezone.utc).isoformat(),
                "last_updated": datetime.now(timezone.utc).isoformat(),
            },
            "descriptions": [],
        }

    def save_tracking(self, data: dict[str, Any]) -> None:
        """Save tracking data."""
        data["metadata"]["last_updated"] = datetime.now(timezone.utc).isoformat()
        with open(self.tracking_file, "w") as f:
            json.dump(data, f, indent=2)

    def status(self, category: str | None = None) -> None:
        """Show current status of description reviews."""
        descriptions = self.extractor.extract_all(category)

        # Analyze all
        for desc in descriptions:
            self.analyzer.analyze(desc)

        # Calculate stats
        total = len(descriptions)
        excellent = sum(1 for d in descriptions if (d.quality_score or 0) >= 8)
        adequate = sum(1 for d in descriptions if 5 <= (d.quality_score or 0) < 8)
        poor = sum(1 for d in descriptions if (d.quality_score or 0) < 5)

        # Issue counts
        issue_counts: dict[str, int] = {}
        for desc in descriptions:
            for issue in desc.issues:
                issue_counts[issue] = issue_counts.get(issue, 0) + 1

        if RICH_AVAILABLE and self.console:
            self._show_rich_status(
                total, excellent, adequate, poor, issue_counts, descriptions
            )
        else:
            self._show_plain_status(total, excellent, adequate, poor, issue_counts)

    def _show_rich_status(
        self,
        total: int,
        excellent: int,
        adequate: int,
        poor: int,
        issue_counts: dict[str, int],
        descriptions: list[TestDescription],
    ) -> None:
        """Show status with rich formatting."""
        # Summary panel
        summary = f"""
Total Tests: {total}

Quality Distribution:
  ✓ Excellent (8-10): {excellent} ({excellent/total*100:.0f}%)
  ~ Adequate (5-7):   {adequate} ({adequate/total*100:.0f}%)
  ✗ Poor (0-4):       {poor} ({poor/total*100:.0f}%)
"""
        self.console.print(
            Panel(summary, title="Description Quality Summary", border_style="blue")
        )

        # Issue table
        table = Table(title="Common Issues", show_header=True)
        table.add_column("Issue", style="red")
        table.add_column("Count", justify="right")
        table.add_column("% of Tests", justify="right")

        for issue, count in sorted(issue_counts.items(), key=lambda x: -x[1]):
            table.add_row(issue, str(count), f"{count/total*100:.0f}%")

        self.console.print(table)

        # Worst descriptions
        worst = sorted(descriptions, key=lambda d: d.quality_score or 0)[:10]
        worst_table = Table(
            title="Lowest Quality Descriptions (Top 10)", show_header=True
        )
        worst_table.add_column("Test", width=50)
        worst_table.add_column("Score", justify="right", width=6)
        worst_table.add_column("Issues", width=30)

        for desc in worst:
            test_name = desc.function_name
            worst_table.add_row(
                test_name,
                str(desc.quality_score),
                ", ".join(desc.issues[:3]),
                style="red" if (desc.quality_score or 0) < 3 else None,
            )

        self.console.print(worst_table)

    def _show_plain_status(
        self,
        total: int,
        excellent: int,
        adequate: int,
        poor: int,
        issue_counts: dict[str, int],
    ) -> None:
        """Show status with plain text."""
        print(f"\n=== Description Quality Summary ===")
        print(f"Total Tests: {total}")
        print(f"\nQuality Distribution:")
        print(f"  ✓ Excellent (8-10): {excellent} ({excellent/total*100:.0f}%)")
        print(f"  ~ Adequate (5-7):   {adequate} ({adequate/total*100:.0f}%)")
        print(f"  ✗ Poor (0-4):       {poor} ({poor/total*100:.0f}%)")
        print(f"\nCommon Issues:")
        for issue, count in sorted(issue_counts.items(), key=lambda x: -x[1]):
            print(f"  {issue}: {count} ({count/total*100:.0f}%)")

    def get_tests_for_review(
        self,
        category: str | None = None,
        limit: int = 10,
        min_score: int = 0,
        max_score: int = 4,
    ) -> list[TestDescription]:
        """Get tests that need description review."""
        descriptions = self.extractor.extract_all(category)

        # Analyze and filter
        result = []
        for desc in descriptions:
            self.analyzer.analyze(desc)
            if min_score <= (desc.quality_score or 0) <= max_score:
                result.append(desc)

        # Sort by score (worst first)
        result.sort(key=lambda d: d.quality_score or 0)
        return result[:limit]

    def generate_review_prompt(self, descriptions: list[TestDescription]) -> str:
        """Generate a prompt for AI review."""
        prompt = """You are reviewing test descriptions for the Traigent optimizer validation suite.

## Your Task
For each test, improve the docstring to clearly communicate:
1. What scenario is tested
2. What behavior is validated
3. Why this matters for users

## Description Template
```
Validates that [COMPONENT] [BEHAVIOR] when [CONDITION].

Scenario: [2-3 sentences describing test setup]

Validates:
- [Assertion 1]
- [Assertion 2]

Why it matters: [1-2 sentences on user impact]
```

## Tests to Review

"""
        for desc in descriptions:
            prompt += f"""
### {desc.test_id}
File: {desc.file_path}
Lines: {desc.line_start}-{desc.line_end}
Current Score: {desc.quality_score}/10
Issues: {', '.join(desc.issues)}

Current docstring:
```
{desc.current_docstring}
```

---
"""

        prompt += """
## Output Format (JSON only)
{
  "reviews": [
    {
      "test_id": "...",
      "improved_docstring": "Validates that..."
    }
  ]
}

Read each test file to understand the actual test logic, then provide improved descriptions.
"""
        return prompt

    async def review_with_claude(
        self, descriptions: list[TestDescription], timeout: int = 300
    ) -> list[dict[str, Any]]:
        """Run review using Claude CLI."""
        prompt = self.generate_review_prompt(descriptions)

        cmd = [
            "claude",
            "-p",
            "--output-format",
            "json",
            "--allowed-tools",
            "Read,Grep",
        ]

        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(self.test_dir.parent.parent),
        )

        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(input=prompt.encode()), timeout=timeout
            )
        except asyncio.TimeoutError:
            process.kill()
            raise TimeoutError(f"Claude CLI timed out after {timeout}s")

        if process.returncode != 0:
            raise RuntimeError(f"Claude CLI failed: {stderr.decode()}")

        output = stdout.decode()
        try:
            response = json.loads(output)
            if "result" in response:
                text = response["result"]
                json_start = text.find("{")
                json_end = text.rfind("}") + 1
                if json_start >= 0:
                    return json.loads(text[json_start:json_end]).get("reviews", [])
            return response.get("reviews", [])
        except json.JSONDecodeError:
            # Try to find JSON in output
            json_start = output.find("{")
            json_end = output.rfind("}") + 1
            if json_start >= 0:
                return json.loads(output[json_start:json_end]).get("reviews", [])
            raise

    def apply_improvements(
        self, reviews: list[dict[str, Any]], dry_run: bool = True
    ) -> None:
        """Apply improved descriptions to test files."""
        for review in reviews:
            test_id = review.get("test_id", "")
            improved = review.get("improved_docstring", "")

            if not improved:
                continue

            # Parse test_id to get file info
            parts = test_id.split("::")
            if len(parts) < 3:
                print(f"Skipping invalid test_id: {test_id}")
                continue

            file_path = Path(parts[0])
            if not file_path.is_absolute():
                file_path = self.test_dir.parent.parent / file_path

            class_name = parts[1]
            func_name = parts[2]

            if dry_run:
                print(f"\n=== Would update: {func_name} ===")
                print(f"File: {file_path}")
                print(f"New docstring:\n{improved[:200]}...")
            else:
                self._update_docstring(file_path, class_name, func_name, improved)
                print(f"✓ Updated: {func_name}")

    def _update_docstring(
        self, file_path: Path, class_name: str, func_name: str, new_docstring: str
    ) -> None:
        """Update a function's docstring in a file."""
        source = file_path.read_text()
        tree = ast.parse(source)

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == class_name:
                for item in node.body:
                    if isinstance(item, ast.FunctionDef | ast.AsyncFunctionDef):
                        if item.name == func_name:
                            # Find the docstring location
                            if (
                                item.body
                                and isinstance(item.body[0], ast.Expr)
                                and isinstance(item.body[0].value, ast.Constant)
                            ):
                                # Has existing docstring
                                doc_node = item.body[0]
                                start_line = doc_node.lineno - 1
                                end_line = doc_node.end_lineno or doc_node.lineno

                                lines = source.split("\n")

                                # Get indentation
                                func_line = lines[item.lineno - 1]
                                indent = len(func_line) - len(func_line.lstrip()) + 4

                                # Format new docstring
                                formatted = self._format_docstring(
                                    new_docstring, indent
                                )

                                # Replace
                                new_lines = (
                                    lines[:start_line] + [formatted] + lines[end_line:]
                                )
                                file_path.write_text("\n".join(new_lines))
                                return

    def _format_docstring(self, docstring: str, indent: int) -> str:
        """Format a docstring with proper indentation."""
        indent_str = " " * indent
        lines = docstring.strip().split("\n")

        if len(lines) == 1:
            return f'{indent_str}"""{lines[0]}"""'

        result = [f'{indent_str}"""']
        for line in lines:
            if line.strip():
                result.append(f"{indent_str}{line}")
            else:
                result.append("")
        result.append(f'{indent_str}"""')

        return "\n".join(result)


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Test Description Reviewer")
    subparsers = parser.add_subparsers(dest="command")

    # Status command
    status_parser = subparsers.add_parser(
        "status", help="Show description quality status"
    )
    status_parser.add_argument("--category", "-c", help="Filter by category")

    # Review command
    review_parser = subparsers.add_parser("review", help="Review descriptions with AI")
    review_parser.add_argument("--category", "-c", help="Filter by category")
    review_parser.add_argument(
        "--limit", "-l", type=int, default=10, help="Number of tests to review"
    )
    review_parser.add_argument(
        "--max-score", type=int, default=4, help="Max quality score to include"
    )

    # Apply command
    apply_parser = subparsers.add_parser("apply", help="Apply improvements")
    apply_parser.add_argument("--input", "-i", type=Path, help="JSON file with reviews")
    apply_parser.add_argument(
        "--dry-run", action="store_true", help="Show changes without applying"
    )

    # Generate prompt command
    prompt_parser = subparsers.add_parser("prompt", help="Generate review prompt")
    prompt_parser.add_argument("--category", "-c", help="Filter by category")
    prompt_parser.add_argument(
        "--limit", "-l", type=int, default=10, help="Number of tests"
    )
    prompt_parser.add_argument("--output", "-o", type=Path, help="Output file")

    args = parser.parse_args()

    test_dir = Path(__file__).parent.parent
    reviewer = DescriptionReviewer(test_dir)

    if args.command == "status":
        reviewer.status(args.category)

    elif args.command == "review":
        tests = reviewer.get_tests_for_review(
            args.category, args.limit, max_score=args.max_score
        )
        if not tests:
            print("No tests found needing review")
            return

        print(f"Reviewing {len(tests)} test descriptions with Claude...")
        reviews = await reviewer.review_with_claude(tests)
        print(f"Got {len(reviews)} improved descriptions")

        # Save to file
        output_file = (
            test_dir
            / f"description_reviews_{datetime.now().strftime('%Y%m%d%H%M%S')}.json"
        )
        with open(output_file, "w") as f:
            json.dump({"reviews": reviews}, f, indent=2)
        print(f"Saved to: {output_file}")

    elif args.command == "apply":
        if not args.input:
            print("Error: --input required")
            return

        with open(args.input) as f:
            data = json.load(f)

        reviewer.apply_improvements(data.get("reviews", []), dry_run=args.dry_run)

    elif args.command == "prompt":
        tests = reviewer.get_tests_for_review(args.category, args.limit)
        prompt = reviewer.generate_review_prompt(tests)

        if args.output:
            args.output.write_text(prompt)
            print(f"Prompt saved to: {args.output}")
        else:
            print(prompt)

    else:
        parser.print_help()


if __name__ == "__main__":
    asyncio.run(main())
