#!/usr/bin/env python3
"""
Broad Test Review Orchestrator

Uses multiple AI models (Codex, Claude Sonnet, Grok) in parallel to review
tests for anti-patterns identified in the META_ANALYSIS_REPORT.md.

Operates in batches to handle the ~9000 tests efficiently.
"""

import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

# Anti-patterns to detect (from lessons learned)
ANTI_PATTERNS = {
    "IT-VRO": {
        "name": "Validator Reliance Only",
        "pattern": r"def test_.*\n(?:.*\n)*?(?:validator|helper)\(.*\)\s*$",
        "keywords": ["validator(", "helper(", "check("],
        "missing": ["assert "],
    },
    "IT-NA": {
        "name": "No Assertions",
        "detection": "no_assert_in_function",
    },
    "IT-VTA": {
        "name": "Vacuous Truth",
        "patterns": [
            r"assert len\([^)]+\) >= 0",
            r"assert .* is not None",  # when field is always present
            r"assert True",
            r"assert 1 == 1",
        ],
    },
    "IT-WA": {
        "name": "Weak Assertions",
        "patterns": [
            r"assert len\([^)]+\) > 0",  # without checking actual value
            r"assert isinstance\([^)]+\)",  # without value check
        ],
    },
}


@dataclass
class ReviewResult:
    """Result from a model review."""

    file_path: str
    model: str
    issues: list[dict] = field(default_factory=list)
    proposed_fixes: list[dict] = field(default_factory=list)
    error: str | None = None


def get_test_files(exclude_dirs: list[str] | None = None) -> list[Path]:
    """Get all test files excluding specified directories."""
    exclude_dirs = exclude_dirs or ["optimizer_validation"]
    tests_dir = Path("tests")
    test_files = []

    for test_file in tests_dir.rglob("test_*.py"):
        if not any(excl in str(test_file) for excl in exclude_dirs):
            test_files.append(test_file)

    return sorted(test_files)


def create_review_prompt(file_path: Path, batch_context: str = "") -> str:
    """Create a prompt for the AI models to review a test file."""
    with open(file_path) as f:
        content = f.read()

    return f"""You are reviewing Python test files for quality issues.

ANTI-PATTERNS TO DETECT (from lessons learned):

1. IT-VRO (Validator Reliance Only): Test only calls a validator/helper without explicit assertions
   - Bad: def test_x(): result = run(); validator(result)  # no assert!
   - Good: def test_x(): result = run(); assert result.success; validator(result)

2. IT-NA (No Assertions): Test function has no assert statements
   - Every test MUST have at least one explicit assert

3. IT-VTA (Vacuous Truth): Assertion always passes
   - Bad: assert len(x) >= 0  # always true
   - Bad: assert result is not None  # when result is always returned
   - Good: assert len(x) == expected_count

4. IT-WA (Weak Assertions): Assertion too weak to catch failures
   - Bad: assert len(x) > 0  # doesn't verify actual content
   - Good: assert len(x) == 5; assert x[0] == expected_value

5. IT-CBM (Condition-Behavior Mismatch): Test setup doesn't trigger intended behavior
   - Example: Testing max_trials=5 but config space has only 2 options (exhausts first)

FILE TO REVIEW: {file_path}

```python
{content}
```

OUTPUT FORMAT (JSON):
{{
  "file": "{file_path}",
  "issues": [
    {{
      "test_name": "test_example",
      "line": 42,
      "issue_type": "IT-VRO",
      "description": "Test only calls validator without explicit assertion",
      "severity": "major",
      "fix": "Add 'assert result.success' before validator call"
    }}
  ],
  "summary": {{
    "total_tests": 10,
    "issues_found": 2,
    "tests_ok": 8
  }}
}}

Only report actual issues. If a test is fine, don't include it in issues.
Focus on the 5 anti-patterns above.
"""


def review_with_codex(file_path: Path, prompt: str) -> ReviewResult:
    """Review a file using Codex CLI."""
    try:
        # Use codex CLI if available
        result = subprocess.run(
            [
                "codex",
                "review",
                "--format",
                "json",
                "--prompt",
                prompt[:4000],  # Limit prompt size
                str(file_path),
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )
        if result.returncode == 0:
            data = json.loads(result.stdout)
            return ReviewResult(
                file_path=str(file_path),
                model="codex",
                issues=data.get("issues", []),
                proposed_fixes=data.get("fixes", []),
            )
        else:
            return ReviewResult(
                file_path=str(file_path),
                model="codex",
                error=f"Codex error: {result.stderr}",
            )
    except FileNotFoundError:
        return ReviewResult(
            file_path=str(file_path),
            model="codex",
            error="Codex CLI not available",
        )
    except Exception as e:
        return ReviewResult(file_path=str(file_path), model="codex", error=str(e))


def review_with_claude(file_path: Path, prompt: str) -> ReviewResult:
    """Review a file using Claude CLI."""
    try:
        # Use claude CLI with haiku for speed
        result = subprocess.run(
            ["claude", "--model", "haiku", "--print", prompt[:8000]],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode == 0:
            # Parse JSON from response
            try:
                # Find JSON in response
                output = result.stdout
                start = output.find("{")
                end = output.rfind("}") + 1
                if start >= 0 and end > start:
                    data = json.loads(output[start:end])
                    return ReviewResult(
                        file_path=str(file_path),
                        model="claude-haiku",
                        issues=data.get("issues", []),
                    )
            except json.JSONDecodeError:
                pass
            return ReviewResult(
                file_path=str(file_path),
                model="claude-haiku",
                error="Could not parse JSON response",
            )
        else:
            return ReviewResult(
                file_path=str(file_path),
                model="claude-haiku",
                error=f"Claude error: {result.stderr}",
            )
    except FileNotFoundError:
        return ReviewResult(
            file_path=str(file_path),
            model="claude-haiku",
            error="Claude CLI not available",
        )
    except Exception as e:
        return ReviewResult(
            file_path=str(file_path), model="claude-haiku", error=str(e)
        )


def static_analysis_review(file_path: Path) -> ReviewResult:
    """Fast static analysis for obvious anti-patterns."""
    issues = []

    with open(file_path) as f:
        content = f.read()
        lines = content.split("\n")

    # Find test functions
    in_test = False
    test_name = ""
    test_start = 0
    test_lines: list[str] = []

    for i, line in enumerate(lines, 1):
        if line.strip().startswith("def test_") or line.strip().startswith(
            "async def test_"
        ):
            # Save previous test if exists
            if in_test and test_lines:
                issues.extend(
                    analyze_test_function(test_name, test_start, test_lines, file_path)
                )
            # Start new test
            in_test = True
            test_name = line.split("(")[0].split()[-1]
            test_start = i
            test_lines = [line]
        elif in_test:
            if (
                line
                and not line.startswith(" ")
                and not line.startswith("\t")
                and not line.startswith("#")
            ):
                # End of function (new top-level definition or class)
                if line.strip() and not line.strip().startswith("@"):
                    issues.extend(
                        analyze_test_function(
                            test_name, test_start, test_lines, file_path
                        )
                    )
                    in_test = False
                    test_lines = []
            else:
                test_lines.append(line)

    # Handle last test
    if in_test and test_lines:
        issues.extend(
            analyze_test_function(test_name, test_start, test_lines, file_path)
        )

    return ReviewResult(
        file_path=str(file_path), model="static-analysis", issues=issues
    )


def analyze_test_function(
    test_name: str, start_line: int, lines: list[str], file_path: Path
) -> list[dict]:
    """Analyze a single test function for anti-patterns."""
    issues = []
    content = "\n".join(lines)

    # IT-NA: No assertions
    if "assert " not in content and "pytest.raises" not in content:
        issues.append(
            {
                "test_name": test_name,
                "line": start_line,
                "issue_type": "IT-NA",
                "description": "Test has no assert statements",
                "severity": "critical",
                "fix": "Add appropriate assertions to verify test behavior",
            }
        )

    # IT-VTA: Vacuous truth assertions
    vacuous_patterns = [
        ("assert len(", ") >= 0", "len >= 0 always true"),
        ("assert True", "", "assert True is meaningless"),
        ("assert 1 == 1", "", "trivial assertion"),
    ]
    for start_pat, end_pat, desc in vacuous_patterns:
        if start_pat in content:
            if not end_pat or end_pat in content:
                issues.append(
                    {
                        "test_name": test_name,
                        "line": start_line,
                        "issue_type": "IT-VTA",
                        "description": f"Vacuous assertion: {desc}",
                        "severity": "major",
                        "fix": "Replace with specific value assertion",
                    }
                )

    # IT-VRO: Validator reliance
    # Check if test ends with validator call without explicit assert
    non_empty_lines = [
        ln for ln in lines if ln.strip() and not ln.strip().startswith("#")
    ]
    if non_empty_lines:
        last_line = non_empty_lines[-1].strip()
        validator_patterns = [
            "validator(",
            "validate(",
            "check_result(",
            "_validator(",
        ]
        if any(p in last_line for p in validator_patterns):
            # Check if there's an assert in the same line
            if "assert" not in last_line:
                # Count asserts in the whole test
                assert_count = content.count("assert ")
                if assert_count == 0:
                    issues.append(
                        {
                            "test_name": test_name,
                            "line": start_line,
                            "issue_type": "IT-VRO",
                            "description": "Test only calls validator without explicit assertions",
                            "severity": "major",
                            "fix": "Add explicit assertions before validator call",
                        }
                    )

    return issues


def process_batch(
    files: list[Path], batch_num: int, use_ai_models: bool = False
) -> list[ReviewResult]:
    """Process a batch of files."""
    results = []

    # Always run static analysis (fast)
    for file_path in files:
        result = static_analysis_review(file_path)
        results.append(result)

    # Optionally use AI models for more thorough review
    if use_ai_models:
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = []
            for file_path in files:
                prompt = create_review_prompt(file_path)
                # Submit to multiple models
                futures.append(executor.submit(review_with_codex, file_path, prompt))
                futures.append(executor.submit(review_with_claude, file_path, prompt))

            for future in as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    print(f"Error in batch {batch_num}: {e}")

    return results


def merge_results(results: list[ReviewResult]) -> dict[str, list[dict]]:
    """Merge results from multiple models, deduplicating issues."""
    merged: dict[str, list[dict]] = {}

    for result in results:
        if result.error:
            continue

        file_path = result.file_path
        if file_path not in merged:
            merged[file_path] = []

        for issue in result.issues:
            # Check for duplicates
            is_dup = False
            for existing in merged[file_path]:
                if existing.get("test_name") == issue.get("test_name") and existing.get(
                    "issue_type"
                ) == issue.get("issue_type"):
                    is_dup = True
                    break
            if not is_dup:
                issue["detected_by"] = result.model
                merged[file_path].append(issue)

    return merged


def update_tracking_file(
    tracking_path: Path,
    batch_num: int,
    files: list[Path],
    issues: dict[str, list[dict]],
) -> None:
    """Update the tracking JSON file with batch results."""
    with open(tracking_path) as f:
        tracking = json.load(f)

    batch_info = {
        "batch_num": batch_num,
        "timestamp": datetime.now().isoformat(),
        "files_reviewed": len(files),
        "files": [str(f) for f in files],
        "issues_found": sum(len(v) for v in issues.values()),
    }
    tracking["batches"].append(batch_info)

    # Add issues
    for file_path, file_issues in issues.items():
        for issue in file_issues:
            tracking["issues_found"].append(
                {
                    "batch": batch_num,
                    "file": file_path,
                    **issue,
                }
            )

    # Update summary
    tracking["summary"]["total_reviewed"] += len(files)
    tracking["summary"]["total_issues"] = len(tracking["issues_found"])

    with open(tracking_path, "w") as f:
        json.dump(tracking, f, indent=2)


def main():
    """Main orchestration function."""
    tracking_path = Path("tests/BROAD_TEST_REVIEW_TRACKING.json")
    batch_size = 50  # Files per batch
    use_ai_models = "--ai" in sys.argv

    print("=" * 60)
    print("Broad Test Review Orchestrator")
    print("=" * 60)

    # Get all test files
    test_files = get_test_files(exclude_dirs=["optimizer_validation"])
    print(f"Found {len(test_files)} test files to review")

    # Process in batches
    total_issues = 0
    for batch_num, i in enumerate(range(0, len(test_files), batch_size), 1):
        batch_files = test_files[i : i + batch_size]
        print(f"\nBatch {batch_num}: Processing {len(batch_files)} files...")

        results = process_batch(batch_files, batch_num, use_ai_models)
        merged = merge_results(results)

        batch_issues = sum(len(v) for v in merged.values())
        total_issues += batch_issues
        print(f"  Found {batch_issues} issues in this batch")

        # Update tracking
        update_tracking_file(tracking_path, batch_num, batch_files, merged)

    print("\n" + "=" * 60)
    print(f"Review complete. Total issues found: {total_issues}")
    print(f"Results saved to: {tracking_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
