#!/usr/bin/env python3
"""
Example Runner - Test and validate Traigent examples

This script runs examples with proper error handling, timing,
and validation to ensure they work correctly.
"""

import argparse
import fnmatch
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List


class ExampleRunner:
    """Runner for Traigent examples with validation and reporting."""

    def __init__(self, base_dir: str = "examples", verbose: bool = False):
        self.verbose = verbose
        self.base_dir = base_dir
        self.results: List[Dict[str, Any]] = []

    def discover_examples(self, pattern: str) -> List[Path]:
        """Discover example files matching the pattern."""
        base_path = Path(self.base_dir)
        if not base_path.exists():
            print(f"❌ Examples directory not found: {base_path}")
            return []

        examples = []
        for file_path in base_path.rglob("*.py"):
            # Skip archive and shared directories
            if "archive" in file_path.parts or "shared" in file_path.parts:
                continue

            if fnmatch.fnmatch(file_path.name, pattern):
                examples.append(file_path)

        return sorted(examples)

    def run_example(self, example_path: Path, timeout: int = 60) -> Dict[str, Any]:
        """Run a single example and return results."""
        if self.verbose:
            print(f"🏃 Running {example_path}")

        start_time = time.time()
        result = {
            "file": str(example_path),
            "success": False,
            "duration": 0,
            "output": "",
            "error": "",
        }

        try:
            # Ensure mock mode is enabled
            env = os.environ.copy()
            env["TRAIGENT_MOCK_MODE"] = "true"

            # Run the example
            process = subprocess.run(
                [sys.executable, str(example_path)],
                capture_output=True,
                text=True,
                timeout=timeout,
                env=env,
                cwd=Path.cwd(),
            )

            result["duration"] = time.time() - start_time
            result["output"] = process.stdout
            result["error"] = process.stderr
            result["return_code"] = process.returncode
            result["success"] = process.returncode == 0

        except subprocess.TimeoutExpired:
            result["duration"] = timeout
            result["error"] = f"Example timed out after {timeout} seconds"
            result["return_code"] = -1

        except Exception as e:
            result["duration"] = time.time() - start_time
            result["error"] = str(e)
            result["return_code"] = -2

        return result

    def validate_example_structure(self, example_path: Path) -> List[str]:
        """Validate example follows the expected structure."""
        issues = []

        try:
            with open(example_path) as f:
                content = f.read()

            # Check for required elements
            if "TRAIGENT_MOCK_MODE" not in content:
                issues.append("Missing mock mode setup")

            if 'if __name__ == "__main__":' not in content:
                issues.append("Missing main section")

            if "@traigent.optimize" not in content and "traigent." not in content:
                issues.append("No Traigent usage detected")

            if not content.startswith("#!/usr/bin/env python3"):
                issues.append("Missing shebang")

            # Check docstring
            lines = content.split("\n")
            if len(lines) < 5 or not (
                lines[1].startswith('"""') or lines[1].startswith("'''")
            ):
                issues.append("Missing or incomplete docstring")

        except Exception as e:
            issues.append(f"File reading error: {e}")

        return issues

    def run_examples(self, pattern: str, validate_structure: bool = True) -> bool:
        """Run all examples matching the pattern."""
        examples = self.discover_examples(pattern)

        if not examples:
            print(f"❌ No examples found matching pattern: {pattern}")
            return False

        print(f"🔍 Found {len(examples)} examples matching pattern: {pattern}")

        success_count = 0

        for example in examples:
            if validate_structure:
                issues = self.validate_example_structure(example)
                if issues:
                    print(f"⚠️  Structure issues in {example.name}: {', '.join(issues)}")

            result = self.run_example(example)
            self.results.append(result)

            if result["success"]:
                status = "✅"
                success_count += 1
            else:
                status = "❌"

            duration_str = f"{result['duration']:.1f}s"
            print(f"{status} {example.name} ({duration_str})")

            if not result["success"] and self.verbose:
                print(f"   Return code: {result.get('return_code', 'unknown')}")
                if result["error"]:
                    print(f"   Error: {result['error'][:200]}...")
                if result["output"]:
                    print(f"   Output: {result['output'][:200]}...")

        return success_count == len(examples)

    def generate_report(self) -> str:
        """Generate a summary report of the test run."""
        if not self.results:
            return "No examples were run."

        total = len(self.results)
        success = sum(1 for r in self.results if r["success"])
        failed = total - success

        avg_duration = sum(r["duration"] for r in self.results) / total

        report = f"""
📊 Example Test Report
=====================
Total Examples: {total}
✅ Successful: {success}
❌ Failed: {failed}
⏱️  Average Duration: {avg_duration:.1f}s

"""

        if failed > 0:
            report += "Failed Examples:\n"
            for result in self.results:
                if not result["success"]:
                    report += f"  ❌ {Path(result['file']).name}: {result['error'][:100]}...\n"

        return report


def main():
    parser = argparse.ArgumentParser(
        description="Run Traigent examples with validation"
    )
    parser.add_argument(
        "--pattern",
        "-p",
        default="run.py",
        help="Pattern to match example files (default: run.py)",
    )
    parser.add_argument(
        "--base",
        "-b",
        default="examples",
        help="Base directory to search for examples (default: examples)",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument(
        "--timeout",
        "-t",
        type=int,
        default=60,
        help="Timeout per example in seconds (default: 60)",
    )
    parser.add_argument(
        "--no-structure-validation",
        action="store_true",
        help="Skip structure validation",
    )

    args = parser.parse_args()

    base_path = Path(args.base)
    if not base_path.exists():
        print(f"❌ Base directory not found: {base_path}")
        return 1

    runner = ExampleRunner(base_dir=str(base_path), verbose=args.verbose)

    print("🚀 Traigent Example Runner")
    print(f"Pattern: {args.pattern}")
    print(f"Timeout: {args.timeout}s")
    print(f"Base directory: {base_path}")
    print("-" * 40)

    success = runner.run_examples(
        args.pattern, validate_structure=not args.no_structure_validation
    )

    print("\n" + runner.generate_report())

    if success:
        print("🎉 All examples passed!")
        return 0
    else:
        print("💥 Some examples failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
