#!/usr/bin/env python3
"""
TraiGent SDK Demo Test Runner

This script runs all demos with basic sanity checks to verify they work correctly.
Use this for quick validation of the entire demo suite.

Usage:
    python demos/test_all_demos.py [options]

Options:
    --verbose, -v       Show detailed output from each demo
    --fast             Skip long-running demos
    --category CATEGORY Only run demos from specific category (basic, advanced, commercial, enterprise, integration)
    --list             List all available demos
    --timeout SECONDS  Set timeout for each demo (default: 60)
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class DemoTestResults:
    """Track demo test results and statistics."""

    def __init__(self):
        self.passed = []
        self.failed = []
        self.skipped = []
        self.warnings = []
        self.start_time = time.time()

    def add_result(
        self, demo_name: str, status: str, message: str = "", duration: float = 0
    ):
        """Add test result."""
        result = {
            "name": demo_name,
            "status": status,
            "message": message,
            "duration": duration,
        }

        if status == "PASS":
            self.passed.append(result)
        elif status == "FAIL":
            self.failed.append(result)
        elif status == "SKIP":
            self.skipped.append(result)
        elif status == "WARN":
            self.warnings.append(result)

    def print_summary(self):
        """Print test summary."""
        total_time = time.time() - self.start_time
        total_tests = len(self.passed) + len(self.failed) + len(self.skipped)

        print("\n" + "=" * 80)
        print("🧪 TRAIGENT DEMO TEST SUMMARY")
        print("=" * 80)

        print(f"📊 Total Demos: {total_tests}")
        print(f"✅ Passed: {len(self.passed)}")
        print(f"❌ Failed: {len(self.failed)}")
        print(f"⏭️  Skipped: {len(self.skipped)}")
        print(f"⚠️  Warnings: {len(self.warnings)}")
        print(f"⏱️  Total Time: {total_time:.1f}s")

        if self.failed:
            print(f"\n❌ FAILED DEMOS ({len(self.failed)}):")
            for result in self.failed:
                print(f"   • {result['name']}: {result['message']}")

        if self.warnings:
            print(f"\n⚠️  WARNINGS ({len(self.warnings)}):")
            for result in self.warnings:
                print(f"   • {result['name']}: {result['message']}")

        if self.skipped:
            print(f"\n⏭️  SKIPPED DEMOS ({len(self.skipped)}):")
            for result in self.skipped:
                print(f"   • {result['name']}: {result['message']}")

        success_rate = (
            len(self.passed) / max(1, len(self.passed) + len(self.failed)) * 100
        )
        print(f"\n🎯 Success Rate: {success_rate:.1f}%")

        if len(self.failed) == 0:
            print("🎉 All demos passed successfully!")

        print("=" * 80)


class DemoRunner:
    """Run and test individual demos."""

    def __init__(self, verbose: bool = False, timeout: int = 60):
        self.verbose = verbose
        self.timeout = timeout
        self.results = DemoTestResults()

    def run_demo(
        self, demo_path: Path, expected_outputs: list[str] = None
    ) -> tuple[bool, str, float]:
        """Run a single demo and check for expected outputs."""
        start_time = time.time()

        try:
            # Set up environment
            env = os.environ.copy()
            env["PYTHONPATH"] = str(PROJECT_ROOT)

            # Run demo with timeout
            process = subprocess.Popen(
                [sys.executable, str(demo_path)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=PROJECT_ROOT,
                env=env,
            )

            try:
                stdout, stderr = process.communicate(timeout=self.timeout)
                duration = time.time() - start_time

                # Check return code
                if process.returncode != 0:
                    error_msg = f"Exit code {process.returncode}"
                    if stderr:
                        error_msg += f": {stderr[:200]}..."
                    return False, error_msg, duration

                # Basic sanity checks
                output = stdout + stderr

                # Check for common error patterns
                error_patterns = [
                    "Traceback (most recent call last):",
                    "ImportError:",
                    "ModuleNotFoundError:",
                    "AttributeError:",
                    "TypeError:",
                    "ValueError:",
                    "RuntimeError:",
                    "Exception:",
                    "Error:",
                ]

                for pattern in error_patterns:
                    if pattern in output and "not available" not in output:
                        # Allow expected dependency warnings
                        if any(
                            ok_msg in output
                            for ok_msg in [
                                "scikit-learn not available",
                                "Security modules not fully available",
                                "Optional dependency",
                                "gracefully handles missing",
                            ]
                        ):
                            continue
                        return False, f"Error pattern found: {pattern}", duration

                # Check for expected positive indicators
                positive_indicators = [
                    "✅",
                    "✓",
                    "SUCCESS",
                    "PASS",
                    "Complete",
                    "Demo",
                    "completed",
                    "SUMMARY",
                    "Results",
                    "optimization",
                    "TraiGent",
                ]

                has_positive = any(
                    indicator in output for indicator in positive_indicators
                )
                if not has_positive:
                    return False, "No positive completion indicators found", duration

                # Check for expected outputs if provided
                if expected_outputs:
                    for expected in expected_outputs:
                        if expected not in output:
                            return (
                                False,
                                f"Expected output not found: {expected}",
                                duration,
                            )

                return True, "Demo completed successfully", duration

            except subprocess.TimeoutExpired:
                process.kill()
                return False, f"Timeout after {self.timeout}s", self.timeout

        except Exception as e:
            duration = time.time() - start_time
            return False, f"Exception: {str(e)}", duration

    def test_demo(
        self,
        demo_path: Path,
        expected_outputs: list[str] = None,
        skip_reason: str = None,
    ):
        """Test a single demo with reporting."""
        demo_name = str(demo_path.relative_to(PROJECT_ROOT))

        if skip_reason:
            print(f"⏭️  SKIP: {demo_name} - {skip_reason}")
            self.results.add_result(demo_name, "SKIP", skip_reason)
            return

        print(f"🧪 TEST: {demo_name}")

        if self.verbose:
            print(f"   Running: python {demo_path}")

        success, message, duration = self.run_demo(demo_path, expected_outputs)

        if success:
            print(f"   ✅ PASS ({duration:.1f}s): {message}")
            self.results.add_result(demo_name, "PASS", message, duration)
        else:
            print(f"   ❌ FAIL ({duration:.1f}s): {message}")
            self.results.add_result(demo_name, "FAIL", message, duration)

        if self.verbose and success:
            print("   📝 Demo output validated successfully")


def get_demo_categories() -> dict[str, list[tuple[Path, list[str], str]]]:
    """Get organized list of demos by category.

    Returns:
        Dict mapping category name to list of (path, expected_outputs, skip_reason) tuples
    """
    demos_dir = PROJECT_ROOT / "demos"

    categories = {
        "basic": [
            (demos_dir / "basic" / "simple_example.py", ["TraiGent"], None),
            (demos_dir / "basic" / "quick_demo.py", ["TraiGent"], None),
            (demos_dir / "basic" / "zero_friction_demo.py", ["TraiGent"], None),
            (demos_dir / "basic" / "adaptive_parameters.py", ["TraiGent"], None),
            (demos_dir / "basic" / "basic_optimization.py", ["TraiGent"], None),
        ],
        "advanced": [
            (demos_dir / "advanced" / "sprint4_features_demo.py", ["Sprint 4"], None),
            (
                demos_dir / "advanced" / "sprint3_developer_experience.py",
                ["Sprint 3"],
                None,
            ),
            (
                demos_dir / "advanced" / "sprint2_production_features.py",
                ["Sprint 2"],
                None,
            ),
            (
                demos_dir / "advanced" / "config_injection_patterns.py",
                ["TraiGent"],
                None,
            ),
            (
                demos_dir / "advanced" / "invoker_evaluator_separation.py",
                ["TraiGent"],
                None,
            ),
            (demos_dir / "advanced" / "batch_processing_simple.py", ["TraiGent"], None),
            (
                demos_dir / "advanced" / "batch_processing_advanced.py",
                ["TraiGent"],
                None,
            ),
        ],
        "commercial": [
            (
                demos_dir / "commercial" / "cloud_execution_simple_demo.py",
                ["TraiGent"],
                None,
            ),
            (demos_dir / "commercial" / "cto_demo.py", ["TraiGent"], None),
            (demos_dir / "commercial" / "simple_cto_demo.py", ["TraiGent"], None),
            (
                demos_dir / "commercial" / "sprint5_cloud_service_demo.py",
                ["TraiGent"],
                None,
            ),
            # Skip full server demo due to complexity
            (
                demos_dir / "commercial" / "run_demo.py",
                ["commercial"],
                "Complex server/client demo - test manually",
            ),
        ],
        "enterprise": [
            (
                demos_dir / "enterprise" / "sprint7_enterprise_security_demo.py",
                ["Sprint 7"],
                "Missing security classes - uses newer root demo",
            ),
        ],
        "integration": [
            (demos_dir / "sprint6_framework_integrations.py", ["SPRINT 6"], None),
            (demos_dir / "sprint7_enterprise_security.py", ["SPRINT 7"], None),
            (demos_dir / "sprint8_advanced_analytics.py", ["SPRINT 8"], None),
        ],
    }

    return categories


def main():
    """Main demo test runner."""
    parser = argparse.ArgumentParser(
        description="Test all TraiGent SDK demos",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python demos/test_all_demos.py                    # Run all demos
    python demos/test_all_demos.py --verbose          # Show detailed output
    python demos/test_all_demos.py --fast             # Skip long demos
    python demos/test_all_demos.py --category basic   # Only basic demos
    python demos/test_all_demos.py --list             # List all demos
    python demos/test_all_demos.py --timeout 30       # 30 second timeout per demo
        """,
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show detailed output from each demo",
    )
    parser.add_argument("--fast", action="store_true", help="Skip long-running demos")
    parser.add_argument(
        "--category",
        choices=["basic", "advanced", "commercial", "enterprise", "integration"],
        help="Only run demos from specific category",
    )
    parser.add_argument("--list", action="store_true", help="List all available demos")
    parser.add_argument(
        "--timeout",
        type=int,
        default=60,
        help="Timeout in seconds for each demo (default: 60)",
    )

    args = parser.parse_args()

    # Get demo categories
    categories = get_demo_categories()

    if args.list:
        print("📋 Available Demo Categories:")
        print("=" * 50)
        for category, demos in categories.items():
            print(f"\n📁 {category.upper()} ({len(demos)} demos):")
            for demo_path, _expected, skip_reason in demos:
                demo_name = demo_path.name
                status = "⏭️  SKIP" if skip_reason else "🧪 TEST"
                print(f"   {status} {demo_name}")
                if skip_reason:
                    print(f"       Reason: {skip_reason}")
        return

    # Initialize runner
    runner = DemoRunner(verbose=args.verbose, timeout=args.timeout)

    print("🚀 TraiGent SDK Demo Test Runner")
    print("=" * 50)
    print(f"📍 Project Root: {PROJECT_ROOT}")
    print(f"⏱️  Timeout: {args.timeout}s per demo")
    if args.category:
        print(f"📁 Category Filter: {args.category}")
    if args.fast:
        print("⚡ Fast Mode: Skipping long demos")
    print("=" * 50)

    # Filter categories if specified
    if args.category:
        if args.category not in categories:
            print(f"❌ Unknown category: {args.category}")
            return
        categories = {args.category: categories[args.category]}

    # Run demos by category
    for category_name, demos in categories.items():
        print(f"\n📁 TESTING {category_name.upper()} DEMOS ({len(demos)} demos)")
        print("-" * 40)

        for demo_path, expected_outputs, skip_reason in demos:
            # Check if file exists
            if not demo_path.exists():
                runner.results.add_result(
                    str(demo_path.relative_to(PROJECT_ROOT)), "SKIP", "File not found"
                )
                print(f"⏭️  SKIP: {demo_path.name} - File not found")
                continue

            # Apply fast mode skipping
            if args.fast and any(
                keyword in demo_path.name.lower()
                for keyword in ["advanced", "batch", "server", "client"]
            ):
                skip_reason = skip_reason or "Fast mode"

            # Run the demo
            runner.test_demo(demo_path, expected_outputs, skip_reason)

    # Print final summary
    runner.results.print_summary()

    # Exit with appropriate code
    if runner.results.failed:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
