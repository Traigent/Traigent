#!/usr/bin/env python3
"""
Test runner for scripts test suite.

This script runs all tests in the scripts/test directory and generates
a coverage report.
"""

import sys
import unittest
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


def run_tests_with_coverage():
    """Run tests with coverage reporting."""
    try:
        import coverage
    except ImportError:
        print("Coverage.py not installed. Running tests without coverage.")
        return run_tests_without_coverage()

    # Initialize coverage
    cov = coverage.Coverage(source=["../"])
    cov.start()

    # Discover and run tests
    loader = unittest.TestLoader()
    start_dir = Path(__file__).parent
    suite = loader.discover(start_dir, pattern="test_*.py")

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Stop coverage and generate report
    cov.stop()
    cov.save()

    print("\n" + "=" * 70)
    print("Coverage Report:")
    print("=" * 70)
    cov.report()

    # Generate HTML report
    html_dir = Path(__file__).parent / "htmlcov"
    cov.html_report(directory=str(html_dir))
    print(f"\nDetailed HTML coverage report generated in: {html_dir}")

    return result.wasSuccessful()


def run_tests_without_coverage():
    """Run tests without coverage reporting."""
    loader = unittest.TestLoader()
    start_dir = Path(__file__).parent
    suite = loader.discover(start_dir, pattern="test_*.py")

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result.wasSuccessful()


def main():
    """Main entry point."""
    print("Running Traigent SDK Scripts Test Suite")
    print("=" * 70)

    success = run_tests_with_coverage()

    if success:
        print("\n✅ All tests passed!")
        return 0
    else:
        print("\n❌ Some tests failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
