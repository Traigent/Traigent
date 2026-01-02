#!/usr/bin/env python3
"""Setup test environment and run tests with coverage."""

import subprocess
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def run_command(cmd, description):
    """Run a command and print status."""
    print(f"\n{'='*60}")
    print(f"🔧 {description}")
    print(f"{'='*60}")

    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

    if result.returncode == 0:
        print("✅ Success")
        if result.stdout:
            print(result.stdout)
    else:
        print("❌ Failed")
        if result.stderr:
            print(f"Error: {result.stderr}")
        if result.stdout:
            print(f"Output: {result.stdout}")

    return result.returncode


def main():
    """Setup and run tests."""
    print("🧪 Traigent SDK Test Setup")

    # 1. Install test dependencies
    if run_command("pip install -e '.[dev]'", "Installing test dependencies") != 0:
        print("Failed to install dependencies")
        return 1

    # 2. Run unit tests with coverage
    run_command(
        "pytest tests/unit -v --cov=traigent --cov-report=term-missing --cov-report=html",
        "Running unit tests with coverage",
    )

    # 3. Run specific test categories
    print("\n📊 Test Summary by Category:")

    # Test categories
    categories = [
        ("API Tests", "tests/unit/api"),
        ("Core Tests", "tests/unit/core"),
        ("Utils Tests", "tests/unit/utils"),
        ("Optimizer Tests", "tests/unit/optimizers"),
        ("Analytics Tests", "tests/unit/analytics"),
        ("Security Tests", "tests/unit/security"),
    ]

    for category_name, test_path in categories:
        cmd = f"pytest {test_path} -q --tb=no"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

        # Parse output for pass/fail counts
        output_lines = result.stdout.strip().split("\n")
        if output_lines and (
            "passed" in output_lines[-1] or "failed" in output_lines[-1]
        ):
            status = output_lines[-1].strip()
            print(f"{category_name:<20} {status}")
        else:
            print(f"{category_name:<20} No tests found or error")

    # 4. Check test coverage threshold
    print("\n📈 Coverage Report:")
    run_command("coverage report --fail-under=70", "Checking coverage threshold (70%)")

    print("\n✨ Test setup complete!")
    print("📁 HTML coverage report: htmlcov/index.html")

    return 0


if __name__ == "__main__":
    sys.exit(main())
