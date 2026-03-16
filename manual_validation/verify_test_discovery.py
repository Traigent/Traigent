#!/usr/bin/env python3
"""Verify pytest test discovery for VSCode"""

import subprocess
import sys


def run_discovery():
    """Run pytest collection as VSCode would"""
    cmd = [
        "traigent_test_env/bin/pytest",
        "-c",
        ".vscode/pytest_minimal.ini",
        "--no-cov",
        "-p",
        "no:warnings",
        "--collect-only",
        "-q",
        "tests",
    ]

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    print(f"\nReturn code: {result.returncode}")
    print(f"\nStdout lines: {len(result.stdout.splitlines())}")

    # Count test types
    lines = result.stdout.splitlines()
    unit_count = sum(1 for line in lines if "tests/unit/" in line)
    integration_count = sum(1 for line in lines if "tests/integration/" in line)
    e2e_count = sum(1 for line in lines if "tests/e2e/" in line)

    print("\nTest distribution:")
    print(f"  Unit tests: {unit_count}")
    print(f"  Integration tests: {integration_count}")
    print(f"  E2E tests: {e2e_count}")

    # Check if stderr has any issues
    if result.stderr:
        print(f"\nStderr output:\n{result.stderr}")

    return result.returncode == 0


if __name__ == "__main__":
    success = run_discovery()
    sys.exit(0 if success else 1)
