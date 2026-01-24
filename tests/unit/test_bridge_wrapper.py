"""Subprocess wrapper for running bridge tests safely.

This module runs the bridge tests in a subprocess with output captured,
avoiding the VSCode terminal rendering issue that can trigger system crashes.

The bridge tests (tests/unit/bridges/) are excluded from VSCode test discovery
to prevent crashes during discovery. This wrapper provides a safe way to run
them from VSCode - it appears as a single test that internally runs all bridge
tests via subprocess.

Usage:
    - From VSCode: Run "test_bridge_tests_pass" like any other test
    - From terminal: pytest tests/unit/test_bridge_wrapper.py -v

The wrapper captures all output, so even if bridge tests produce heavy output,
it won't be rendered to the terminal until after completion (and only on failure).

NOTE: In CI (GitHub Actions), bridge tests run directly without this wrapper
since headless environments don't have terminal rendering issues. The bridge
tests use mocked subprocesses and don't require Node.js to be installed.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

# Get the project root (where pytest.ini lives)
PROJECT_ROOT = Path(__file__).parent.parent.parent


@pytest.mark.timeout(300)  # 5 minute timeout for all bridge tests
def test_bridge_tests_pass():
    """Run all bridge tests in a subprocess with captured output.

    This test:
    1. Runs bridge tests via subprocess (not in-process)
    2. Captures stdout/stderr to avoid terminal rendering crashes
    3. Reports pass/fail based on subprocess exit code
    4. On failure, prints captured output for debugging

    The subprocess approach prevents VSCode terminal rendering from
    triggering GPU-related system instability.
    """
    bridge_tests_path = PROJECT_ROOT / "tests" / "unit" / "bridges"

    if not bridge_tests_path.exists():
        pytest.skip(f"Bridge tests directory not found: {bridge_tests_path}")

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            str(bridge_tests_path),
            "-q",  # Quiet output (less terminal spam)
            "--tb=short",  # Short tracebacks on failure
            "-x",  # Stop on first failure
        ],
        capture_output=True,  # Capture stdout/stderr (no terminal rendering)
        text=True,
        timeout=300,  # 5 minute timeout
        cwd=str(PROJECT_ROOT),  # Run from project root
        env={
            **dict(__import__("os").environ),
            # Ensure tests run in mock mode
            "TRAIGENT_MOCK_LLM": "true",
            "TRAIGENT_OFFLINE_MODE": "true",
        },
    )

    # On failure, show the captured output for debugging
    if result.returncode != 0:
        print("\n" + "=" * 60)
        print("BRIDGE TESTS FAILED")
        print("=" * 60)
        if result.stdout:
            print("\n--- STDOUT ---")
            print(result.stdout)
        if result.stderr:
            print("\n--- STDERR ---")
            print(result.stderr)
        print("=" * 60 + "\n")

        pytest.fail(
            f"Bridge tests failed with exit code {result.returncode}. "
            "See captured output above."
        )


@pytest.mark.timeout(60)
def test_bridge_tests_discoverable():
    """Verify bridge tests exist and can be collected (dry-run).

    This is a quick sanity check that bridge tests are properly structured
    and can be discovered by pytest without actually running them.
    """
    bridge_tests_path = PROJECT_ROOT / "tests" / "unit" / "bridges"

    if not bridge_tests_path.exists():
        pytest.skip(f"Bridge tests directory not found: {bridge_tests_path}")

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            str(bridge_tests_path),
            "--collect-only",  # Only collect, don't run
            "-q",
        ],
        capture_output=True,
        text=True,
        timeout=60,
        cwd=str(PROJECT_ROOT),
    )

    # Parse output to count collected tests
    if result.returncode != 0:
        print(f"Collection failed:\n{result.stderr}")
        pytest.fail("Bridge test collection failed")

    # Look for "X tests collected" in output
    output = result.stdout + result.stderr
    if "no tests ran" in output.lower() or "0 tests" in output.lower():
        pytest.fail("No bridge tests were collected")

    print(f"Bridge tests collected successfully:\n{result.stdout}")
