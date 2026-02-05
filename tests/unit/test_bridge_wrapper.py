"""Subprocess wrapper for running JS-related tests safely.

This module runs JS-related tests (bridges, JS evaluators) in a subprocess with
output captured, avoiding the VSCode terminal rendering issue that can trigger
system crashes due to GPU-related instability.

The following tests are excluded from VSCode test discovery and must be run
via this wrapper:
- tests/unit/bridges/ (JSBridge, JSProcessPool)
- tests/unit/evaluators/test_js_evaluator*.py (JSEvaluator tests)

IMPORTANT: This file is EXCLUDED from VSCode test discovery (.vscode/settings.json)
to prevent system crashes. Run JS tests from the terminal only:

    pytest tests/unit/test_bridge_wrapper.py -v

The wrapper captures all output, so even if tests produce heavy async output,
it won't be rendered to the terminal until after completion (and only on failure).

WARNING: DO NOT run JS tests directly! Always use this wrapper:
    WRONG:  pytest tests/unit/bridges/ -v          # May crash system!
    RIGHT:  pytest tests/unit/test_bridge_wrapper.py -v

NOTE: In CI (GitHub Actions), these tests run directly without this wrapper
since headless environments don't have terminal rendering issues.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

# Get the project root (where pytest.ini lives)
PROJECT_ROOT = Path(__file__).parent.parent.parent


def _run_tests_safely(test_paths: list, test_name: str, timeout: int = 300):
    """Run tests in a subprocess with captured output.

    Args:
        test_paths: List of test paths to run
        test_name: Human-readable name for error messages
        timeout: Timeout in seconds (default 5 minutes)
    """
    import os

    # Filter to existing paths
    existing_paths = [p for p in test_paths if p.exists()]
    if not existing_paths:
        pytest.skip(f"No {test_name} found at: {test_paths}")

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            *[str(p) for p in existing_paths],
            "-q",  # Quiet output (less terminal spam)
            "--tb=short",  # Short tracebacks on failure
            "-x",  # Stop on first failure
        ],
        capture_output=True,  # Capture stdout/stderr (no terminal rendering)
        text=True,
        timeout=timeout,
        cwd=str(PROJECT_ROOT),
        env={
            **dict(os.environ),
            # Ensure tests run in mock mode
            "TRAIGENT_MOCK_LLM": "true",
            "TRAIGENT_OFFLINE_MODE": "true",
            # Signal to conftest.py that we're running via subprocess wrapper
            # This enables the JS tests which are otherwise skipped
            "TRAIGENT_JS_TEST_SUBPROCESS": "1",
        },
    )

    # On failure, show the captured output for debugging
    if result.returncode != 0:
        print("\n" + "=" * 60)
        print(f"{test_name.upper()} FAILED")
        print("=" * 60)
        if result.stdout:
            print("\n--- STDOUT ---")
            print(result.stdout)
        if result.stderr:
            print("\n--- STDERR ---")
            print(result.stderr)
        print("=" * 60 + "\n")

        pytest.fail(
            f"{test_name} failed with exit code {result.returncode}. "
            "See captured output above."
        )


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
    _run_tests_safely([bridge_tests_path], "Bridge tests", timeout=300)


@pytest.mark.timeout(300)  # 5 minute timeout for JS evaluator tests
def test_js_evaluator_tests_pass():
    """Run all JS evaluator tests in a subprocess with captured output.

    These tests cover JSEvaluator which integrates with the JS bridge.
    They must run via subprocess to avoid terminal rendering crashes.
    """
    evaluator_tests = [
        PROJECT_ROOT / "tests" / "unit" / "evaluators" / "test_js_evaluator.py",
        PROJECT_ROOT / "tests" / "unit" / "evaluators" / "test_js_evaluator_budget.py",
        PROJECT_ROOT
        / "tests"
        / "unit"
        / "evaluators"
        / "test_js_evaluator_stop_conditions.py",
    ]
    _run_tests_safely(evaluator_tests, "JS Evaluator tests", timeout=300)


@pytest.mark.timeout(600)  # 10 minute timeout for all JS tests combined
def test_all_js_tests_pass():
    """Run ALL JS-related tests in a single subprocess.

    This is the comprehensive test that runs both bridge and JS evaluator tests
    together. Use this for full validation before commits.
    """
    all_js_tests = [
        PROJECT_ROOT / "tests" / "unit" / "bridges",
        PROJECT_ROOT / "tests" / "unit" / "evaluators" / "test_js_evaluator.py",
        PROJECT_ROOT / "tests" / "unit" / "evaluators" / "test_js_evaluator_budget.py",
        PROJECT_ROOT
        / "tests"
        / "unit"
        / "evaluators"
        / "test_js_evaluator_stop_conditions.py",
    ]
    _run_tests_safely(all_js_tests, "All JS tests", timeout=600)


@pytest.mark.timeout(60)
def test_js_tests_discoverable():
    """Verify JS-related tests exist and can be collected (dry-run).

    This is a quick sanity check that JS tests are properly structured
    and can be discovered by pytest without actually running them.
    """
    all_js_tests = [
        PROJECT_ROOT / "tests" / "unit" / "bridges",
        PROJECT_ROOT / "tests" / "unit" / "evaluators" / "test_js_evaluator.py",
        PROJECT_ROOT / "tests" / "unit" / "evaluators" / "test_js_evaluator_budget.py",
        PROJECT_ROOT
        / "tests"
        / "unit"
        / "evaluators"
        / "test_js_evaluator_stop_conditions.py",
    ]

    existing_paths = [p for p in all_js_tests if p.exists()]
    if not existing_paths:
        pytest.skip("No JS tests found")

    import os

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            *[str(p) for p in existing_paths],
            "--collect-only",  # Only collect, don't run
            "-q",
        ],
        capture_output=True,
        text=True,
        timeout=60,
        cwd=str(PROJECT_ROOT),
        env={
            **dict(os.environ),
            # Must set subprocess flag even for collection
            "TRAIGENT_JS_TEST_SUBPROCESS": "1",
        },
    )

    # Parse output to count collected tests
    if result.returncode != 0:
        print(f"Collection failed:\n{result.stderr}")
        pytest.fail("JS test collection failed")

    # Look for "X tests collected" in output
    output = result.stdout + result.stderr
    if "no tests ran" in output.lower() or "0 tests" in output.lower():
        pytest.fail("No JS tests were collected")

    print(f"JS tests collected successfully:\n{result.stdout}")
