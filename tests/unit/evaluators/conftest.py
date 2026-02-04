"""Conftest for evaluator tests.

CRITICAL: JS evaluator tests (test_js_evaluator*.py) spawn Node.js processes
that can crash VSCode terminals. They MUST run via test_bridge_wrapper.py.

DO NOT run directly:
    pytest tests/unit/evaluators/test_js_evaluator.py -v  # WILL CRASH!

Safe way:
    pytest tests/unit/test_bridge_wrapper.py -v
"""

import os
import sys

import pytest

# Environment variable set by test_bridge_wrapper.py when running via subprocess
_SUBPROCESS_ENV_VAR = "TRAIGENT_JS_TEST_SUBPROCESS"


def pytest_configure(config):
    """HARD FAIL if JS evaluator tests are run directly."""
    if os.environ.get(_SUBPROCESS_ENV_VAR) == "1":
        return

    args = config.args if hasattr(config, "args") else []

    # Detect if JS evaluator tests are being run directly
    is_js_evaluator_targeted = any("test_js_evaluator" in str(arg) for arg in args)

    if is_js_evaluator_targeted:
        print("\n" + "=" * 70, file=sys.stderr)
        print("FATAL: JS evaluator tests cannot be run directly!", file=sys.stderr)
        print("=" * 70, file=sys.stderr)
        print(
            "\nThese tests spawn Node.js processes that WILL CRASH your terminal.",
            file=sys.stderr,
        )
        print("\nSafe way to run JS tests:", file=sys.stderr)
        print("    pytest tests/unit/test_bridge_wrapper.py -v", file=sys.stderr)
        print("\n" + "=" * 70 + "\n", file=sys.stderr)
        pytest.exit(
            "JS evaluator tests must run via test_bridge_wrapper.py", returncode=1
        )


def pytest_collection_modifyitems(config, items):
    """Additional safety: skip any JS evaluator tests that slip through."""
    if os.environ.get(_SUBPROCESS_ENV_VAR) == "1":
        return

    for item in items:
        if "test_js_evaluator" in item.fspath.basename:
            item.add_marker(
                pytest.mark.skip(reason="BLOCKED: Run via test_bridge_wrapper.py only")
            )
