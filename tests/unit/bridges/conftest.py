"""Conftest for JS bridge tests.

CRITICAL: These tests spawn Node.js processes that can crash VSCode terminals
and cause system instability. They MUST run via test_bridge_wrapper.py subprocess.

DO NOT run directly:
    pytest tests/unit/bridges/ -v  # WILL CRASH YOUR SYSTEM!

Safe way:
    pytest tests/unit/test_bridge_wrapper.py -v
"""

import os
import sys

import pytest

# Environment variable set by test_bridge_wrapper.py when running via subprocess
_SUBPROCESS_ENV_VAR = "TRAIGENT_JS_TEST_SUBPROCESS"


def pytest_configure(config):
    """HARD FAIL if JS bridge tests are run directly (not via subprocess wrapper).

    This runs BEFORE test collection, preventing any JS bridge code from executing.
    """
    if os.environ.get(_SUBPROCESS_ENV_VAR) == "1":
        # Running via wrapper subprocess - allow tests
        return

    # Check if we're collecting tests in this directory
    # pytest_configure runs for ALL conftest.py files, so we need to check
    # if this specific directory's tests are being targeted
    args = config.args if hasattr(config, "args") else []
    invocation_dir = (
        str(config.invocation_params.dir)
        if hasattr(config, "invocation_params")
        else ""
    )

    # Detect if bridges tests are being run directly
    is_bridges_targeted = (
        any("bridges" in str(arg) for arg in args) or "bridges" in invocation_dir
    )

    if is_bridges_targeted:
        print("\n" + "=" * 70, file=sys.stderr)
        print("FATAL: JS bridge tests cannot be run directly!", file=sys.stderr)
        print("=" * 70, file=sys.stderr)
        print(
            "\nThese tests spawn Node.js processes that WILL CRASH your terminal.",
            file=sys.stderr,
        )
        print("\nSafe way to run JS tests:", file=sys.stderr)
        print("    pytest tests/unit/test_bridge_wrapper.py -v", file=sys.stderr)
        print("\n" + "=" * 70 + "\n", file=sys.stderr)
        pytest.exit("JS bridge tests must run via test_bridge_wrapper.py", returncode=1)


def pytest_collection_modifyitems(config, items):
    """Additional safety: skip any JS bridge tests that slip through."""
    if os.environ.get(_SUBPROCESS_ENV_VAR) == "1":
        return

    for item in items:
        if "/bridges/" in str(item.fspath):
            # Hard fail instead of skip - this should never happen if pytest_configure works
            item.add_marker(
                pytest.mark.skip(reason="BLOCKED: Run via test_bridge_wrapper.py only")
            )
