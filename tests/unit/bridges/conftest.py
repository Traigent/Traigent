"""Conftest for JS bridge tests.

These tests should only run via test_bridge_wrapper.py subprocess to avoid
terminal rendering crashes that can cause system instability.

When running `pytest tests/unit/` directly, these tests are SKIPPED.
To run them, use: `pytest tests/unit/test_bridge_wrapper.py -v`
"""

import os

import pytest

# Environment variable set by test_bridge_wrapper.py when running via subprocess
_SUBPROCESS_ENV_VAR = "TRAIGENT_JS_TEST_SUBPROCESS"


def pytest_collection_modifyitems(config, items):
    """Skip JS bridge tests unless running via subprocess wrapper."""
    if os.environ.get(_SUBPROCESS_ENV_VAR) == "1":
        # Running via wrapper subprocess - allow all tests
        return

    skip_marker = pytest.mark.skip(
        reason=(
            "JS bridge tests must run via test_bridge_wrapper.py to avoid "
            "terminal rendering crashes. Run: pytest tests/unit/test_bridge_wrapper.py -v"
        )
    )

    for item in items:
        # Only skip tests in this directory (bridges/)
        if "/bridges/" in str(item.fspath):
            item.add_marker(skip_marker)
