"""Conftest for evaluator tests.

JS evaluator tests (test_js_evaluator*.py) should only run via
test_bridge_wrapper.py subprocess to avoid terminal rendering crashes.

Non-JS evaluator tests run normally.
"""

import os

import pytest

# Environment variable set by test_bridge_wrapper.py when running via subprocess
_SUBPROCESS_ENV_VAR = "TRAIGENT_JS_TEST_SUBPROCESS"


def pytest_collection_modifyitems(config, items):
    """Skip JS evaluator tests unless running via subprocess wrapper."""
    if os.environ.get(_SUBPROCESS_ENV_VAR) == "1":
        # Running via wrapper subprocess - allow all tests
        return

    skip_marker = pytest.mark.skip(
        reason=(
            "JS evaluator tests must run via test_bridge_wrapper.py to avoid "
            "terminal rendering crashes. Run: pytest tests/unit/test_bridge_wrapper.py -v"
        )
    )

    for item in items:
        # Only skip JS evaluator tests (test_js_evaluator*.py)
        if "test_js_evaluator" in item.fspath.basename:
            item.add_marker(skip_marker)
