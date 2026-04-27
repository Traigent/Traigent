"""Conftest for JS evaluator tests.

JS evaluator tests may exercise the JS bridge in integration-style paths. The
bridge now owns subprocess isolation and cleanup, so direct test collection is
allowed. The subprocess wrapper is still available when captured output is
preferred:
    pytest tests/unit/test_bridge_wrapper.py -v
"""

import os

os.environ.setdefault("TRAIGENT_JS_TEST_SUBPROCESS", "1")
