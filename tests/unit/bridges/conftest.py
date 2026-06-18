"""Conftest for JS bridge tests.

Bridge unit tests mock subprocess creation, while integration paths can spawn
real Node.js runners. The bridge now isolates spawned runners in their own
process group on POSIX systems and terminates that group during cleanup.

The wrapper remains useful for captured-output runs:
    pytest tests/unit/test_bridge_wrapper.py -v
"""

import os

os.environ.setdefault("TRAIGENT_JS_TEST_SUBPROCESS", "1")
