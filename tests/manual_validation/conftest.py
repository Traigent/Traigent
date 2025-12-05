"""Pytest configuration for manual validation tests.

Skips these tests by default since they require a local backend service.
Enable by setting RUN_MANUAL_VALIDATION=1 in the environment.
"""

import os

import pytest

pytestmark = pytest.mark.skipif(
    not os.getenv("RUN_MANUAL_VALIDATION"),
    reason="Manual validation tests require a local backend (set RUN_MANUAL_VALIDATION=1)",
)
