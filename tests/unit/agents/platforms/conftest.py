"""Fixtures for the agents/platforms test package.

Provides test isolation for the global PlatformRegistry._executors dict.
Without this, tests that register custom executors (e.g. test_register_executor,
test_custom_platform_integration) leak state into subsequent tests running in the
same xdist worker process, causing spurious failures such as:

    assert 'custom' not in get_supported_platforms()
"""

import pytest

from traigent.agents.platforms import PlatformRegistry


@pytest.fixture(autouse=True)
def _restore_executor_registry():
    """Snapshot and restore PlatformRegistry._executors around every test.

    The registry is a class-level dict (global state). Tests that call
    register_executor() would otherwise leak custom platforms across xdist
    worker boundaries.  This fixture is autouse so it guards every test in
    this package without requiring individual tests to opt in.
    """
    snapshot = dict(PlatformRegistry._executors)
    yield
    PlatformRegistry._executors = snapshot
