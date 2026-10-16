"""Observability unit-test fixtures."""

from __future__ import annotations

import pytest

from traigent.identity.workload import reset_environment_warnings


@pytest.fixture(autouse=True)
def _fresh_environment_warnings():
    """Environment warnings fire once per process; give each test a clean slate."""
    reset_environment_warnings()
    yield
