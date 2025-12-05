"""Tests for constraint handling inside the optimization orchestrator."""

from __future__ import annotations

import pytest

from traigent.core.orchestrator_helpers import (
    constraint_requires_metrics,
    enforce_constraints,
)
from traigent.utils.exceptions import TVLConstraintError


def test_constraint_requires_metrics_checks_metadata() -> None:
    """Helper respects metadata flag."""

    def constraint(config, metrics):
        return True

    constraint.__tvl_constraint__ = {"requires_metrics": True}
    assert constraint_requires_metrics(constraint) is True

    constraint.__tvl_constraint__["requires_metrics"] = False
    assert constraint_requires_metrics(constraint) is False


def test_enforce_constraints_raises_on_failure() -> None:
    """enforce_constraints raises TVLConstraintError with helpful detail."""

    def bad_constraint(config, metrics=None):
        return config.get("temperature", 0) < 0.5

    bad_constraint.__tvl_constraint__ = {"id": "temp", "message": "Too hot"}

    with pytest.raises(TVLConstraintError, match="Too hot"):
        enforce_constraints({"temperature": 0.9}, None, [bad_constraint], stage="pre")
