"""Regression coverage for the retired legacy analytics surface."""

from __future__ import annotations

import importlib.util

from traigent.analytics import __all__ as analytics_exports
from traigent.cli.main import cli


def test_legacy_next_steps_package_and_cli_surfaces_are_absent() -> None:
    """The SDK exposes neither the retired module nor its CLI command."""
    assert importlib.util.find_spec("traigent.analytics.next_steps") is None
    assert "NextStepsClient" not in analytics_exports
    assert "next-steps" not in cli.commands
