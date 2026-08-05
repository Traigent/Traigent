"""Regression coverage for the removed Planner V2 public boundary."""

from __future__ import annotations

import importlib.util

from click.testing import CliRunner

import traigent.analytics as analytics
from traigent.cli.main import cli


def test_planner_v2_module_export_and_cli_are_absent() -> None:
    """Planner V2 has no importable package, export, or CLI command."""
    assert importlib.util.find_spec("traigent.analytics.planner") is None
    assert not hasattr(analytics, "PlannerV2Client")

    result = CliRunner().invoke(cli, ["guidance"])

    assert result.exit_code != 0
    assert "No such command 'guidance'" in result.output
