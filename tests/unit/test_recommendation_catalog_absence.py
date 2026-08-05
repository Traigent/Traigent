"""Regression coverage for the removed public recommendation catalogs."""

from __future__ import annotations

import importlib.util

from click.testing import CliRunner

import traigent
import traigent.api
import traigent.evaluators
from traigent.cli.main import cli
from traigent.mcp.tools import V1_TOOL_NAMES


REMOVED_MODULES = (
    "traigent.config_generator.recommendations",
    "traigent.evaluators.catalog_loader",
    "traigent.evaluators.recommendations",
)
REMOVED_ROOT_AND_API_EXPORTS = (
    "list_recommendation_agent_types",
    "recommend_configuration_space",
)
REMOVED_EVALUATOR_EXPORTS = (
    "EVAL_RECOMMENDATION_CAVEAT",
    "list_eval_recommendation_task_types",
    "recommend_evaluator",
    "recommend_metrics",
)
REMOVED_MCP_TOOL_NAMES = {
    "list_recommendation_agent_types",
    "recommend_configuration_space",
}


def test_public_recommendation_catalog_surfaces_are_absent() -> None:
    for module_name in REMOVED_MODULES:
        assert importlib.util.find_spec(module_name) is None

    for export_name in REMOVED_ROOT_AND_API_EXPORTS:
        assert export_name not in traigent.__all__
        assert export_name not in traigent.api.__all__
        assert not hasattr(traigent, export_name)
        assert not hasattr(traigent.api, export_name)

    for export_name in REMOVED_EVALUATOR_EXPORTS:
        assert export_name not in traigent.evaluators.__all__
        assert not hasattr(traigent.evaluators, export_name)

    for command_name in ("recommend", "recommend-eval"):
        result = CliRunner().invoke(cli, [command_name])
        assert result.exit_code != 0
        assert f"No such command '{command_name}'" in result.output

    assert REMOVED_MCP_TOOL_NAMES.isdisjoint(V1_TOOL_NAMES)
