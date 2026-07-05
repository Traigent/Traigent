"""Tests for CLI command: traigent algorithms."""

from click.testing import CliRunner

from traigent.cli.main import cli
from traigent.config.types import accepted_algorithm_values


def test_algorithms_command_lists_public_accepted_algorithm_surface() -> None:
    result = CliRunner().invoke(cli, ["algorithms"])

    assert result.exit_code == 0, result.output
    for algorithm in accepted_algorithm_values():
        assert algorithm in result.output

    for runtime_only_name in (
        "parallel_batch",
        "multi_objective_batch",
        "adaptive_batch",
        "remote",
    ):
        assert runtime_only_name not in result.output

    assert "auto" in result.output
    assert "TPE" in result.output
    assert "local" in result.output
    assert "connected" in result.output
