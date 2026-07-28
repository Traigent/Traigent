"""Tests for CLI command: traigent algorithms."""

from collections.abc import Callable

from click.testing import CliRunner

from traigent.cli.main import cli
from traigent.config.types import accepted_algorithm_values


def test_algorithms_command_lists_public_accepted_algorithm_surface(
    plain: Callable[[str], str],
) -> None:
    result = CliRunner().invoke(cli, ["algorithms"])

    # Normalized like every other substring assertion in this package: the
    # negatives below matter most here, and a style boundary landing inside a
    # runtime-only name would hide it from a raw ``not in`` while the CLI
    # printed it in full.
    output = plain(result.output)

    assert result.exit_code == 0, result.output
    for algorithm in accepted_algorithm_values():
        assert algorithm in output

    for runtime_only_name in (
        "parallel_batch",
        "multi_objective_batch",
        "adaptive_batch",
        "remote",
    ):
        assert runtime_only_name not in output

    assert "auto" in output
    assert "TPE" in output
    assert "local" in output
    assert "connected" in output
