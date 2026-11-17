"""Non-finite flush intervals and watch intervals are rejected (#3136)."""

import math

import pytest
from click.testing import CliRunner

from traigent.cli.requirements_commands import requirements
from traigent.cloud.optimizer_client import OptimizerDirectClient


@pytest.mark.parametrize("bad", [math.nan, math.inf, 0, -1.0])
def test_set_flush_interval_rejects_non_finite_and_non_positive(bad):
    client = OptimizerDirectClient("https://api.example.com/opt", "tok")
    before = client._flush_interval
    with pytest.raises(ValueError):
        client.set_flush_interval(bad)
    assert client._flush_interval == before


def test_set_flush_interval_accepts_finite_positive():
    client = OptimizerDirectClient("https://api.example.com/opt", "tok")
    client.set_flush_interval(0.5)
    assert client._flush_interval == 0.5


@pytest.mark.parametrize("bad", ["nan", "inf"])
def test_requirements_watch_rejects_non_finite_interval(tmp_path, bad):
    src = tmp_path / "agent.py"
    src.write_text("def fn():\n    pass\n")
    result = CliRunner().invoke(
        requirements, ["watch", str(src), "-f", "fn", "--interval", bad]
    )
    assert result.exit_code == 2
    assert "--interval" in result.output
    assert not isinstance(result.exception, ValueError)
