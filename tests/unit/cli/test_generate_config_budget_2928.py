"""generate-config --budget must be finite and > 0 (#2928)."""

from __future__ import annotations

import pytest
from click.testing import CliRunner

from traigent.cli.generate_config_command import generate_config


@pytest.mark.parametrize("bad", ["nan", "inf", "-inf", "0", "-0.5"])
def test_budget_rejected(tmp_path, bad):
    src = tmp_path / "app.py"
    src.write_text("def f(x):\n    return x\n")
    result = CliRunner().invoke(
        generate_config, [str(src), "--enrich", "--budget", bad]
    )
    assert result.exit_code == 2, result.output
    assert "budget" in result.output.lower()
