"""Security regression tests for CLI optimize command."""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from click.testing import CliRunner

from traigent.cli.main import cli


def test_optimize_rejects_paths_outside_workspace(tmp_path):
    """Passing a file outside the repository workspace must fail fast."""
    external_module = tmp_path / "malicious.py"
    external_module.write_text("print('malicious')")

    runner = CliRunner()
    result = runner.invoke(cli, ["optimize", str(external_module)])

    assert result.exit_code != 0
    assert "workspace" in result.output.lower()


def test_optimize_rejects_output_directory_outside_workspace():
    """--output-dir must be constrained to the repository workspace."""
    modules_dir = Path(__file__).resolve().parent / "temp_modules"
    modules_dir.mkdir(exist_ok=True)
    module_path = modules_dir / f"module_{uuid4().hex}.py"
    module_path.write_text("def plain_function():\n    return 42\n")

    try:
        runner = CliRunner()
        outside_output = Path("/tmp") / f"traigent_outside_{uuid4().hex}"
        result = runner.invoke(
            cli,
            [
                "optimize",
                str(module_path),
                "--output-dir",
                str(outside_output),
            ],
        )

        assert result.exit_code != 0
        assert "output directory" in result.output.lower()
    finally:
        module_path.unlink(missing_ok=True)
