"""CLI optimize path handling regression tests."""

from __future__ import annotations

import sys
from pathlib import Path
from uuid import uuid4

from click.testing import CliRunner

from traigent.cli.main import cli


def _write_optimizable_module(root: Path) -> Path:
    module_path = root / f"module_{uuid4().hex}.py"
    module_path.write_text("""
last_kwargs = {}


class DummyResult:
    def __init__(self):
        self.status = "COMPLETED"
        trial = DummyTrial()
        self.trials = [trial]
        self.successful_trials = [trial]
        self.best_config = {"param": 1}
        self.best_score = 0.75
        self.metadata = {"function_name": "fake_optimized_function"}
        self.algorithm = "grid"
        self.objectives = ["accuracy"]
        self.preset_selection = None
        self.success_rate = 1.0
        self.duration = 0.01
        self.convergence_info = {}


class DummyTrial:
    def __init__(self):
        self.config = {"param": 1}
        self.metrics = {"accuracy": 0.75}
        self.duration = 0.01
        self.status = "completed"
        self.timestamp = None
        self.metadata = {}


async def _optimize_stub(*args, **kwargs):
    global last_kwargs
    last_kwargs = dict(kwargs)
    return DummyResult()


def fake_optimized_function():
    return "ok"


fake_optimized_function.optimize = _optimize_stub
""")
    return module_path


def test_optimize_accepts_user_project_file_outside_package_dir(tmp_path):
    module_path = _write_optimizable_module(tmp_path)
    output_dir = tmp_path / "results"

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "optimize",
            str(module_path),
            "--output-dir",
            str(output_dir),
        ],
    )

    try:
        assert result.exit_code == 0, result.output
        assert "Optimization Complete" in result.output
        assert "workspace" not in result.output.lower()
    finally:
        sys.modules.pop("user_module", None)


def test_optimize_accepts_user_output_directory_outside_package_dir(tmp_path):
    module_path = _write_optimizable_module(tmp_path)
    output_dir = tmp_path / "user_results"

    try:
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "optimize",
                str(module_path),
                "--output-dir",
                str(output_dir),
            ],
        )

        assert result.exit_code == 0, result.output
        assert "Results saved to:" in result.output
        assert output_dir.exists()
    finally:
        sys.modules.pop("user_module", None)
