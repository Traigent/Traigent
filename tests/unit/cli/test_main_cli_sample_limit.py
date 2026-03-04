"""CLI tests for sample-budget flags."""

from __future__ import annotations

import sys
from pathlib import Path
from uuid import uuid4

from click.testing import CliRunner

from traigent.cli.main import cli


def _write_temp_module(root: Path) -> Path:
    module_dir = root / "temp_modules"
    module_dir.mkdir(exist_ok=True)
    module_path = module_dir / f"module_{uuid4().hex}.py"

    module_path.write_text(
        """
last_kwargs = {}


class DummyResult:
    def __init__(self):
        self.status = "COMPLETED"
        self.trials = [object(), object()]
        self.successful_trials = [object()]
        self.best_config = {"param": 1}
        self.best_score = 0.75


async def _optimize_stub(*args, **kwargs):
    global last_kwargs
    last_kwargs = dict(kwargs)
    return DummyResult()


def fake_optimized_function():
    return "ok"


fake_optimized_function.optimize = _optimize_stub
"""
    )

    return module_path


def test_cli_passes_sample_limit_options():
    modules_root = Path(__file__).resolve().parent
    module_path = _write_temp_module(modules_root)

    try:
        sys.modules.pop("user_module", None)
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "optimize",
                str(module_path),
                "--max-total-examples",
                "4",
                "--samples-exclude-pruned",
            ],
        )

        assert result.exit_code == 0
        assert "Sample budget" in result.output

        user_module = sys.modules.get("user_module")
        assert user_module is not None
        recorded = getattr(user_module, "last_kwargs", None)
        assert recorded is not None
        assert recorded.get("max_total_examples") == 4
        assert recorded.get("samples_include_pruned") is False
    finally:
        sys.modules.pop("user_module", None)
        if module_path.exists():
            module_path.unlink()
        module_dir = module_path.parent
        if module_dir.exists() and not any(module_dir.iterdir()):
            module_dir.rmdir()
