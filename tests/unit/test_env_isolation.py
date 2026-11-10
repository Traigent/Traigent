"""Regression: env-mutating test modules must not leak TRAIGENT_* variables."""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

LEAKY_MODULES = [
    "tests/unit/core/test_cost_enforcement_stateful.py",
    "tests/test_backend_config.py",
]

PLUGIN = textwrap.dedent(
    """
    import json, os

    def _snap():
        return {k: v for k, v in os.environ.items() if k.startswith("TRAIGENT_")}

    def pytest_configure(config):
        config._env_before = _snap()

    def pytest_sessionfinish(session, exitstatus):
        # runs after every test and module-scoped teardown
        out = os.environ["ENV_PROBE_OUT"]
        json.dump(
            {"before": session.config._env_before, "after": _snap()}, open(out, "w")
        )
    """
)


@pytest.mark.parametrize("module", LEAKY_MODULES)
def test_module_restores_traigent_env(module: str, tmp_path: Path) -> None:
    (tmp_path / "env_probe_plugin.py").write_text(PLUGIN)
    out = tmp_path / "probe.json"
    env = {
        "PATH": __import__("os").environ["PATH"],
        "PYTHONPATH": f"{tmp_path}:{REPO_ROOT}",
        "ENV_PROBE_OUT": str(out),
        "HOME": str(tmp_path),
    }
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            module,
            "-p",
            "env_probe_plugin",
            "-p",
            "no:cacheprovider",
            "-n0",
            "-q",
        ],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    snap = json.loads(out.read_text())
    assert snap["after"] == snap["before"], f"{module} leaked TRAIGENT_* env"
