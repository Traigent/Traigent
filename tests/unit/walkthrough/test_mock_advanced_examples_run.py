"""Documented mock walkthrough scripts must run to completion in mock mode."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
ADVANCED_DIR = REPO_ROOT / "walkthrough" / "mock" / "advanced"


@pytest.mark.parametrize("script", ["01_tuned_variables.py", "03_multi_agent.py"])
def test_mock_advanced_script_exits_zero(script: str) -> None:
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env.update(
        {
            "TRAIGENT_MOCK_LLM": "true",
            "TRAIGENT_OFFLINE_MODE": "true",
            "TRAIGENT_DATASET_ROOT": str(REPO_ROOT),
            "PYTHONPATH": str(REPO_ROOT),
        }
    )
    proc = subprocess.run(
        [sys.executable, script],
        cwd=ADVANCED_DIR,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert proc.returncode == 0, (proc.stdout + proc.stderr)[-2000:]
    assert "Traceback" not in proc.stderr
