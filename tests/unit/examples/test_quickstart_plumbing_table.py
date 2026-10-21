"""The keyless quickstart's results table names no winner (#1690).

The demo is a plumbing check with simulated scores, so the rendered table must
not star a trial or print the "Overall Best" legend. Runs the real entry point
in a subprocess so the demo's mock-mode activation stays out of this process.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]


def test_keyless_quickstart_table_has_no_overall_best(tmp_path: Path) -> None:
    env = {k: v for k, v in os.environ.items() if k != "TRAIGENT_API_KEY"}
    env["PYTHONPATH"] = str(REPO_ROOT)
    proc = subprocess.run(
        [sys.executable, "-m", "traigent.examples.quickstart"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    output = proc.stdout + proc.stderr

    # The table itself rendered, so the absence checks below are meaningful.
    assert "Trial Results" in output
    assert "plumbing check" in output

    assert "Overall Best" not in output
    assert "★" not in output
