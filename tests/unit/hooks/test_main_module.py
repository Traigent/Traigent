"""Regression test for #2634: ``python -m traigent`` must work.

The installed git hooks fall back to ``python -m traigent`` when the
``traigent`` console script is not on PATH.
"""

# Traceability: CONC-Quality-Reliability FUNC-API-ENTRY REQ-API-001

from __future__ import annotations

import subprocess
import sys


def test_python_dash_m_traigent_prints_version() -> None:
    proc = subprocess.run(
        [sys.executable, "-m", "traigent", "--version"],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr[-500:]
    assert "traigent" in proc.stdout.lower()
