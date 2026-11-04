"""The local-guards example runs end to end and blocks what it should."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("tvl", reason="needs the runtime-tvl extra (tvl-spec)")

RUN_PATH = Path(__file__).resolve().parents[2] / "examples/advanced/local-guards/run.py"


@pytest.mark.timeout(60)
def test_local_guards_example_blocks_unapproved_deletes() -> None:
    result = subprocess.run(
        [sys.executable, str(RUN_PATH)], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
    out = result.stdout.splitlines()
    assert "  REFUSED (local_guard_blocked); side effect ran: False" in out
    assert out.count("  deleted vol-1") == 2
    assert "  deleted vol-2" not in out
    assert out.count("  volumes deleted: []") == 1
    assert out.count("  volumes deleted: ['vol-1']") == 2
