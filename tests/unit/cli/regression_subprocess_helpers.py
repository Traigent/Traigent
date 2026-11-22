"""Run the real ``traigent regression`` CLI in a subprocess (no tests here).

A subprocess with stdout and stderr as pipes is the only way to see what
reaches the real descriptors: an in-process runner replaces ``sys.stdout``.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from tests.unit.cli.regression_run_helpers import MODULE_SOURCE, make_row, write_rows


def make_workspace(root: Path, source: str = MODULE_SOURCE) -> Path:
    (root / "agent.py").write_text(source, encoding="utf-8")
    (root / "config.json").write_text(json.dumps({"mode": "m"}), encoding="utf-8")
    write_rows(root, [make_row("P", {"q": "pass"}, "ok")])
    return root


def run_cli(
    root: Path, verb: str, *extra: str, pre: str = ""
) -> subprocess.CompletedProcess[str]:
    """Run ``traigent regression <verb>`` in ``root``; ``pre`` runs first."""
    env = dict(os.environ)
    env.update(
        TRAIGENT_DATASET_ROOT=str(root),
        TRAIGENT_BACKEND_URL="http://127.0.0.1:9",
        TRAIGENT_API_URL="http://127.0.0.1:9",
        TRAIGENT_MOCK_LLM="true",
        TRAIGENT_OFFLINE_MODE="true",
        PYTHONPATH=os.pathsep.join(p for p in sys.path if p),
    )
    code = (
        pre
        + "\nfrom traigent.cli.main import cli\n"
        + f"cli(['regression', {verb!r}, '--module', 'agent.py:answer', "
        + "'--config', 'config.json', '--dir', 'reg'"
        + "".join(f", {arg!r}" for arg in extra)
        + "])\n"
    )
    return subprocess.run(  # noqa: S603 - fixed argv, synthetic workspace
        [sys.executable, "-c", code],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
