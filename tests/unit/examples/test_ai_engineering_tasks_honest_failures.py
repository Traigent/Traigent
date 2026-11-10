"""ai-engineering-tasks examples must not invent results or write datasets the SDK rejects."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
EXAMPLES_DIR = REPO_ROOT / "examples" / "advanced" / "ai-engineering-tasks"
MAINS = sorted(EXAMPLES_DIR.glob("p*/main.py"))

_PROBE = """
import importlib.util, os, sys
from pathlib import Path
script = Path(sys.argv[1])
sys.path.insert(0, str(script.parent))
spec = importlib.util.spec_from_file_location("example_main", script)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
from traigent.evaluators.base import _get_dataset_root
root = _get_dataset_root()
datasets = sorted(Path(root).glob("*.jsonl"))
assert datasets, "example wrote no dataset under the dataset root"
print("OK", root)
"""


def test_all_six_examples_found() -> None:
    assert len(MAINS) == 6


@pytest.mark.parametrize("main", MAINS, ids=lambda p: p.parent.name)
def test_example_does_not_fabricate_results(main: Path) -> None:
    source = main.read_text(encoding="utf-8")
    assert "simulated_results" not in source
    assert "SimulatedResults" not in source
    assert "tempfile.NamedTemporaryFile" not in source


@pytest.mark.parametrize("main", MAINS, ids=lambda p: p.parent.name)
def test_example_dataset_is_loadable_from_any_cwd(main: Path, tmp_path: Path) -> None:
    env = {
        k: v
        for k, v in os.environ.items()
        if k not in {"TRAIGENT_DATASET_ROOT", "PYTHONPATH"}
    }
    env.update(
        {
            "TRAIGENT_MOCK_LLM": "true",
            "TRAIGENT_OFFLINE_MODE": "true",
            "PYTHONPATH": str(REPO_ROOT),
        }
    )
    proc = subprocess.run(
        [sys.executable, "-c", _PROBE, str(main)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    root = Path(proc.stdout.split()[-1]).resolve()
    assert root == main.parent.resolve()
