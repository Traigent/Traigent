"""multi-objective-tradeoff examples: mock quick start runs and unmeasured cost is not shown as $0."""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
EXAMPLE_DIR = REPO_ROOT / "examples" / "core" / "multi-objective-tradeoff"

_SCRIPTS = [
    "run_anthropic.py",
    "run_many_providers.py",
    "run_openai.py",
    "run_openai_backend_smart.py",
    "run_openai_backend_smart_concurrency.py",
]

# Provider client packages are optional and unused in mock mode; stub them so the
# scripts import in a minimal environment.
_RUNNER = """
import runpy, sys, types
from unittest import mock
for name in ("langchain_anthropic", "langchain_openai", "langchain_core",
             "langchain_core.messages"):
    try:
        __import__(name)
    except ImportError:
        sys.modules[name] = mock.MagicMock()
sys.argv = [sys.argv[1], "--max-trials", "3"]
runpy.run_path(sys.argv[0], run_name="__main__")
"""


def _load_format_cost():
    spec = importlib.util.spec_from_file_location(
        "_cost_display", EXAMPLE_DIR / "_cost_display.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.format_cost


def test_unmeasured_cost_is_not_presented_as_zero() -> None:
    format_cost = _load_format_cost()
    assert format_cost(None) == "unmeasured"
    assert "$0.000000" not in format_cost(None)
    assert "at least $0.250000" in format_cost(None, 0.25)
    assert format_cost(0.0) == "$0.000000"
    assert format_cost(1.5) == "$1.500000"


@pytest.mark.parametrize("script", _SCRIPTS)
def test_mock_path_reports_a_measured_cost(script: str) -> None:
    source = (EXAMPLE_DIR / script).read_text(encoding="utf-8")
    assert "with_usage(_evaluate_expression(question), total_cost=0.0)" in source
    assert 'float(summary.get("total_cost", 0.0) or 0.0)' not in source


@pytest.mark.parametrize("script", ["run_anthropic.py", "run_many_providers.py"])
def test_documented_mock_quick_start_exits_zero(script: str) -> None:
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env.update(
        {
            "TRAIGENT_MOCK_LLM": "true",
            "TRAIGENT_OFFLINE_MODE": "true",
            "TRAIGENT_COST_APPROVED": "true",
            "TRAIGENT_DATASET_ROOT": str(REPO_ROOT),
            "PYTHONPATH": str(REPO_ROOT),
        }
    )
    proc = subprocess.run(
        [sys.executable, "-c", _RUNNER, str(EXAMPLE_DIR / script)],
        cwd=EXAMPLE_DIR,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert proc.returncode == 0, (proc.stdout + proc.stderr)[-2000:]
