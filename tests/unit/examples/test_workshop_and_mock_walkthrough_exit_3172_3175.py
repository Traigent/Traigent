"""Workshop quickstart runs clean; mock walkthroughs fail loudly (#3172, #3175)."""

from __future__ import annotations

import importlib.util
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
ADVANCED = ROOT / "walkthrough" / "mock" / "advanced"
WORKSHOP = ROOT / "examples" / "quickstart" / "01_simple_qa_workshop.py"


# Run a script as __main__ with dotenv neutralised so a discoverable .env can
# never supply a key or point the demo at a real backend.
_RUNNER = (
    "import runpy, sys\n"
    "try:\n"
    "    import dotenv\n"
    "    dotenv.load_dotenv = lambda *a, **k: False\n"
    "except ImportError:\n"
    "    pass\n"
    "runpy.run_path(sys.argv[1], run_name='__main__')\n"
)


def _run(script: Path, tmp_path: Path, **env: str) -> subprocess.CompletedProcess[str]:
    child_env = {
        k: v
        for k, v in os.environ.items()
        if not k.endswith("_API_KEY") and k != "TRAIGENT_MOCK_LLM"
    }
    child_env.update(
        PYTHONPATH=str(ROOT),
        TRAIGENT_RESULTS_FOLDER=str(tmp_path / "results"),
        TRAIGENT_DATASET_ROOT=str(ROOT),
        **env,
    )
    return subprocess.run(
        [sys.executable, "-c", _RUNNER, str(script)],
        cwd=ROOT,
        env=child_env,
        capture_output=True,
        text=True,
        timeout=170,
        check=False,
    )


def test_workshop_quickstart_runs_9_trials_with_numeric_cost_3172(
    tmp_path: Path,
) -> None:
    source = WORKSHOP.read_text()
    assert "_merge_comprehensive_metrics" not in source
    assert "_compute_aggregated_custom_metrics" not in source

    proc = _run(WORKSHOP, tmp_path)
    assert proc.returncode == 0, proc.stdout[-800:] + proc.stderr[-800:]
    payload = json.loads(proc.stdout[proc.stdout.index('{\n  "example"') :])
    assert payload["trialCount"] == 9
    cost = payload["bestMetrics"]["cost"]
    questions = re.findall(r'"question": "([^"]*)"', source)
    # Trial cost is the sum of the per-call simulated costs over the dataset.
    expected = sum(0.0012 if "capital" in q else 0.0005 for q in questions)
    assert cost == pytest.approx(expected)


@pytest.mark.parametrize(
    "script", ["04_workflow_traces_demo.py", "05_langgraph_multiagent_demo.py"]
)
def test_mock_walkthrough_without_api_key_exits_nonzero_3175(
    script: str, tmp_path: Path
) -> None:
    if script.startswith("05_") and not all(
        importlib.util.find_spec(m)
        for m in ("langgraph", "langchain_core", "langchain_openai", "opentelemetry")
    ):
        pytest.skip("05 needs langgraph/langchain/opentelemetry to reach the key check")
    proc = _run(ADVANCED / script, tmp_path, TRAIGENT_MOCK_LLM="true")
    assert proc.returncode != 0, proc.stdout[-800:] + proc.stderr[-800:]
    assert "TRAIGENT_API_KEY" in proc.stdout


# --- 05 success / failure paths (no network: tracker and optimize are stubbed) ---


class _FakeResponse:
    def __init__(self, success: bool) -> None:
        self.success = success
        self.error = None if success else "stub failure"
        self.graph_id = "g-1"


def _load_05(monkeypatch: pytest.MonkeyPatch):
    for mod in ("langgraph", "langchain_core", "langchain_openai", "opentelemetry"):
        if importlib.util.find_spec(mod) is None:
            pytest.skip("05 needs langgraph/langchain/opentelemetry")
    monkeypatch.setenv("TRAIGENT_MOCK_LLM", "true")
    monkeypatch.setenv("TRAIGENT_API_KEY", "fake-key-for-test")
    path = ADVANCED / "05_langgraph_multiagent_demo.py"
    spec = importlib.util.spec_from_file_location("_demo05_3175", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _stub_05(module, monkeypatch, *, ingest_ok: bool, metadata: dict, best=True):
    import types

    sent: list[object] = []

    class _Client:
        def ingest_traces(self, graph):
            sent.append(graph)
            return _FakeResponse(ingest_ok)

    class _Tracker:
        def __init__(self, *a, **k) -> None:
            self.client = _Client()

    class _Workflow:
        async def optimize(self, **kwargs):
            return types.SimpleNamespace(
                metadata=metadata,
                best_config={"model": "m", "temperature": 0.0} if best else {},
                best_metrics={"accuracy": 1.0, "cost": 0.0},
            )

    monkeypatch.setattr(module, "WorkflowTracesTracker", _Tracker)
    monkeypatch.setattr(module, "run_rag_workflow", _Workflow())
    return sent


def test_05_success_path_exits_zero_3175(monkeypatch: pytest.MonkeyPatch) -> None:
    import asyncio

    module = _load_05(monkeypatch)
    sent = _stub_05(
        module, monkeypatch, ingest_ok=True, metadata={"experiment_id": "exp-1"}
    )
    assert asyncio.run(module.main()) == 0
    assert len(sent) == 1


def test_05_failed_graph_send_exits_one_3175(monkeypatch: pytest.MonkeyPatch) -> None:
    import asyncio

    module = _load_05(monkeypatch)
    _stub_05(module, monkeypatch, ingest_ok=False, metadata={"experiment_id": "exp-1"})
    assert asyncio.run(module.main()) == 1


def test_05_offline_run_without_experiment_id_is_not_a_failure_3175(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import asyncio

    module = _load_05(monkeypatch)
    sent = _stub_05(module, monkeypatch, ingest_ok=True, metadata={})
    assert asyncio.run(module.main()) == 0
    assert sent == []


def test_05_no_best_config_exits_one_3175(monkeypatch: pytest.MonkeyPatch) -> None:
    import asyncio

    module = _load_05(monkeypatch)
    _stub_05(module, monkeypatch, ingest_ok=True, metadata={}, best=False)
    assert asyncio.run(module.main()) == 1
