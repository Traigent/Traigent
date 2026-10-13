"""Honest cost round 16: simulated example costs are labelled, not presented as spend.

Offline stand-in LLMs in the advanced examples price *estimated* tokens at list
price. The run output and the exported results must say so, and the export must
carry a machine-readable marker next to the cost totals.
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from traigent.cloud.session_budgets import (
    ensure_cost_metric_for_budgeted_completed_submission as ensure,
)
from traigent.utils.trial_costs import lift_flagged_cost_to_floor

REPO = Path(__file__).resolve().parents[3]
ADVANCED = REPO / "examples" / "advanced"
RUNNERS = [
    "optimization-routing/ex01-local-search",
    "optimization-routing/ex02-offline-no-egress",
    "results-analysis/ex01-viewing-results",
    "results-analysis/ex02-performance-metrics",
]


@pytest.mark.parametrize("example", RUNNERS)
def test_advanced_example_labels_simulated_cost(example: str, tmp_path: Path) -> None:
    env = {
        **os.environ,
        "TRAIGENT_MOCK_LLM": "true",
        "TRAIGENT_OFFLINE_MODE": "true",
        "PYTHONPATH": str(REPO),
        "TRAIGENT_DATASET_ROOT": str(ADVANCED / example),
    }
    proc = subprocess.run(
        [sys.executable, str(ADVANCED / example / "run.py")],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    banner = "SIMULATED estimates for the offline stand-in model"
    assert banner in proc.stdout
    # The notice comes before the printed results.
    assert proc.stdout.index(banner) < proc.stdout.index('"total_cost"')
    exported = json.loads((tmp_path / "results.json").read_text(encoding="utf-8"))
    assert exported["cost_source"] == "simulated_estimate"
    assert "total_cost" in exported


def test_marker_absent_with_real_client() -> None:
    path = ADVANCED / RUNNERS[0] / "local_search.py"
    sys.path.insert(0, str(path.parent))
    try:
        spec = importlib.util.spec_from_file_location("_r16_local_search", path)
        assert spec and spec.loader
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        sys.path.remove(str(path.parent))
    mod._USING_FALLBACK_LLM = True
    assert mod.simulated_cost_provenance() == {"cost_source": "simulated_estimate"}
    mod._USING_FALLBACK_LLM = False
    assert mod.simulated_cost_provenance() == {}


# --- follow-up 1: a numeric string cost never survives the floor lift --------
def test_lift_replaces_numeric_string_cost_with_floor() -> None:
    metrics = {"cost": "0.4", "cost_lower_bound": 0.4, "cost_unmeasured": 1}
    assert lift_flagged_cost_to_floor(metrics) is True
    assert metrics["cost"] == pytest.approx(0.4)
    assert isinstance(metrics["cost"], float)


def test_budget_writer_cloud_lifts_numeric_string_cost() -> None:
    client = SimpleNamespace(_cost_budget_armed_sessions={"s1"})
    metrics = {"cost": "0.4", "cost_lower_bound": 0.4, "cost_unmeasured": 1}
    ensure(client=client, session_id="s1", metrics=metrics, status="COMPLETED")
    assert isinstance(metrics["cost"], float)
    assert metrics["cost"] == pytest.approx(0.4)


def test_budget_writer_session_manager_lifts_numeric_string_cost() -> None:
    from traigent.api.types import TrialStatus
    from traigent.core.backend_session_manager import BackendSessionManager

    manager = BackendSessionManager.__new__(BackendSessionManager)
    manager._session_cost_budget_armed = {"s1"}
    payload = {"cost": "0.4", "cost_lower_bound": 0.4, "cost_unmeasured": 1.0}
    trial = SimpleNamespace(
        status=TrialStatus.COMPLETED, metrics=dict(payload), metadata={}
    )
    manager._ensure_budget_cost_metric(
        session_id="s1", trial_result=trial, metrics_payload=payload, metadata={}
    )
    assert isinstance(payload["cost"], float)
    assert payload["cost"] == pytest.approx(0.4)


def test_lift_keeps_numeric_original_above_floor() -> None:
    metrics = {"cost": 0.9, "cost_lower_bound": 0.4, "cost_unmeasured": 1}
    assert lift_flagged_cost_to_floor(metrics) is False
    assert metrics["cost"] == 0.9


# --- follow-up 2: walkthrough 06 scorer unwraps the with_usage envelope ------
def test_walkthrough_06_scorer_unwraps_usage_envelope(monkeypatch) -> None:
    monkeypatch.delenv("TRAIGENT_MOCK_LLM", raising=False)
    path = REPO / "walkthrough" / "mock" / "06_custom_evaluator.py"
    monkeypatch.syspath_prepend(str(REPO))
    monkeypatch.syspath_prepend(str(path.parent))
    spec = importlib.util.spec_from_file_location("_r16_walkthrough_06", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    # Importing the walkthrough turns mock mode on; the heuristic branch is the
    # one that reads ``output`` and the one that crashed on the envelope.
    monkeypatch.delenv("TRAIGENT_MOCK_LLM", raising=False)
    code = 'def f():\n    """doc"""\n    return 1  # c\n'
    bare = mod.code_evaluator(code, "")
    wrapped = mod.code_evaluator({"text": code, "usage": {}}, "")
    assert bare == pytest.approx(1.0)
    assert wrapped == bare
    assert mod.code_evaluator({"text": None}, "") == 0.0
