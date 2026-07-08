"""Live-backend contract test for edge-analytics sync (#868, gates #1211/#1213).

Boots nothing itself: requires a running TraigentBackend + a scoped API key. It
builds a finalized cost-bearing local session and syncs it through the
content-free typed-session path (POST /sessions with
tracking_mode=native_local -> per-trial POST /sessions/{id}/results ->
POST /sessions/{id}/finalize). NO benchmark is created or bound, so the
backend's empty-dataset guard takes its no-dataset pass-through and an empty
server-side dataset never blocks the import. It then asserts the full chain
landed — session -> experiment -> experiment-run -> per-trial
configuration_run with cost INSIDE measures (metrics submitted per trial are
materialized by the backend's native_local handler). This is the only test
that catches sync<->backend contract drift end-to-end; mocked-transport unit
tests cannot.

Enable with:
    TRAIGENT_LIVE_SYNC_E2E=1 \
    TRAIGENT_BACKEND_URL=http://localtest.me:5001 \
    TRAIGENT_API_KEY=<scoped key: sessions/experiments :write> \
    pytest tests/cloud/test_sync_live_contract.py -m contract
"""

from __future__ import annotations

import os
import shutil
import tempfile

import pytest
import requests

pytestmark = [
    pytest.mark.contract,
    pytest.mark.skipif(
        not os.getenv("TRAIGENT_LIVE_SYNC_E2E"),
        reason="live backend not configured (set TRAIGENT_LIVE_SYNC_E2E=1 + URL + key)",
    ),
]

_BACKEND = os.getenv("TRAIGENT_BACKEND_URL", "http://localtest.me:5001")
_KEY = os.getenv("TRAIGENT_API_KEY", "")


def _api(path: str) -> str:
    return f"{_BACKEND.rstrip('/')}/api/v1{path}"


def test_edge_analytics_sync_lands_cost_in_configuration_run():
    """A finalized cost session syncs with zero step errors and lands cost>0."""
    store = tempfile.mkdtemp(prefix="traigent-sync-contract-")
    os.environ.update(
        TRAIGENT_BACKEND_URL=_BACKEND,
        TRAIGENT_API_URL=_BACKEND,
        TRAIGENT_ALLOW_INSECURE_BACKEND="true",
        ENVIRONMENT="development",
        TRAIGENT_RESULTS_FOLDER=store,
        TRAIGENT_API_KEY=_KEY,
    )
    try:
        from traigent.cloud.sync_manager import SyncManager
        from traigent.config.types import TraigentConfig
        from traigent.storage.local_storage import LocalStorageManager

        lsm = LocalStorageManager(storage_path=store)
        sid = lsm.create_session(
            "contract_probe",
            optimization_config={
                "objectives": ["accuracy"],
                "search_space": {"model": ["a", "b"]},
            },
        )
        lsm.add_trial_result(
            sid,
            config={"model": "a"},
            score=1.0,
            cost=0.000123,
            total_cost=0.000123,
            metadata={"accuracy": 1.0},
        )
        lsm.add_trial_result(
            sid,
            config={"model": "b"},
            score=1.0,
            cost=0.000456,
            total_cost=0.000456,
            metadata={"accuracy": 1.0},
        )
        lsm.update_session_status(sid, "completed")

        result = SyncManager(
            TraigentConfig.from_environment(), _KEY
        ).sync_session_to_cloud(sid)

        # 1) Zero step errors, all trials converted.
        assert result.get("status") == "success", result.get("errors")
        assert result.get("trials_converted") == 2
        experiment_id = result.get("cloud_experiment_id")
        assert experiment_id

        # 2) Read the cost back through the API and assert it landed in measures.
        headers = {"x-api-key": _KEY}
        runs = requests.get(
            _api(f"/experiment-runs/{experiment_id}/runs"), headers=headers, timeout=10
        )
        assert runs.status_code == 200, runs.text
        runs_body = runs.json()
        run_items = (
            runs_body.get("runs")
            or (runs_body.get("data") or {}).get("items")
            or runs_body.get("data")
            or []
        )
        first_run = run_items[0] if run_items else {}
        run_id = (
            first_run.get("run_id")
            or first_run.get("id")
            or first_run.get("experiment_run_id")
        )
        assert run_id, runs.text

        configs = requests.get(
            _api(f"/configuration-runs/runs/{run_id}/configurations"),
            headers=headers,
            timeout=10,
        )
        assert configs.status_code == 200, configs.text
        cfg_items = (
            (configs.json().get("data") or {}).get("items")
            or configs.json().get("data")
            or []
        )

        # The live backend serves configuration-run measures as a LIST of
        # measure dicts; older responses used a single dict. Accept both.
        def _iter_measures(config_run: dict) -> list[dict]:
            measures = config_run.get("measures")
            if isinstance(measures, dict):
                return [measures]
            if isinstance(measures, list):
                return [m for m in measures if isinstance(m, dict)]
            return []

        costs = [float(m.get("cost", 0)) for c in cfg_items for m in _iter_measures(c)]
        assert any(cost > 0 for cost in costs), (
            f"no configuration_run carried cost>0: {cfg_items}"
        )
    finally:
        shutil.rmtree(store, ignore_errors=True)
