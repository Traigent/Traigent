"""Round-4 sol findings: budget submission and offline sync follow tracked spend."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from traigent.api.types import TrialStatus
from traigent.cloud.sync_manager import SyncManager
from traigent.core.backend_session_manager import BackendSessionManager
from traigent.storage.local_storage import TrialResult as StoredTrial


def _manager(armed: set[str]) -> BackendSessionManager:
    manager = BackendSessionManager.__new__(BackendSessionManager)
    manager._session_cost_budget_armed = armed
    manager._cost_budget_zero_backfill_logged = False
    return manager


def _trial(metrics: dict, tracked: dict | None, metadata: dict | None = None):
    return SimpleNamespace(
        status=TrialStatus.COMPLETED,
        metrics=metrics,
        metadata=metadata or {},
        tracked_spend=tracked,
    )


def _submit(armed: bool, metrics: dict, tracked: dict | None) -> dict:
    payload = dict(metrics)
    _manager({"s1"} if armed else set())._ensure_budget_cost_metric(
        session_id="s1",
        trial_result=_trial(metrics, tracked),
        metrics_payload=payload,
        metadata={},
    )
    return payload


def test_live_submission_sends_tracker_cost_not_custom_zero() -> None:
    payload = _submit(
        True,
        {"accuracy": 1.0, "cost": 0.0, "total_cost": 0.0},
        {"unmeasured": False, "cost": 0.20, "lower_bound": 0.0},
    )
    assert payload["cost"] == pytest.approx(0.20)
    assert payload["total_cost"] == pytest.approx(0.20)


def test_live_submission_never_sends_custom_inflated_cost() -> None:
    payload = _submit(
        True,
        {"accuracy": 1.0, "cost": 99.0, "total_cost": 99.0},
        {"unmeasured": False, "cost": 0.20, "lower_bound": 0.0},
    )
    assert payload["cost"] == pytest.approx(0.20)
    assert payload["total_cost"] == pytest.approx(0.20)


def test_live_submission_unknown_branch_sends_tracker_floor() -> None:
    payload = _submit(
        True,
        {"accuracy": 1.0, "cost": 99.0, "total_cost": 99.0, "cost_unmeasured": 1.0},
        {"unmeasured": True, "cost": None, "lower_bound": 0.05},
    )
    assert payload["cost_unmeasured"] == 1.0
    assert payload["cost"] == pytest.approx(0.05)
    assert payload.get("total_cost") in (None, pytest.approx(0.05))


def test_live_submission_unknown_unarmed_drops_custom_cost() -> None:
    payload = _submit(
        False,
        {"accuracy": 1.0, "cost": 99.0, "total_cost": 99.0, "cost_unmeasured": 1.0},
        {"unmeasured": True, "cost": None, "lower_bound": 0.05},
    )
    assert "cost" not in payload
    assert "total_cost" not in payload


# --- offline persistence + sync ---------------------------------------------
def test_offline_metadata_persists_tracked_spend() -> None:
    manager = BackendSessionManager.__new__(BackendSessionManager)
    manager._backend_client = None
    storage = MagicMock()
    manager._local_storage = lambda: storage
    manager._egress_disabled = lambda: True
    tracked = {"unmeasured": False, "cost": 0.20, "lower_bound": 0.0}
    trial = SimpleNamespace(
        trial_id="t1",
        config={"m": 1},
        error_message=None,
        tracked_spend=tracked,
    )
    manager._persist_trial_locally(
        session_id="s1", trial_result=trial, score=0.5, metadata={"x": 1}
    )
    metadata = storage.add_trial_result.call_args.kwargs["metadata"]
    assert metadata["tracked_spend"] == tracked


@pytest.fixture
def sync_manager() -> SyncManager:
    with patch("traigent.cloud.sync_manager.LocalStorageManager"):
        return SyncManager(
            config=MagicMock(),
            api_key="tg_" + "a" * 61,  # pragma: allowlist secret
        )


def _stored(metadata: dict) -> StoredTrial:
    return StoredTrial(
        trial_id=1,
        config={"m": 1},
        score=0.5,
        timestamp="2026-06-09T10:00:00Z",
        metadata=metadata,
    )


def test_sync_uses_tracked_spend_over_custom_cost(sync_manager) -> None:
    trial = _stored(
        {
            "tracked_spend": {"unmeasured": False, "cost": 0.20, "lower_bound": 0.0},
            "all_metrics": {"cost": 99.0, "total_cost": 99.0},
            "total_example_cost": 99.0,
        }
    )
    measures = sync_manager._convert_trials_to_results([trial])[0]["measures"]
    assert measures["cost"] == pytest.approx(0.20)
    assert measures["total_cost"] == pytest.approx(0.20)


def test_sync_unknown_tracked_spend_sends_floor_only(sync_manager) -> None:
    trial = _stored(
        {
            "tracked_spend": {"unmeasured": True, "cost": None, "lower_bound": 0.05},
            "all_metrics": {"cost": 99.0, "total_cost": 99.0},
            "total_example_cost": 99.0,
        }
    )
    measures = sync_manager._convert_trials_to_results([trial])[0]["measures"]
    assert measures.get("cost") is None
    assert measures.get("total_cost") is None
    assert measures["cost_unmeasured"] == 1.0
    assert measures["cost_lower_bound"] == pytest.approx(0.05)


def test_sync_old_record_without_tracked_spend_falls_back(sync_manager) -> None:
    trial = _stored({"total_example_cost": 0.04})
    measures = sync_manager._convert_trials_to_results([trial])[0]["measures"]
    assert measures["cost"] == 0.04
