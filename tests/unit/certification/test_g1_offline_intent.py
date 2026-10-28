"""Offline capture keeps G1 intent: a pending marker syncs as the closed flag.

An offline run cannot declare ``snapshot_g1_required`` at registration. It
persists a local ``snapshot_g1_pending`` marker with the session so a later
``traigent sync`` registers WITH the flag after the readiness probe says
``available``, and refuses closed (never registers plain) when it does not.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

import traigent
from tests.unit.certification.fakes import FakeG1Backend
from traigent.api.functions import with_usage
from traigent.certification import snapshot_binding
from traigent.cloud.sync_manager import SyncManager
from traigent.config.types import TraigentConfig
from traigent.evaluators.base import Dataset, EvaluationExample
from traigent.snapshot.backends import LocalDirStore
from traigent.storage.local_storage import LocalStorageManager

PENDING = "snapshot_g1_pending"


@pytest.fixture(autouse=True)
def _env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("TRAIGENT_RESULTS_FOLDER", str(tmp_path / "results"))
    monkeypatch.setenv("TRAIGENT_COST_APPROVED", "true")
    monkeypatch.setenv("TRAIGENT_RUN_COST_LIMIT", "100.0")
    for name in ("TRAIGENT_REQUIRE_CLOUD", "TRAIGENT_OFFLINE", "TRAIGENT_OFFLINE_MODE"):
        monkeypatch.delenv(name, raising=False)


def _scorer(prediction: str, expected: str) -> float:
    return 1.0 if prediction == expected else 0.0


def _run(tmp_path: Path, *, with_store: bool = True) -> Any:
    execution: dict[str, Any] = {"offline": True}
    if with_store:
        execution["snapshot_store"] = LocalDirStore(tmp_path / "store")

    @traigent.optimize(
        eval_dataset=Dataset([EvaluationExample({"text": "q"}, "YES")], name="g1"),
        objectives=["accuracy"],
        configuration_space={"temperature": [0.1, 0.2]},
        scoring_function=_scorer,
        execution=execution,
    )
    def agent(text: str) -> Any:
        return with_usage("YES", total_cost=0.0)

    return agent.optimize_sync(max_trials=1)


def _sync_manager(tmp_path: Path) -> SyncManager:
    return SyncManager(
        TraigentConfig(),
        api_key="k",  # pragma: allowlist secret
    )


def test_offline_capture_persists_a_pending_marker_with_the_session(
    tmp_path: Path,
) -> None:
    result = _run(tmp_path)
    storage = LocalStorageManager(str(tmp_path / "results"))
    session = storage.load_session(result.sync_session_id)
    assert session is not None
    assert session.metadata.get(PENDING) is True


def test_offline_run_without_snapshot_store_has_no_marker(tmp_path: Path) -> None:
    result = _run(tmp_path, with_store=False)
    storage = LocalStorageManager(str(tmp_path / "results"))
    session = storage.load_session(result.sync_session_id)
    assert session is not None
    assert PENDING not in session.metadata


def _capture_create(
    manager: SyncManager, monkeypatch: pytest.MonkeyPatch
) -> list[dict[str, Any]]:
    seen: list[dict[str, Any]] = []

    def fake_create(payload: dict[str, Any]) -> dict[str, Any]:
        seen.append(payload)
        return {"success": False, "error": "stop"}

    monkeypatch.setattr(manager, "_sync_create_session", fake_create)
    return seen


def test_sync_registers_with_the_flag_after_readiness(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result = _run(tmp_path)
    fake = FakeG1Backend()
    monkeypatch.setattr(snapshot_binding, "default_transport", lambda: fake)
    manager = _sync_manager(tmp_path)
    seen = _capture_create(manager, monkeypatch)
    manager.sync_session_to_cloud(result.sync_session_id)
    assert [m for m, _, _ in fake.requests] == ["GET"]
    assert len(seen) == 1
    assert seen[0]["snapshot_g1_required"] is True


@pytest.mark.parametrize("readiness", ["unavailable", "absent"])
def test_sync_refuses_closed_when_readiness_is_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, readiness: str
) -> None:
    result = _run(tmp_path)
    fake = FakeG1Backend()
    fake.readiness = readiness
    monkeypatch.setattr(snapshot_binding, "default_transport", lambda: fake)
    manager = _sync_manager(tmp_path)
    seen = _capture_create(manager, monkeypatch)
    outcome = manager.sync_session_to_cloud(result.sync_session_id)
    assert seen == []  # never registered plain
    assert outcome["status"] == "error"
    assert outcome["error_code"] == "READINESS_UNAVAILABLE"


def test_sync_without_marker_never_declares_or_probes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result = _run(tmp_path, with_store=False)

    def _no_probe() -> Any:
        raise AssertionError("no readiness probe without a pending marker")

    monkeypatch.setattr(snapshot_binding, "default_transport", _no_probe)
    manager = _sync_manager(tmp_path)
    seen = _capture_create(manager, monkeypatch)
    manager.sync_session_to_cloud(result.sync_session_id)
    assert len(seen) == 1
    assert "snapshot_g1_required" not in seen[0]
