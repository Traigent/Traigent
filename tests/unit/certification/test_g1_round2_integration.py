"""Round-2 integration seams for G1 registration (deadline, readiness auth, run id).

Each test drives the real object chain and fakes only the network edge.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import requests

from tests.unit.certification.fakes import SESSIONS
from tests.unit.certification.test_g1_offline_intent import (
    PENDING,
    _run,
    _sync_manager,
)
from tests.unit.certification.test_g1_registration_flag import _create, _manager
from traigent.cloud.auth import AuthManager
from traigent.cloud.backend_client import BackendIntegratedClient
from traigent.cloud.models import session_g1_required_to_wire
from traigent.cloud.sync_manager import SyncManager
from traigent.storage.local_storage import LocalStorageManager

_KEY = "tg_" + "y" * 61  # pragma: allowlist secret
_AMBIENT = (
    "TRAIGENT_API_KEY",
    "OPTIGEN_API_KEY",
    "TRAIGENT_OFFLINE",
    "TRAIGENT_OFFLINE_MODE",
    "TRAIGENT_REQUIRE_RUN_ID",
    "TRAIGENT_REQUIRE_CLOUD",
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("TRAIGENT_RESULTS_FOLDER", str(tmp_path / "results"))
    monkeypatch.setenv("TRAIGENT_COST_APPROVED", "true")
    monkeypatch.setenv("TRAIGENT_RUN_COST_LIMIT", "100.0")
    for name in _AMBIENT:
        monkeypatch.delenv(name, raising=False)


# ---- 1. deadline survives the real manager -> BackendIntegratedClient chain


def test_deadline_reaches_the_wire_through_the_real_client_chain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = BackendIntegratedClient(api_key=_KEY)
    captured: list[Any] = []

    async def fake_via_api(session_request: Any) -> tuple[str, str, str]:
        captured.append(session_request)
        return ("sess-1", "exp-1", "run-1")

    monkeypatch.setattr(client, "_create_traigent_session_via_api", fake_via_api)
    monkeypatch.setattr(client, "_raise_if_backend_egress_disabled", lambda *_: None)
    manager = _manager(client)
    manager.set_snapshot_g1_required(12)
    _create(manager)
    assert len(captured) == 1
    assert session_g1_required_to_wire(captured[0]) == {
        "snapshot_g1_required": True,
        "snapshot_g1_deadline_days": 12,
    }


# ---- 2. offline readiness probe uses the sync caller's explicit key


class _Resp:
    def __init__(self, status: int, body: Any) -> None:
        self.status_code = status
        self._body = body

    def json(self) -> Any:
        return self._body


def test_readiness_probe_uses_the_sync_callers_explicit_api_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def _accept(self: Any, api_key: str) -> None:
        return None

    # The key-validation call is a network edge; the readiness probe is real.
    monkeypatch.setattr(AuthManager, "_validate_api_key_with_backend", _accept)
    result = _run(tmp_path)
    probes: list[dict[str, str]] = []

    def fake_request(method: str, url: str, **kwargs: Any) -> _Resp:
        assert method == "GET" and url.endswith(f"{SESSIONS}/readiness")
        probes.append(dict(kwargs.get("headers") or {}))
        return _Resp(200, {"readiness_version": 1, "g1_issuance": "available"})

    monkeypatch.setattr(requests, "request", fake_request)
    manager = SyncManager(
        _sync_manager(tmp_path).config,
        api_key=_KEY,
    )
    seen: list[dict[str, Any]] = []

    def fake_create(payload: dict[str, Any]) -> dict[str, Any]:
        seen.append(payload)
        return {"success": False, "error": "stop"}

    monkeypatch.setattr(manager, "_sync_create_session", fake_create)
    outcome = manager.sync_session_to_cloud(result.sync_session_id)
    assert outcome.get("error_code") != "READINESS_UNAVAILABLE"
    assert len(probes) == 1
    assert probes[0].get("X-API-Key") == _KEY
    assert len(seen) == 1 and seen[0]["snapshot_g1_required"] is True
    session = LocalStorageManager(str(tmp_path / "results")).load_session(
        result.sync_session_id
    )
    assert session is not None and session.metadata.get(PENDING) is True


# ---- 3. offline G1 registration needs the authoritative run id


@pytest.mark.parametrize("require_env", [None, "false"])
def test_offline_g1_registration_requires_a_run_id_regardless_of_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, require_env: str | None
) -> None:
    if require_env is not None:
        monkeypatch.setenv("TRAIGENT_REQUIRE_RUN_ID", require_env)
    manager = _sync_manager(tmp_path)
    posts: list[dict[str, Any]] = []

    def fake_post(url: str, **kwargs: Any) -> _Resp:
        posts.append(kwargs["json"])
        return _Resp(201, {"session_id": "s-1", "metadata": {}})

    monkeypatch.setattr(manager._session, "post", fake_post)
    g1 = manager._sync_create_session(
        {"function_name": "fn", "snapshot_g1_required": True}
    )
    assert g1["success"] is False
    assert "experiment_run_id" in g1["error"]
    assert posts  # the registration request really went through the real parser
    plain = manager._sync_create_session({"function_name": "fn"})
    assert plain["success"] is True and plain["experiment_run_id"] is None
