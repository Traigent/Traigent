"""G1 intent keys cross the optimization-only session-create envelope, closed.

``snapshot_g1_required`` (exactly ``True``) and ``snapshot_g1_deadline_days`` (an integer
1-30, only with the flag) are the whole G1 surface a registration body may add. Under
enforcement the envelope is default-deny, so without these entries a G1 run is refused at
session creation, online and through offline sync.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from tests.egress_canary.harness import capture_egress
from tests.egress_canary.scenario import BACKEND_URL, FAKE_API_KEY
from traigent.cloud.models import SessionCreationRequest
from traigent.utils import optimization_only_egress as oo

pytestmark = pytest.mark.backend_online

FLAG = "snapshot_g1_required"
DAYS = "snapshot_g1_deadline_days"

_PROJECTORS = [
    pytest.param(oo.project_session_create, "configuration_space", id="typed"),
    pytest.param(oo.project_legacy_session_create, "search_space", id="legacy"),
]


@pytest.fixture(autouse=True)
def _clean(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    for name in (oo.ENV_VAR, "TRAIGENT_CONTENT_LESS", "TRAIGENT_SESSION_CONTRACT"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("TRAIGENT_RESULTS_FOLDER", str(tmp_path))
    oo.reset_latch()
    yield
    oo.reset_latch()


@pytest.mark.parametrize(("project", "space_key"), _PROJECTORS)
@pytest.mark.parametrize("days", [1, 7, 30])
def test_flag_and_deadline_project_unchanged(
    project: Any, space_key: str, days: int
) -> None:
    body = {space_key: {"a": [1, 2]}, FLAG: True, DAYS: days}
    out = project(body)
    assert out[FLAG] is True
    assert out[DAYS] == days
    assert type(out[DAYS]) is int


@pytest.mark.parametrize(("project", "space_key"), _PROJECTORS)
def test_flag_alone_projects_unchanged(project: Any, space_key: str) -> None:
    out = project({space_key: {"a": [1]}, FLAG: True})
    assert out[FLAG] is True
    assert DAYS not in out


@pytest.mark.parametrize(("project", "space_key"), _PROJECTORS)
@pytest.mark.parametrize("days", [0, 31, -1, True, False, "7", 7.0, None, [7]])
def test_bad_deadline_is_refused(project: Any, space_key: str, days: Any) -> None:
    with pytest.raises(oo.MetadataContractViolation):
        project({space_key: {"a": [1]}, FLAG: True, DAYS: days})


@pytest.mark.parametrize(("project", "space_key"), _PROJECTORS)
@pytest.mark.parametrize("flag", [False, 1, 0, "true", "True", None, {"x": 1}])
def test_non_true_flag_is_refused(project: Any, space_key: str, flag: Any) -> None:
    with pytest.raises(oo.MetadataContractViolation):
        project({space_key: {"a": [1]}, FLAG: flag})


@pytest.mark.parametrize(("project", "space_key"), _PROJECTORS)
def test_deadline_without_the_flag_is_refused(project: Any, space_key: str) -> None:
    with pytest.raises(oo.MetadataContractViolation):
        project({space_key: {"a": [1]}, DAYS: 7})


@pytest.mark.parametrize(("project", "space_key"), _PROJECTORS)
def test_unrelated_unknown_key_still_refuses_next_to_g1(
    project: Any, space_key: str
) -> None:
    with pytest.raises(oo.MetadataContractViolation):
        project({space_key: {"a": [1]}, FLAG: True, DAYS: 7, "snapshot_g1_extra": "x"})


def _g1_request() -> SessionCreationRequest:
    return SessionCreationRequest(
        function_name="pkg.mod.fn",
        configuration_space={"temperature": [0.1, 0.9]},
        objectives=["accuracy"],
        dataset_metadata={"size": 1},
        max_trials=2,
        snapshot_g1_required=True,
        snapshot_g1_deadline_days=7,
    )


def test_online_create_reaches_the_wire_with_both_fields_under_enforcement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from traigent.cloud.backend_client import (
        BackendClientConfig,
        BackendIntegratedClient,
    )

    monkeypatch.setenv("TRAIGENT_API_KEY", FAKE_API_KEY)
    monkeypatch.setenv("TRAIGENT_BACKEND_URL", BACKEND_URL)
    monkeypatch.setenv("TRAIGENT_OFFLINE_MODE", "false")
    monkeypatch.setenv("TRAIGENT_OFFLINE", "false")
    monkeypatch.setenv(oo.ENV_VAR, "1")
    client = BackendIntegratedClient(
        api_key=FAKE_API_KEY,
        backend_config=BackendClientConfig(backend_base_url=BACKEND_URL),
    )
    payload = client._api_ops._build_session_payload(_g1_request(), max_trials=2)
    assert payload[FLAG] is True
    assert payload[DAYS] == 7
    with capture_egress() as cap:
        asyncio.run(
            client._api_ops._post_session_creation(
                json.loads(json.dumps(payload)),
                {"Content-Type": "application/json"},
                None,
            )
        )
    posts = [
        r for r in cap.requests if r.method == "POST" and r.path.endswith("/sessions")
    ]
    assert len(posts) == 1
    sent = json.loads(posts[0].body)
    assert sent[FLAG] is True
    assert sent[DAYS] == 7


def test_offline_sync_create_reaches_the_wire_with_both_fields_under_enforcement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from traigent.cloud.sync_manager import SyncManager
    from traigent.config.types import TraigentConfig

    monkeypatch.setenv("TRAIGENT_BACKEND_URL", BACKEND_URL)
    monkeypatch.setenv("TRAIGENT_OFFLINE_MODE", "false")
    monkeypatch.setenv("TRAIGENT_OFFLINE", "false")
    monkeypatch.setenv(oo.ENV_VAR, "1")
    manager = SyncManager(TraigentConfig(), api_key=FAKE_API_KEY)
    body = {
        "function_name": "pkg.mod.fn",
        "configuration_space": {"temperature": [0.1, 0.9]},
        "objectives": ["accuracy"],
        "dataset_metadata": {"size": 1},
        "max_trials": 2,
        FLAG: True,
        DAYS: 7,
    }
    with capture_egress() as cap:
        manager._sync_create_session(body)
    posts = [
        r for r in cap.requests if r.method == "POST" and r.path.endswith("/sessions")
    ]
    assert len(posts) == 1
    sent = json.loads(posts[0].body)
    assert sent[FLAG] is True
    assert sent[DAYS] == 7
