"""Content-less compatibility matrix, SDK side (phase 3 group 1).

A content-less SDK against a Backend that does not advertise ``content_less_v1`` refuses
locally with ``backend_upgrade_required``: no plaintext fallback, no legacy-shape retry,
nothing more sent.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from .canaries import CANARIES
from .scenario import run_scenario

pytestmark = pytest.mark.backend_online

IDENTITY_KINDS = (
    "agent_name",
    "evaluation_set",
    "dataset_identity",
    "evaluator_id",
    "dataset_file_name",
)


def _creates(result):
    return [
        r
        for r in result.requests
        if r.method == "POST" and r.path.endswith("/sessions")
    ]


def test_backend_without_capability_gets_one_create_and_nothing_more(
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    result = run_scenario(
        Path(tmp_path_factory.mktemp("cl-old-backend")),
        direct_paths=False,
        content_less=True,
        advertise_content_less=False,
    )
    assert result.optimization_error is not None
    assert "ContentLessRefusal" in result.optimization_error
    assert "upgrade" in result.optimization_error.lower()
    creates = _creates(result)
    assert len(creates) == 1, "no legacy-shape retry after backend_upgrade_required"
    later = [
        r
        for r in result.requests
        if any(v in r.path for v in ("next-trial", "results", "finalize"))
    ]
    assert later == [], "requests followed the local refusal"
    body = creates[0].body.decode("utf-8")
    assert '"content_less_v1"' in body
    for kind in IDENTITY_KINDS:
        assert CANARIES[kind] not in body


def test_capable_backend_run_sends_capability_environment_and_no_identity(
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    result = run_scenario(
        Path(tmp_path_factory.mktemp("cl-capable")),
        direct_paths=False,
        content_less=True,
    )
    assert result.optimization_error is None, result.optimization_error
    creates = _creates(result)
    assert len(creates) == 1
    import json

    body = json.loads(creates[0].body)
    assert body["capabilities"] == ["content_less_v1"]
    assert body["environment"] == "development"  # TRAIGENT_ENVIRONMENT of the scenario
    everything = b"\n".join(r.body or b"" for r in result.requests).decode(
        "utf-8", errors="replace"
    )
    for kind in IDENTITY_KINDS:
        assert CANARIES[kind] not in everything, kind
    # never the agents.id the fake Backend used to return
    assert "agent-canary" not in everything


# ---------------------------------------------------------------- direct client / replay


def _direct_create(tmp_path: Path, *, advertise: bool, env_opt_in: bool = False):
    """Direct ``TraigentCloudClient.create_session`` under capture."""

    import asyncio
    import os

    from traigent.cloud.client import TraigentCloudClient
    from traigent.cloud.content_less import ContentLessOptions
    from traigent.cloud.models import SessionCreationRequest

    from .harness import EgressCapture
    from .scenario import BACKEND_URL, FAKE_API_KEY, scenario_env

    c = CANARIES
    capture = EgressCapture()
    capture.backend.advertise_content_less = advertise
    request = SessionCreationRequest(
        function_name="pkg.mod.fn (abc123)",
        agent_key=c["agent_name"],
        configuration_space={"temperature": [0.1, 0.9]},
        objectives=["accuracy"],
        dataset_metadata={"size": 2, "name": c["evaluation_set"]},
        metadata={
            "evaluation_set": c["evaluation_set"],
            "function_name": "pkg/mod.py_fn",
        },
        dataset_id=c["dataset_identity"],
        evaluator_id=c["evaluator_id"],
        content_less=None if env_opt_in else ContentLessOptions(enabled=True),
    )
    error: BaseException | None = None

    async def run() -> None:
        client = TraigentCloudClient(api_key=FAKE_API_KEY, base_url=BACKEND_URL)
        try:
            await client.create_session(request, {}, [])
        finally:
            await client.close() if hasattr(client, "close") else None

    saved = os.environ.get("TRAIGENT_CONTENT_LESS")
    if env_opt_in:
        os.environ["TRAIGENT_CONTENT_LESS"] = "1"
    try:
        with scenario_env(tmp_path / "r", tmp_path / "d"), capture:
            try:
                asyncio.run(run())
            except BaseException as exc:  # noqa: BLE001
                error = exc
    finally:
        if saved is None:
            os.environ.pop("TRAIGENT_CONTENT_LESS", None)
        else:
            os.environ["TRAIGENT_CONTENT_LESS"] = saved
    return capture, error


@pytest.mark.parametrize("env_opt_in", [False, True])
def test_direct_client_create_is_content_less_when_opted_in(
    tmp_path: Path, env_opt_in: bool
) -> None:
    import json

    capture, error = _direct_create(tmp_path, advertise=True, env_opt_in=env_opt_in)
    assert error is None, repr(error)
    creates = _creates(capture)
    assert len(creates) == 1
    body = json.loads(creates[0].body)
    assert body["capabilities"] == ["content_less_v1"]
    text = creates[0].body.decode()
    for kind in IDENTITY_KINDS:
        assert CANARIES[kind] not in text, kind
    for key in ("agent_key", "dataset_id", "evaluator_id", "user_id", "billing_tier"):
        assert key not in body


def test_direct_client_refuses_a_backend_without_the_capability(
    tmp_path: Path,
) -> None:
    capture, error = _direct_create(tmp_path, advertise=False)
    # the client's error wrapper re-raises the closed refusal text
    assert error is not None and "Backend upgrade required." in str(error), repr(error)
    assert len(_creates(capture)) == 1
    assert [r for r in capture.requests if "/sessions/" in r.path] == []


def test_offline_replay_of_a_content_less_session_is_refused(
    tmp_path: Path,
) -> None:
    from traigent.cloud.content_less import RECORD_KEY, REPLAY_UNSUPPORTED
    from traigent.cloud.sync_manager import SyncManager
    from traigent.config.types import TraigentConfig
    from traigent.storage.local_storage import LocalStorageManager

    from .direct_paths import build_offline_session
    from .harness import EgressCapture
    from .scenario import FAKE_API_KEY, scenario_env

    store = tmp_path / "store"
    plain_id = build_offline_session(str(store))
    storage = LocalStorageManager(str(store))
    marked = storage.load_session(plain_id)
    marked_id = storage.create_session(
        function_name=marked.function_name,
        optimization_config=dict(marked.optimization_config or {}),
        metadata={**dict(marked.metadata or {}), RECORD_KEY: True},
    )
    storage.finalize_session(marked_id)
    capture = EgressCapture()
    with scenario_env(tmp_path / "r", tmp_path / "d"), capture:
        manager = SyncManager(
            TraigentConfig(local_storage_path=str(store)), api_key=FAKE_API_KEY
        )
        refused = manager.sync_session_to_cloud(marked_id, force=True)
        dry = manager.sync_session_to_cloud(marked_id, dry_run=True)
    assert refused["status"] == "error"
    assert refused["error_code"] == REPLAY_UNSUPPORTED
    assert dry["error_code"] == REPLAY_UNSUPPORTED
    assert capture.requests == []


def test_env_opt_in_refuses_replay_of_any_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from traigent.cloud.content_less import REPLAY_UNSUPPORTED
    from traigent.cloud.sync_manager import SyncManager
    from traigent.config.types import TraigentConfig

    from .direct_paths import build_offline_session
    from .harness import EgressCapture
    from .scenario import FAKE_API_KEY, scenario_env

    store = tmp_path / "store"
    sid = build_offline_session(str(store))
    monkeypatch.setenv("TRAIGENT_CONTENT_LESS", "1")
    capture = EgressCapture()
    with scenario_env(tmp_path / "r", tmp_path / "d"), capture:
        manager = SyncManager(
            TraigentConfig(local_storage_path=str(store)), api_key=FAKE_API_KEY
        )
        result = manager.sync_session_to_cloud(sid, force=True)
        monkeypatch.setenv("TRAIGENT_CONTENT_LESS", "1")
    assert result["error_code"] == REPLAY_UNSUPPORTED
    assert capture.requests == []


def test_opted_in_offline_run_persists_the_mode(tmp_path: Path) -> None:
    """An offline optimize(content_less=True) record is marked, so replay refuses it."""
    import os

    from traigent import get_config, optimize
    from traigent.cloud.content_less import RECORD_KEY
    from traigent.storage.local_storage import LocalStorageManager

    results = tmp_path / "offline"
    saved = {
        k: os.environ.get(k) for k in ("TRAIGENT_OFFLINE_MODE", "TRAIGENT_OFFLINE")
    }
    os.environ.update({"TRAIGENT_OFFLINE_MODE": "true", "TRAIGENT_OFFLINE": "true"})
    try:

        @optimize(
            eval_dataset=[{"input": {"q": "a"}, "output": "a"}],
            objectives=["accuracy"],
            configuration_space={"t": [0.1, 0.9]},
            algorithm="grid",
            max_trials=1,
            offline=True,
            content_less=True,
            local_storage_path=str(results),
        )
        def app(example: dict) -> str:
            get_config()
            return "a"

        app.optimize_sync(progress_bar=False)
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    sessions = LocalStorageManager(str(results)).list_sessions()
    assert sessions, "offline run left no local session"
    assert all((s.metadata or {}).get(RECORD_KEY) is True for s in sessions)
