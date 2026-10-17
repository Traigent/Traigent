"""Every best-config load path is bound to the request (sol F1c review, round F1d).

A runtime must never serve a best config approved for another environment or another
function, whichever source it loads from: cloud fetch, cloud cache, repo manifest, an
explicit file (``load_from`` / ``TRAIGENT_CONFIG_PATH``) or the development log.

* P2-1: repo and file loading honour the requested environment.
* P2-2: when the caller names a function, a config with no ``function_ref`` is refused
  (explicitly unbound reads keep accepting legacy unbound configs).
* P3-4: the transport client validates the identity it returns.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from traigent.core.best_config_runtime import (
    BestConfigSource,
    compute_spec_hash,
    function_ref_for,
    load_best_config_spec,
    resolve_cloud_cache_best_config,
    resolve_repo_best_config,
    snapshot_from_spec,
    write_cloud_cache_best_config,
    write_repo_best_config,
)
from traigent.core.config_state_manager import (
    CloudBestConfigIntegrityError,
    ConfigStateManager,
)
from traigent.utils.exceptions import ConfigurationError


def _agent(text: str) -> str:
    return text


_REF = function_ref_for(_agent)


def _spec(**override: Any) -> dict[str, Any]:
    spec: dict[str, Any] = {
        "schema_version": "traigent.best_config.v1",
        "config_id": "agent",
        "function_ref": _REF,
        "environment": "staging",
        "config": {"t": 2},
    }
    spec.update(override)
    return {key: value for key, value in spec.items() if value != "__omit__"}


def _manager(
    *,
    environment: str = "prod",
    source: str = "repo",
    strict: bool = False,
    load_from: str | None = None,
    cache_dir: Path | None = None,
    dev_logs: bool = False,
) -> ConfigStateManager:
    return ConfigStateManager(
        func=_agent,
        default_config={"t": 1},
        local_storage_path=None,
        configuration_space={"t": [1, 2]},
        auto_load_best=dev_logs,
        load_from=load_from,
        setup_wrapper_callback=lambda: None,
        config_id="agent",
        best_config_source=source,
        best_config_strict=strict,
        best_config_cache_dir=str(cache_dir) if cache_dir else None,
        best_config_environment=environment,
        enable_auto_load_dev_logs=dev_logs,
    )


def _write_repo(root: Path, spec: dict[str, Any]) -> None:
    write_repo_best_config(
        root / ".traigent" / "best-configs",
        config_id=spec["config_id"],
        config=spec["config"],
        function_ref=spec.get("function_ref"),
        environment=spec.get("environment"),
    )


def _write_file(path: Path, spec: dict[str, Any]) -> str:
    path.write_text(json.dumps(spec), encoding="utf-8")
    return str(path)


@pytest.fixture(autouse=True)
def _isolated_env(monkeypatch, tmp_path):
    monkeypatch.delenv("TRAIGENT_CONFIG_PATH", raising=False)
    monkeypatch.chdir(tmp_path)


# --- P2-1: repo and file loading honour the requested environment ---------------------


def test_load_best_config_spec_binds_the_requested_environment(tmp_path) -> None:
    path = _write_file(tmp_path / "s.json", _spec())
    with pytest.raises(ConfigurationError, match="environment"):
        load_best_config_spec(path, expected_environment="prod", strict=True)
    ok = load_best_config_spec(path, expected_environment="staging", strict=True)
    assert ok is not None and ok.config["t"] == 2


def test_resolve_repo_best_config_binds_the_requested_environment(tmp_path) -> None:
    _write_repo(tmp_path, _spec())
    kwargs = {
        "config_id": "agent",
        "repo_root": tmp_path,
        "expected_function_ref": _REF,
    }
    with pytest.raises(ConfigurationError, match="environment"):
        resolve_repo_best_config(strict=True, expected_environment="prod", **kwargs)
    assert (
        resolve_repo_best_config(strict=False, expected_environment="prod", **kwargs)
        is None
    )
    ok = resolve_repo_best_config(expected_environment="staging", **kwargs)
    assert ok is not None and ok.config["t"] == 2


def test_repo_source_never_serves_another_environment(tmp_path) -> None:
    _write_repo(tmp_path, _spec())
    strict = _manager(strict=True)
    with pytest.raises(ConfigurationError, match="environment"):
        strict.maybe_auto_load_config()
    loose = _manager(strict=False)
    loose.maybe_auto_load_config()
    snap = loose._best_config_snapshot
    assert snap.source == BestConfigSource.DEFAULT.value
    assert snap.config["t"] == 1


def test_repo_source_serves_the_matching_environment(tmp_path) -> None:
    _write_repo(tmp_path, _spec())
    manager = _manager(environment="staging")
    manager.maybe_auto_load_config()
    assert manager._best_config_snapshot.config["t"] == 2
    assert manager._best_config_snapshot.source == BestConfigSource.REPO.value


@pytest.mark.parametrize(("requested", "loaded"), [("prod", 1), ("default", 2)])
def test_repo_spec_without_environment_is_the_default_environment(
    tmp_path, requested, loaded
) -> None:
    _write_repo(tmp_path, _spec(environment="__omit__"))
    manager = _manager(environment=requested)
    manager.maybe_auto_load_config()
    assert manager._best_config_snapshot.config["t"] == loaded


def test_load_from_never_serves_another_environment(tmp_path) -> None:
    path = _write_file(tmp_path / "staging.json", _spec())
    with pytest.raises(ConfigurationError, match="environment"):
        _manager(source="off", strict=True, load_from=path).maybe_auto_load_config()
    loose = _manager(source="off", strict=False, load_from=path)
    loose.maybe_auto_load_config()
    assert loose._best_config_snapshot.source == BestConfigSource.DEFAULT.value
    ok = _manager(source="off", environment="staging", load_from=path)
    ok.maybe_auto_load_config()
    assert ok._best_config_snapshot.config["t"] == 2


def test_traigent_config_path_never_serves_another_environment(
    monkeypatch, tmp_path
) -> None:
    path = _write_file(tmp_path / "staging.json", _spec())
    monkeypatch.setenv("TRAIGENT_CONFIG_PATH", path)
    with pytest.raises(ConfigurationError, match="environment"):
        _manager(source="off", strict=True).maybe_auto_load_config()
    loose = _manager(source="off", strict=False)
    loose.maybe_auto_load_config()
    assert loose._best_config_snapshot.source == BestConfigSource.DEFAULT.value
    ok = _manager(source="off", environment="staging")
    ok.maybe_auto_load_config()
    assert ok._best_config_snapshot.config["t"] == 2


def test_development_log_never_serves_another_environment(tmp_path) -> None:
    run = (
        tmp_path
        / ".traigent"
        / "optimization_logs"
        / "experiments"
        / "_agent"
        / "runs"
        / "r1"
    )
    (run / "artifacts").mkdir(parents=True)
    _write_file(run / "artifacts" / "best_config_v2.json", _spec())
    loose = _manager(source="off", strict=False, dev_logs=True)
    loose.maybe_auto_load_config()
    assert loose._best_config_snapshot.source == BestConfigSource.DEFAULT.value
    with pytest.raises(ConfigurationError, match="environment"):
        _manager(source="off", strict=True, dev_logs=True).maybe_auto_load_config()


def test_cloud_then_repo_does_not_fall_through_to_another_environment(
    monkeypatch, tmp_path
) -> None:
    """The fetch is refused for a staging answer; the repo must not then serve staging."""
    import traigent.cloud.backend_client as backend_client

    spec = _spec()
    data = {
        "spec": spec,
        "spec_hash": compute_spec_hash(spec),
        "config_hash": "sha256:"
        + hashlib.sha256(
            json.dumps(
                spec["config"],
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest(),
        "etag": "e1",
        "version": 1,
    }

    class Client:
        def fetch_best_config_sync(self, config_id, **kwargs):
            return data

    monkeypatch.setattr(backend_client, "get_backend_client", lambda **kw: Client())
    _write_repo(tmp_path, spec)
    manager = _manager(source="cloud_then_repo", strict=False)
    manager.maybe_auto_load_config()
    assert manager._best_config_snapshot.source == BestConfigSource.DEFAULT.value
    assert manager._best_config_snapshot.config["t"] == 1


# --- P2-2: a missing / null function_ref never satisfies a bound request ----------------


@pytest.mark.parametrize("shape", ["omitted", "null"])
def test_snapshot_from_spec_refuses_an_unbound_config_for_a_bound_request(
    shape,
) -> None:
    spec = _spec(environment="prod")
    if shape == "omitted":
        spec.pop("function_ref")
    else:
        spec["function_ref"] = None
    with pytest.raises(ConfigurationError, match="function_ref"):
        snapshot_from_spec(
            spec, expected_function_ref=_REF, expected_environment="prod"
        )


def test_an_explicitly_unbound_read_keeps_accepting_legacy_unbound_configs() -> None:
    spec = _spec(environment="prod")
    spec.pop("function_ref")
    snap = snapshot_from_spec(spec, expected_environment="prod")
    assert snap.config["t"] == 2


def _fetch_response(spec: dict[str, Any], **top: Any) -> dict[str, Any]:
    data = {
        "spec": spec,
        "spec_hash": compute_spec_hash(spec),
        "config_hash": "sha256:"
        + hashlib.sha256(
            json.dumps(
                spec["config"],
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest(),
        "etag": "e1",
        "version": 1,
    }
    data.update(top)
    return data


def _fetching(monkeypatch, tmp_path, returned) -> ConfigStateManager:
    import traigent.cloud.backend_client as backend_client

    class Client:
        def fetch_best_config_sync(self, config_id, **kwargs):
            return returned

    monkeypatch.setattr(backend_client, "get_backend_client", lambda **kw: Client())
    return _manager(
        source="cloud", environment="prod", cache_dir=tmp_path / "cache", strict=False
    )


@pytest.mark.parametrize("shape", ["omitted", "null"])
@pytest.mark.parametrize("top_level", ["absent", "null"])
def test_fetch_without_function_ref_is_refused_for_a_bound_request(
    monkeypatch, tmp_path, shape, top_level
) -> None:
    spec = _spec(environment="prod")
    if shape == "omitted":
        spec.pop("function_ref")
    else:
        spec["function_ref"] = None
    extra = {"function_ref": None} if top_level == "null" else {}
    manager = _fetching(monkeypatch, tmp_path, _fetch_response(spec, **extra))
    with pytest.raises(CloudBestConfigIntegrityError, match="function_ref"):
        manager._resolve_cloud_fetch_best_config()
    assert not (tmp_path / "cache").exists() or not any((tmp_path / "cache").iterdir())


def test_fetch_with_the_requested_function_ref_still_loads(
    monkeypatch, tmp_path
) -> None:
    manager = _fetching(
        monkeypatch, tmp_path, _fetch_response(_spec(environment="prod"))
    )
    snap = manager._resolve_cloud_fetch_best_config()
    assert snap is not None and snap.config["t"] == 2


@pytest.mark.parametrize("shape", ["omitted", "null"])
def test_cloud_cache_without_function_ref_is_refused_for_a_bound_request(
    tmp_path, shape
) -> None:
    spec = _spec(environment="prod")
    if shape == "omitted":
        spec.pop("function_ref")
    else:
        spec["function_ref"] = None
    write_cloud_cache_best_config(tmp_path, spec, version=1)
    kwargs = {"config_id": "agent", "cache_dir": tmp_path, "strict": True}
    with pytest.raises(ConfigurationError, match="function_ref"):
        resolve_cloud_cache_best_config(
            expected_function_ref=_REF, expected_environment="prod", **kwargs
        )
    # Explicitly unbound read: the legacy unbound entry is still served.
    snap = resolve_cloud_cache_best_config(expected_environment="prod", **kwargs)
    assert snap is not None and snap.config["t"] == 2


def test_repo_without_function_ref_is_refused_for_a_bound_request(tmp_path) -> None:
    _write_repo(tmp_path, _spec(environment="prod", function_ref="__omit__"))
    kwargs = {
        "config_id": "agent",
        "repo_root": tmp_path,
        "expected_environment": "prod",
    }
    with pytest.raises(ConfigurationError, match="function_ref"):
        resolve_repo_best_config(expected_function_ref=_REF, strict=True, **kwargs)
    assert resolve_repo_best_config(expected_function_ref=_REF, **kwargs) is None
    snap = resolve_repo_best_config(**kwargs)  # explicitly unbound
    assert snap is not None and snap.config["t"] == 2


def test_explicit_file_without_function_ref_is_refused_for_a_bound_manager(
    tmp_path,
) -> None:
    path = _write_file(
        tmp_path / "unbound.json", _spec(environment="prod", function_ref="__omit__")
    )
    with pytest.raises(ConfigurationError, match="function_ref"):
        _manager(source="off", strict=True, load_from=path).maybe_auto_load_config()
    loose = _manager(source="off", strict=False, load_from=path)
    loose.maybe_auto_load_config()
    assert loose._best_config_snapshot.source == BestConfigSource.DEFAULT.value


# --- P3-4: the transport client validates the identity it returns -----------------------


def _client():
    from traigent.cloud.backend_client import BackendIntegratedClient

    return BackendIntegratedClient(
        api_key="tg_" + "x" * 61,  # pragma: allowlist secret
        base_url="https://api.test",
    )


def _http(status: int, data: dict[str, Any]) -> MagicMock:
    response = MagicMock(status_code=status, text="ok", headers={})
    response.json.return_value = {"success": True, "data": data}
    return response


def _authed(client):
    from unittest.mock import AsyncMock

    return patch.object(
        client.auth_manager.auth,
        "get_headers",
        AsyncMock(return_value={"Authorization": "Bearer t"}),
    )


@pytest.mark.backend_online
@pytest.mark.parametrize(
    "returned",
    [
        {"config_id": "other", "function_ref": "other:agent", "environment": "staging"},
        {"config_id": "agent", "function_ref": "pkg:agent", "environment": "staging"},
        {"config_id": "agent", "function_ref": "other:agent", "environment": "prod"},
        {"config_id": "other", "function_ref": "pkg:agent", "environment": "prod"},
    ],
    ids=["all", "environment", "function_ref", "config_id"],
)
def test_client_fetch_refuses_a_response_for_another_identity(returned) -> None:
    spec = {**_spec(), **returned}
    client = _client()
    with (
        patch("requests.get", return_value=_http(200, {**returned, "spec": spec})),
        _authed(client),
        pytest.raises(CloudBestConfigIntegrityError),
    ):
        client.fetch_best_config_sync(
            "agent", function_ref="pkg:agent", environment="prod"
        )


@pytest.mark.backend_online
def test_client_fetch_refuses_top_level_identity_without_a_spec() -> None:
    client = _client()
    body = {
        "config_id": "other",
        "function_ref": "other:agent",
        "environment": "staging",
    }
    with (
        patch("requests.get", return_value=_http(200, body)),
        _authed(client),
        pytest.raises(CloudBestConfigIntegrityError),
    ):
        client.fetch_best_config_sync(
            "agent", function_ref="pkg:agent", environment="prod"
        )


@pytest.mark.backend_online
def test_client_fetch_refuses_an_unbound_spec_for_a_bound_request() -> None:
    spec = _spec(function_ref="__omit__", environment="prod")
    client = _client()
    with (
        patch("requests.get", return_value=_http(200, {"spec": spec})),
        _authed(client),
        pytest.raises(CloudBestConfigIntegrityError, match="function_ref"),
    ):
        client.fetch_best_config_sync(
            "agent", function_ref="pkg:agent", environment="prod"
        )


@pytest.mark.backend_online
def test_client_fetch_returns_a_matching_response_unchanged() -> None:
    spec = _spec(function_ref="pkg:agent", environment="prod")
    body = {
        "config_id": "agent",
        "function_ref": "pkg:agent",
        "environment": "PROD ",
        "spec": spec,
    }
    client = _client()
    with patch("requests.get", return_value=_http(200, body)), _authed(client):
        out = client.fetch_best_config_sync(
            "agent", function_ref="pkg:agent", environment="prod"
        )
    assert out == body
    # An explicitly unbound / default-environment request is not over-constrained.
    spec2 = _spec(environment="__omit__")
    with (
        patch("requests.get", return_value=_http(200, {"spec": spec2})),
        _authed(client),
    ):
        assert client.fetch_best_config_sync("agent") == {"spec": spec2}


@pytest.mark.backend_online
def test_client_publish_refuses_a_response_for_another_identity() -> None:
    sent = _spec(function_ref="pkg:agent", environment="prod")
    stored = {**sent, "config_id": "other", "environment": "staging"}
    client = _client()
    with (
        patch(
            "requests.post",
            return_value=_http(
                201,
                {
                    "config_id": "other",
                    "environment": "staging",
                    "function_ref": "pkg:agent",
                    "spec": stored,
                },
            ),
        ),
        _authed(client),
        pytest.raises(CloudBestConfigIntegrityError),
    ):
        client.publish_best_config_sync(sent, environment="prod")


@pytest.mark.backend_online
def test_client_publish_refuses_a_contradictory_top_level_identity() -> None:
    sent = _spec(function_ref="pkg:agent", environment="prod")
    client = _client()
    with (
        patch(
            "requests.post",
            return_value=_http(201, {"environment": "staging", "spec": sent}),
        ),
        _authed(client),
        pytest.raises(CloudBestConfigIntegrityError),
    ):
        client.publish_best_config_sync(sent, environment="prod")


@pytest.mark.backend_online
def test_client_publish_without_a_spec_in_the_response_still_returns_it() -> None:
    sent = _spec(function_ref="pkg:agent", environment="prod")
    client = _client()
    body = {"config_id": "agent", "version": 1}
    with patch("requests.post", return_value=_http(201, body)), _authed(client):
        assert client.publish_best_config_sync(sent, environment="prod") == body


@pytest.mark.backend_online
def test_client_publish_uses_the_envelope_environment_when_the_spec_has_none() -> None:
    sent = _spec(function_ref="pkg:agent", environment="__omit__")
    stored = {**sent, "environment": "prod"}
    client = _client()
    with (
        patch(
            "requests.post",
            return_value=_http(201, {"environment": "prod", "spec": stored}),
        ),
        _authed(client),
    ):
        out = client.publish_best_config_sync(sent, environment="prod")
    assert out["spec"]["environment"] == "prod"
