"""One canary, planted in every customer-controlled input, never reaches any rendering.

Customer-controlled text on the best-config paths: config ids, function refs,
environments, file paths and directories, cache dirs, manifest/spec keys, schema
versions, timestamps, transport exception messages and HTTP error bodies. For every
entry point (load_from, TRAIGENT_CONFIG_PATH, dev-log, repo, cloud cache, cloud fetch,
publish, the raw backend client; strict and non-strict) the canary must not appear in:

* caplog at DEBUG (message, args, exc_text, handler-formatted text),
* ``str`` / ``repr`` / ``args`` of any raised exception,
* ``traceback.format_exception`` of the raised exception (chained causes included),
* ``logger.exception`` output for the raised exception.

Rejected values live on ``exc.details`` only.
"""

from __future__ import annotations

import json
import logging
import traceback
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from collections.abc import Callable
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from traigent.cloud.backend_client import BackendIntegratedClient
from traigent.core.best_config_runtime import (
    canonical_json,
    compute_spec_hash,
    function_ref_for,
    load_best_config_spec,
    resolve_cloud_cache_best_config,
    resolve_repo_best_config,
    sha256_digest,
    source_order_for_mode,
    write_cloud_cache_best_config,
    write_repo_best_config,
)
from traigent.core.config_state_manager import ConfigStateManager
from traigent.utils.exceptions import ConfigurationError

CANARY = "zq-acme-private-canary-91"
CID = f"id-{CANARY}"
FAKE_KEY = "tg_" + "x" * 61  # pragma: allowlist secret


def _agent(text: str) -> str:
    return text


_REF = function_ref_for(_agent)


def _spec(**override: Any) -> dict[str, Any]:
    spec: dict[str, Any] = {
        "schema_version": "traigent.best_config.v1",
        "config_id": CID,
        "function_ref": _REF,
        "environment": "prod",
        "config": {"t": 2},
    }
    spec.update(override)
    return spec


SPEC_VARIANTS: dict[str, Any] = {
    "schema_version": _spec(schema_version=f"/work/{CANARY}"),
    "unknown_top_level_key": _spec(**{f"x_{CANARY}": 1}),
    "unknown_nested_key": _spec(config={"t": 2, f"k_{CANARY}": 1}),
    "safety_nested_key": _spec(config={"t": 2, f"auth_{CANARY}": 1}),
    "function_ref": _spec(function_ref=f"mod:{CANARY}"),
    "config_id": _spec(config_id=f"other-{CANARY}"),
    "environment": _spec(environment=f"env-{CANARY}"),
    "legacy_function_ref": {"function_ref": f"mod:{CANARY}", "config": {"t": 2}},
    "legacy_config_id": {"config_id": f"id-{CANARY}x", "config": {"t": 2}},
    "legacy_unknown_key": {"t": 2, f"k_{CANARY}": 1},
    "missing_fields": _spec(config=None, **{f"x_{CANARY}": 1}),
    "provenance_loaded_at": _spec(provenance={"loaded_at": CANARY}),
}


@pytest.fixture(autouse=True)
def _env(monkeypatch, tmp_path):
    monkeypatch.delenv("TRAIGENT_CONFIG_PATH", raising=False)
    monkeypatch.setenv("TRAIGENT_RESULTS_FOLDER", str(tmp_path / "results"))
    monkeypatch.chdir(tmp_path)


def _manager(**kw: Any) -> ConfigStateManager:
    return ConfigStateManager(
        func=_agent,
        default_config={"t": 1},
        local_storage_path=None,
        configuration_space={"t": [1, 2]},
        auto_load_best=False,
        load_from=kw.get("load_from"),
        setup_wrapper_callback=lambda: None,
        config_id=kw.get("config_id", CID),
        best_config_source=kw.get("source", "off"),
        best_config_strict=kw.get("strict", False),
        best_config_environment="prod",
        best_config_cache_dir=kw.get("cache_dir"),
        enable_auto_load_dev_logs=kw.get("dev_logs", False),
    )


def _renderings(exc: BaseException) -> list[str]:
    out = [str(exc), repr(exc), "".join(map(str, exc.args))]
    out.append("".join(traceback.format_exception(exc)))
    # ``logger.exception`` as a caller would write it.
    stream = _Capture()
    probe = logging.getLogger("leak-class-probe")
    probe.addHandler(stream)
    probe.propagate = False
    try:
        probe.exception("caught", exc_info=exc)
    finally:
        probe.removeHandler(stream)
    out.append(stream.text)
    return out


class _Capture(logging.Handler):
    def __init__(self) -> None:
        super().__init__(logging.DEBUG)
        self.text = ""
        self.setFormatter(logging.Formatter("%(message)s"))

    def emit(self, record: logging.LogRecord) -> None:
        self.text += self.format(record)


def _check(caplog, raised: list[BaseException]) -> None:
    for rec in caplog.records:
        formatted = logging.Formatter("%(message)s").format(rec)
        assert CANARY not in formatted, f"log leak: {formatted!r}"
        assert CANARY not in str(rec.args), f"log args leak: {rec.args!r}"
        assert CANARY not in (rec.exc_text or "")
    assert CANARY not in caplog.text
    for exc in raised:
        for text in _renderings(exc):
            assert CANARY not in text, f"exception leak: {text[:400]!r}"


# --------------------------------------------------------------------- scenarios
Scenario = Callable[[Path, Any, bool], Any]


def _write(path: Path, data: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        data if isinstance(data, str) else json.dumps(data), encoding="utf-8"
    )
    return path


def _scn_manager_file(entry: str, payload: Any) -> Scenario:
    def run(tmp_path, monkeypatch, strict):
        if entry == "dev-log":
            path = _write(
                tmp_path
                / ".traigent/optimization_logs/experiments/_agent/runs/r1/artifacts"
                / "best_config_v2.json",
                payload,
            )
        else:
            path = _write(tmp_path / f"dir-{CANARY}" / f"f-{CANARY}.json", payload)
        if entry == "load_from":
            manager = _manager(load_from=str(path), strict=strict)
        elif entry == "env":
            monkeypatch.setenv("TRAIGENT_CONFIG_PATH", str(path))
            manager = _manager(strict=strict)
        else:
            manager = _manager(dev_logs=True, strict=strict)
        manager.maybe_auto_load_config()

    return run


def _scn_manager_missing(entry: str) -> Scenario:
    def run(tmp_path, monkeypatch, strict):
        path = str(tmp_path / f"missing-{CANARY}" / "c.json")
        if entry == "load_from":
            manager = _manager(load_from=path, strict=strict)
        else:
            monkeypatch.setenv("TRAIGENT_CONFIG_PATH", path)
            manager = _manager(strict=strict)
        manager.maybe_auto_load_config()

    return run


def _scn_load_spec(payload: Any) -> Scenario:
    def run(tmp_path, monkeypatch, strict):
        path = _write(tmp_path / f"f-{CANARY}.json", payload)
        load_best_config_spec(
            path,
            configuration_space={"t": [1, 2]},
            expected_config_id=CID,
            expected_function_ref=_REF,
            expected_environment="prod",
            strict=strict,
        )

    return run


def _scn_load_missing(tmp_path, monkeypatch, strict):
    load_best_config_spec(tmp_path / f"missing-{CANARY}" / "c.json", strict=strict)


def _scn_repo(tamper: str) -> Scenario:
    def run(tmp_path, monkeypatch, strict):
        root = tmp_path / ".traigent" / "best-configs"
        write_repo_best_config(
            root, config_id=CID, config={"t": 2}, function_ref=_REF, environment="prod"
        )
        manifest_path = root / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        if tamper == "entry_path":
            manifest["configs"][CID]["path"] = f"../{CANARY}.json"
        elif tamper == "entry_missing_file":
            manifest["configs"][CID]["path"] = f"gone-{CANARY}.json"
        elif tamper == "hash":
            manifest["configs"][CID]["spec_hash"] = f"sha256:{CANARY}"
        elif tamper == "manifest_version":
            manifest["schema_version"] = f"v-{CANARY}"
        elif tamper == "manifest_key":
            manifest[f"k_{CANARY}"] = 1
            manifest["configs"][f"k_{CANARY}"] = "x"
        if tamper == "manifest_json":
            manifest_path.write_text('{"a": "' + CANARY, encoding="utf-8")
        else:
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        resolve_repo_best_config(
            config_id=CID,
            repo_root=tmp_path,
            configuration_space={"t": [1, 2]},
            expected_function_ref=_REF,
            expected_environment="prod",
            strict=strict,
        )

    return run


def _scn_repo_via_manager(tmp_path, monkeypatch, strict):
    root = tmp_path / ".traigent" / "best-configs"
    write_repo_best_config(
        root, config_id=CID, config={"t": 2}, function_ref=_REF, environment="prod"
    )
    spec_file = root / f"{CID}.json"
    spec = json.loads(spec_file.read_text())
    spec["config"] = {"t": 2, f"k_{CANARY}": 1}
    spec_file.write_text(json.dumps(spec), encoding="utf-8")
    _manager(source="repo", strict=strict)._resolve_mode_snapshot()


def _scn_cache(tamper: str) -> Scenario:
    def run(tmp_path, monkeypatch, strict):
        cache = tmp_path / f"cache-{CANARY}"
        loaded = datetime.now(UTC)
        if tamper == "stale":
            loaded = loaded - timedelta(days=30)
        write_cloud_cache_best_config(cache, _spec(), etag="e1", loaded_at=loaded)
        manifest_path = cache / "manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            entry = manifest["configs"][CID]
        else:
            manifest, entry = None, None
        meta_path = cache / f"{CID}.metadata.json"
        if entry is None and meta_path.exists():
            entry = json.loads(meta_path.read_text())
        if tamper == "loaded_at":
            entry["loaded_at"] = CANARY
        elif tamper == "missing_spec":
            (cache / f"{CID}.json").unlink()
        elif tamper == "entry_path":
            entry["path"] = f"../{CANARY}.json"
        elif tamper == "no_etag":
            entry.pop("etag", None)
            entry.pop("version", None)
        elif tamper == "spec_json":
            (cache / f"{CID}.json").write_text('{"a": "' + CANARY, encoding="utf-8")
        if manifest is not None:
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        else:
            meta_path.write_text(json.dumps(entry), encoding="utf-8")
        resolve_cloud_cache_best_config(
            config_id=CID,
            cache_dir=cache,
            configuration_space={"t": [1, 2]},
            strict=strict,
            expected_function_ref=_REF,
            expected_environment="prod",
        )

    return run


def _data_for(spec: dict[str, Any]) -> dict[str, Any]:
    return {
        "spec": spec,
        "spec_hash": compute_spec_hash(spec),
        "config_hash": sha256_digest(canonical_json(spec["config"])),
        "etag": "e1",
        "version": 1,
    }


class _FakeClient:
    def __init__(self, fetch: Any = None, publish: Any = None) -> None:
        self._fetch, self._publish = fetch, publish

    @staticmethod
    def _go(what: Any) -> Any:
        if isinstance(what, BaseException):
            raise what
        return what

    def fetch_best_config_sync(self, *a: Any, **k: Any) -> Any:
        return self._go(self._fetch)

    def publish_best_config_sync(self, *a: Any, **k: Any) -> Any:
        return self._go(self._publish)


def _scn_fetch(kind: str, mode: str = "cloud") -> Scenario:
    def run(tmp_path, monkeypatch, strict):
        cache_dir: str | None = str(tmp_path / f"cache-{CANARY}")
        fetch: Any
        if kind == "transport":
            fetch = ConnectionError(f"failed URL /work/{CANARY}")
        elif kind == "runtime":
            fetch = RuntimeError(CANARY)
        elif kind in ("config_error", "client_config_error"):
            fetch = ConfigurationError(f"Invalid config {CANARY}")
        elif kind == "bad_hash":
            data = _data_for(_spec())
            data["spec_hash"] = f"sha256:{CANARY}"
            fetch = data
        elif kind == "other_id":
            fetch = _data_for(_spec(config_id=f"other-{CANARY}"))
        elif kind == "other_ref":
            fetch = _data_for(_spec(function_ref=f"mod:{CANARY}"))
        elif kind == "other_env":
            fetch = _data_for(_spec(environment=f"env-{CANARY}"))
        elif kind == "spec_keys":
            fetch = _data_for(_spec(config={"t": 2, f"k_{CANARY}": 1}))
        elif kind == "cache_unwritable":
            fetch = _data_for(_spec())
            cache_dir = f"/proc/{CANARY}"
        elif kind == "cache_is_file":
            fetch = _data_for(_spec())
            blocker = tmp_path / f"blocker-{CANARY}"
            blocker.write_text("x", encoding="utf-8")
            cache_dir = str(blocker / "sub")
        else:  # pragma: no cover
            raise AssertionError(kind)
        manager = _manager(source=mode, strict=strict, cache_dir=cache_dir)
        if kind == "client_config_error":
            target = patch(
                "traigent.cloud.backend_client.get_backend_client",
                side_effect=fetch,
            )
        else:
            target = patch(
                "traigent.cloud.backend_client.get_backend_client",
                return_value=_FakeClient(fetch=fetch),
            )
        with target:
            manager._resolve_mode_snapshot()

    return run


def _scn_publish(kind: str) -> Scenario:
    def run(tmp_path, monkeypatch, strict):
        manager = _manager(strict=strict)
        manager.set_config({"t": 2})
        publish: Any
        if kind == "transport":
            publish = ConnectionError(f"failed URL /work/{CANARY}")
        elif kind == "runtime":
            publish = RuntimeError(CANARY)
        elif kind == "other_id":
            publish = _data_for(_spec(config_id=f"other-{CANARY}"))
        elif kind == "bad_hash":
            data = _data_for(_spec())
            data["spec_hash"] = f"sha256:{CANARY}"
            publish = data
        else:  # pragma: no cover
            raise AssertionError(kind)
        with patch(
            "traigent.cloud.backend_client.get_backend_client",
            return_value=_FakeClient(fetch=None, publish=publish),
        ):
            manager.publish_best_config()

    return run


def _scn_source_mode(tmp_path, monkeypatch, strict):
    source_order_for_mode(f"x-{CANARY}")


def _response(status: int, text: str) -> MagicMock:
    response = MagicMock()
    response.status_code = status
    response.text = text
    response.headers = {}
    response.json.return_value = {"success": False, "error": text}
    return response


def _scn_client(op: str, kind: str) -> Scenario:
    def run(tmp_path, monkeypatch, strict):
        client = BackendIntegratedClient(api_key=FAKE_KEY, base_url="https://api.test")
        verb = "requests.get" if op == "fetch" else "requests.post"
        if kind == "transport":
            side: dict[str, Any] = {
                "side_effect": ConnectionError(f"failed URL /work/{CANARY}")
            }
        elif kind == "http_500":
            side = {"return_value": _response(500, f"boom {CANARY}")}
        else:
            side = {"return_value": _response(400, f"bad {CANARY}")}
        with (
            patch(verb, **side),
            patch("traigent.utils.retry.time.sleep", return_value=None),
            patch.object(
                client.auth_manager.auth,
                "get_headers",
                AsyncMock(return_value={"Authorization": "Bearer t"}),
            ),
        ):
            if op == "fetch":
                client.fetch_best_config_sync(
                    CID, environment=f"env-{CANARY}", function_ref=f"mod:{CANARY}"
                )
            else:
                client.publish_best_config_sync(
                    _spec(), environment=f"env-{CANARY}", if_match=f'W/"{CANARY}"'
                )

    return run


_AGENT_REF = "agentref_" + "a" * 32


def _client() -> BackendIntegratedClient:
    return BackendIntegratedClient(api_key=FAKE_KEY, base_url="https://api.test")


def _auth(client: BackendIntegratedClient, **kw: Any):
    return patch.object(client.auth_manager.auth, "get_headers", AsyncMock(**kw))


def _scn_client_invalid_header(op: str, field: str) -> Scenario:
    """A real ``requests`` header validation failure (``InvalidHeader``); nothing is sent."""

    def run(tmp_path, monkeypatch, strict):
        import requests

        client = _client()
        bad = f"\n{CANARY}"
        with (
            _auth(client, return_value={"Authorization": "Bearer t"}),
            patch.object(
                requests.adapters.HTTPAdapter,
                "send",
                side_effect=AssertionError("must not send"),
            ),
        ):
            if op == "fetch":
                client.fetch_best_config_sync(CID, etag=bad)
            else:
                client.publish_best_config_sync(_spec(), if_match=bad)

    return run


def _scn_client_protected(op: str) -> Scenario:
    def run(tmp_path, monkeypatch, strict):
        import requests

        client = _client()
        with (
            _auth(client, return_value={"Authorization": "Bearer t"}),
            patch(
                "traigent.cloud.backend_client.gate_session",
                lambda sess, **kw: sess,
            ),
            patch.object(
                requests.Session,
                "request",
                side_effect=requests.exceptions.ConnectionError(f"/work/{CANARY}"),
            ),
        ):
            if op == "publish":
                client.publish_best_config_sync(
                    _spec(),
                    environment="prod",
                    agent_ref=_AGENT_REF,
                    expected_head_generation=1,
                )
            else:
                client.fetch_agent_head_generation_by_ref_sync(
                    _AGENT_REF, environment="production"
                )

    return run


def _scn_client_auth_failure(tmp_path, monkeypatch, strict):
    client = _client()
    with _auth(client, side_effect=RuntimeError(f"refresh URL /work/{CANARY}")):
        client.fetch_best_config_sync(CID)


def _long(tmp_path: Path) -> Path:
    return tmp_path / (CANARY + "x" * 300)


def _scn_repo_overlong(tmp_path, monkeypatch, strict):
    resolve_repo_best_config(config_id=CID, repo_root=_long(tmp_path), strict=strict)


def _scn_cache_overlong(tmp_path, monkeypatch, strict):
    resolve_cloud_cache_best_config(
        config_id=CID, cache_dir=_long(tmp_path), strict=strict
    )


def _scn_manager_overlong_cache(tmp_path, monkeypatch, strict):
    manager = _manager(source="cache", strict=strict, cache_dir=str(_long(tmp_path)))
    manager._resolve_mode_snapshot()


def _scn_whole_config(tmp_path, monkeypatch, strict):
    manager = _manager()
    manager.set_config({"t": 2, "prompt": CANARY})
    results = MagicMock()
    results.best_config = {"t": 2, "prompt": CANARY}
    results.metadata = {}
    manager.apply_best_config(results)


def _scn_load_results_missing(tmp_path, monkeypatch, strict):
    _manager().load_optimization_results(str(tmp_path / f"missing-{CANARY}.json"))


def _scn_load_results_bad_json(tmp_path, monkeypatch, strict):
    target = _write(tmp_path / f"r-{CANARY}.json", '{"a": "' + CANARY)
    _manager().load_optimization_results(str(target))


def _scn_load_results_ok_log(tmp_path, monkeypatch, strict):
    # a valid file whose name carries the canary: the success log must not name it
    manager = _manager()
    src = tmp_path / f"saved-{CANARY}.json"
    manager.set_config({"t": 2})
    manager._optimization_results = _real_results()
    saved = _real_results()
    manager._optimization_results = saved
    manager.save_optimization_results(str(src))
    assert src.exists(), "the save must succeed so the load below is reached"
    manager._optimization_results = None
    manager.load_optimization_results(str(src))
    loaded = manager._optimization_results
    assert loaded is not None, "the load must produce a result"
    assert loaded == saved, "the loaded object must equal the saved one"


def _real_results():
    from traigent.api.types import OptimizationResult, OptimizationStatus

    return OptimizationResult(
        trials=[],
        best_config={"t": 2},
        best_score=1.0,
        optimization_id="opt-1",
        duration=1.0,
        convergence_info={},
        status=OptimizationStatus.COMPLETED,
        objectives=["accuracy"],
        algorithm="grid",
        timestamp=datetime.now(UTC),
    )


def _scn_save_overlong_path(tmp_path, monkeypatch, strict):
    manager = _manager()
    manager._optimization_results = _real_results()
    manager.save_optimization_results(str(_long(tmp_path) / "r.json"))


def _scn_save_overlong_name(tmp_path, monkeypatch, strict):
    manager = _manager()
    manager._optimization_results = _real_results()
    manager.save_optimization_results(str(tmp_path / (CANARY + "y" * 300 + ".json")))


def _scn_save_bad_timestamp(tmp_path, monkeypatch, strict):
    manager = _manager()
    results = _real_results()
    results.timestamp = f"ts-{CANARY}"
    manager._optimization_results = results
    manager.save_optimization_results(str(tmp_path / "r.json"))


def _scn_save_bad_status(tmp_path, monkeypatch, strict):
    manager = _manager()
    results = _real_results()
    results.status = f"status-{CANARY}"
    manager._optimization_results = results
    manager.save_optimization_results(str(tmp_path / "r.json"))


def _scn_client_overlong_local_storage(tmp_path, monkeypatch, strict):
    BackendIntegratedClient(
        api_key=FAKE_KEY,
        base_url="https://api.test",
        enable_fallback=True,
        local_storage_path=str(_long(tmp_path)),
    )


def _scn_client_local_storage_dir_logged(tmp_path, monkeypatch, strict):
    BackendIntegratedClient(
        api_key=FAKE_KEY,
        base_url="https://api.test",
        enable_fallback=True,
        local_storage_path=str(tmp_path / f"store-{CANARY}"),
    )


def _scn_client_submit_result_storage_failure(tmp_path, monkeypatch, strict):
    client = BackendIntegratedClient(
        api_key=FAKE_KEY,
        base_url="https://api.test",
        enable_fallback=True,
        local_storage_path=str(tmp_path / "store"),
    )
    assert client.local_storage is not None
    err = OSError(36, f"File name too long: '/work/{CANARY}'")
    with patch.object(client.local_storage, "add_trial_result", side_effect=err):
        try:
            client.submit_result("opaque-session", {"t": 2}, 1.0)
        finally:
            assert client.local_storage.add_trial_result.call_count == 1


def _scn_client_async_auth_failure(tmp_path, monkeypatch, strict):
    import asyncio

    client = _client()
    with _auth(client, side_effect=RuntimeError(f"refresh URL /work/{CANARY}")):
        asyncio.run(client._build_http_session())


def _scn_load_results_unknown_status(tmp_path, monkeypatch, strict):
    manager = _manager()
    src = tmp_path / "saved.json"
    manager.set_config({"t": 2})
    manager._optimization_results = _real_results()
    manager.save_optimization_results(str(src))
    data = json.loads(src.read_text())
    data["status"] = f"status-{CANARY}"
    src.write_text(json.dumps(data))
    manager.load_optimization_results(str(src))


def _scn_load_results_legacy(tmp_path, monkeypatch, strict):
    # an unversioned (legacy) artifact named with the canary decodes with a warning
    from traigent.utils.optimization_result_persistence import SCHEMA_VERSION_KEY

    manager = _manager()
    src = tmp_path / f"legacy-{CANARY}.json"
    manager.set_config({"t": 2})
    manager._optimization_results = _real_results()
    manager.save_optimization_results(str(src))
    data = json.loads(src.read_text())
    data.pop(SCHEMA_VERSION_KEY, None)
    for field in ("source", "warnings", "stop_reason", "total_cost", "metadata"):
        data.pop(field, None)
    src.write_text(json.dumps(data))
    manager.load_optimization_results(str(src))


def _scn_client_protected_invalid_header(tmp_path, monkeypatch, strict):
    import requests

    client = _client()
    with (
        _auth(client, return_value={"Authorization": "Bearer t"}),
        patch("traigent.cloud.backend_client.gate_session", lambda sess, **kw: sess),
        patch.object(
            requests.adapters.HTTPAdapter,
            "send",
            side_effect=AssertionError("must not send"),
        ),
    ):
        client.publish_best_config_sync(
            _spec(),
            environment="prod",
            agent_ref=_AGENT_REF,
            expected_head_generation=0,
            if_match="\n" + CANARY,
        )


def _table_missing():
    raise FileNotFoundError(f"/work/{CANARY}/table.json")


def _scn_gate_refusal_chain(tmp_path, monkeypatch, strict):
    from traigent.labels import transport_gate as tg

    with (
        patch.object(tg, "current_mode", return_value="refuse"),
        patch.object(tg, "applies_to", return_value=True),
        patch.object(tg, "get_table", _table_missing),
    ):
        tg.inspect_outgoing("GET", "https://api.test/x", None)


def _scn_client_gate_refusal_chain(tmp_path, monkeypatch, strict):
    import requests
    from traigent.labels import transport_gate as tg

    client = _client()
    with (
        _auth(client, return_value={"Authorization": "Bearer t"}),
        patch.object(tg, "current_mode", return_value="refuse"),
        patch.object(tg, "applies_to", return_value=True),
        patch.object(tg, "get_table", _table_missing),
        patch.object(
            requests.adapters.HTTPAdapter,
            "send",
            side_effect=AssertionError("must not send"),
        ),
    ):
        client.fetch_agent_head_generation_by_ref_sync(
            _AGENT_REF, environment="production"
        )


def _scn_apply_best_config_key(tmp_path, monkeypatch, strict):
    manager = _manager()
    manager.set_config({"t": 2})
    results = MagicMock()
    results.best_config = {"t": 2, f"k_{CANARY}": 1}
    results.metadata = {}
    manager.apply_best_config(results)


def _long_dir(tmp_path: Path) -> Path:
    return tmp_path / (CANARY + "x" * 300)


def _scn_write_repo_overlong(tmp_path, monkeypatch, strict):
    write_repo_best_config(_long_dir(tmp_path), config_id=CID, config={"t": 2})


def _scn_write_cache_overlong(tmp_path, monkeypatch, strict):
    write_cloud_cache_best_config(
        _long_dir(tmp_path), _spec(), etag="e", loaded_at=datetime.now(UTC)
    )


def _scn_load_manifest_overlong(tmp_path, monkeypatch, strict):
    from traigent.core.best_config_runtime import load_manifest

    load_manifest(_long_dir(tmp_path))


def _scn_export_overlong(tmp_path, monkeypatch, strict):
    manager = _manager()
    manager.set_config({"t": 2})
    manager.export_best_config(str(_long_dir(tmp_path)))


def _scn_write_manifest_is_dir(tmp_path, monkeypatch, strict):
    target = tmp_path / f"d-{CANARY}"
    (target / "manifest.json").mkdir(parents=True)
    write_repo_best_config(target, config_id=CID, config={"t": 2})


def _scn_save_traversal(tmp_path, monkeypatch, strict):
    monkeypatch.chdir(tmp_path)
    manager = _manager()
    manager._optimization_results = _real_results()
    manager.save_optimization_results(f"../{CANARY}/escape.json")


def _scn_save_base_is_file(tmp_path, monkeypatch, strict):
    blocker = tmp_path / f"file-{CANARY}"
    blocker.write_text("x")
    manager = _manager()
    manager._optimization_results = _real_results()
    manager.save_optimization_results(str(blocker / "r.json"))


class _RaisingDeepcopy:
    def __deepcopy__(self, memo):
        raise RuntimeError(f"deepcopy of /work/{CANARY}")


class _RaisingStr:
    def __deepcopy__(self, memo):
        return self

    def __str__(self) -> str:
        raise RuntimeError(f"str of /work/{CANARY}")


def _scn_save_deepcopy_raises(tmp_path, monkeypatch, strict):
    manager = _manager()
    results = _real_results()
    results.best_config = {"t": _RaisingDeepcopy()}
    manager._optimization_results = results
    manager.save_optimization_results(str(tmp_path / "r.json"))


def _scn_save_str_raises(tmp_path, monkeypatch, strict):
    manager = _manager()
    results = _real_results()
    results.best_config = {"t": _RaisingStr()}
    manager._optimization_results = results
    manager.save_optimization_results(str(tmp_path / "r.json"))


def _scn_client_submit_result_load_session_permission(tmp_path, monkeypatch, strict):
    client = BackendIntegratedClient(
        api_key=FAKE_KEY,
        base_url="https://api.test",
        enable_fallback=True,
        local_storage_path=str(tmp_path / "store"),
    )
    assert client.local_storage is not None
    sessions = tmp_path / "store" / "sessions"
    sessions.mkdir(parents=True, exist_ok=True)
    (sessions / "opaque-session.json").write_text("{}")
    err = PermissionError(
        13, "Permission denied", f"/work/{CANARY}/sessions/opaque-session.json"
    )
    with patch("traigent.storage.local_storage.safe_open", side_effect=err):
        client.submit_result("opaque-session", {"t": 2}, 1.0)


# ------------------------------------------------------- F1m (terra F1k/F1l + sol)
def _store(tmp_path):
    from traigent.storage.local_storage import LocalStorageManager

    return LocalStorageManager(str(tmp_path / "f1m-store"))


def _scn_store_banded_objective_name(tmp_path, monkeypatch, strict):
    # terra P1: the objective name is stored caller config; the band refusal quoted it
    store = _store(tmp_path)
    sid = store.create_session(
        "fn",
        optimization_config={
            "objective_schema": {
                "objectives": [{"name": f"obj-{CANARY}", "orientation": "band"}]
            }
        },
    )
    store.add_trial_result(sid, {"t": 1}, 0.5)
    store.add_trial_result(sid, {"t": 2}, 0.6)


def _scn_store_status_logs(tmp_path, monkeypatch, strict):
    # terra P1: a caller-supplied status string was logged verbatim
    store = _store(tmp_path)
    sid = store.create_session("fn")
    store.update_session_status(sid, f"status-{CANARY}")
    store.finalize_session(sid, f"status-{CANARY}")


def _scn_store_session_id_not_found(tmp_path, monkeypatch, strict):
    store = _store(tmp_path)
    sid = f"sid-{CANARY}"
    store.finalize_session(sid)
    store.update_sync_state(sid, {"a": 1})
    store.delete_session(sid)
    store.add_trial_result(sid, {"t": 1}, 1.0)


def _scn_store_session_id_load_failure(tmp_path, monkeypatch, strict):
    store = _store(tmp_path)
    sid = f"sid-{CANARY}"
    (tmp_path / "f1m-store" / "sessions" / f"{sid}.json").write_text("{not json")
    store.load_session(sid)
    store.export_session(sid, str(tmp_path / "out.json"))


def _scn_store_session_id_created_and_saved(tmp_path, monkeypatch, strict):
    store = _store(tmp_path)
    sid = store.create_session("fn", session_id=f"sid-{CANARY}")
    store.add_trial_result(sid, {"t": 1}, 1.0)
    store.export_session(sid, str(tmp_path / "out.json"))
    store.finalize_session(sid)
    store.delete_session(sid)


def _scn_store_save_failure_session_id(tmp_path, monkeypatch, strict):
    store = _store(tmp_path)
    sid = store.create_session("fn", session_id=f"sid-{CANARY}")
    with patch("traigent.storage.local_storage.safe_open", side_effect=OSError("x")):
        store.update_session_status(sid, "running")


def _scn_store_export_format(tmp_path, monkeypatch, strict):
    store = _store(tmp_path)
    sid = store.create_session("fn")
    store.export_session(sid, str(tmp_path / "o.json"), format=f"fmt-{CANARY}")


def _scn_manager_captured_environment(tmp_path, monkeypatch, strict):
    # terra P2: a captured head's environment reached a debug log
    ConfigStateManager._promotion_precondition_kwargs(
        {"agent_id": "a1", "generation": 3, "environment": f"env-{CANARY}"},
        "prod",
    )
    ConfigStateManager._promotion_precondition_kwargs(
        {"agent_id": "a1", "generation": 3, "environment": "prod"},
        f"env-{CANARY}",
    )


class _StrRaisesConfigurationError:
    def __deepcopy__(self, memo):
        return self

    def __str__(self) -> str:
        raise ConfigurationError(f"value /work/{CANARY}")


def _scn_save_str_raises_configuration_error(tmp_path, monkeypatch, strict):
    # sol P3: a value's own __str__ raising ConfigurationError(value) escaped unchanged
    manager = _manager()
    results = _real_results()
    results.best_config = {"t": _StrRaisesConfigurationError()}
    manager._optimization_results = results
    manager.save_optimization_results(str(tmp_path / "r.json"))


def _scn_persistence_encode_timestamp_value(tmp_path, monkeypatch, strict):
    # sol P2: value-bearing ValueErrors outside the save boundary
    from traigent.utils.optimization_result_persistence import encode_result_fields

    results = _real_results()
    results.timestamp = f"ts-{CANARY}"
    encode_result_fields(results)


def _scn_persistence_encode_status_value(tmp_path, monkeypatch, strict):
    from traigent.utils.optimization_result_persistence import encode_result_fields

    results = _real_results()
    results.status = f"status-{CANARY}"
    encode_result_fields(results)


def _scn_persistence_decode_timestamp_value(tmp_path, monkeypatch, strict):
    from traigent.utils.optimization_result_persistence import _decode_timestamp

    _decode_timestamp({"k": CANARY})


def _scn_persistence_manager_save_foreign_result(tmp_path, monkeypatch, strict):
    from traigent.utils.persistence import PersistenceManager

    real = _real_results()

    class Foreign:
        """Duck-typed result (not an OptimizationResult) with a bad timestamp."""

        timestamp = f"ts-{CANARY}"

        def __getattr__(self, name):
            return getattr(real, name)

    PersistenceManager(str(tmp_path / "pm")).save_result(Foreign(), "named")


def _scn_client_submit_result_session_id(tmp_path, monkeypatch, strict):
    client = BackendIntegratedClient(
        api_key=FAKE_KEY,
        base_url="https://api.test",
        enable_fallback=True,
        local_storage_path=str(tmp_path / "store"),
    )
    client.submit_result(f"sid-{CANARY}", {"t": 2}, 1.0)


def _scn_manager_export_config_format(tmp_path, monkeypatch, strict):
    # F1m audit: the caller-supplied format token was echoed in the error text
    manager = _manager()
    manager.set_config({"t": 2})
    manager.export_config(tmp_path / "c.json", format=f"fmt-{CANARY}")


def _encoded_result_fields():
    from traigent.utils.optimization_result_persistence import (
        SCHEMA_VERSION_KEY,
        encode_result_fields,
    )

    fields = encode_result_fields(_real_results())
    fields[SCHEMA_VERSION_KEY] = 2
    return fields


def _scn_persistence_decode_version_value(tmp_path, monkeypatch, strict):
    from traigent.utils.optimization_result_persistence import (
        SCHEMA_VERSION_KEY,
        decode_result,
    )

    fields = _encoded_result_fields()
    fields[SCHEMA_VERSION_KEY] = f"v-{CANARY}"
    decode_result(fields, trials=[], artifact_name="named")


def _scn_persistence_envelope_version_value(tmp_path, monkeypatch, strict):
    from traigent.utils.optimization_result_persistence import (
        SCHEMA_VERSION_KEY,
        verify_envelope_version,
    )

    inner = {SCHEMA_VERSION_KEY: f"in-{CANARY}"}
    verify_envelope_version({SCHEMA_VERSION_KEY: f"out-{CANARY}"}, inner)


def _scn_persistence_predating_fields_label(tmp_path, monkeypatch, strict):
    from traigent.utils.optimization_result_persistence import (
        FIELD_INTRODUCED_IN,
        SCHEMA_VERSION_KEY,
        decode_result,
    )

    fields = _encoded_result_fields()
    fields[SCHEMA_VERSION_KEY] = 1
    for name, introduced in FIELD_INTRODUCED_IN.items():
        if introduced > 1:
            fields.pop(name, None)
    decode_result(fields, trials=[], artifact_name=f"name-{CANARY}")


F1M_SCENARIOS: dict[str, Scenario] = {
    "manager/export_config_format": _scn_manager_export_config_format,
    "persistence/decode_version_value": _scn_persistence_decode_version_value,
    "persistence/envelope_version_value": _scn_persistence_envelope_version_value,
    "persistence/predating_fields_label": _scn_persistence_predating_fields_label,
    "store/banded_objective_name": _scn_store_banded_objective_name,
    "store/status_logs": _scn_store_status_logs,
    "store/session_id_not_found": _scn_store_session_id_not_found,
    "store/session_id_load_failure": _scn_store_session_id_load_failure,
    "store/session_id_created_and_saved": _scn_store_session_id_created_and_saved,
    "store/save_failure_session_id": _scn_store_save_failure_session_id,
    "store/export_format": _scn_store_export_format,
    "manager/captured_environment": _scn_manager_captured_environment,
    "results/save_str_raises_configuration_error": (
        _scn_save_str_raises_configuration_error
    ),
    "persistence/encode_timestamp_value": _scn_persistence_encode_timestamp_value,
    "persistence/encode_status_value": _scn_persistence_encode_status_value,
    "persistence/decode_timestamp_value": _scn_persistence_decode_timestamp_value,
    "persistence/manager_save_foreign_result": (
        _scn_persistence_manager_save_foreign_result
    ),
    "client/submit_result_session_id": _scn_client_submit_result_session_id,
}


def _scenarios() -> dict[str, Scenario]:
    s: dict[str, Scenario] = {}
    s["results/save_traversal"] = _scn_save_traversal
    s["results/save_base_is_file"] = _scn_save_base_is_file
    s["results/save_deepcopy_raises"] = _scn_save_deepcopy_raises
    s["results/save_str_raises"] = _scn_save_str_raises
    s["client/submit_result_load_session_permission"] = (
        _scn_client_submit_result_load_session_permission
    )
    for entry in ("load_from", "env", "dev-log"):
        for name, payload in SPEC_VARIANTS.items():
            s[f"{entry}/{name}"] = _scn_manager_file(entry, payload)
        s[f"{entry}/bad_json"] = _scn_manager_file(entry, '{"a": "' + CANARY)
    s["load_from/missing"] = _scn_manager_missing("load_from")
    s["env/missing"] = _scn_manager_missing("env")
    for name, payload in SPEC_VARIANTS.items():
        s[f"load_spec/{name}"] = _scn_load_spec(payload)
    s["load_spec/missing"] = _scn_load_missing
    for tamper in (
        "entry_path",
        "entry_missing_file",
        "hash",
        "manifest_version",
        "manifest_key",
        "manifest_json",
    ):
        s[f"repo/{tamper}"] = _scn_repo(tamper)
    s["repo/manager_nested_key"] = _scn_repo_via_manager
    for tamper in (
        "stale",
        "loaded_at",
        "missing_spec",
        "entry_path",
        "no_etag",
        "spec_json",
    ):
        s[f"cache/{tamper}"] = _scn_cache(tamper)
    for kind in (
        "transport",
        "runtime",
        "bad_hash",
        "other_id",
        "other_ref",
        "other_env",
        "spec_keys",
        "cache_unwritable",
        "cache_is_file",
        "config_error",
        "client_config_error",
    ):
        for mode in ("cloud", "repo_then_cloud"):
            s[f"fetch/{mode}/{kind}"] = _scn_fetch(kind, mode)
    for kind in ("transport", "runtime", "other_id", "bad_hash"):
        s[f"publish/{kind}"] = _scn_publish(kind)
    s["source_mode"] = _scn_source_mode
    for op in ("fetch", "publish"):
        for kind in ("transport", "http_500", "http_400"):
            s[f"client/{op}/{kind}"] = _scn_client(op, kind)
    for op in ("fetch", "publish"):
        s[f"client/{op}/invalid_header"] = _scn_client_invalid_header(op, "x")
    s["client/protected/publish"] = _scn_client_protected("publish")
    s["client/protected/head_fetch"] = _scn_client_protected("head")
    s["client/auth_failure"] = _scn_client_auth_failure
    s["repo/overlong_root"] = _scn_repo_overlong
    s["cache/overlong_dir"] = _scn_cache_overlong
    s["manager/overlong_cache_dir"] = _scn_manager_overlong_cache
    s["manager/whole_config_logging"] = _scn_whole_config
    s["results/load_missing"] = _scn_load_results_missing
    s["results/load_bad_json"] = _scn_load_results_bad_json
    s["results/save_load_named_file"] = _scn_load_results_ok_log
    s["results/save_overlong_dir"] = _scn_save_overlong_path
    s["results/save_overlong_name"] = _scn_save_overlong_name
    s["results/save_bad_timestamp"] = _scn_save_bad_timestamp
    s["results/save_bad_status"] = _scn_save_bad_status
    s["client/overlong_local_storage"] = _scn_client_overlong_local_storage
    s["client/local_storage_dir_logged"] = _scn_client_local_storage_dir_logged
    s["client/submit_result_storage_failure"] = (
        _scn_client_submit_result_storage_failure
    )
    s["client/async_auth_failure"] = _scn_client_async_auth_failure
    s["results/legacy_named_file"] = _scn_load_results_legacy
    s["results/unknown_status"] = _scn_load_results_unknown_status
    s["client/protected/publish_invalid_header"] = _scn_client_protected_invalid_header
    s["gate/refusal_chain"] = _scn_gate_refusal_chain
    s["client/gate_refusal_chain"] = _scn_client_gate_refusal_chain
    s["manager/apply_best_config_key"] = _scn_apply_best_config_key
    s["write/repo_overlong"] = _scn_write_repo_overlong
    s["write/cache_overlong"] = _scn_write_cache_overlong
    s["write/load_manifest_overlong"] = _scn_load_manifest_overlong
    s["write/export_overlong"] = _scn_export_overlong
    s["write/manifest_is_dir"] = _scn_write_manifest_is_dir
    s.update(F1M_SCENARIOS)
    return s


SCENARIOS = _scenarios()


@pytest.mark.backend_online
@pytest.mark.parametrize("strict", [False, True], ids=["lenient", "strict"])
@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_canary_never_appears_in_any_rendering(
    name, strict, tmp_path, monkeypatch, caplog
) -> None:
    caplog.set_level(logging.DEBUG)
    raised: list[BaseException] = []
    try:
        SCENARIOS[name](tmp_path, monkeypatch, strict)
    except AssertionError:
        # a failed scenario precondition (setup / round-trip check) is a test failure,
        # never a "rendered nothing" pass
        raise
    except Exception as exc:  # noqa: BLE001 - the property is about what it renders
        raised.append(exc)
    _check(caplog, raised)


def test_scenario_matrix_is_not_empty() -> None:
    # inputs x entry points x renderings: guard against a silently shrunk matrix.
    assert len(SCENARIOS) >= 136


def test_helper_keeps_rejected_values_on_details_only() -> None:
    from traigent.core import best_config_diagnostics as diag

    inner = OSError(f"/work/{CANARY}")
    error = diag.refusal(
        "Fixed message",
        category=diag.Category.CACHE_WRITE_FAILED,
        cause=inner,
        cache_dir=f"/work/{CANARY}",
    )
    assert error.details["cache_dir"] == f"/work/{CANARY}"
    assert error.details["error_type"] == "OSError"
    assert error.details["category"] == "cache_write_failed"
    assert error.__cause__ is None and error.__suppress_context__ is True
    for text in _renderings(error):
        assert CANARY not in text
    assert diag.fixed_reason(inner) == "OSError"
    assert diag.fixed_reason(error) == "Fixed message"


def test_cache_write_failure_details_stay_available(tmp_path) -> None:
    blocker = tmp_path / "blocker"
    blocker.write_text("x", encoding="utf-8")
    manager = _manager(
        source="cloud", strict=True, cache_dir=str(blocker / f"sub-{CANARY}")
    )
    with patch(
        "traigent.cloud.backend_client.get_backend_client",
        return_value=_FakeClient(fetch=_data_for(_spec())),
    ):
        with pytest.raises(Exception) as info:
            manager._resolve_mode_snapshot()
    assert info.value.details["error_type"] == "NotADirectoryError"


def test_unmarked_configuration_error_message_is_never_a_reason() -> None:
    from traigent.core import best_config_diagnostics as diag

    assert diag.fixed_reason(ConfigurationError(f"Invalid config {CANARY}")) == (
        "ConfigurationError"
    )


@pytest.mark.parametrize("strict", [False, True], ids=["lenient", "strict"])
def test_overlong_path_components_give_fixed_refusal_or_none(strict, tmp_path) -> None:
    for call in (
        lambda: resolve_repo_best_config(
            config_id=CID, repo_root=_long(tmp_path), strict=strict
        ),
        lambda: resolve_cloud_cache_best_config(
            config_id=CID, cache_dir=_long(tmp_path), strict=strict
        ),
    ):
        if strict:
            with pytest.raises(ConfigurationError) as info:
                call()
            assert info.value.details["category"] in {"repo_invalid", "cache_invalid"}
        else:
            assert call() is None


def test_whole_config_values_are_not_logged(tmp_path, caplog) -> None:
    caplog.set_level(logging.DEBUG)
    _scn_whole_config(tmp_path, None, False)
    assert CANARY not in caplog.text


def test_successful_load_scenario_cannot_pass_without_loading(
    tmp_path, monkeypatch, caplog
) -> None:
    """A setup failure before the save/load must fail the property test, not pass it."""
    with patch(
        f"{__name__}._real_results", side_effect=AssertionError("setup failed")
    ) as setup:
        with pytest.raises(AssertionError, match="setup failed"):
            test_canary_never_appears_in_any_rendering(
                "results/save_load_named_file", False, tmp_path, monkeypatch, caplog
            )
    assert setup.call_count == 1


def test_loaded_result_equals_saved_result(tmp_path) -> None:
    manager = _manager()
    saved = _real_results()
    target = tmp_path / f"saved-{CANARY}.json"
    manager._optimization_results = saved
    manager.save_optimization_results(str(target))
    manager._optimization_results = None
    manager.load_optimization_results(str(target))
    assert manager._optimization_results == saved
