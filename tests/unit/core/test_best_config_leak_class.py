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
        with patch(
            "traigent.cloud.backend_client.get_backend_client",
            return_value=_FakeClient(fetch=fetch),
        ):
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


def _scenarios() -> dict[str, Scenario]:
    s: dict[str, Scenario] = {}
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
    ):
        for mode in ("cloud", "repo_then_cloud"):
            s[f"fetch/{mode}/{kind}"] = _scn_fetch(kind, mode)
    for kind in ("transport", "runtime", "other_id", "bad_hash"):
        s[f"publish/{kind}"] = _scn_publish(kind)
    s["source_mode"] = _scn_source_mode
    for op in ("fetch", "publish"):
        for kind in ("transport", "http_500", "http_400"):
            s[f"client/{op}/{kind}"] = _scn_client(op, kind)
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
    except Exception as exc:  # noqa: BLE001 - the property is about what it renders
        raised.append(exc)
    _check(caplog, raised)


def test_scenario_matrix_is_not_empty() -> None:
    # inputs x entry points x renderings: guard against a silently shrunk matrix.
    assert len(SCENARIOS) >= 90


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
