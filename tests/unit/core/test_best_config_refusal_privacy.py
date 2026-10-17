"""Refusals on the bound best-config load paths never print customer content.

A rejected file can carry customer-chosen text: a schema version, a key name, an
identity value, a path. Log text and exception text use fixed wording; the details stay
in structured exception attributes only.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pytest

from traigent.core.best_config_runtime import (
    BestConfigSource,
    function_ref_for,
    load_best_config_spec,
    write_repo_best_config,
)
from traigent.core.config_state_manager import ConfigStateManager
from traigent.utils.exceptions import ConfigurationError

CANARY = "zq-acme-private-canary-91"


def _agent(text: str) -> str:
    return text


_REF = function_ref_for(_agent)


def _spec(**override: Any) -> dict[str, Any]:
    spec: dict[str, Any] = {
        "schema_version": "traigent.best_config.v1",
        "config_id": "agent",
        "function_ref": _REF,
        "environment": "prod",
        "config": {"t": 2},
    }
    spec.update(override)
    return spec


CASES: dict[str, dict[str, Any]] = {
    "schema_version": _spec(schema_version=f"/work/{CANARY}"),
    "schema_version_bare": {
        "schema_version": f"/work/{CANARY}",
        "config_id": "agent",
        "config": {"t": 2},
    },
    "unknown_top_level_key": _spec(**{f"x_{CANARY}": 1}),
    "unknown_nested_key": _spec(config={"t": 2, f"k_{CANARY}": 1}),
    "unknown_nested_key_forward_compat": _spec(
        config={"t": 2, f"k_{CANARY}": 1}, forward_compat=True
    ),
    "safety_sensitive_nested_key": _spec(config={"t": 2, f"auth_{CANARY}": 1}),
    "function_ref": _spec(function_ref=f"mod:{CANARY}"),
    "config_id": _spec(config_id=f"id-{CANARY}"),
    "environment": _spec(environment=f"env-{CANARY}"),
    "legacy_function_ref": {"function_ref": f"mod:{CANARY}", "config": {"t": 2}},
    "legacy_config_id": {"config_id": f"id-{CANARY}", "config": {"t": 2}},
    "legacy_unknown_key": {"t": 2, f"k_{CANARY}": 1},
    "missing_fields": _spec(config=None, **{f"x_{CANARY}": 1}),
}


def _manager(**kwargs: Any) -> ConfigStateManager:
    return ConfigStateManager(
        func=_agent,
        default_config={"t": 1},
        local_storage_path=None,
        configuration_space={"t": [1, 2]},
        auto_load_best=kwargs.get("dev_logs", False),
        load_from=kwargs.get("load_from"),
        setup_wrapper_callback=lambda: None,
        config_id="agent",
        best_config_source="off",
        best_config_strict=kwargs.get("strict", False),
        best_config_environment="prod",
        enable_auto_load_dev_logs=kwargs.get("dev_logs", False),
    )


@pytest.fixture(autouse=True)
def _env(monkeypatch, tmp_path):
    monkeypatch.delenv("TRAIGENT_CONFIG_PATH", raising=False)
    monkeypatch.setenv("TRAIGENT_RESULTS_FOLDER", str(tmp_path / "results"))
    monkeypatch.chdir(tmp_path)


def _place(entry: str, tmp_path: Path, monkeypatch, data: Any, **kw: Any):
    directory = tmp_path / f"dir-{CANARY}"
    if entry == "dev-log":
        directory = (
            tmp_path / ".traigent" / "optimization_logs" / "experiments"
            / "_agent" / "runs" / "r1" / "artifacts"
        )
        name = "best_config_v2.json"
    else:
        name = f"file-{CANARY}.json"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_text(json.dumps(data), encoding="utf-8")
    if entry == "load_from":
        return _manager(load_from=str(path), **kw)
    if entry == "env":
        monkeypatch.setenv("TRAIGENT_CONFIG_PATH", str(path))
        return _manager(**kw)
    return _manager(dev_logs=True, **kw)


def _assert_clean(caplog, exc: BaseException | None = None) -> None:
    text = caplog.text + "".join(
        f"{r.getMessage()}|{r.args}|{r.exc_text}" for r in caplog.records
    )
    assert CANARY not in text
    if exc is not None:
        assert CANARY not in str(exc)
        assert CANARY not in repr(exc)
        assert CANARY not in "".join(map(str, exc.args))


@pytest.mark.parametrize("entry", ["load_from", "env", "dev-log"])
@pytest.mark.parametrize("case", sorted(CASES))
@pytest.mark.parametrize("strict", [False, True])
def test_refusals_never_print_customer_content(
    entry, case, strict, tmp_path, monkeypatch, caplog
) -> None:
    caplog.set_level(logging.DEBUG)
    manager = _place(entry, tmp_path, monkeypatch, CASES[case], strict=strict)
    raised: BaseException | None = None
    try:
        manager.maybe_auto_load_config()
    except ConfigurationError as exc:
        raised = exc
    _assert_clean(caplog, raised)


@pytest.mark.parametrize("strict", [False, True])
@pytest.mark.parametrize("case", sorted(CASES))
def test_load_best_config_spec_never_prints_customer_content(
    case, strict, tmp_path, caplog
) -> None:
    caplog.set_level(logging.DEBUG)
    path = tmp_path / f"file-{CANARY}.json"
    path.write_text(json.dumps(CASES[case]), encoding="utf-8")
    raised: BaseException | None = None
    try:
        load_best_config_spec(
            path,
            configuration_space={"t": [1, 2]},
            expected_config_id="agent",
            expected_function_ref=_REF,
            expected_environment="prod",
            strict=strict,
        )
    except ConfigurationError as exc:
        raised = exc
    _assert_clean(caplog, raised)


@pytest.mark.parametrize("case", ["unknown_nested_key", "safety_sensitive_nested_key"])
def test_repo_mode_never_prints_customer_content(case, tmp_path, caplog) -> None:
    caplog.set_level(logging.DEBUG)
    root = tmp_path / ".traigent" / "best-configs"
    write_repo_best_config(
        root, config_id="agent", config={"t": 2}, function_ref=_REF,
        environment="prod",
    )
    spec_file = root / "agent.json"
    spec = json.loads(spec_file.read_text())
    spec["config"] = CASES[case]["config"]
    spec_file.write_text(json.dumps(spec), encoding="utf-8")
    for strict in (False, True):
        manager = _manager(strict=strict)
        manager.best_config_source = "repo"
        raised = None
        try:
            manager._resolve_mode_snapshot()
        except ConfigurationError as exc:
            raised = exc
        _assert_clean(caplog, raised)


def test_the_rejected_details_stay_available_as_attributes(tmp_path) -> None:
    path = tmp_path / "x.json"
    path.write_text(json.dumps(CASES["schema_version"]), encoding="utf-8")
    manager = _manager(load_from=str(path), strict=True)
    with pytest.raises(ConfigurationError) as info:
        manager.maybe_auto_load_config()
    assert info.value.details["path"] == str(path)
    assert BestConfigSource.LOAD_FROM.value
