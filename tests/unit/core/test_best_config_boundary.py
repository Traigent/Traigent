"""One diagnostic boundary on every public best-config entry point.

A leak-class table closes the shapes someone thought of. This test closes the class: for
every public entry point of the hardened surface, an arbitrary exception carrying the
canary is injected at an inner seam (a helper, an attribute, an iterator the entry point
touches) and the exception that escapes must be a fixed refusal. The canary must be absent
from ``str`` / ``repr`` / ``args`` / ``details`` / ``__cause__`` / ``__context__`` / the
formatted traceback and from captured logs. ``KeyboardInterrupt`` and friends
(``BaseException``) are never swallowed, and trusted refusals pass through unchanged.
"""

from __future__ import annotations

import logging
import traceback
from collections.abc import Callable
from datetime import UTC, datetime
from types import MappingProxyType
from typing import Any

import pytest

from traigent.core import best_config_diagnostics as diag
from traigent.core import best_config_runtime as runtime
from traigent.core import config_state_manager as csm_module
from traigent.core.best_config_runtime import (
    CloudPublishUnavailable,
    SafetySensitiveBestConfigError,
)
from traigent.core.config_state_manager import ConfigStateManager
from traigent.utils.exceptions import ConfigurationError

CANARY = "zq-acme-private-canary-91"
CID = f"id-{CANARY}"


def _agent(text: str) -> str:
    return text


# ----------------------------------------------------------------- injected failures
class _CustomError(Exception):
    """Carries the canary in every rendering and an empty ``args``."""

    def __str__(self) -> str:
        return CANARY

    def __repr__(self) -> str:
        return f"_CustomError({CANARY!r})"


def _chained() -> BaseException:
    try:
        try:
            raise ValueError(CANARY)
        except ValueError as inner:
            raise RuntimeError("outer") from inner
    except RuntimeError as outer:
        return outer


EXCEPTIONS: dict[str, Callable[[], BaseException]] = {
    "runtime": lambda: RuntimeError(CANARY),
    "key": lambda: KeyError(CANARY),
    "type": lambda: TypeError(CANARY),
    "value": lambda: ValueError(CANARY),
    "unicode_decode": lambda: UnicodeDecodeError(
        "utf-8", b"\xff" + CANARY.encode(), 0, 1, "invalid start byte"
    ),
    "unicode_encode": lambda: UnicodeEncodeError(
        "utf-8", "\ud800" + CANARY, 0, 1, "surrogates not allowed"
    ),
    "os": lambda: OSError(2, CANARY, CANARY),
    "configuration": lambda: ConfigurationError(CANARY, details={"k": CANARY}),
    "custom": _CustomError,
    "chained": _chained,
}


class _Boom:
    """Stands in for a module global: any use of it raises the injected exception."""

    def __init__(self, exc: BaseException) -> None:
        self._exc = exc

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        raise self._exc

    def __getattr__(self, name: str) -> Any:
        raise self._exc


class _BoomDict(dict):  # type: ignore[type-arg]
    """A dict whose every read raises the injected exception."""

    def __init__(self, exc: BaseException) -> None:
        super().__init__()
        self._exc = exc

    def __iter__(self):  # type: ignore[no-untyped-def]
        raise self._exc

    def get(self, *args: Any, **kwargs: Any) -> Any:
        raise self._exc

    def items(self):  # type: ignore[no-untyped-def]
        raise self._exc

    def __getitem__(self, key: Any) -> Any:
        raise self._exc


class _BoomBool:
    def __init__(self, exc: BaseException) -> None:
        self._exc = exc

    def __bool__(self) -> bool:
        raise self._exc

    def __getattr__(self, name: str) -> Any:
        raise self._exc


# ------------------------------------------------------------------------ entry points
def _manager(**kw: Any) -> ConfigStateManager:
    return ConfigStateManager(
        func=_agent,
        default_config={"t": 1},
        local_storage_path=None,
        configuration_space={"t": [1, 2]},
        auto_load_best=False,
        load_from=None,
        setup_wrapper_callback=lambda: None,
        config_id=CID,
        best_config_source=kw.get("source", "off"),
        best_config_strict=kw.get("strict", False),
        best_config_environment="prod",
        best_config_cache_dir=kw.get("cache_dir"),
        enable_auto_load_dev_logs=False,
    )


def _results() -> Any:
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


# A case is ``(expected_class, setup)``; ``setup(exc, monkeypatch, tmp_path)`` injects
# ``exc`` at the seam and returns the zero-argument call.
Setup = Callable[[BaseException, Any, Any], Callable[[], Any]]


def _patch_runtime(name: str, call: Callable[[], Any]) -> Setup:
    def setup(exc: BaseException, monkeypatch: Any, tmp_path: Any) -> Callable[[], Any]:
        monkeypatch.setattr(runtime, name, _Boom(exc))
        return call

    return setup


def _patch_csm(name: str, build: Callable[[Any], Callable[[], Any]]) -> Setup:
    def setup(exc: BaseException, monkeypatch: Any, tmp_path: Any) -> Callable[[], Any]:
        monkeypatch.setattr(csm_module, name, _Boom(exc))
        return build(tmp_path)

    return setup


def _boom_arg(call: Callable[[Any], Any]) -> Setup:
    def setup(exc: BaseException, monkeypatch: Any, tmp_path: Any) -> Callable[[], Any]:
        return lambda: call(exc)

    return setup


def _tuple_boom(exc: BaseException) -> Any:
    class _T(tuple):  # type: ignore[type-arg]
        def __iter__(self):  # type: ignore[no-untyped-def]
            raise exc

    return runtime.thaw_config(_T((1,)))


def _manager_attr(method: str, attr: str, build: Callable[..., Any]) -> Setup:
    """Patch ``attr`` on a fresh manager to raise, then call ``method``."""

    def setup(exc: BaseException, monkeypatch: Any, tmp_path: Any) -> Callable[[], Any]:
        manager = _manager()
        manager.set_config({"t": 2})
        manager._optimization_results = _results()
        manager._override_sticky = False  # maybe_auto_load_config returns early if set
        monkeypatch.setattr(manager, attr, _Boom(exc), raising=False)
        return lambda: build(manager, tmp_path)

    return setup


def _publish_setup(exc: BaseException, monkeypatch: Any, tmp_path: Any):  # type: ignore[no-untyped-def]
    manager = _manager()
    manager.set_config({"t": 2})
    monkeypatch.setattr(csm_module, "check_publishable_spec", _Boom(exc))
    return lambda: manager.publish_best_config()


def _get_best_setup(exc: BaseException, monkeypatch: Any, tmp_path: Any):  # type: ignore[no-untyped-def]
    manager = _manager()
    manager._optimization_results = _BoomBool(exc)  # type: ignore[assignment]
    return lambda: manager.get_best_config()


CASES: dict[str, tuple[type[BaseException], Setup]] = {
    # --- best_config_runtime ---
    "runtime.canonical_json": (
        ConfigurationError,
        _patch_runtime("_normalize_json_value", lambda: runtime.canonical_json({})),
    ),
    "runtime.sha256_digest": (
        ConfigurationError,
        _patch_runtime("hashlib", lambda: runtime.sha256_digest("x")),
    ),
    "runtime.thaw_config": (ConfigurationError, _boom_arg(_tuple_boom)),
    "runtime.function_ref_for": (
        ConfigurationError,
        _patch_runtime("inspect", lambda: runtime.function_ref_for(_agent)),
    ),
    "runtime.source_order_for_mode": (
        ConfigurationError,
        _patch_runtime(
            "BestConfigSource", lambda: runtime.source_order_for_mode("repo")
        ),
    ),
    "runtime.load_best_config_spec": (
        ConfigurationError,
        _patch_runtime(
            "Path", lambda: runtime.load_best_config_spec("x.json", strict=True)
        ),
    ),
    "runtime.snapshot_from_spec": (
        ConfigurationError,
        _boom_arg(lambda exc: runtime.snapshot_from_spec(_BoomDict(exc))),
    ),
    "runtime.compute_spec_hash": (
        ConfigurationError,
        _boom_arg(lambda exc: runtime.compute_spec_hash(_BoomDict(exc))),
    ),
    "runtime.check_publishable_spec": (
        ConfigurationError,
        _boom_arg(lambda exc: runtime.check_publishable_spec(_BoomDict(exc))),
    ),
    "runtime.load_manifest": (
        ConfigurationError,
        _patch_runtime("Path", lambda: runtime.load_manifest("d")),
    ),
    "runtime.write_repo_best_config": (
        ConfigurationError,
        _patch_runtime(
            "_validate_config_id",
            lambda: runtime.write_repo_best_config("d", config_id="a", config={}),
        ),
    ),
    "runtime.write_cloud_cache_best_config": (
        ConfigurationError,
        _patch_runtime(
            "snapshot_from_spec",
            lambda: runtime.write_cloud_cache_best_config("d", {}, etag="e"),
        ),
    ),
    "runtime.resolve_repo_best_config": (
        ConfigurationError,
        _patch_runtime(
            "_validate_config_id",
            lambda: runtime.resolve_repo_best_config(config_id="a", strict=False),
        ),
    ),
    "runtime.resolve_cloud_cache_best_config": (
        ConfigurationError,
        _patch_runtime(
            "_validate_config_id",
            lambda: runtime.resolve_cloud_cache_best_config(
                config_id="a", cache_dir="d", strict=False
            ),
        ),
    ),
    "runtime.BestConfigSnapshot.from_config": (
        ConfigurationError,
        _patch_runtime(
            "_normalize_json_value",
            lambda: runtime.BestConfigSnapshot.from_config(
                {}, config_id="a", source="repo"
            ),
        ),
    ),
    # --- ConfigStateManager ---
    "csm.get_best_config": (ConfigurationError, _get_best_setup),
    "csm.set_config": (
        ConfigurationError,
        _manager_attr(
            "set_config",
            "_snapshot_from_plain_config",
            lambda m, t: m.set_config({"t": 2}),
        ),
    ),
    "csm.apply_best_config": (
        ConfigurationError,
        _manager_attr(
            "apply_best_config",
            "_snapshot_from_plain_config",
            lambda m, t: m.apply_best_config(),
        ),
    ),
    "csm.clear_override": (
        ConfigurationError,
        _manager_attr(
            "clear_override", "maybe_auto_load_config", lambda m, t: m.clear_override()
        ),
    ),
    "csm.save_optimization_results": (
        ConfigurationError,
        _patch_csm(
            "_redact_example_content_for_disk_save",
            lambda tmp: _save_call(tmp),
        ),
    ),
    "csm.load_optimization_results": (
        ConfigurationError,
        _patch_csm(
            "validate_path",
            lambda tmp: lambda: _manager().load_optimization_results("r.json"),
        ),
    ),
    "csm.maybe_auto_load_config": (
        ConfigurationError,
        _manager_attr(
            "maybe_auto_load_config",
            "_resolve_mode_snapshot",
            lambda m, t: m.maybe_auto_load_config(),
        ),
    ),
    "csm.export_config": (
        ConfigurationError,
        _patch_csm(
            "validate_user_path",
            lambda tmp: lambda: _exportable().export_config(str(tmp / "o.json")),
        ),
    ),
    "csm.export_best_config": (
        ConfigurationError,
        _patch_csm(
            "write_repo_best_config",
            lambda tmp: lambda: _exportable().export_best_config(str(tmp / "o")),
        ),
    ),
    "csm.publish_best_config": (CloudPublishUnavailable, _publish_setup),
}


def _exportable() -> ConfigStateManager:
    manager = _manager()
    manager.set_config({"t": 2})
    return manager


def _save_call(tmp_path: Any) -> Callable[[], Any]:
    manager = _manager()
    manager._optimization_results = _results()
    return lambda: manager.save_optimization_results(str(tmp_path / "r.json"))


# ------------------------------------------------------------------------ assertions
def _chain(exc: BaseException) -> list[BaseException]:
    seen: list[BaseException] = []
    stack = [exc]
    while stack:
        item = stack.pop()
        if item is None or any(item is s for s in seen):
            continue
        seen.append(item)
        stack.extend([item.__cause__, item.__context__])  # type: ignore[list-item]
    return seen


def _assert_clean(exc: BaseException, caplog: pytest.LogCaptureFixture) -> None:
    for item in _chain(exc):
        for text in (
            str(item),
            repr(item),
            "".join(map(str, item.args)),
            "".join(traceback.format_exception(item)),
            # ``details['reason']`` of load_optimization_results is the one documented
            # details-only carrier of the decoder's message (pinned by the reload tests)
            str(
                {
                    k: v
                    for k, v in (getattr(item, "details", None) or {}).items()
                    if k != "reason"
                }
            ),
        ):
            assert CANARY not in text, f"leak in {type(item).__name__}: {text[:300]!r}"
    assert exc.__cause__ is None, "refusal must not chain a cause"
    assert exc.__context__ is None, "refusal must not chain a context"
    assert CANARY not in caplog.text
    for rec in caplog.records:
        assert CANARY not in logging.Formatter("%(message)s").format(rec)
        assert CANARY not in str(rec.args)


@pytest.fixture(autouse=True)
def _env(monkeypatch, tmp_path):
    monkeypatch.delenv("TRAIGENT_CONFIG_PATH", raising=False)
    monkeypatch.setenv("TRAIGENT_RESULTS_FOLDER", str(tmp_path / "results"))
    monkeypatch.chdir(tmp_path)


@pytest.mark.parametrize("kind", sorted(EXCEPTIONS))
@pytest.mark.parametrize("case", sorted(CASES))
def test_any_exception_at_an_inner_seam_leaves_as_a_fixed_refusal(
    case, kind, monkeypatch, tmp_path, caplog
) -> None:
    caplog.set_level(logging.DEBUG)
    expected, setup = CASES[case]
    call = setup(EXCEPTIONS[kind](), monkeypatch, tmp_path)
    with pytest.raises(Exception) as info:
        call()
    exc = info.value
    assert isinstance(exc, expected), f"{case}: raised {type(exc).__name__}"
    assert diag.is_safe(exc), f"{case}: escaped as an untrusted {type(exc).__name__}"
    _assert_clean(exc, caplog)


def test_entry_point_table_names_every_public_function() -> None:
    names = {c.split(".", 1)[1] for c in CASES if c.startswith("runtime.")}
    public = {
        n
        for n in runtime.__all__
        if callable(getattr(runtime, n))
        and not isinstance(getattr(runtime, n), type)
        and not n.startswith("_")
    }
    assert public <= names, (
        f"public runtime functions without a boundary case: {public - names}"
    )
    manager_methods = {c.split(".", 1)[1] for c in CASES if c.startswith("csm.")}
    assert {
        "set_config",
        "apply_best_config",
        "clear_override",
        "save_optimization_results",
        "load_optimization_results",
        "maybe_auto_load_config",
        "export_config",
        "export_best_config",
        "publish_best_config",
        "get_best_config",
    } <= manager_methods


@pytest.mark.parametrize(
    "interrupt",
    [KeyboardInterrupt, SystemExit, GeneratorExit],
    ids=lambda c: c.__name__,
)
@pytest.mark.parametrize("case", sorted(CASES))
def test_base_exceptions_are_never_swallowed(
    case, interrupt, monkeypatch, tmp_path
) -> None:
    _, setup = CASES[case]
    sentinel = interrupt()
    call = setup(sentinel, monkeypatch, tmp_path)
    with pytest.raises(interrupt) as info:
        call()
    assert info.value is sentinel


def test_trusted_refusals_pass_through_the_boundary_unchanged() -> None:
    refusal = diag._safe_error("fixed")

    @diag.boundary(diag.Category.LOAD_FAILED, "boundary message")
    def raises() -> None:
        raise refusal

    with pytest.raises(ConfigurationError) as info:
        raises()
    assert info.value is refusal
    assert str(info.value) == "fixed"


def test_untrusted_configuration_error_is_converted() -> None:
    @diag.boundary(diag.Category.LOAD_FAILED, "boundary message")
    def raises() -> None:
        raise ConfigurationError(CANARY)

    with pytest.raises(ConfigurationError) as info:
        raises()
    assert CANARY not in str(info.value)
    assert info.value.details["error_type"] == "ConfigurationError"


def test_safety_sensitive_error_keeps_its_class(tmp_path) -> None:
    with pytest.raises(SafetySensitiveBestConfigError) as info:
        runtime.snapshot_from_spec(
            {
                "schema_version": runtime.BEST_CONFIG_SCHEMA_VERSION,
                "config_id": "a",
                "config": {"auth_token": 1},
            },
            configuration_space={"t": [1]},
        )
    assert diag.is_safe(info.value)


def test_snapshot_config_is_frozen_after_boundary() -> None:
    # the wrapped classmethod still builds an immutable snapshot
    snap = runtime.BestConfigSnapshot.from_config(
        {"t": 1}, config_id="a", source="repo"
    )
    assert isinstance(snap.config, MappingProxyType)
