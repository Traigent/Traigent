"""Characterization of SeamlessParameterProvider's context-config merge.

Pins how ``SeamlessParameterProvider.inject_config`` folds the active
``config_context`` value into a copy of the wrap-time config before running
the function, using a spy in place of ``_seamless_run``.
"""

from __future__ import annotations

import asyncio
from contextvars import ContextVar
from typing import Any

import pytest

from traigent.config import context as context_module
from traigent.config import providers as providers_module
from traigent.config.context import config_context
from traigent.config.providers import SeamlessParameterProvider
from traigent.config.types import TraigentConfig


class _Recorder:
    """Spy standing in for ``SeamlessParameterProvider._seamless_run``."""

    def __init__(self, var: ContextVar | None = None) -> None:
        self.var = var if var is not None else config_context
        self.calls: list[dict[str, Any]] = []
        self.error: BaseException | None = None

    def __call__(self, func: Any, active_config: dict[str, Any], args, kwargs):
        self.calls.append(
            {
                "active": active_config,
                "snapshot": dict(active_config),
                "ctx": self.var.get(None),
                "args": args,
                "kwargs": kwargs,
            }
        )
        if self.error is not None:
            raise self.error
        return func(*args, **kwargs)


@pytest.fixture(autouse=True)
def _unset_context():
    """The suite-wide conftest installs a default config; start from unset."""
    token = config_context.set(None)
    yield
    config_context.reset(token)


@pytest.fixture
def set_ctx():
    tokens = []

    def _set(value: Any) -> None:
        tokens.append(config_context.set(value))

    yield _set
    for token in reversed(tokens):
        config_context.reset(token)


@pytest.fixture
def from_dict_calls(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    original = TraigentConfig.from_dict

    def _counting(cls: type, config_dict: dict[str, Any]) -> TraigentConfig:
        calls.append(config_dict)
        return original(config_dict)

    monkeypatch.setattr(TraigentConfig, "from_dict", classmethod(_counting))
    return calls


def _build(config: dict[str, Any], *, is_async: bool = False):
    provider = SeamlessParameterProvider()
    spy = _Recorder()
    provider._seamless_run = spy  # type: ignore[method-assign]
    ran: list[tuple[Any, ...]] = []

    if is_async:

        async def func(*args: Any, **kwargs: Any) -> str:
            ran.append((args, kwargs))
            return "async-result"

    else:

        def func(*args: Any, **kwargs: Any) -> str:
            ran.append((args, kwargs))
            return "result"

    return provider.inject_config(func, config), spy, ran


class _ExplodingConfig(TraigentConfig):
    def to_dict(self) -> dict[str, Any]:  # type: ignore[override]
        raise RuntimeError("to_dict broke")


class _FalseyConfig(TraigentConfig):
    def to_dict(self) -> dict[str, Any]:  # type: ignore[override]
        return {}


class _NoneConfig(TraigentConfig):
    def to_dict(self) -> dict[str, Any]:  # type: ignore[override]
        return None  # type: ignore[return-value]


class TestNoContext:
    def test_absent_context_uses_copy_of_config(self):
        source = {"a": 1}
        wrapped, spy, _ = _build(source)

        assert wrapped(1, k=2) == "result"

        call = spy.calls[0]
        assert call["active"] == {"a": 1}
        assert call["active"] is not source
        assert call["args"] == (1,)
        assert call["kwargs"] == {"k": 2}
        assert source == {"a": 1}

    def test_absent_context_never_calls_get_config(self, monkeypatch):
        def _boom():
            raise AssertionError("get_config must not be called")

        monkeypatch.setattr(providers_module, "get_config", _boom)
        wrapped, spy, _ = _build({"a": 1})

        wrapped()

        assert len(spy.calls) == 1

    def test_empty_config_and_no_context_calls_original(self, from_dict_calls):
        wrapped, spy, ran = _build({})

        assert wrapped("x") == "result"

        assert spy.calls == []
        assert ran == [(("x",), {})]
        assert from_dict_calls == []

    def test_none_config_is_treated_as_empty(self):
        wrapped, spy, ran = _build(None)  # type: ignore[arg-type]

        wrapped()

        assert spy.calls == []
        assert len(ran) == 1


class TestContextMerge:
    def test_dict_context_overrides_and_extends_in_order(self, set_ctx):
        set_ctx({"b": 9, "c": 3})
        wrapped, spy, _ = _build({"a": 1, "b": 2})

        wrapped()

        active = spy.calls[0]["active"]
        assert active == {"a": 1, "b": 9, "c": 3}
        assert list(active) == ["a", "b", "c"]

    def test_dict_context_is_not_mutated(self, set_ctx):
        ctx = {"b": 9}
        set_ctx(ctx)
        wrapped, _, _ = _build({"a": 1})

        wrapped()

        assert ctx == {"b": 9}

    def test_wrap_time_config_is_not_mutated(self, set_ctx):
        source = {"a": 1}
        set_ctx({"a": 5, "z": 1})
        wrapped, _, _ = _build(source)

        wrapped()

        assert source == {"a": 1}

    def test_traigent_config_context_uses_to_dict(self, set_ctx):
        set_ctx(TraigentConfig(model="ctx-model", temperature=0.5))
        wrapped, spy, _ = _build({"model": "wrap", "extra": 1})

        wrapped()

        active = spy.calls[0]["active"]
        assert active["model"] == "ctx-model"
        assert active["temperature"] == 0.5
        assert active["extra"] == 1

    def test_default_traigent_config_context_adds_nothing(self, set_ctx):
        set_ctx(TraigentConfig())
        wrapped, spy, _ = _build({"a": 1})

        wrapped()

        assert spy.calls[0]["active"] == {"a": 1}

    def test_empty_dict_context_adds_nothing(self, set_ctx):
        set_ctx({})
        wrapped, spy, _ = _build({"a": 1})

        wrapped()

        assert spy.calls[0]["active"] == {"a": 1}

    def test_falsey_to_dict_adds_nothing(self, set_ctx):
        set_ctx(_FalseyConfig())
        wrapped, spy, _ = _build({"a": 1})

        wrapped()

        assert spy.calls[0]["active"] == {"a": 1}

    def test_none_to_dict_adds_nothing(self, set_ctx):
        set_ctx(_NoneConfig())
        wrapped, spy, _ = _build({"a": 1})

        wrapped()

        assert spy.calls[0]["active"] == {"a": 1}

    @pytest.mark.parametrize("unsupported", ["text", 7, ["a"], object()])
    def test_unsupported_context_value_is_ignored(self, set_ctx, unsupported):
        set_ctx(unsupported)
        wrapped, spy, _ = _build({"a": 1})

        wrapped()

        assert spy.calls[0]["active"] == {"a": 1}

    def test_empty_config_with_context_dict_is_not_empty(self, set_ctx):
        set_ctx({"only": "ctx"})
        wrapped, spy, ran = _build({})

        wrapped()

        assert spy.calls[0]["active"] == {"only": "ctx"}

    def test_empty_config_with_empty_context_calls_original(self, set_ctx):
        set_ctx({})
        wrapped, spy, ran = _build({})

        wrapped()

        assert spy.calls == []
        assert len(ran) == 1


class TestRunContract:
    def test_active_dict_identity_reaches_from_dict_and_run(
        self, set_ctx, from_dict_calls
    ):
        set_ctx({"b": 2})
        wrapped, spy, _ = _build({"a": 1})

        wrapped()

        assert from_dict_calls[-1] is spy.calls[0]["active"]
        assert from_dict_calls[-1] == {"a": 1, "b": 2}

    def test_context_set_during_run_is_traigent_config(self, set_ctx):
        set_ctx({"model": "ctx-model"})
        wrapped, spy, _ = _build({"temperature": 0.3})

        wrapped()

        inside = spy.calls[0]["ctx"]
        assert isinstance(inside, TraigentConfig)
        assert inside.model == "ctx-model"
        assert inside.temperature == 0.3

    def test_context_restored_after_success(self, set_ctx):
        previous = {"model": "before"}
        set_ctx(previous)
        wrapped, _, _ = _build({"a": 1})

        wrapped()

        assert config_context.get(None) is previous

    def test_context_reset_to_unset_after_success(self):
        wrapped, _, _ = _build({"a": 1})

        wrapped()

        assert config_context.get(None) is None

    def test_context_restored_after_run_exception(self, set_ctx):
        previous = {"model": "before"}
        set_ctx(previous)
        wrapped, spy, _ = _build({"a": 1})
        spy.error = RuntimeError("run failed")

        with pytest.raises(RuntimeError, match="run failed"):
            wrapped()

        assert config_context.get(None) is previous

    def test_run_spy_and_function_each_invoked_once(self):
        wrapped, spy, ran = _build({"a": 1})

        wrapped()

        assert len(spy.calls) == 1
        assert len(ran) == 1


class TestErrorsBeforeSet:
    def test_get_config_error_propagates_before_any_set(
        self, set_ctx, monkeypatch, from_dict_calls
    ):
        previous = {"model": "before"}
        set_ctx(previous)

        def _boom():
            raise RuntimeError("get_config broke")

        monkeypatch.setattr(providers_module, "get_config", _boom)
        wrapped, spy, ran = _build({"a": 1})

        with pytest.raises(RuntimeError, match="get_config broke"):
            wrapped()

        assert spy.calls == []
        assert ran == []
        assert from_dict_calls == []
        assert config_context.get(None) is previous

    def test_to_dict_error_propagates_before_any_set(self, set_ctx, from_dict_calls):
        bad = _ExplodingConfig()
        set_ctx(bad)
        wrapped, spy, ran = _build({"a": 1})

        with pytest.raises(RuntimeError, match="to_dict broke"):
            wrapped()

        assert spy.calls == []
        assert ran == []
        assert config_context.get(None) is bad


class TestLocalContextImport:
    def test_wrapper_resolves_context_var_at_call_time(self, monkeypatch):
        fake = ContextVar("fake_traigent_config", default=None)
        monkeypatch.setattr(context_module, "config_context", fake)
        provider = SeamlessParameterProvider()
        spy = _Recorder(var=fake)
        provider._seamless_run = spy  # type: ignore[method-assign]
        wrapped = provider.inject_config(lambda: "ok", {"a": 1})

        assert wrapped() == "ok"

        assert isinstance(spy.calls[0]["ctx"], TraigentConfig)
        assert fake.get(None) is None
        assert config_context.get(None) is None

    def test_patched_context_value_is_merged(self, monkeypatch):
        fake = ContextVar("fake_traigent_config", default=None)
        fake.set({"from_fake": 1})
        monkeypatch.setattr(context_module, "config_context", fake)
        provider = SeamlessParameterProvider()
        spy = _Recorder(var=fake)
        provider._seamless_run = spy  # type: ignore[method-assign]
        wrapped = provider.inject_config(lambda: "ok", {"a": 1})

        wrapped()

        assert spy.calls[0]["active"] == {"a": 1, "from_fake": 1}


class TestAsyncSeamless:
    def test_async_function_merges_and_awaits(self, set_ctx):
        set_ctx({"b": 2})
        wrapped, spy, ran = _build({"a": 1}, is_async=True)

        result = asyncio.run(wrapped("x"))

        assert result == "async-result"
        assert spy.calls[0]["active"] == {"a": 1, "b": 2}
        assert ran == [(("x",), {})]

    def test_async_empty_config_calls_original(self):
        wrapped, spy, ran = _build({}, is_async=True)

        assert asyncio.run(wrapped()) == "async-result"

        assert spy.calls == []
        assert len(ran) == 1

    def test_async_context_restored(self, set_ctx):
        previous = {"model": "before"}
        set_ctx(previous)
        wrapped, _, _ = _build({"a": 1}, is_async=True)

        asyncio.run(wrapped())

        assert config_context.get(None) is previous
