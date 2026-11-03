"""Characterization of ParameterBasedProvider's applied-vs-wrap-time selection.

Pins the current behaviour of the wrapper built by
``ParameterBasedProvider.inject_config``: which configuration object is
injected, when the wrap-time snapshot is converted, and when a
``ConfigurationContext`` is entered.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from traigent.config import providers as providers_module
from traigent.config.context import (
    ConfigurationContext,
    applied_config_context,
    config_context,
    get_applied_config,
)
from traigent.config.providers import ParameterBasedProvider
from traigent.config.types import TraigentConfig
from traigent.utils.exceptions import ConfigurationError

WRAP_TIME = {"model": "wrap-model", "temperature": 0.25}


@pytest.fixture
def from_dict_calls(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Record every dict handed to ``TraigentConfig.from_dict``."""
    calls: list[dict[str, Any]] = []
    original = TraigentConfig.from_dict

    def _counting(cls: type, config_dict: dict[str, Any]) -> TraigentConfig:
        calls.append(config_dict)
        return original(config_dict)

    monkeypatch.setattr(TraigentConfig, "from_dict", classmethod(_counting))
    return calls


@pytest.fixture
def applied():
    """Set ``applied_config_context`` directly and reset it afterwards."""
    tokens = []

    def _set(value: Any) -> None:
        tokens.append(applied_config_context.set(value))

    yield _set
    for token in reversed(tokens):
        applied_config_context.reset(token)


def _wrap_sync(**provider_kwargs: Any):
    seen: list[dict[str, Any]] = []

    def func(query: str, config: Any = None, **extra: Any) -> str:
        seen.append(
            {
                "config": config,
                "applied_inside": get_applied_config(),
                "extra": extra,
            }
        )
        return f"result:{query}"

    wrapped = ParameterBasedProvider(**provider_kwargs).inject_config(
        func, dict(WRAP_TIME)
    )
    return wrapped, seen


def _wrap_async():
    seen: list[dict[str, Any]] = []

    async def func(query: str, config: Any = None) -> str:
        seen.append({"config": config, "applied_inside": get_applied_config()})
        return f"async:{query}"

    wrapped = ParameterBasedProvider().inject_config(func, dict(WRAP_TIME))
    return wrapped, seen


class TestWrapTimeSnapshot:
    def test_wrap_time_conversion_happens_once_at_wrap(self, from_dict_calls):
        wrapped, _ = _wrap_sync()
        assert from_dict_calls == [WRAP_TIME]

        wrapped("a")
        wrapped("b")

        assert from_dict_calls == [WRAP_TIME]

    def test_no_applied_context_injects_same_wrap_time_object(self):
        wrapped, seen = _wrap_sync()

        assert wrapped("a") == "result:a"
        assert wrapped("b") == "result:b"

        first, second = seen[0]["config"], seen[1]["config"]
        assert isinstance(first, TraigentConfig)
        assert first is second
        assert first.model == "wrap-model"
        assert first.temperature == 0.25

    def test_wrap_time_dict_mutation_does_not_leak(self):
        source = dict(WRAP_TIME)
        seen: list[Any] = []

        def func(config: Any = None) -> None:
            seen.append(config)

        wrapped = ParameterBasedProvider().inject_config(func, source)
        source["model"] = "mutated"
        wrapped()

        assert seen[0].model == "wrap-model"

    def test_custom_param_name_is_used(self):
        seen: list[Any] = []

        def func(cfg: Any = None) -> None:
            seen.append(cfg)

        ParameterBasedProvider().inject_config(func, dict(WRAP_TIME), "cfg")()

        assert seen[0].model == "wrap-model"

    def test_missing_param_raises_configuration_error(self):
        def func(query: str) -> str:
            return query

        with pytest.raises(ConfigurationError, match="does not have parameter"):
            ParameterBasedProvider().inject_config(func, dict(WRAP_TIME))


class TestActiveAppliedSelection:
    def test_traigent_config_applied_is_injected_by_identity(
        self, applied, from_dict_calls
    ):
        wrapped, seen = _wrap_sync()
        before = len(from_dict_calls)
        mine = TraigentConfig(model="applied-model")
        applied(mine)

        wrapped("a")

        assert seen[0]["config"] is mine
        assert len(from_dict_calls) == before

    def test_dict_applied_is_converted_each_call(self, applied, from_dict_calls):
        wrapped, seen = _wrap_sync()
        before = len(from_dict_calls)
        mine = {"model": "dict-model", "temperature": 0.9}
        applied(mine)

        wrapped("a")
        wrapped("b")

        assert from_dict_calls[before:] == [mine, mine]
        first, second = seen[0]["config"], seen[1]["config"]
        assert isinstance(first, TraigentConfig)
        assert first.model == "dict-model"
        assert first is not second
        assert second.model == "dict-model"

    def test_empty_dict_applied_is_still_converted(self, applied, from_dict_calls):
        wrapped, seen = _wrap_sync()
        before = len(from_dict_calls)
        applied({})

        wrapped("a")

        assert from_dict_calls[before:] == [{}]
        assert isinstance(seen[0]["config"], TraigentConfig)
        assert seen[0]["config"].model != "wrap-model"

    @pytest.mark.parametrize("unsupported", [object(), "text", 0, ["x"]])
    def test_unsupported_applied_falls_back_to_wrap_time(
        self, applied, from_dict_calls, unsupported
    ):
        wrapped, seen = _wrap_sync()
        before = len(from_dict_calls)
        applied(unsupported)

        wrapped("a")

        assert seen[0]["config"].model == "wrap-model"
        assert len(from_dict_calls) == before

    def test_unsupported_applied_returns_the_wrap_time_object_itself(self, applied):
        wrapped, seen = _wrap_sync()
        wrapped("a")
        wrap_time_obj = seen[0]["config"]
        applied(object())

        wrapped("b")

        assert seen[1]["config"] is wrap_time_obj


class TestExplicitKwarg:
    def test_explicit_kwarg_is_not_overridden(self):
        wrapped, seen = _wrap_sync()
        explicit = TraigentConfig(model="explicit")

        wrapped("a", config=explicit)

        assert seen[0]["config"] is explicit

    def test_explicit_kwarg_with_applied_context_wins(self, applied):
        wrapped, seen = _wrap_sync()
        explicit = {"anything": 1}
        applied(TraigentConfig(model="applied"))

        wrapped("a", config=explicit)

        assert seen[0]["config"] is explicit

    def test_other_kwargs_pass_through(self):
        wrapped, seen = _wrap_sync()

        wrapped("a", flag=True)

        assert seen[0]["extra"] == {"flag": True}


class TestContextEntry:
    def test_context_entered_when_nothing_applied(self):
        wrapped, seen = _wrap_sync()
        assert get_applied_config() is None

        wrapped("a")

        inside = seen[0]["applied_inside"]
        assert inside is seen[0]["config"]
        assert get_applied_config() is None

    def test_context_entered_for_explicit_kwarg_too(self):
        wrapped, seen = _wrap_sync()
        wrapped("a", config={"x": 1})

        assert isinstance(seen[0]["applied_inside"], TraigentConfig)
        assert seen[0]["applied_inside"].model == "wrap-model"

    def test_context_not_entered_when_applied(self, applied, monkeypatch):
        entered: list[Any] = []
        real = providers_module.ConfigurationContext

        class _Spy(real):
            def __init__(self, cfg: Any) -> None:
                entered.append(cfg)
                super().__init__(cfg)

        monkeypatch.setattr(providers_module, "ConfigurationContext", _Spy)
        wrapped, seen = _wrap_sync()
        mine = TraigentConfig(model="applied")
        applied(mine)

        wrapped("a")

        assert entered == []
        assert seen[0]["applied_inside"] is mine

    def test_context_entered_with_effective_config(self, monkeypatch):
        entered: list[Any] = []
        real = providers_module.ConfigurationContext

        class _Spy(real):
            def __init__(self, cfg: Any) -> None:
                entered.append(cfg)
                super().__init__(cfg)

        monkeypatch.setattr(providers_module, "ConfigurationContext", _Spy)
        wrapped, seen = _wrap_sync()

        wrapped("a")

        assert len(entered) == 1
        assert entered[0] is seen[0]["config"]

    def test_context_reset_after_exception(self):
        def func(config: Any = None) -> None:
            raise RuntimeError("boom")

        wrapped = ParameterBasedProvider().inject_config(func, dict(WRAP_TIME))
        before = config_context.get(None)

        with pytest.raises(RuntimeError, match="boom"):
            wrapped()

        assert get_applied_config() is None
        assert config_context.get(None) is before

    def test_exception_propagates_when_applied(self, applied):
        def func(config: Any = None) -> None:
            raise ValueError("inner")

        wrapped = ParameterBasedProvider().inject_config(func, dict(WRAP_TIME))
        applied(TraigentConfig(model="m"))

        with pytest.raises(ValueError, match="inner"):
            wrapped()

    def test_user_context_applied_is_respected(self, from_dict_calls):
        wrapped, seen = _wrap_sync()
        mine = TraigentConfig(model="user-ctx")

        with ConfigurationContext(mine):
            wrapped("a")

        assert seen[0]["config"] is mine


class _RaisingVar:
    def __init__(self, error: BaseException) -> None:
        self._error = error

    def get(self) -> Any:
        raise self._error


class TestAppliedLookup:
    def test_lookup_error_means_nothing_applied(self, monkeypatch):
        monkeypatch.setattr(
            providers_module, "applied_config_context", _RaisingVar(LookupError())
        )
        wrapped, seen = _wrap_sync()

        wrapped("a")

        assert seen[0]["config"].model == "wrap-model"

    def test_other_errors_propagate_and_func_not_called(self, monkeypatch):
        monkeypatch.setattr(
            providers_module,
            "applied_config_context",
            _RaisingVar(RuntimeError("lookup broke")),
        )
        wrapped, seen = _wrap_sync()

        with pytest.raises(RuntimeError, match="lookup broke"):
            wrapped("a")

        assert seen == []


class TestAsyncWrapper:
    def test_async_function_gets_wrap_time_default(self):
        wrapped, seen = _wrap_async()

        result = asyncio.run(wrapped("a"))

        assert result == "async:a"
        assert seen[0]["config"].model == "wrap-model"
        assert seen[0]["applied_inside"] is seen[0]["config"]
        assert get_applied_config() is None

    def test_async_function_prefers_applied(self, applied):
        wrapped, seen = _wrap_async()
        mine = TraigentConfig(model="async-applied")
        applied(mine)

        asyncio.run(wrapped("a"))

        assert seen[0]["config"] is mine
        assert seen[0]["applied_inside"] is mine

    def test_async_dict_applied_is_converted(self, applied):
        wrapped, seen = _wrap_async()
        applied({"model": "async-dict"})

        asyncio.run(wrapped("a"))

        assert seen[0]["config"].model == "async-dict"

    def test_async_unsupported_applied_falls_back(self, applied):
        wrapped, seen = _wrap_async()
        applied(object())

        asyncio.run(wrapped("a"))

        assert seen[0]["config"].model == "wrap-model"

    def test_async_explicit_kwarg_wins(self):
        wrapped, seen = _wrap_async()
        explicit = object()

        asyncio.run(wrapped("a", config=explicit))

        assert seen[0]["config"] is explicit

    def test_async_exception_resets_context(self):
        async def func(config: Any = None) -> None:
            raise RuntimeError("async boom")

        wrapped = ParameterBasedProvider().inject_config(func, dict(WRAP_TIME))

        with pytest.raises(RuntimeError, match="async boom"):
            asyncio.run(wrapped())

        assert get_applied_config() is None

    def test_async_wrapper_is_coroutine_function(self):
        import inspect

        wrapped, _ = _wrap_async()
        sync_wrapped, _ = _wrap_sync()

        assert inspect.iscoroutinefunction(wrapped)
        assert not inspect.iscoroutinefunction(sync_wrapped)
