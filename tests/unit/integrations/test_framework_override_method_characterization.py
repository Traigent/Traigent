"""Characterization tests for FrameworkOverrideManager._create_override_method."""

from __future__ import annotations

import asyncio

import pytest

from traigent.config.context import config_context, config_space_context
from traigent.config.types import TraigentConfig
from traigent.integrations import framework_override as fo
from traigent.integrations.framework_override import FrameworkOverrideManager

CLS = "test.Client"
METHOD = "gen"


@pytest.fixture
def manager():
    mgr = FrameworkOverrideManager()
    mgr._method_mappings[CLS] = {METHOD: ["temperature", "model", "max_tokens"]}
    mgr._parameter_mappings[CLS] = {"model": "model_name"}
    mgr._method_parameter_translations[CLS] = {METHOD: {"max_tokens": "max_new_tokens"}}
    yield mgr
    mgr._override_active.enabled = False


@pytest.fixture
def ctx():
    space_token = config_space_context.set(None)
    try:
        yield
    finally:
        config_space_context.reset(space_token)


def _sync_method(instance, *args, **kwargs):
    return ("sync", args, kwargs)


async def _async_method(instance, *args, **kwargs):
    return ("async", args, kwargs)


def _call(manager, config, *, space=None, enabled=True, is_async=False, **kwargs):
    original = _async_method if is_async else _sync_method
    wrapped = manager._create_override_method(original, CLS, METHOD)
    manager._override_active.enabled = enabled
    cfg_token = config_context.set(config)
    space_token = config_space_context.set(space)
    try:
        if is_async:
            return asyncio.run(wrapped(object(), "a", **kwargs))
        return wrapped(object(), "a", **kwargs)
    finally:
        config_context.reset(cfg_token)
        config_space_context.reset(space_token)


@pytest.mark.parametrize("is_async", [False, True])
def test_dict_config_overrides_with_translation(manager, ctx, is_async):
    kind, args, kw = _call(
        manager,
        {"temperature": 0.3, "model": "m", "max_tokens": 5, "unrelated": 1},
        is_async=is_async,
        temperature=0.9,
    )
    assert kind == ("async" if is_async else "sync")
    assert args == ("a",)
    assert kw == {"temperature": 0.3, "model_name": "m", "max_new_tokens": 5}


@pytest.mark.parametrize("is_async", [False, True])
def test_traigent_config_uses_custom_params(manager, ctx, is_async):
    cfg = TraigentConfig(custom_params={"temperature": 0.1})
    _, _, kw = _call(manager, cfg, is_async=is_async)
    assert kw["temperature"] == 0.1


@pytest.mark.parametrize("is_async", [False, True])
def test_config_space_restricts_overrides(manager, ctx, is_async):
    _, _, kw = _call(
        manager,
        {"temperature": 0.3, "model": "m"},
        space={"temperature": [0.1]},
        is_async=is_async,
        model_name="orig",
    )
    assert kw == {"temperature": 0.3, "model_name": "orig"}


@pytest.mark.parametrize("is_async", [False, True])
def test_disabled_override_passes_through(manager, ctx, is_async):
    _, _, kw = _call(
        manager, {"temperature": 0.3}, enabled=False, is_async=is_async, x=1
    )
    assert kw == {"x": 1}


@pytest.mark.parametrize("is_async", [False, True])
@pytest.mark.parametrize("config", [{}, ["temperature"], "temperature", 7])
def test_empty_or_unsupported_config_passes_through(manager, ctx, config, is_async):
    _, _, kw = _call(manager, config, is_async=is_async, temperature=0.9)
    assert kw == {"temperature": 0.9}


def test_caller_kwargs_not_mutated(manager, ctx):
    caller_kwargs = {"temperature": 0.9}
    wrapped = manager._create_override_method(_sync_method, CLS, METHOD)
    manager._override_active.enabled = True
    token = config_context.set({"temperature": 0.5})
    try:
        wrapped(object(), **caller_kwargs)
    finally:
        config_context.reset(token)
    assert caller_kwargs == {"temperature": 0.9}


def test_async_wrapper_returned_for_coroutine_function(manager):
    wrapped = manager._create_override_method(_async_method, CLS, METHOD)
    import inspect

    assert inspect.iscoroutinefunction(wrapped)
    assert not inspect.iscoroutinefunction(
        manager._create_override_method(_sync_method, CLS, METHOD)
    )


def test_unknown_method_uses_no_overrides(manager, ctx):
    wrapped = manager._create_override_method(_sync_method, CLS, "nope")
    manager._override_active.enabled = True
    token = config_context.set({"temperature": 0.5})
    try:
        _, _, kw = wrapped(object(), temperature=0.9)
    finally:
        config_context.reset(token)
    assert kw == {"temperature": 0.9}


@pytest.mark.parametrize("is_async", [False, True])
def test_openai_usage_capture_receives_call_kwargs(monkeypatch, is_async, ctx):
    mgr = FrameworkOverrideManager()
    cls = "openai.OpenAI"
    mgr._method_mappings[cls] = {METHOD: ["temperature"]}
    captured = []
    monkeypatch.setattr(
        fo, "_capture_openai_usage", lambda resp, kw: captured.append((resp, dict(kw)))
    )

    async def aorig(instance, *args, **kwargs):
        return "resp"

    def sorig(instance, *args, **kwargs):
        return "resp"

    wrapped = mgr._create_override_method(aorig if is_async else sorig, cls, METHOD)
    mgr._override_active.enabled = True
    token = config_context.set({"temperature": 0.5})
    try:
        if is_async:
            result = asyncio.run(wrapped(object()))
        else:
            result = wrapped(object())
    finally:
        config_context.reset(token)
        mgr._override_active.enabled = False
    assert result == "resp"
    assert captured == [("resp", {"temperature": 0.5})]
