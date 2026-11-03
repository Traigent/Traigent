"""Characterization tests for FrameworkOverrideManager._create_resilient_override."""

from __future__ import annotations

import asyncio
from unittest.mock import patch

import pytest

from traigent.config.context import config_context
from traigent.config.types import TraigentConfig
from traigent.integrations import framework_override as fo
from traigent.integrations.framework_override import FrameworkOverrideManager

MAPPING = {"model": "model_name", "temperature": "temp"}


@pytest.fixture
def manager():
    mgr = FrameworkOverrideManager()
    original = mgr._override_active
    mgr._override_active.enabled = True
    yield mgr
    original.enabled = False


def _sync(*args, **kwargs):
    return ("sync", args, kwargs)


async def _async(*args, **kwargs):
    return ("async", args, kwargs)


def _call(wrapper, *args, **kwargs):
    if asyncio.iscoroutinefunction(wrapper):
        return asyncio.run(wrapper(*args, **kwargs))
    return wrapper(*args, **kwargs)


def _with_config(cfg):
    token = config_context.set(cfg)
    return token


@pytest.fixture(params=[_sync, _async], ids=["sync", "async"])
def method(request):
    return request.param


def _run(manager, method, cfg, *args, mapping=MAPPING, **kwargs):
    wrapper = manager._create_resilient_override(method, "Cls", mapping)
    token = _with_config(cfg)
    try:
        return _call(wrapper, *args, **kwargs)
    finally:
        config_context.reset(token)


def test_wrapper_kind_follows_original(manager):
    assert not asyncio.iscoroutinefunction(
        manager._create_resilient_override(_sync, "C", {})
    )
    assert asyncio.iscoroutinefunction(
        manager._create_resilient_override(_async, "C", {})
    )


def test_disabled_passes_through(manager, method):
    manager._override_active.enabled = False
    out = _run(manager, method, {"model": "m"}, 1, x=2)
    assert out[1:] == ((1,), {"x": 2})


def test_missing_attribute_enabled_passes_through(manager, method):
    manager._override_active = object()  # no ``enabled`` attribute
    wrapper = manager._create_resilient_override(method, "C", MAPPING)
    token = _with_config({"model": "m"})
    try:
        out = _call(wrapper, y=1)
    finally:
        config_context.reset(token)
    assert out[2] == {"y": 1}


@pytest.mark.parametrize("cfg", [None, {}])
def test_no_or_empty_config_passes_through(manager, method, cfg):
    assert _run(manager, method, cfg, a=1)[2] == {"a": 1}


def test_non_dict_non_config_value_passes_through(manager, method):
    assert _run(manager, method, ["not", "a", "config"], a=1)[2] == {"a": 1}


def test_mapping_then_exact_names(manager, method):
    out = _run(manager, method, {"model": "m", "temperature": 0.1, "top_p": 0.9})
    assert out[2] == {
        "model_name": "m",
        "temp": 0.1,
        "model": "m",
        "temperature": 0.1,
        "top_p": 0.9,
    }


def test_mapping_overrides_explicit_kwarg_but_exact_name_does_not(manager, method):
    out = _run(
        manager,
        method,
        {"model": "cfg", "top_p": 0.9},
        1,
        model_name="explicit",
        top_p="explicit",
    )
    assert out[1] == (1,)
    assert out[2]["model_name"] == "cfg"
    assert out[2]["top_p"] == "explicit"
    assert out[2]["model"] == "cfg"


def test_mapping_skips_params_absent_from_config(manager, method):
    out = _run(manager, method, {"model": "m"})
    assert out[2] == {"model_name": "m", "model": "m"}


def test_original_kwargs_not_mutated(manager):
    wrapper = manager._create_resilient_override(_sync, "C", MAPPING)
    kwargs = {"a": 1}
    token = _with_config({"model": "m"})
    try:
        wrapper(**kwargs)
    finally:
        config_context.reset(token)
    assert kwargs == {"a": 1}


def test_traigent_config_includes_custom_params(manager, method):
    cfg = TraigentConfig(model="gpt", custom_params={"special": 7})
    out = _run(manager, method, cfg)
    assert out[2]["model_name"] == "gpt"
    assert out[2]["special"] == 7


def test_get_config_called_once_per_call(manager):
    wrapper = manager._create_resilient_override(_sync, "C", MAPPING)
    with patch.object(fo, "get_config", return_value={"model": "m"}) as gc:
        wrapper()
    assert gc.call_count == 1
    manager._override_active.enabled = False
    with patch.object(fo, "get_config", return_value={"model": "m"}) as gc:
        wrapper()
    assert gc.call_count == 0
