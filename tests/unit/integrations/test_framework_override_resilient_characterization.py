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
    # Changed (#3150): config keys the vendor does not name (temperature,
    # top_p on a **kwargs-only stub) are no longer blanket-copied.
    out = _run(manager, method, {"model": "m", "temperature": 0.1, "top_p": 0.9})
    assert out[2] == {"model_name": "m", "temp": 0.1}


def _sync_named(*, model=None, top_p=None, **kwargs):
    return ("sync", (), {"model": model, "top_p": top_p, **kwargs})


def test_exact_names_forwarded_only_when_vendor_names_them(manager):
    # New (#3150): unmapped keys are forwarded only if the signature names them.
    out = _run(
        manager,
        _sync_named,
        {"model": "m", "top_p": 0.9, "unknown": 1},
        mapping={},
    )
    assert out[2] == {"model": "m", "top_p": 0.9}


def test_mapping_overrides_explicit_kwarg_but_exact_name_does_not(manager, method):
    # Changed (#3150): top_p is not named by the stub, so it is not copied and
    # the explicit kwarg stays; mapped model_name still wins; no raw "model".
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
    assert "model" not in out[2]


def test_named_exact_name_does_not_override_explicit_kwarg(manager):
    # New (#3150): an exactly-named parameter never clobbers an explicit kwarg.
    out = _run(manager, _sync_named, {"top_p": 0.9}, mapping={}, top_p="explicit")
    assert out[2]["top_p"] == "explicit"


def test_mapping_skips_params_absent_from_config(manager, method):
    # Changed (#3150): only the mapped name is sent, not the raw "model" key.
    out = _run(manager, method, {"model": "m"})
    assert out[2] == {"model_name": "m"}


def test_original_kwargs_not_mutated(manager):
    wrapper = manager._create_resilient_override(_sync, "C", MAPPING)
    kwargs = {"a": 1}
    token = _with_config({"model": "m"})
    try:
        wrapper(**kwargs)
    finally:
        config_context.reset(token)
    assert kwargs == {"a": 1}


def test_traigent_config_custom_params_not_blanket_copied(manager, method):
    cfg = TraigentConfig(model="gpt", custom_params={"special": 7})
    out = _run(manager, method, cfg)
    assert out[2]["model_name"] == "gpt"
    # Changed (#3150): custom params are not forwarded unless the vendor names them.
    assert "special" not in out[2]


def test_get_config_called_once_per_call(manager):
    wrapper = manager._create_resilient_override(_sync, "C", MAPPING)
    with patch.object(fo, "get_config", return_value={"model": "m"}) as gc:
        wrapper()
    assert gc.call_count == 1
    manager._override_active.enabled = False
    with patch.object(fo, "get_config", return_value={"model": "m"}) as gc:
        wrapper()
    assert gc.call_count == 0
