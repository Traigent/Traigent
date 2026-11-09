"""Regression tests for #2923 and #2925: tuned params must reach the real SDKs.

Drives the real ``anthropic`` / ``openai`` clients through an in-process
``httpx.MockTransport`` (no network, no keys) and asserts on the request body.
"""

# Traceability: CONC-Layer-Integration FUNC-INTEGRATIONS REQ-INJ-002

from __future__ import annotations

import importlib
import json

import httpx
import pytest

from traigent.config.context import set_config, set_config_space  # noqa: E402
from traigent.config.types import TraigentConfig  # noqa: E402
from traigent.integrations.framework_override import (  # noqa: E402
    FrameworkOverrideManager,
)

OPENAI_SPACE = {
    "model": ["gpt-4o-mini"],
    "max_tokens": [77],
    "frequency_penalty": [0.5],
    "presence_penalty": [0.4],
    "seed": [9],
    "stop": [["ZZ"]],
}


def _httpx_module_for(sdk: str):
    """Return the httpx flavour the installed ``sdk`` accepts as ``http_client``.

    Newer anthropic releases reject a plain ``httpx.Client`` (``_reject_httpx_object``)
    and require ``httpx2``; older releases and openai accept ``httpx``.
    """
    base_client = importlib.import_module(f"{sdk}._base_client")
    if hasattr(base_client, "_reject_httpx_object"):
        return importlib.import_module("httpx2")
    return httpx


def _http_client(response_body: dict, bodies: list[dict], sdk: str = "openai"):
    """Build an offline http client that records request bodies."""
    lib = _httpx_module_for(sdk)

    def handler(request):
        bodies.append(json.loads(request.content))
        return lib.Response(200, json=response_body)

    return lib.Client(transport=lib.MockTransport(handler))


@pytest.fixture
def manager():
    mgr = FrameworkOverrideManager()
    yield mgr
    mgr.deactivate_overrides()


def test_anthropic_max_tokens_reaches_messages_create(manager):
    anthropic = pytest.importorskip("anthropic")
    bodies: list[dict] = []
    http_client = _http_client(
        {
            "id": "m",
            "type": "message",
            "role": "assistant",
            "model": "x",
            "content": [{"type": "text", "text": "hi"}],
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {"input_tokens": 1, "output_tokens": 1},
        },
        bodies,
        "anthropic",
    )
    client = anthropic.Anthropic(api_key="k", http_client=http_client)
    manager.activate_overrides(["anthropic.Anthropic"])
    set_config(
        TraigentConfig(
            model="claude-sonnet-4-5",
            max_tokens=77,
            custom_params={"stop_sequences": ["ZZ"]},
        )
    )
    set_config_space(
        {"model": ["claude-sonnet-4-5"], "max_tokens": [77], "stop_sequences": [["ZZ"]]}
    )
    try:
        client.messages.create(
            model="x", max_tokens=5, messages=[{"role": "user", "content": "q"}]
        )
    finally:
        set_config_space(None)
    assert bodies[0]["max_tokens"] == 77
    assert "max_tokens_to_sample" not in bodies[0]
    assert bodies[0]["stop_sequences"] == ["ZZ"]


def test_anthropic_class_mapping_has_no_legacy_max_tokens_name():
    from traigent.integrations.mappings import PARAMETER_MAPPINGS

    for cls in ("anthropic.Anthropic", "anthropic.AsyncAnthropic"):
        assert "max_tokens" not in PARAMETER_MAPPINGS[cls]


def test_openai_tuned_penalties_seed_stop_are_sent(manager):
    openai = pytest.importorskip("openai")
    bodies: list[dict] = []
    http_client = _http_client(
        {
            "id": "c",
            "object": "chat.completion",
            "created": 0,
            "model": "x",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": "hi"},
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        },
        bodies,
    )
    client = openai.OpenAI(api_key="k", http_client=http_client)
    manager.activate_overrides(["openai.OpenAI"])
    set_config(
        TraigentConfig(
            model="gpt-4o-mini",
            max_tokens=77,
            custom_params={
                "frequency_penalty": 0.5,
                "presence_penalty": 0.4,
                "seed": 9,
                "stop": ["ZZ"],
            },
        )
    )
    set_config_space(OPENAI_SPACE)
    try:
        client.chat.completions.create(
            model="x", messages=[{"role": "user", "content": "q"}]
        )
    finally:
        set_config_space(None)
    body = bodies[0]
    assert body["frequency_penalty"] == 0.5
    assert body["presence_penalty"] == 0.4
    assert body["seed"] == 9
    assert body["stop"] == ["ZZ"]
    assert body["max_tokens"] == 77


def test_langchain_openai_methods_accept_seed_and_penalties():
    from traigent.integrations.mappings import METHOD_MAPPINGS

    for method in ("invoke", "stream", "astream"):
        params = METHOD_MAPPINGS["langchain_openai.ChatOpenAI"][method]
        for name in ("seed", "frequency_penalty", "presence_penalty"):
            assert name in params


ANTHROPIC_RESPONSE = {
    "id": "m",
    "type": "message",
    "role": "assistant",
    "model": "x",
    "content": [{"type": "text", "text": "hi"}],
    "stop_reason": "end_turn",
    "stop_sequence": None,
    "usage": {"input_tokens": 1, "output_tokens": 1},
}
OPENAI_RESPONSE = {
    "id": "c",
    "object": "chat.completion",
    "created": 0,
    "model": "x",
    "choices": [
        {
            "index": 0,
            "finish_reason": "stop",
            "message": {"role": "assistant", "content": "hi"},
        }
    ],
    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
}


@pytest.mark.parametrize("name", ["stop", "stop_sequences"])
def test_stop_alias_reaches_anthropic_as_stop_sequences(manager, name):
    anthropic = pytest.importorskip("anthropic")
    bodies: list[dict] = []
    client = anthropic.Anthropic(
        api_key="k", http_client=_http_client(ANTHROPIC_RESPONSE, bodies, "anthropic")
    )
    manager.activate_overrides(["anthropic.Anthropic"])
    set_config(TraigentConfig(model="claude-sonnet-4-5", custom_params={name: ["ZZ"]}))
    set_config_space({"model": ["claude-sonnet-4-5"], name: [["ZZ"]]})
    try:
        client.messages.create(
            model="x", max_tokens=5, messages=[{"role": "user", "content": "q"}]
        )
    finally:
        set_config_space(None)
    assert bodies[0]["stop_sequences"] == ["ZZ"]
    assert "stop" not in bodies[0]


@pytest.mark.parametrize("name", ["stop", "stop_sequences"])
def test_stop_alias_reaches_openai_as_stop(manager, name):
    openai = pytest.importorskip("openai")
    bodies: list[dict] = []
    client = openai.OpenAI(
        api_key="k", http_client=_http_client(OPENAI_RESPONSE, bodies)
    )
    manager.activate_overrides(["openai.OpenAI"])
    set_config(TraigentConfig(model="gpt-4o-mini", custom_params={name: ["ZZ"]}))
    set_config_space({"model": ["gpt-4o-mini"], name: [["ZZ"]]})
    try:
        client.chat.completions.create(
            model="x", messages=[{"role": "user", "content": "q"}]
        )
    finally:
        set_config_space(None)
    assert bodies[0]["stop"] == ["ZZ"]
    assert "stop_sequences" not in bodies[0]


@pytest.mark.parametrize("method", ["invoke", "ainvoke"])
def test_langchain_openai_request_body_gets_seed_penalties_stop(manager, method):
    import asyncio

    pytest.importorskip("langchain_openai")
    from langchain_openai import ChatOpenAI

    bodies: list[dict] = []
    sync_client = _http_client(OPENAI_RESPONSE, bodies)

    def handler(request):
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json=OPENAI_RESPONSE)

    async_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        api_key="k",
        http_client=sync_client,
        http_async_client=async_client,
    )
    manager.activate_overrides(["langchain_openai.ChatOpenAI"])
    set_config(
        TraigentConfig(
            model="gpt-4o-mini",
            custom_params={
                "seed": 9,
                "frequency_penalty": 0.5,
                "presence_penalty": 0.4,
                "stop": ["ZZ"],
            },
        )
    )
    set_config_space(
        {
            "model": ["gpt-4o-mini"],
            "seed": [9],
            "frequency_penalty": [0.5],
            "presence_penalty": [0.4],
            "stop": [["ZZ"]],
        }
    )
    try:
        if method == "invoke":
            llm.invoke("q")
        else:
            asyncio.run(llm.ainvoke("q"))
    finally:
        set_config_space(None)
    body = bodies[0]
    assert body["seed"] == 9
    assert body["frequency_penalty"] == 0.5
    assert body["presence_penalty"] == 0.4
    assert body["stop"] == ["ZZ"]


def test_canonical_stop_wins_over_alias_for_every_provider(manager):
    from traigent.integrations.mappings import METHOD_MAPPINGS

    for cls, methods in METHOD_MAPPINGS.items():
        for method, params in methods.items():
            if "stop" in params and "stop_sequences" in params:
                assert params.index("stop") > params.index("stop_sequences"), (
                    cls,
                    method,
                )


def test_openai_both_stop_names_send_canonical_value(manager):
    openai = pytest.importorskip("openai")
    bodies: list[dict] = []
    client = openai.OpenAI(
        api_key="k", http_client=_http_client(OPENAI_RESPONSE, bodies)
    )
    manager.activate_overrides(["openai.OpenAI"])
    set_config(
        TraigentConfig(
            model="m", custom_params={"stop": ["A"], "stop_sequences": ["B"]}
        )
    )
    set_config_space({"model": ["m"], "stop": [["A"]], "stop_sequences": [["B"]]})
    try:
        client.chat.completions.create(
            model="x", messages=[{"role": "user", "content": "q"}]
        )
    finally:
        set_config_space(None)
    assert bodies[0]["stop"] == ["A"]


def test_cached_property_annotation_resolves_via_lazy_module_getattr(
    manager, monkeypatch
):
    """openai>=3 exposes resource classes through a module ``__getattr__``."""
    import functools
    import sys
    import types

    class Resource:
        def create(self):
            return "ok"

    fake = types.ModuleType("lazysdk.resources")
    fake.__getattr__ = lambda name: Resource if name == "Resource" else _missing(name)

    def _missing(name):
        raise AttributeError(name)

    monkeypatch.setitem(sys.modules, "lazysdk", types.ModuleType("lazysdk"))
    monkeypatch.setitem(sys.modules, "lazysdk.resources", fake)

    def resource(self) -> Resource:  # noqa: F821 - resolved lazily by design
        raise NotImplementedError

    resource.__module__ = "lazysdk.client"
    descriptor = functools.cached_property(resource)

    assert manager._cached_property_return_class(descriptor) is Resource
