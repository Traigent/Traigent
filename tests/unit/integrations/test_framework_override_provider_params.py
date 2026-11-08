"""Regression tests for #2923 and #2925: tuned params must reach the real SDKs.

Drives the real ``anthropic`` / ``openai`` clients through an in-process
``httpx.MockTransport`` (no network, no keys) and asserts on the request body.
"""

# Traceability: CONC-Layer-Integration FUNC-INTEGRATIONS REQ-INJ-002

from __future__ import annotations

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


def _http_client(response_body: dict, bodies: list[dict]):
    """Build an offline http client that records request bodies."""

    def handler(request):
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json=response_body)

    return httpx.Client(transport=httpx.MockTransport(handler))


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
