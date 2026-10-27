"""Characterization tests for AnthropicPlugin.apply_overrides."""

from __future__ import annotations

import pytest

from traigent.integrations.llms.anthropic_plugin import AnthropicPlugin


@pytest.fixture
def plugin(monkeypatch) -> AnthropicPlugin:
    instance = AnthropicPlugin()
    # Config validation is covered elsewhere; these tests target the
    # Anthropic-specific post-processing of the overridden kwargs.
    monkeypatch.setattr(instance, "validate_config", lambda _config: True)
    return instance


def _run(plugin, kwargs, config=None):
    return plugin.apply_overrides(kwargs, config if config is not None else {})


def test_custom_system_param_is_injected(plugin):
    out = _run(plugin, {"messages": []}, {"custom_params": {"system": "be nice"}})
    assert out["system"] == "be nice"


def test_custom_system_param_does_not_replace_existing(plugin):
    out = _run(
        plugin,
        {"system": "mine", "messages": []},
        {"custom_params": {"system": "other"}},
    )
    assert out["system"] == "mine"


def test_leading_system_message_promoted(plugin):
    msgs = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hi"},
    ]
    out = _run(plugin, {"messages": msgs})
    assert out["messages"] == [{"role": "user", "content": "hi"}]
    assert out["system"] == "sys"


def test_leading_system_message_without_content_gives_empty_system(plugin):
    out = _run(plugin, {"messages": [{"role": "system"}, {"role": "user"}]})
    assert out["system"] == ""
    assert out["messages"] == [{"role": "user"}]


def test_leading_system_message_dropped_but_existing_system_kept(plugin):
    out = _run(
        plugin,
        {"system": "keep", "messages": [{"role": "system", "content": "x"}]},
    )
    assert out["system"] == "keep"
    assert out["messages"] == []


def test_non_system_first_message_untouched(plugin):
    msgs = [{"role": "user", "content": "hi"}]
    out = _run(plugin, {"messages": msgs})
    assert out["messages"] == msgs
    assert "system" not in out


@pytest.mark.parametrize("messages", ["text", b"bytes", [], None, 5])
def test_odd_messages_untouched(plugin, messages):
    out = _run(plugin, {"messages": messages})
    assert out["messages"] == messages
    assert "system" not in out


def test_non_mapping_first_message_untouched(plugin):
    out = _run(plugin, {"messages": ["system", "x"]})
    assert out["messages"] == ["system", "x"]


def test_stop_renamed_to_stop_sequences(plugin):
    out = _run(plugin, {}, {"custom_params": {"stop": ["END"]}})
    assert out["stop_sequences"] == ["END"]


def test_stop_does_not_replace_existing_stop_sequences(plugin):
    out = _run(
        plugin,
        {"stop_sequences": ["a"]},
        {"custom_params": {"stop": ["END"]}},
    )
    assert out["stop_sequences"] == ["a"]


@pytest.mark.parametrize(
    "model,expected",
    [
        ("claude-3-opus", 4096),
        ("claude-2.1", 4096),
        ("claude-instant", 1024),
        ("", 1024),
    ],
)
def test_default_max_tokens_by_model(plugin, model, expected):
    out = _run(plugin, {"model": model})
    assert out["max_tokens"] == expected


def test_default_max_tokens_without_model(plugin):
    assert _run(plugin, {})["max_tokens"] == 1024


def test_existing_max_tokens_kept(plugin):
    assert _run(plugin, {"max_tokens": 7})["max_tokens"] == 7


def test_non_string_model_raises_type_error(plugin):
    with pytest.raises(TypeError):
        _run(plugin, {"model": None})


def test_openai_tools_converted(plugin):
    tools = [
        {
            "type": "function",
            "function": {
                "name": "f",
                "description": "d",
                "parameters": {"type": "object"},
            },
        },
        {"function": {"name": "g"}},
        {"function": {"description": "nameless"}},
        "not-a-mapping",
        {"name": "native", "input_schema": {}},
    ]
    out = _run(plugin, {"tools": tools})
    assert out["tools"] == [
        {"name": "f", "description": "d", "input_schema": {"type": "object"}},
        {"name": "g", "description": "", "input_schema": {}},
        {"name": "native", "input_schema": {}},
    ]


def test_all_tools_invalid_leaves_original_list(plugin):
    tools = ["x", 3]
    assert _run(plugin, {"tools": tools})["tools"] == tools


@pytest.mark.parametrize("tools", ["str", b"b", None, 4, []])
def test_odd_tools_untouched(plugin, tools):
    assert _run(plugin, {"tools": tools})["tools"] == tools
