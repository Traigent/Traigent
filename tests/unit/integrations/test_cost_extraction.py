"""Integration-format cost extraction tests."""

from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest

from traigent.evaluators.metrics_tracker import extract_llm_metrics
from traigent.integrations.bedrock_client import BedrockChatResponse


@pytest.fixture(autouse=True)
def _disable_mock_mode(monkeypatch):
    monkeypatch.setenv("TRAIGENT_MOCK_LLM", "")
    monkeypatch.setenv("TRAIGENT_GENERATE_MOCKS", "")


def test_openai_response_cost_extraction() -> None:
    usage = SimpleNamespace(prompt_tokens=100, completion_tokens=50, total_tokens=150)
    response = SimpleNamespace(usage=usage)

    metrics = extract_llm_metrics(response, model_name="gpt-4o")

    assert metrics.tokens.input_tokens == 100
    assert metrics.tokens.output_tokens == 50
    assert metrics.cost.total_cost > 0


def test_anthropic_response_cost_extraction() -> None:
    usage = SimpleNamespace(input_tokens=100, output_tokens=50)
    response = SimpleNamespace(usage=usage, model="claude-3-5-sonnet-20241022")

    metrics = extract_llm_metrics(response, model_name="claude-3-5-sonnet-20241022")

    assert metrics.tokens.input_tokens == 100
    assert metrics.tokens.output_tokens == 50
    assert metrics.cost.total_cost > 0


def test_langchain_response_cost_extraction() -> None:
    response = SimpleNamespace(
        llm_output={
            "token_usage": {
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "total_tokens": 150,
            }
        }
    )

    metrics = extract_llm_metrics(response, model_name="gpt-4o")

    assert metrics.tokens.total_tokens == 150
    assert metrics.cost.total_cost > 0


def test_bedrock_non_streaming_cost_extraction() -> None:
    response = BedrockChatResponse(
        text="ok",
        raw={"usage": {"inputTokens": 100, "outputTokens": 50}},
        usage={"inputTokens": 100, "outputTokens": 50},
    )

    metrics = extract_llm_metrics(response, model_name="claude-3-5-sonnet-20241022")

    assert metrics.tokens.input_tokens == 100
    assert metrics.tokens.output_tokens == 50
    assert metrics.cost.total_cost > 0


def test_bedrock_streaming_without_usage_warns(caplog) -> None:
    response = BedrockChatResponse(text="ok", raw={"streamed": True}, usage=None)

    with caplog.at_level(logging.WARNING, logger="traigent.evaluators.metrics_tracker"):
        metrics = extract_llm_metrics(response, model_name="claude-3-5-sonnet-20241022")

    assert metrics.cost.total_cost == 0.0
    assert any("No token usage extracted" in r.message for r in caplog.records)


def test_unknown_model_with_usage_warns(caplog) -> None:
    usage = SimpleNamespace(prompt_tokens=100, completion_tokens=50, total_tokens=150)
    response = SimpleNamespace(usage=usage)

    with caplog.at_level(logging.WARNING):
        metrics = extract_llm_metrics(response, model_name="totally-fake-model-xyz")

    assert metrics.tokens.total_tokens == 150
    assert metrics.cost.total_cost == 0.0
    assert any("Unknown model" in r.message for r in caplog.records)


def test_unknown_model_strict_accounting_false_integration_path(
    monkeypatch,
    caplog,
) -> None:
    """Unknown model with strict mode disabled should warn and stay non-fatal."""
    monkeypatch.setenv("TRAIGENT_STRICT_COST_ACCOUNTING", "false")
    usage = SimpleNamespace(prompt_tokens=120, completion_tokens=30, total_tokens=150)
    response = SimpleNamespace(usage=usage)

    with caplog.at_level(logging.WARNING):
        metrics = extract_llm_metrics(response, model_name="nonexistent-model-for-test")

    assert metrics.tokens.total_tokens == 150
    assert metrics.cost.total_cost == 0.0
    assert any("Unknown model" in r.message for r in caplog.records)
