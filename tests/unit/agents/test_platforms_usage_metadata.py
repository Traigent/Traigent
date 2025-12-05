from typing import Any

import pytest

from traigent.agents.platforms import LangChainAgentExecutor
from traigent.cloud.models import AgentSpecification


class _StubAIMessage:
    def __init__(
        self,
        content: str,
        usage_metadata: dict[str, Any] | None = None,
        response_metadata: dict[str, Any] | None = None,
    ):
        self.content = content
        self.usage_metadata = usage_metadata or {}
        self.response_metadata = response_metadata or {}


class _StubChatOpenAI:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    async def ainvoke(self, messages):
        # Return a stub with token usage so executor can surface it
        usage = {"input_tokens": 12, "output_tokens": 7, "total_tokens": 19}
        resp_meta = {
            "model": self.kwargs.get("model")
            or self.kwargs.get("model_name", "unknown")
        }
        return _StubAIMessage(
            content="ok", usage_metadata=usage, response_metadata=resp_meta
        )


class _StubHumanMessage:
    def __init__(self, content: str):
        self.content = content


@pytest.mark.asyncio
async def test_langchain_executor_surfaces_usage_metadata(monkeypatch):
    executor = LangChainAgentExecutor()

    # Monkeypatch import to use stubs
    def _fake_import(self):
        return _StubChatOpenAI, _StubHumanMessage, True

    monkeypatch.setattr(
        LangChainAgentExecutor, "_import_langchain_components", _fake_import
    )

    agent_spec = AgentSpecification(
        agent_platform="langchain",
        agent_type="conversational",
        prompt_template="{{text}}",
        model_parameters={"model": "gpt-4o-mini"},
    )

    await executor.initialize()
    # Set _langchain_available AFTER initialize(), since _platform_initialize resets it
    executor._langchain_available = True  # type: ignore[attr-defined]
    result = await executor.execute(agent_spec, {"text": "say ok"})

    assert result.output is not None
    # Should surface usage metadata and total_tokens in metadata/tokens_used
    usage = (
        result.metadata.get("usage_metadata")
        if isinstance(result.metadata, dict)
        else None
    )
    assert usage and usage.get("total_tokens") == 19
    assert result.tokens_used == 19
