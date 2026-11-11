"""Drive PydanticAIHandler against a real ``pydantic_ai.Agent`` (no network).

The mock-based handler tests give ``result.usage`` a callable shape; in the pinned
pydantic-ai major ``AgentRunResult.usage`` is a property returning ``RunUsage``. These
tests use a scripted ``FunctionModel`` so the handler reads the genuine result types.
"""

from __future__ import annotations

import asyncio

import pytest

pydantic_ai = pytest.importorskip("pydantic_ai")

from pydantic_ai import Agent  # noqa: E402
from pydantic_ai.messages import ModelResponse, TextPart  # noqa: E402
from pydantic_ai.models.function import (  # noqa: E402
    AgentInfo,
    FunctionModel,
)

from traigent.integrations.pydantic_ai.handler import (  # noqa: E402
    create_pydantic_ai_handler,
)

pytestmark = pytest.mark.unit


def _respond(messages, info: AgentInfo) -> ModelResponse:
    return ModelResponse(parts=[TextPart("hello")])


async def _stream(messages, info: AgentInfo):
    yield "hel"
    yield "lo"


def _handler():
    agent = Agent(FunctionModel(_respond, stream_function=_stream))
    return create_pydantic_ai_handler(agent, traigent_config={"temperature": 0.31})


def _assert_usage_captured(handler) -> None:
    measures = handler.get_measures_dict()
    assert measures["pydantic_ai_request_count"] == 1
    assert measures["pydantic_ai_total_input_tokens"] > 0
    assert measures["pydantic_ai_total_output_tokens"] > 0
    assert measures["pydantic_ai_total_tokens"] > 0


def test_run_sync_captures_usage_from_real_result() -> None:
    handler = _handler()
    handler.run_sync("q")
    _assert_usage_captured(handler)


def test_run_captures_usage_from_real_result() -> None:
    handler = _handler()
    asyncio.run(handler.run("q"))
    _assert_usage_captured(handler)


def test_run_stream_captures_usage_from_real_stream_result() -> None:
    handler = _handler()

    async def drive() -> None:
        async with handler.run_stream("q") as stream:
            await stream.get_output()

    asyncio.run(drive())
    _assert_usage_captured(handler)


def test_install_hints_match_pinned_major() -> None:
    import traigent.integrations as integrations_pkg
    import traigent.integrations.pydantic_ai as pkg
    from traigent.integrations.pydantic_ai import handler as handler_mod
    from traigent.integrations.pydantic_ai.plugin import PydanticAIPlugin

    meta = PydanticAIPlugin().metadata
    assert meta.requires_packages == ["pydantic-ai>=2.53.0,<3"]
    assert meta.supports_versions == {"pydantic_ai": "2."}
    import inspect

    for mod in (integrations_pkg, pkg, handler_mod):
        src = inspect.getsource(mod)
        assert ">=1,<2" not in src, mod.__name__
