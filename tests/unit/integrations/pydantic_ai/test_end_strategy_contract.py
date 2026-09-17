"""Agent contract test pinning pydantic-ai tool-execution semantics.

PR #2170 raised the ``pydantic-ai-slim`` floor to ``>=2.28.0,<3`` to close a CVE. That is a
behavioural major, not just an import-compatible bump: ``pydantic_ai.Agent``'s ``end_strategy``
default flipped from ``"early"`` (v1) to ``"graceful"`` (v2). Under the old default, a function
tool called alongside an output tool that already succeeded was *skipped*; under the new default
it *runs*. See #2172.

The existing adapter tests (``test_handler.py``, ``test_plugin.py``, ``test_imports.py``) mock
``pydantic_ai.Agent`` entirely, so none of them would notice this flip. This module drives a real
``Agent`` with a scripted ``FunctionModel`` to pin both the semantics Traigent relies on
(``end_strategy="early"``) and the new installed default, so a future dependency bump that changes
this behaviour again fails loudly here instead of silently.
"""

from __future__ import annotations

import pytest

pydantic_ai = pytest.importorskip("pydantic_ai")

from pydantic import BaseModel  # noqa: E402
from pydantic_ai import Agent  # noqa: E402
from pydantic_ai.messages import ModelMessage, ModelResponse, ToolCallPart  # noqa: E402
from pydantic_ai.models.function import AgentInfo, FunctionModel  # noqa: E402


class _Answer(BaseModel):
    value: str


def _make_agent(end_strategy: str) -> tuple[Agent, list[str]]:
    """Build an Agent whose scripted model response calls a function tool *and* the
    output tool in the same turn, so `end_strategy` alone decides whether the function
    tool runs.
    """
    calls: list[str] = []

    def model_function(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(
            parts=[
                # Function tool call listed first ("precedes" the output tool, per
                # pydantic_ai's own EndStrategy docstring) so `graceful` would run it
                # before ending, and `early` must skip it outright.
                ToolCallPart(tool_name="side_effect", args={}, tool_call_id="fn-1"),
                ToolCallPart(
                    tool_name="final_result",
                    args={"value": "done"},
                    tool_call_id="out-1",
                ),
            ]
        )

    agent: Agent = Agent(
        FunctionModel(model_function),
        output_type=_Answer,
        end_strategy=end_strategy,
    )

    @agent.tool_plain
    def side_effect() -> str:
        calls.append("ran")
        return "did it"

    return agent, calls


class TestEndStrategyContract:
    """Pin the `end_strategy` tool-execution semantics Traigent's handler layer relies on.

    `PydanticAIHandler` (traigent/integrations/pydantic_ai/handler.py) forwards whatever
    `Agent` the caller constructed; it does not set `end_strategy` itself. These tests do not
    change that — they document, at the `pydantic_ai.Agent` boundary, which behaviour is v1
    ("early", the semantics Traigent's docs/examples assumed) and which is now the installed
    default ("graceful"), so the two can never silently drift back together unnoticed.
    """

    def test_early_skips_function_tool_once_output_tool_succeeds(self) -> None:
        """v1 semantics: `end_strategy="early"` must not execute a function tool that was
        requested alongside an output tool call that already succeeded.
        """
        agent, calls = _make_agent("early")

        result = agent.run_sync("go")

        assert calls == []
        assert result.output == _Answer(value="done")

    def test_graceful_runs_function_tool_before_ending(self) -> None:
        """v2 default: `end_strategy="graceful"` runs a function tool that precedes the
        output tool call before the run ends. This is the flipped default #2172 warns about —
        code written against the v1 default must now pass `end_strategy="early"` explicitly to
        keep the old behaviour.
        """
        agent, calls = _make_agent("graceful")

        result = agent.run_sync("go")

        assert calls == ["ran"]
        assert result.output == _Answer(value="done")

    def test_agent_default_end_strategy_is_graceful(self) -> None:
        """Pin the installed library's default so a future pydantic-ai bump that reverts or
        changes it again fails here first, rather than silently in production behaviour.
        """
        agent = Agent(FunctionModel(lambda messages, info: ModelResponse(parts=[])))

        assert agent.end_strategy == "graceful"
