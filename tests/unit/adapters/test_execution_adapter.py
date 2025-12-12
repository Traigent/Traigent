"""Unit tests for LocalExecutionAdapter (TDD: expose current issues)."""

import asyncio
from typing import Any

import pytest

from traigent.adapters.execution_adapter import (
    HybridPlatformAdapter,
    LocalExecutionAdapter,
)
from traigent.evaluators.base import Dataset, EvaluationExample


class _AgentBuilder:
    def __init__(self, agent: Any):
        self._agent = agent

    def build_agent(self, spec: dict[str, Any]):
        # In tests we just return a prepared agent regardless of spec
        return self._agent


class _SyncAgentReturnsAwaitable:
    """Agent whose execute is a sync function returning an awaitable/coroutine."""

    def execute(self, _input):  # noqa: D401
        # Return a coroutine even though this is a sync function
        return asyncio.sleep(0, result="OK")


class _SimpleAgent:
    async def execute(self, _input):  # noqa: D401
        return "OK"


class _HybridAgentBuilder(_AgentBuilder):
    def __init__(self, agent: Any, call_log: list[Any]):
        super().__init__(agent)
        self._call_log = call_log

    def build_platform_agent(self, spec: dict[str, Any], platform_client: Any):
        self._call_log.append(
            ("build_platform_agent", spec.get("platform"), platform_client)
        )
        return self._agent


@pytest.mark.asyncio
async def test_local_adapter_awaits_returned_awaitable_and_evaluates_correctly():
    # Arrange: agent.execute is not a coroutinefunction but returns a coroutine
    agent = _SyncAgentReturnsAwaitable()
    adapter = LocalExecutionAdapter(_AgentBuilder(agent))

    dataset = {
        "examples": [
            {
                "input": {"text": "irrelevant"},
                "expected_output": "OK",
                "metadata": {"evaluation_type": "exact_match"},
            }
        ]
    }

    # Act
    result = await adapter.execute_configuration(
        agent_spec={}, dataset=dataset, trial_id="t1"
    )

    # Assert: accuracy should be 1.0 if the awaitable was awaited and output == expected
    assert result["metrics"]["accuracy"] == 1.0


@pytest.mark.asyncio
async def test_local_adapter_preserves_evaluation_type_and_reports_per_type_accuracy():
    agent = _SimpleAgent()
    adapter = LocalExecutionAdapter(_AgentBuilder(agent))

    dataset = {
        "examples": [
            {
                "input": {"text": "a"},
                "expected_output": "OK",
                "metadata": {"evaluation_type": "exact_match"},
            },
            {
                "input": {"text": "b"},
                "expected_output": "OK",
                "metadata": {"evaluation_type": "exact_match"},
            },
        ]
    }

    result = await adapter.execute_configuration(
        agent_spec={}, dataset=dataset, trial_id="t2"
    )
    # Should include per-type accuracy keyed by evaluation_type
    assert result["metrics"]["accuracy_exact_match"] == 1.0


@pytest.mark.asyncio
async def test_local_adapter_accepts_dataset_object_from_core_types():
    agent = _SimpleAgent()
    adapter = LocalExecutionAdapter(_AgentBuilder(agent))

    examples = [
        EvaluationExample(
            input_data={"text": "x"},
            expected_output="OK",
            metadata={"evaluation_type": "exact_match"},
        ),
        EvaluationExample(
            input_data={"text": "y"},
            expected_output="OK",
            metadata={"evaluation_type": "exact_match"},
        ),
    ]
    ds = Dataset(examples=examples, name="testds")

    result = await adapter.execute_configuration(
        agent_spec={}, dataset=ds, trial_id="t3"
    )
    assert result["metrics"]["accuracy"] == 1.0


@pytest.mark.asyncio
async def test_local_adapter_validates_missing_input_and_raises_clear_error():
    agent = _SimpleAgent()
    adapter = LocalExecutionAdapter(_AgentBuilder(agent))

    bad_dataset = {"examples": [{"expected_output": "OK", "metadata": {}}]}

    with pytest.raises(ValueError) as ei:
        await adapter.execute_configuration(
            agent_spec={}, dataset=bad_dataset, trial_id="t4"
        )
    assert "missing 'input'" in str(ei.value)


@pytest.mark.asyncio
async def test_local_adapter_logs_progress_at_debug_with_configurable_interval(caplog):
    agent = _SimpleAgent()
    # Set interval to 1 so every example triggers a debug log
    adapter = LocalExecutionAdapter(_AgentBuilder(agent), progress_interval=1)

    dataset = {
        "examples": [
            {
                "input": {"text": "a"},
                "expected_output": "OK",
                "metadata": {"evaluation_type": "exact_match"},
            },
            {
                "input": {"text": "b"},
                "expected_output": "OK",
                "metadata": {"evaluation_type": "exact_match"},
            },
        ]
    }

    caplog.set_level("DEBUG")
    await adapter.execute_configuration(agent_spec={}, dataset=dataset, trial_id="t5")
    debug_logs = [
        rec
        for rec in caplog.records
        if rec.levelname == "DEBUG" and "Processed" in rec.getMessage()
    ]
    assert len(debug_logs) >= 2


@pytest.mark.asyncio
async def test_local_adapter_redacts_sensitive_agent_spec_fields():
    agent = _SimpleAgent()
    adapter = LocalExecutionAdapter(_AgentBuilder(agent))

    examples = [
        {
            "input": {"text": "a"},
            "expected_output": "OK",
            "metadata": {"evaluation_type": "exact_match"},
        },
    ]
    dataset = {"examples": examples}

    agent_spec = {"model": "x", "api_key": "SECRET", "nested": {"token": "abc"}}
    result = await adapter.execute_configuration(
        agent_spec=agent_spec, dataset=dataset, trial_id="t6"
    )
    sanitized = result["metadata"].get("agent_spec", {})

    # Values should be redacted
    flat = str(sanitized)
    assert "SECRET" not in flat and "abc" not in flat


@pytest.mark.asyncio
async def test_hybrid_adapter_reuses_prebuilt_agent_and_enriches_metadata():
    call_log: list[Any] = []
    agent = _SimpleAgent()
    builder = _HybridAgentBuilder(agent, call_log)
    platform_client = object()
    adapter = HybridPlatformAdapter(
        platform_client=platform_client, agent_builder=builder
    )

    dataset = {
        "examples": [
            {
                "input": {"text": "hybrid"},
                "expected_output": "OK",
                "metadata": {"evaluation_type": "exact_match"},
            }
        ]
    }
    agent_spec = {
        "platform": "openai",
        "platform_features": ["json_schema"],
    }

    result = await adapter.execute_configuration(
        agent_spec=agent_spec, dataset=dataset, trial_id="hybrid-1"
    )

    # ensure the platform-specific builder was invoked
    assert call_log == [("build_platform_agent", "openai", platform_client)]

    # metadata should include platform context injected by the adapter
    assert result["metadata"]["platform"] == "openai"
    assert result["metadata"]["platform_features"] == ["json_schema"]

    # sanity check: execution completed via local adapter path
    assert result["metrics"]["accuracy"] == 1.0
