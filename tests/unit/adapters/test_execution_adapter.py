"""Unit tests for LocalExecutionAdapter (TDD: expose current issues)."""

import asyncio
from typing import Any
from unittest.mock import AsyncMock, Mock

import pytest

from traigent.adapters.execution_adapter import (
    HybridPlatformAdapter,
    LocalExecutionAdapter,
    RemoteExecutionAdapter,
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
async def test_local_adapter_unknown_evaluation_type_fails_closed():
    agent = _SimpleAgent()
    adapter = LocalExecutionAdapter(_AgentBuilder(agent))

    dataset = {
        "examples": [
            {
                "input": {"text": "valid"},
                "expected_output": "OK",
                "metadata": {"evaluation_type": "exact_match"},
            },
            {
                "input": {"text": "a"},
                "expected_output": "OK",
                "metadata": {"evaluation_type": "bogus"},
            },
        ]
    }

    result = await adapter.execute_configuration(
        agent_spec={}, dataset=dataset, trial_id="unknown-eval"
    )

    assert result["metrics"]["accuracy"] == 1.0
    assert result["metrics"]["success_rate"] == 0.5
    assert result["metrics"]["examples_with_ground_truth"] == 1.0
    assert result["metrics"]["accuracy_exact_match"] == 1.0
    assert "accuracy_bogus" not in result["metrics"]
    assert result["metadata"]["failures"] == 1


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


# ==============================================================================
# RemoteExecutionAdapter Tests
# ==============================================================================


@pytest.mark.asyncio
async def test_remote_adapter_executes_configuration_via_backend():
    """Test RemoteExecutionAdapter delegates to backend client."""
    # Mock backend client
    backend_client = Mock()
    backend_client.execute_configuration = AsyncMock(
        return_value={
            "trial_id": "remote-1",
            "metrics": {"accuracy": 0.85, "latency": 1.2},
            "execution_time": 5.5,
            "metadata": {"backend": "cloud", "examples_processed": 10},
        }
    )

    adapter = RemoteExecutionAdapter(backend_client)

    # Dataset with dataset_id (required for remote execution)
    dataset = {"dataset_id": "ds-12345"}
    agent_spec = {"model": "gpt-4", "temperature": 0.7}

    # Execute
    result = await adapter.execute_configuration(
        agent_spec=agent_spec, dataset=dataset, trial_id="remote-1"
    )

    # Verify backend client was called correctly
    backend_client.execute_configuration.assert_called_once_with(
        trial_id="remote-1", agent_spec=agent_spec, dataset_id="ds-12345"
    )

    # Verify result structure
    assert result["trial_id"] == "remote-1"
    assert result["metrics"]["accuracy"] == 0.85
    assert result["metadata"]["backend"] == "cloud"


@pytest.mark.asyncio
async def test_remote_adapter_raises_error_without_dataset_id():
    """Test RemoteExecutionAdapter requires dataset_id."""
    backend_client = Mock()
    adapter = RemoteExecutionAdapter(backend_client)

    # Dataset without dataset_id
    dataset = {"examples": [{"input": "test"}]}
    agent_spec = {"model": "gpt-4"}

    # Should raise ValueError
    with pytest.raises(ValueError) as exc_info:
        await adapter.execute_configuration(
            agent_spec=agent_spec, dataset=dataset, trial_id="remote-2"
        )

    assert "Dataset ID required for remote execution" in str(exc_info.value)


@pytest.mark.asyncio
async def test_remote_adapter_get_execution_mode():
    """Test RemoteExecutionAdapter returns hybrid_api execution mode."""
    backend_client = Mock()
    adapter = RemoteExecutionAdapter(backend_client)

    mode = await adapter.get_execution_mode()
    assert mode == "hybrid_api"


@pytest.mark.asyncio
async def test_remote_adapter_handles_backend_errors():
    """Test RemoteExecutionAdapter propagates backend errors."""
    backend_client = Mock()
    backend_client.execute_configuration = AsyncMock(
        side_effect=Exception("Backend unavailable")
    )

    adapter = RemoteExecutionAdapter(backend_client)
    dataset = {"dataset_id": "ds-12345"}
    agent_spec = {"model": "gpt-4"}

    # Should propagate the exception
    with pytest.raises(Exception) as exc_info:
        await adapter.execute_configuration(
            agent_spec=agent_spec, dataset=dataset, trial_id="remote-3"
        )

    assert "Backend unavailable" in str(exc_info.value)


# ==============================================================================
# Additional Evaluation Types Tests
# ==============================================================================


@pytest.mark.asyncio
async def test_local_adapter_contains_evaluation_type():
    """Test LocalExecutionAdapter handles 'contains' evaluation type."""
    agent = _SimpleAgent()
    adapter = LocalExecutionAdapter(_AgentBuilder(agent))

    dataset = {
        "examples": [
            {
                "input": {"text": "test"},
                "expected_output": "OK",
                "metadata": {"evaluation_type": "contains"},
            }
        ]
    }

    result = await adapter.execute_configuration(
        agent_spec={}, dataset=dataset, trial_id="contains-1"
    )

    # "OK" is contained in "OK", so accuracy should be 1.0
    assert result["metrics"]["accuracy"] == 1.0
    assert result["metrics"]["accuracy_contains"] == 1.0


@pytest.mark.asyncio
async def test_local_adapter_numeric_evaluation_type():
    """Test LocalExecutionAdapter handles 'numeric' evaluation type."""

    class NumericAgent:
        async def execute(self, _input):
            return "42.5"

    agent = NumericAgent()
    adapter = LocalExecutionAdapter(_AgentBuilder(agent))

    dataset = {
        "examples": [
            {
                "input": {"text": "calculate"},
                "expected_output": "42.49",
                "metadata": {"evaluation_type": "numeric", "tolerance": 0.1},
            }
        ]
    }

    result = await adapter.execute_configuration(
        agent_spec={}, dataset=dataset, trial_id="numeric-1"
    )

    # 42.5 is within 0.1 of 42.49
    assert result["metrics"]["accuracy"] == 1.0
    assert result["metrics"]["accuracy_numeric"] == 1.0


@pytest.mark.asyncio
async def test_local_adapter_numeric_evaluation_with_failure():
    """Test LocalExecutionAdapter handles numeric evaluation failures."""

    class NumericAgent:
        async def execute(self, _input):
            return "100.0"

    agent = NumericAgent()
    adapter = LocalExecutionAdapter(_AgentBuilder(agent))

    dataset = {
        "examples": [
            {
                "input": {"text": "calculate"},
                "expected_output": "42.0",
                "metadata": {"evaluation_type": "numeric", "tolerance": 0.01},
            }
        ]
    }

    result = await adapter.execute_configuration(
        agent_spec={}, dataset=dataset, trial_id="numeric-2"
    )

    # 100.0 is not within 0.01 of 42.0
    assert result["metrics"]["accuracy"] == 0.0
    assert result["metrics"]["accuracy_numeric"] == 0.0


@pytest.mark.asyncio
async def test_local_adapter_semantic_eval_fails_loud_without_configured_evaluator(
    caplog,
):
    """Issue #891: semantic evaluation_type must fail loudly when the
    LocalExecutionAdapter has no semantic evaluator configured. Previously
    the adapter silently set ``correct=None`` so paraphrased answers were
    scored 0/1 with no visible error signal.
    """
    agent = _SimpleAgent()
    adapter = LocalExecutionAdapter(_AgentBuilder(agent))

    dataset = {
        "examples": [
            {
                "input": {"text": "test"},
                "expected_output": "similar meaning",
                "metadata": {"evaluation_type": "semantic"},
            }
        ]
    }

    caplog.set_level("ERROR")
    result = await adapter.execute_configuration(
        agent_spec={}, dataset=dataset, trial_id="semantic-fail-loud"
    )

    # The example is counted and visibly marked as failed.
    assert result["metrics"]["total_examples"] == 1.0
    assert result["metrics"]["success_rate"] == 0.0
    assert result["metrics"]["accuracy_semantic"] == 0.0
    assert result["metrics"]["successful_executions"] == 0.0

    # An ERROR-level log was emitted naming the missing capability and the
    # remediation (configure a scoring_function).
    error_messages = [
        rec.getMessage() for rec in caplog.records if rec.levelname == "ERROR"
    ]
    assert any(
        "semantic" in msg.lower() and "scoring_function" in msg.lower()
        for msg in error_messages
    ), f"expected fail-loud ERROR log, got: {error_messages}"
    assert any(
        "trial_id=semantic-fail-loud" in msg and "example_index=0" in msg
        for msg in error_messages
    ), f"expected per-example ERROR context, got: {error_messages}"


@pytest.mark.asyncio
async def test_local_adapter_paraphrased_answers_fail_under_exact_match():
    """Issue #891: paraphrased answers (e.g. "Paris is the capital" vs
    "Paris") must score 0.0 under the default exact-match contract. This
    pins the honest behaviour so docs cannot drift back to claiming
    semantic similarity is the default."""

    class ParaphraseAgent:
        async def execute(self, _input):
            # The agent's answer is semantically equivalent but not a
            # character-for-character match with the expected output.
            return "Paris is the capital of France"

    adapter = LocalExecutionAdapter(_AgentBuilder(ParaphraseAgent()))

    dataset = {
        "examples": [
            {
                "input": {"question": "What is the capital of France?"},
                "expected_output": "Paris",
                "metadata": {"evaluation_type": "exact_match"},
            },
            {
                "input": {"question": "Capital of France?"},
                "expected_output": "The capital is Paris",
                "metadata": {"evaluation_type": "exact_match"},
            },
        ]
    }

    result = await adapter.execute_configuration(
        agent_spec={}, dataset=dataset, trial_id="paraphrase-exact"
    )

    # Both paraphrases must fail under exact_match. If this assertion ever
    # flips, either the contract was changed or someone silently re-introduced
    # semantic scoring under the default name.
    assert result["metrics"]["accuracy"] == 0.0
    assert result["metrics"]["accuracy_exact_match"] == 0.0


@pytest.mark.asyncio
async def test_local_adapter_exact_match_is_case_insensitive_after_stripping():
    """Issue #891 regression: docs and `LocalEvaluator` both promise that
    the default scorer treats `"paris"` and `"Paris"` as equal. The
    adapter's `exact_match` branch must match — otherwise the public
    contract diverges between the two code paths.

    This test pins the case-insensitive-after-strip contract for
    `LocalExecutionAdapter._evaluate_output` specifically (the
    `LocalEvaluator` path is covered by
    `test_local_evaluator_paraphrases_*` in test_local_evaluator_accuracy)."""

    class LowercaseAgent:
        async def execute(self, _input):
            # Returns the answer in lowercase with surrounding whitespace;
            # the expected_output is the canonical, title-cased form.
            return "  paris  "

    adapter = LocalExecutionAdapter(_AgentBuilder(LowercaseAgent()))

    dataset = {
        "examples": [
            {
                "input": {"question": "What is the capital of France?"},
                "expected_output": "Paris",
                "metadata": {"evaluation_type": "exact_match"},
            },
        ]
    }

    result = await adapter.execute_configuration(
        agent_spec={}, dataset=dataset, trial_id="paris-case-insensitive"
    )

    assert result["metrics"]["accuracy"] == 1.0
    assert result["metrics"]["accuracy_exact_match"] == 1.0


@pytest.mark.asyncio
async def test_local_adapter_paraphrased_answers_pass_under_non_exact_match_type():
    """Issue #891 (paired with the exact-match test above): the same
    paraphrased answers that fail under the default `exact_match`
    contract pass under a non-exact-match `evaluation_type`. Here we use
    the built-in `contains` branch as a stand-in: it is *not* a semantic
    scorer (Traigent does not ship one), but it makes the same
    "paraphrase counts as correct" point on the adapter's public
    surface. The documented path for real semantic scoring is a
    user-supplied `scoring_function` (see
    docs/user-guide/evaluation_guide.md Method 2)."""

    class ParaphraseAgent:
        async def execute(self, _input):
            return "Paris is the capital of France"

    adapter = LocalExecutionAdapter(_AgentBuilder(ParaphraseAgent()))

    dataset = {
        "examples": [
            {
                "input": {"question": "What is the capital of France?"},
                "expected_output": "Paris",
                # `contains` is a built-in non-exact-match branch — used
                # here purely as a surrogate to show that paraphrases
                # pass under a non-exact-match scorer. It is NOT a
                # semantic scorer.
                "metadata": {"evaluation_type": "contains"},
            },
        ]
    }

    result = await adapter.execute_configuration(
        agent_spec={}, dataset=dataset, trial_id="paraphrase-contains"
    )

    assert result["metrics"]["accuracy"] == 1.0
    assert result["metrics"]["accuracy_contains"] == 1.0


@pytest.mark.asyncio
async def test_local_adapter_mixed_evaluation_types():
    """Test LocalExecutionAdapter handles multiple evaluation types."""

    class MixedAgent:
        async def execute(self, _input):
            text = _input.get("text", "")
            if "exact" in text:
                return "EXACT"
            elif "contains" in text:
                return "This contains PARTIAL"
            elif "numeric" in text:
                return "99.5"
            return "DEFAULT"

    agent = MixedAgent()
    adapter = LocalExecutionAdapter(_AgentBuilder(agent))

    dataset = {
        "examples": [
            {
                "input": {"text": "exact"},
                "expected_output": "EXACT",
                "metadata": {"evaluation_type": "exact_match"},
            },
            {
                "input": {"text": "contains"},
                "expected_output": "PARTIAL",
                "metadata": {"evaluation_type": "contains"},
            },
            {
                "input": {"text": "numeric"},
                "expected_output": "100.0",
                "metadata": {"evaluation_type": "numeric", "tolerance": 1.0},
            },
        ]
    }

    result = await adapter.execute_configuration(
        agent_spec={}, dataset=dataset, trial_id="mixed-1"
    )

    # All three should be correct
    assert result["metrics"]["accuracy"] == 1.0
    assert result["metrics"]["accuracy_exact_match"] == 1.0
    assert result["metrics"]["accuracy_contains"] == 1.0
    assert result["metrics"]["accuracy_numeric"] == 1.0
    assert result["metrics"]["total_examples"] == 3.0


@pytest.mark.asyncio
async def test_local_adapter_invalid_numeric_values():
    """Test LocalExecutionAdapter handles invalid numeric values gracefully."""

    class InvalidNumericAgent:
        async def execute(self, _input):
            return "not a number"

    agent = InvalidNumericAgent()
    adapter = LocalExecutionAdapter(_AgentBuilder(agent))

    dataset = {
        "examples": [
            {
                "input": {"text": "test"},
                "expected_output": "42.0",
                "metadata": {"evaluation_type": "numeric"},
            }
        ]
    }

    result = await adapter.execute_configuration(
        agent_spec={}, dataset=dataset, trial_id="invalid-numeric-1"
    )

    # Invalid numeric should fail evaluation
    assert result["metrics"]["accuracy"] == 0.0
    assert result["metrics"]["accuracy_numeric"] == 0.0
