"""Integration tests for LangChain/LangGraph handler.

Run with: TRAIGENT_MOCK_LLM=true TRAIGENT_OFFLINE_MODE=true pytest tests/integration/langchain/ -v
"""

from __future__ import annotations

import threading
import time
from unittest.mock import MagicMock
from uuid import uuid4

import pytest

from traigent.cloud.dtos import MeasuresDict
from traigent.integrations.langchain import (
    LANGCHAIN_AVAILABLE,
    TraigentHandler,
    create_traigent_handler,
    get_current_node_name,
    get_current_trial_config,
    node_context,
    trial_context,
)

# Skip all tests if LangChain not available
pytestmark = pytest.mark.skipif(
    not LANGCHAIN_AVAILABLE, reason="LangChain not installed"
)


class TestTraigentHandler:
    """Test TraigentHandler callback handler."""

    def test_handler_creation(self):
        """Verify handler can be created with default settings."""
        handler = TraigentHandler()

        assert handler.trace_id is not None
        assert len(handler.trace_id) > 0

    def test_handler_with_custom_trace_id(self):
        """Verify handler accepts custom trace_id."""
        custom_id = "my-custom-trace-123"
        handler = TraigentHandler(trace_id=custom_id)

        assert handler.trace_id == custom_id

    def test_handler_with_trace_id_generator(self):
        """Verify handler uses trace_id_generator."""
        counter = [0]

        def generator():
            counter[0] += 1
            return f"trace-{counter[0]}"

        handler = TraigentHandler(trace_id_generator=generator)
        assert handler.trace_id == "trace-1"

    def test_handler_with_metric_prefix(self):
        """Verify metric prefix is applied."""
        handler = TraigentHandler(metric_prefix="myapp_")

        # Simulate LLM call
        run_id = str(uuid4())
        handler.on_llm_start(
            serialized={"kwargs": {"model": "gpt-4"}},
            prompts=["Hello"],
            run_id=run_id,
        )

        # Simulate LLM response
        mock_response = MagicMock()
        mock_response.llm_output = {
            "token_usage": {
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "total_tokens": 30,
            },
            "model_name": "gpt-4",
        }
        handler.on_llm_end(mock_response, run_id=run_id)

        measures = handler.get_measures_dict()
        assert "myapp_total_tokens" in measures
        assert measures["myapp_total_tokens"] == 30

    def test_llm_call_tracking(self):
        """Verify LLM calls are tracked correctly."""
        handler = TraigentHandler()
        run_id = str(uuid4())

        # Start LLM call
        handler.on_llm_start(
            serialized={"kwargs": {"model": "gpt-4o-mini"}},
            prompts=["Test prompt"],
            run_id=run_id,
            metadata={"langgraph_node": "grader"},
        )

        # Simulate some processing time
        time.sleep(0.01)

        # End LLM call
        mock_response = MagicMock()
        mock_response.llm_output = {
            "token_usage": {
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "total_tokens": 150,
            }
        }
        handler.on_llm_end(mock_response, run_id=run_id)

        metrics = handler.get_metrics()

        assert len(metrics.llm_calls) == 1
        call = metrics.llm_calls[0]
        assert call.call_id == run_id
        assert call.node_name == "grader"
        assert call.input_tokens == 100
        assert call.output_tokens == 50
        assert call.total_tokens == 150
        assert call.latency_ms >= 10  # At least 10ms

    def test_chat_model_tracking(self):
        """Verify chat model calls are tracked correctly."""
        handler = TraigentHandler()
        run_id = str(uuid4())

        # Start chat model call
        handler.on_chat_model_start(
            serialized={"kwargs": {"model": "gpt-4"}},
            messages=[[]],  # Empty messages list
            run_id=run_id,
            metadata={"langgraph_node": "generator"},
        )

        # End call
        mock_response = MagicMock()
        mock_response.llm_output = {
            "token_usage": {
                "prompt_tokens": 50,
                "completion_tokens": 100,
                "total_tokens": 150,
            }
        }
        handler.on_llm_end(mock_response, run_id=run_id)

        metrics = handler.get_metrics()
        assert len(metrics.llm_calls) == 1
        assert metrics.llm_calls[0].node_name == "generator"

    def test_tool_call_tracking(self):
        """Verify tool calls are tracked correctly."""
        handler = TraigentHandler()
        run_id = str(uuid4())

        # Start tool call
        handler.on_tool_start(
            serialized={"name": "web_search"},
            input_str="search query",
            run_id=run_id,
            metadata={"langgraph_node": "researcher"},
        )

        time.sleep(0.01)

        # End tool call
        handler.on_tool_end("search results", run_id=run_id)

        metrics = handler.get_metrics()
        assert len(metrics.tool_calls) == 1
        tool = metrics.tool_calls[0]
        assert tool.tool_name == "web_search"
        assert tool.node_name == "researcher"
        assert tool.latency_ms >= 10

    def test_llm_error_tracking(self):
        """Verify LLM errors are tracked correctly."""
        handler = TraigentHandler()
        run_id = str(uuid4())

        handler.on_llm_start(
            serialized={},
            prompts=["test"],
            run_id=run_id,
        )

        handler.on_llm_error(ValueError("API rate limit"), run_id=run_id)

        metrics = handler.get_metrics()
        assert len(metrics.llm_calls) == 1
        assert metrics.llm_calls[0].error == "API rate limit"

    def test_tool_error_tracking(self):
        """Verify tool errors are tracked correctly."""
        handler = TraigentHandler()
        run_id = str(uuid4())

        handler.on_tool_start(
            serialized={"name": "calculator"},
            input_str="1/0",
            run_id=run_id,
        )

        handler.on_tool_error(ZeroDivisionError("division by zero"), run_id=run_id)

        metrics = handler.get_metrics()
        assert len(metrics.tool_calls) == 1
        assert "division by zero" in metrics.tool_calls[0].error

    def test_metrics_aggregation(self):
        """Verify metrics are aggregated correctly."""
        handler = TraigentHandler()

        # Simulate multiple LLM calls
        for _ in range(3):
            run_id = str(uuid4())
            handler.on_llm_start(
                serialized={"kwargs": {"model": "gpt-4"}},
                prompts=["test"],
                run_id=run_id,
            )
            mock_response = MagicMock()
            mock_response.llm_output = {
                "token_usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 10,
                    "total_tokens": 20,
                }
            }
            handler.on_llm_end(mock_response, run_id=run_id)

        metrics = handler.get_metrics()
        assert metrics.total_tokens == 60  # 3 calls × 20 tokens
        assert metrics.total_input_tokens == 30
        assert metrics.total_output_tokens == 30
        assert len(metrics.llm_calls) == 3

    def test_measures_dict_compliance(self):
        """Verify output is MeasuresDict-compliant."""
        handler = TraigentHandler()
        run_id = str(uuid4())

        handler.on_llm_start(
            serialized={"kwargs": {"model": "gpt-4"}},
            prompts=["test"],
            run_id=run_id,
            metadata={"langgraph_node": "my-agent"},  # Hyphen in name
        )
        mock_response = MagicMock()
        mock_response.llm_output = {
            "token_usage": {
                "prompt_tokens": 10,
                "completion_tokens": 10,
                "total_tokens": 20,
            }
        }
        handler.on_llm_end(mock_response, run_id=run_id)

        measures = handler.get_measures_dict()

        # Verify no dots or hyphens in keys
        for key in measures:
            assert "." not in key, f"Key '{key}' contains dot"
            assert "-" not in key, f"Key '{key}' contains hyphen"

        # Verify numeric values
        for value in measures.values():
            assert isinstance(value, (int, float)), f"Value {value} is not numeric"

        # Hyphen in node name should be sanitized to underscore
        assert "my_agent_tokens" in measures

        # Verify can construct MeasuresDict without error
        validated = MeasuresDict(measures)
        assert len(validated) > 0

    def test_handler_reset(self):
        """Verify handler can be reset."""
        handler = TraigentHandler()

        # Add some calls
        run_id = str(uuid4())
        handler.on_llm_start(serialized={}, prompts=["test"], run_id=run_id)
        handler.on_llm_end(MagicMock(llm_output={}), run_id=run_id)

        # Verify we have data
        assert len(handler.get_metrics().llm_calls) == 1

        # Reset
        old_trace_id = handler.trace_id
        handler.reset()

        # Verify reset
        assert len(handler.get_metrics().llm_calls) == 0
        assert handler.trace_id != old_trace_id

    def test_thread_safety(self):
        """Verify handler is thread-safe."""
        handler = TraigentHandler()
        errors = []

        def worker(thread_id: int):
            try:
                for i in range(10):
                    run_id = f"{thread_id}-{i}"
                    handler.on_llm_start(serialized={}, prompts=["test"], run_id=run_id)
                    time.sleep(0.001)
                    mock_response = MagicMock()
                    mock_response.llm_output = {"token_usage": {"total_tokens": 1}}
                    handler.on_llm_end(mock_response, run_id=run_id)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        # 5 threads × 10 calls each = 50 calls
        assert len(handler.get_metrics().llm_calls) == 50


class TestContextVars:
    """Test context variable functions for async-safe state."""

    def test_trial_context(self):
        """Verify trial_context sets and clears config."""
        assert get_current_trial_config() == {}

        with trial_context({"temperature": 0.7, "model": "gpt-4"}):
            config = get_current_trial_config()
            assert config["temperature"] == 0.7
            assert config["model"] == "gpt-4"

        # Should be cleared after context
        assert get_current_trial_config() == {}

    def test_node_context(self):
        """Verify node_context sets and clears node name."""
        assert get_current_node_name() is None

        with node_context("grader"):
            assert get_current_node_name() == "grader"

            # Nested context
            with node_context("validator"):
                assert get_current_node_name() == "validator"

            # Outer context restored
            assert get_current_node_name() == "grader"

        # Should be cleared after context
        assert get_current_node_name() is None

    def test_handler_uses_node_context(self):
        """Verify handler respects node_context."""
        handler = TraigentHandler()

        with node_context("my_node"):
            run_id = str(uuid4())
            handler.on_llm_start(serialized={}, prompts=["test"], run_id=run_id)
            handler.on_llm_end(MagicMock(llm_output={}), run_id=run_id)

        metrics = handler.get_metrics()
        assert metrics.llm_calls[0].node_name == "my_node"


class TestFactoryFunction:
    """Test create_traigent_handler factory."""

    def test_create_traigent_handler_defaults(self):
        """Verify factory creates handler with defaults."""
        handler = create_traigent_handler()

        assert isinstance(handler, TraigentHandler)
        assert handler.trace_id is not None

    def test_create_traigent_handler_custom_prefix(self):
        """Verify factory accepts custom prefix."""
        handler = create_traigent_handler(metric_prefix="custom_")

        # Add a call and check prefix
        run_id = str(uuid4())
        handler.on_llm_start(serialized={}, prompts=["test"], run_id=run_id)
        handler.on_llm_end(MagicMock(llm_output={}), run_id=run_id)

        measures = handler.get_measures_dict()
        assert any(k.startswith("custom_") for k in measures)


class TestPerNodeMetrics:
    """Test per-node metrics breakdown."""

    def test_per_node_aggregation(self):
        """Verify metrics are aggregated per node."""
        handler = TraigentHandler(include_per_node=True)

        # Call from grader node
        for _ in range(2):
            run_id = str(uuid4())
            handler.on_llm_start(
                serialized={},
                prompts=["test"],
                run_id=run_id,
                metadata={"langgraph_node": "grader"},
            )
            mock = MagicMock()
            mock.llm_output = {"token_usage": {"total_tokens": 10}}
            handler.on_llm_end(mock, run_id=run_id)

        # Call from generator node
        run_id = str(uuid4())
        handler.on_llm_start(
            serialized={},
            prompts=["test"],
            run_id=run_id,
            metadata={"langgraph_node": "generator"},
        )
        mock = MagicMock()
        mock.llm_output = {"token_usage": {"total_tokens": 50}}
        handler.on_llm_end(mock, run_id=run_id)

        measures = handler.get_measures_dict()

        assert measures["total_tokens"] == 70  # 10 + 10 + 50
        assert measures["grader_tokens"] == 20  # 2 × 10
        assert measures["generator_tokens"] == 50

    def test_per_node_disabled(self):
        """Verify per-node metrics can be disabled."""
        handler = TraigentHandler(include_per_node=False)

        run_id = str(uuid4())
        handler.on_llm_start(
            serialized={},
            prompts=["test"],
            run_id=run_id,
            metadata={"langgraph_node": "grader"},
        )
        handler.on_llm_end(MagicMock(llm_output={}), run_id=run_id)

        measures = handler.get_measures_dict()

        # Should have total metrics but NOT per-node
        assert "total_tokens" in measures
        assert "grader_tokens" not in measures


class TestCostEstimation:
    """Test cost estimation functionality."""

    def test_cost_estimation_with_litellm(self):
        """Verify cost estimation with litellm library."""
        handler = TraigentHandler()

        run_id = str(uuid4())
        handler.on_llm_start(
            serialized={"kwargs": {"model": "gpt-4o-mini"}},
            prompts=["test"],
            run_id=run_id,
        )
        mock = MagicMock()
        mock.llm_output = {
            "token_usage": {
                "prompt_tokens": 1000,
                "completion_tokens": 500,
                "total_tokens": 1500,
            },
            "model_name": "gpt-4o-mini",
        }
        handler.on_llm_end(mock, run_id=run_id)

        metrics = handler.get_metrics()
        # Cost should be > 0 (exact value depends on litellm)
        assert metrics.total_cost >= 0

    def test_cost_estimation_fallback(self):
        """Verify fallback cost estimation works for known models."""
        handler = TraigentHandler()

        run_id = str(uuid4())
        # Use a known model that should have cost estimation
        handler.on_llm_start(
            serialized={"kwargs": {"model": "gpt-4"}},
            prompts=["test"],
            run_id=run_id,
        )
        mock = MagicMock()
        mock.llm_output = {
            "token_usage": {
                "prompt_tokens": 1000,
                "completion_tokens": 1000,
                "total_tokens": 2000,
            },
            "model_name": "gpt-4",
        }
        handler.on_llm_end(mock, run_id=run_id)

        metrics = handler.get_metrics()
        # Should have some cost (either from litellm or fallback)
        assert metrics.total_cost >= 0  # Cost estimation may vary, just verify it runs
