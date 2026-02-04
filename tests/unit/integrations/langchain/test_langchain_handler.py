"""Unit tests for LangChain/LangGraph callback handler.

Tests the TraigentHandler class and related utilities.
Run with: TRAIGENT_MOCK_LLM=true pytest tests/unit/integrations/langchain/ -v
"""

from __future__ import annotations

import threading
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest


class TestLLMCallMetrics:
    """Test LLMCallMetrics dataclass."""

    def test_basic_creation(self):
        """Test creating basic LLMCallMetrics."""
        from traigent.integrations.langchain.handler import LLMCallMetrics

        metrics = LLMCallMetrics(
            call_id="call-123",
            node_name="agent",
            model="gpt-4",
            start_time=1000.0,
        )
        assert metrics.call_id == "call-123"
        assert metrics.node_name == "agent"
        assert metrics.model == "gpt-4"
        assert metrics.start_time == 1000.0
        assert metrics.end_time is None
        assert metrics.input_tokens == 0
        assert metrics.output_tokens == 0
        assert metrics.error is None

    def test_latency_ms_without_end_time(self):
        """Test latency calculation when end_time is None."""
        from traigent.integrations.langchain.handler import LLMCallMetrics

        metrics = LLMCallMetrics(
            call_id="call-123",
            node_name=None,
            model=None,
            start_time=1000.0,
        )
        assert metrics.latency_ms == 0.0

    def test_latency_ms_with_end_time(self):
        """Test latency calculation with end_time."""
        from traigent.integrations.langchain.handler import LLMCallMetrics

        metrics = LLMCallMetrics(
            call_id="call-123",
            node_name=None,
            model=None,
            start_time=1000.0,
            end_time=1002.5,  # 2.5 seconds later
        )
        assert metrics.latency_ms == 2500.0

    def test_full_metrics(self):
        """Test LLMCallMetrics with all fields populated."""
        from traigent.integrations.langchain.handler import LLMCallMetrics

        metrics = LLMCallMetrics(
            call_id="call-123",
            node_name="grader",
            model="gpt-4o-mini",
            start_time=1000.0,
            end_time=1001.0,
            input_tokens=100,
            output_tokens=50,
            total_tokens=150,
            cost=0.001,
            error=None,
        )
        assert metrics.total_tokens == 150
        assert metrics.cost == 0.001
        assert metrics.latency_ms == 1000.0


class TestToolCallMetrics:
    """Test ToolCallMetrics dataclass."""

    def test_basic_creation(self):
        """Test creating basic ToolCallMetrics."""
        from traigent.integrations.langchain.handler import ToolCallMetrics

        metrics = ToolCallMetrics(
            call_id="tool-123",
            node_name="search_agent",
            tool_name="web_search",
            start_time=1000.0,
        )
        assert metrics.call_id == "tool-123"
        assert metrics.node_name == "search_agent"
        assert metrics.tool_name == "web_search"
        assert metrics.end_time is None
        assert metrics.error is None

    def test_latency_ms(self):
        """Test tool call latency calculation."""
        from traigent.integrations.langchain.handler import ToolCallMetrics

        metrics = ToolCallMetrics(
            call_id="tool-123",
            node_name=None,
            tool_name="calculator",
            start_time=1000.0,
            end_time=1000.5,
        )
        assert metrics.latency_ms == 500.0


class TestTraigentHandlerMetrics:
    """Test TraigentHandlerMetrics dataclass."""

    def test_basic_creation(self):
        """Test creating TraigentHandlerMetrics."""
        from traigent.integrations.langchain.handler import TraigentHandlerMetrics

        metrics = TraigentHandlerMetrics(trace_id="trace-123")
        assert metrics.trace_id == "trace-123"
        assert metrics.llm_calls == []
        assert metrics.tool_calls == []
        assert metrics.total_cost == 0.0
        assert metrics.total_latency_ms == 0.0

    def test_to_measures_dict_basic(self):
        """Test converting to measures dict."""
        from traigent.integrations.langchain.handler import TraigentHandlerMetrics

        metrics = TraigentHandlerMetrics(
            trace_id="trace-123",
            total_cost=0.01,
            total_latency_ms=1500.0,
            total_input_tokens=100,
            total_output_tokens=50,
            total_tokens=150,
        )
        measures = metrics.to_measures_dict()
        assert measures["total_cost"] == 0.01
        assert measures["total_latency_ms"] == 1500.0
        assert measures["total_tokens"] == 150
        assert measures["llm_call_count"] == 0
        assert measures["tool_call_count"] == 0

    def test_to_measures_dict_with_prefix(self):
        """Test measures dict with prefix."""
        from traigent.integrations.langchain.handler import TraigentHandlerMetrics

        metrics = TraigentHandlerMetrics(
            trace_id="trace-123",
            total_cost=0.005,
        )
        measures = metrics.to_measures_dict(prefix="langchain_")
        assert "langchain_total_cost" in measures
        assert measures["langchain_total_cost"] == 0.005

    def test_to_measures_dict_with_per_node(self):
        """Test measures dict with per-node breakdown."""
        from traigent.integrations.langchain.handler import (
            LLMCallMetrics,
            TraigentHandlerMetrics,
        )

        llm_call = LLMCallMetrics(
            call_id="call-1",
            node_name="grader",
            model="gpt-4",
            start_time=1000.0,
            end_time=1001.0,
            total_tokens=100,
            cost=0.01,
        )
        metrics = TraigentHandlerMetrics(
            trace_id="trace-123",
            llm_calls=[llm_call],
        )
        measures = metrics.to_measures_dict(include_per_node=True)
        assert "grader_cost" in measures
        assert "grader_latency_ms" in measures
        assert "grader_tokens" in measures

    def test_to_measures_dict_without_per_node(self):
        """Test measures dict without per-node breakdown."""
        from traigent.integrations.langchain.handler import (
            LLMCallMetrics,
            TraigentHandlerMetrics,
        )

        llm_call = LLMCallMetrics(
            call_id="call-1",
            node_name="grader",
            model="gpt-4",
            start_time=1000.0,
            end_time=1001.0,
            total_tokens=100,
            cost=0.01,
        )
        metrics = TraigentHandlerMetrics(
            trace_id="trace-123",
            llm_calls=[llm_call],
        )
        measures = metrics.to_measures_dict(include_per_node=False)
        assert "grader_cost" not in measures
        assert "grader_latency_ms" not in measures

    def test_to_measures_dict_sanitizes_node_names(self):
        """Test that node names are sanitized for MeasuresDict."""
        from traigent.integrations.langchain.handler import (
            LLMCallMetrics,
            TraigentHandlerMetrics,
        )

        llm_call = LLMCallMetrics(
            call_id="call-1",
            node_name="my.agent-name",  # Contains dot and dash
            model="gpt-4",
            start_time=1000.0,
            end_time=1001.0,
            total_tokens=100,
            cost=0.01,
        )
        metrics = TraigentHandlerMetrics(
            trace_id="trace-123",
            llm_calls=[llm_call],
        )
        measures = metrics.to_measures_dict(include_per_node=True)
        # Should have sanitized names (dots and dashes → underscores)
        assert "my_agent_name_cost" in measures


class TestContextManagers:
    """Test context managers for trial and node context."""

    def test_trial_context(self):
        """Test trial_context context manager."""
        from traigent.integrations.langchain.handler import (
            get_current_trial_config,
            trial_context,
        )

        assert get_current_trial_config() == {}

        with trial_context({"temperature": 0.7, "model": "gpt-4"}):
            config = get_current_trial_config()
            assert config["temperature"] == 0.7
            assert config["model"] == "gpt-4"

        assert get_current_trial_config() == {}

    def test_node_context(self):
        """Test node_context context manager."""
        from traigent.integrations.langchain.handler import (
            get_current_node_name,
            node_context,
        )

        assert get_current_node_name() is None

        with node_context("grader"):
            assert get_current_node_name() == "grader"

        assert get_current_node_name() is None

    def test_nested_node_context(self):
        """Test nested node contexts."""
        from traigent.integrations.langchain.handler import (
            get_current_node_name,
            node_context,
        )

        with node_context("outer"):
            assert get_current_node_name() == "outer"
            with node_context("inner"):
                assert get_current_node_name() == "inner"
            assert get_current_node_name() == "outer"


class TestTraigentHandlerInit:
    """Test TraigentHandler initialization."""

    def test_init_without_langchain(self):
        """Test that init raises ImportError when langchain not available."""
        with patch(
            "traigent.integrations.langchain.handler.LANGCHAIN_AVAILABLE", False
        ):
            from traigent.integrations.langchain.handler import TraigentHandler

            with pytest.raises(ImportError, match="LangChain is required"):
                TraigentHandler()

    def test_init_with_defaults(self):
        """Test handler initialization with defaults."""
        pytest.importorskip("langchain_core")
        from traigent.integrations.langchain.handler import TraigentHandler

        handler = TraigentHandler()
        assert handler.trace_id is not None
        assert len(handler.trace_id) > 0

    def test_init_with_trace_id(self):
        """Test handler initialization with explicit trace_id."""
        pytest.importorskip("langchain_core")
        from traigent.integrations.langchain.handler import TraigentHandler

        handler = TraigentHandler(trace_id="my-trace-123")
        assert handler.trace_id == "my-trace-123"

    def test_init_with_trace_id_generator(self):
        """Test handler initialization with trace_id generator."""
        pytest.importorskip("langchain_core")
        from traigent.integrations.langchain.handler import TraigentHandler

        counter = [0]

        def generator():
            counter[0] += 1
            return f"generated-{counter[0]}"

        handler = TraigentHandler(trace_id_generator=generator)
        assert handler.trace_id == "generated-1"

    def test_init_with_options(self):
        """Test handler initialization with custom options."""
        pytest.importorskip("langchain_core")
        from traigent.integrations.langchain.handler import TraigentHandler

        handler = TraigentHandler(
            trace_id="test-trace",
            metric_prefix="custom_",
            include_per_node=False,
        )
        assert handler._metric_prefix == "custom_"
        assert handler._include_per_node is False


class TestTraigentHandlerLLMCallbacks:
    """Test TraigentHandler LLM callback methods."""

    @pytest.fixture
    def handler(self):
        """Create a handler for testing."""
        pytest.importorskip("langchain_core")
        from traigent.integrations.langchain.handler import TraigentHandler

        return TraigentHandler(trace_id="test-trace")

    def test_on_llm_start(self, handler):
        """Test on_llm_start callback."""
        run_id = str(uuid4())
        handler.on_llm_start(
            serialized={"kwargs": {"model": "gpt-4"}},
            prompts=["Hello"],
            run_id=run_id,
        )
        assert run_id in handler._llm_calls
        call = handler._llm_calls[run_id]
        assert call.model == "gpt-4"
        assert call.start_time > 0

    def test_on_llm_start_with_metadata_node(self, handler):
        """Test on_llm_start extracts node from metadata."""
        run_id = str(uuid4())
        handler.on_llm_start(
            serialized={},
            prompts=["Hello"],
            run_id=run_id,
            metadata={"langgraph_node": "grader"},
        )
        call = handler._llm_calls[run_id]
        assert call.node_name == "grader"

    def test_on_llm_end(self, handler):
        """Test on_llm_end callback."""
        run_id = str(uuid4())
        handler.on_llm_start(
            serialized={},
            prompts=["Hello"],
            run_id=run_id,
        )

        # Create mock response
        mock_response = MagicMock()
        mock_response.llm_output = {
            "token_usage": {
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "total_tokens": 30,
            }
        }

        handler.on_llm_end(response=mock_response, run_id=run_id)

        assert run_id not in handler._llm_calls
        assert len(handler._completed_llm_calls) == 1
        call = handler._completed_llm_calls[0]
        assert call.total_tokens == 30
        assert call.end_time is not None

    def test_on_llm_end_unknown_call(self, handler):
        """Test on_llm_end for unknown call doesn't crash."""
        mock_response = MagicMock()
        mock_response.llm_output = {}

        # Should not raise
        handler.on_llm_end(response=mock_response, run_id="unknown-id")

    def test_on_llm_error(self, handler):
        """Test on_llm_error callback."""
        run_id = str(uuid4())
        handler.on_llm_start(
            serialized={},
            prompts=["Hello"],
            run_id=run_id,
        )

        handler.on_llm_error(error=Exception("Test error"), run_id=run_id)

        assert run_id not in handler._llm_calls
        assert len(handler._completed_llm_calls) == 1
        call = handler._completed_llm_calls[0]
        assert call.error == "Test error"


class TestTraigentHandlerChatModelCallbacks:
    """Test TraigentHandler chat model callback methods."""

    @pytest.fixture
    def handler(self):
        """Create a handler for testing."""
        pytest.importorskip("langchain_core")
        from traigent.integrations.langchain.handler import TraigentHandler

        return TraigentHandler(trace_id="test-trace")

    def test_on_chat_model_start(self, handler):
        """Test on_chat_model_start callback."""
        run_id = str(uuid4())
        handler.on_chat_model_start(
            serialized={"kwargs": {"model": "gpt-4o-mini"}},
            messages=[[]],
            run_id=run_id,
        )
        assert run_id in handler._llm_calls
        call = handler._llm_calls[run_id]
        assert call.model == "gpt-4o-mini"

    def test_on_chat_model_end(self, handler):
        """Test on_chat_model_end callback."""
        run_id = str(uuid4())
        handler.on_chat_model_start(
            serialized={},
            messages=[[]],
            run_id=run_id,
        )

        mock_response = MagicMock()
        mock_response.llm_output = {
            "token_usage": {
                "prompt_tokens": 50,
                "completion_tokens": 100,
                "total_tokens": 150,
            },
            "model_name": "gpt-4o",
        }

        handler.on_chat_model_end(response=mock_response, run_id=run_id)

        assert len(handler._completed_llm_calls) == 1
        call = handler._completed_llm_calls[0]
        assert call.total_tokens == 150
        assert call.model == "gpt-4o"

    def test_on_chat_model_error(self, handler):
        """Test on_chat_model_error callback."""
        run_id = str(uuid4())
        handler.on_chat_model_start(
            serialized={},
            messages=[[]],
            run_id=run_id,
        )

        handler.on_chat_model_error(error=Exception("Chat model failed"), run_id=run_id)

        assert len(handler._completed_llm_calls) == 1
        call = handler._completed_llm_calls[0]
        assert call.error == "Chat model failed"


class TestTraigentHandlerToolCallbacks:
    """Test TraigentHandler tool callback methods."""

    @pytest.fixture
    def handler(self):
        """Create a handler for testing."""
        pytest.importorskip("langchain_core")
        from traigent.integrations.langchain.handler import TraigentHandler

        return TraigentHandler(trace_id="test-trace")

    def test_on_tool_start(self, handler):
        """Test on_tool_start callback."""
        run_id = str(uuid4())
        handler.on_tool_start(
            serialized={"name": "web_search"},
            input_str="search query",
            run_id=run_id,
        )
        assert run_id in handler._tool_calls
        call = handler._tool_calls[run_id]
        assert call.tool_name == "web_search"

    def test_on_tool_end(self, handler):
        """Test on_tool_end callback."""
        run_id = str(uuid4())
        handler.on_tool_start(
            serialized={"name": "calculator"},
            input_str="2+2",
            run_id=run_id,
        )

        handler.on_tool_end(output="4", run_id=run_id)

        assert run_id not in handler._tool_calls
        assert len(handler._completed_tool_calls) == 1

    def test_on_tool_error(self, handler):
        """Test on_tool_error callback."""
        run_id = str(uuid4())
        handler.on_tool_start(
            serialized={"name": "api_call"},
            input_str="request",
            run_id=run_id,
        )

        handler.on_tool_error(error=Exception("API failed"), run_id=run_id)

        assert len(handler._completed_tool_calls) == 1
        call = handler._completed_tool_calls[0]
        assert call.error == "API failed"


class TestTraigentHandlerChainCallbacks:
    """Test TraigentHandler chain callback methods."""

    @pytest.fixture
    def handler(self):
        """Create a handler for testing."""
        pytest.importorskip("langchain_core")
        from traigent.integrations.langchain.handler import TraigentHandler

        return TraigentHandler(trace_id="test-trace")

    def test_on_chain_start_sets_node_context(self, handler):
        """Test on_chain_start sets node context from metadata."""
        from traigent.integrations.langchain.handler import get_current_node_name

        run_id = str(uuid4())
        handler.on_chain_start(
            serialized={},
            inputs={},
            run_id=run_id,
            metadata={"langgraph_node": "grader"},
        )
        assert get_current_node_name() == "grader"

        # Cleanup
        handler.on_chain_end(outputs={}, run_id=run_id)
        assert get_current_node_name() is None

    def test_on_chain_end_restores_context(self, handler):
        """Test on_chain_end restores previous node context."""
        from traigent.integrations.langchain.handler import get_current_node_name

        run_id = str(uuid4())
        handler.on_chain_start(
            serialized={},
            inputs={},
            run_id=run_id,
            metadata={"langgraph_node": "inner"},
        )
        handler.on_chain_end(outputs={}, run_id=run_id)
        assert get_current_node_name() is None

    def test_on_chain_error_restores_context(self, handler):
        """Test on_chain_error restores previous node context."""
        from traigent.integrations.langchain.handler import get_current_node_name

        run_id = str(uuid4())
        handler.on_chain_start(
            serialized={},
            inputs={},
            run_id=run_id,
            metadata={"langgraph_node": "test_node"},
        )
        handler.on_chain_error(error=Exception("Chain failed"), run_id=run_id)
        assert get_current_node_name() is None


class TestTraigentHandlerCostEstimation:
    """Test TraigentHandler cost estimation."""

    @pytest.fixture
    def handler(self):
        """Create a handler for testing."""
        pytest.importorskip("langchain_core")
        from traigent.integrations.langchain.handler import TraigentHandler

        return TraigentHandler(trace_id="test-trace")

    def test_estimate_cost_returns_float(self, handler):
        """Test cost estimation returns a float."""
        cost = handler._estimate_cost("gpt-4o", 1000, 500)
        assert isinstance(cost, float)
        # Cost should be non-negative
        assert cost >= 0.0

    def test_estimate_cost_with_zero_tokens(self, handler):
        """Test cost estimation with zero tokens."""
        cost = handler._estimate_cost("gpt-4", 0, 0)
        assert cost == 0.0

    def test_estimate_cost_increases_with_tokens(self, handler):
        """Test that cost increases with more tokens."""
        cost_small = handler._estimate_cost("gpt-4o", 100, 50)
        cost_large = handler._estimate_cost("gpt-4o", 10000, 5000)
        # Larger token count should result in equal or higher cost
        assert cost_large >= cost_small


class TestTraigentHandlerMetricsAggregation:
    """Test TraigentHandler metrics aggregation."""

    @pytest.fixture
    def handler(self):
        """Create a handler for testing."""
        pytest.importorskip("langchain_core")
        from traigent.integrations.langchain.handler import TraigentHandler

        return TraigentHandler(trace_id="test-trace", metric_prefix="test_")

    def test_get_metrics_empty(self, handler):
        """Test get_metrics with no calls."""
        metrics = handler.get_metrics()
        assert metrics.trace_id == "test-trace"
        assert len(metrics.llm_calls) == 0
        assert len(metrics.tool_calls) == 0
        assert metrics.total_cost == 0.0

    def test_get_metrics_with_calls(self, handler):
        """Test get_metrics with LLM calls."""
        run_id = str(uuid4())
        handler.on_llm_start(
            serialized={"kwargs": {"model": "gpt-4o-mini"}},
            prompts=["Hello"],
            run_id=run_id,
        )

        mock_response = MagicMock()
        mock_response.llm_output = {
            "token_usage": {
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "total_tokens": 150,
            }
        }
        handler.on_llm_end(response=mock_response, run_id=run_id)

        metrics = handler.get_metrics()
        assert len(metrics.llm_calls) == 1
        assert metrics.total_tokens == 150

    def test_get_measures_dict(self, handler):
        """Test get_measures_dict convenience method."""
        measures = handler.get_measures_dict()
        assert "test_total_cost" in measures
        assert "test_total_latency_ms" in measures


class TestTraigentHandlerReset:
    """Test TraigentHandler reset functionality."""

    def test_reset_clears_state(self):
        """Test that reset clears all handler state."""
        pytest.importorskip("langchain_core")
        from traigent.integrations.langchain.handler import TraigentHandler

        handler = TraigentHandler(trace_id="original-trace")
        original_trace_id = handler.trace_id

        # Add some calls
        run_id = str(uuid4())
        handler.on_llm_start(
            serialized={},
            prompts=["Hello"],
            run_id=run_id,
        )

        handler.reset()

        # Verify state is cleared
        assert handler.trace_id != original_trace_id
        assert len(handler._llm_calls) == 0
        assert len(handler._completed_llm_calls) == 0
        assert len(handler._tool_calls) == 0
        assert len(handler._completed_tool_calls) == 0


class TestTraigentHandlerThreadSafety:
    """Test TraigentHandler thread safety."""

    def test_handler_has_lock(self):
        """Test that handler has a threading lock."""
        pytest.importorskip("langchain_core")
        from traigent.integrations.langchain.handler import TraigentHandler

        handler = TraigentHandler()
        assert hasattr(handler, "_lock")
        assert isinstance(handler._lock, type(threading.Lock()))


class TestCreateTraigentHandler:
    """Test factory function."""

    def test_create_traigent_handler_defaults(self):
        """Test factory function with defaults."""
        pytest.importorskip("langchain_core")
        from traigent.integrations.langchain.handler import create_traigent_handler

        handler = create_traigent_handler()
        assert handler._metric_prefix == "langchain_"
        assert handler._include_per_node is True

    def test_create_traigent_handler_custom(self):
        """Test factory function with custom options."""
        pytest.importorskip("langchain_core")
        from traigent.integrations.langchain.handler import create_traigent_handler

        handler = create_traigent_handler(
            trace_id="custom-trace",
            metric_prefix="myapp_",
            include_per_node=False,
        )
        assert handler.trace_id == "custom-trace"
        assert handler._metric_prefix == "myapp_"
        assert handler._include_per_node is False


class TestTraigentHandlerModelExtraction:
    """Test model extraction from various sources."""

    @pytest.fixture
    def handler(self):
        """Create a handler for testing."""
        pytest.importorskip("langchain_core")
        from traigent.integrations.langchain.handler import TraigentHandler

        return TraigentHandler(trace_id="test-trace")

    def test_model_from_serialized_kwargs(self, handler):
        """Test extracting model from serialized kwargs."""
        run_id = str(uuid4())
        handler.on_llm_start(
            serialized={"kwargs": {"model": "gpt-4"}},
            prompts=["Test"],
            run_id=run_id,
        )
        assert handler._llm_calls[run_id].model == "gpt-4"

    def test_model_from_serialized_model_name(self, handler):
        """Test extracting model from serialized model_name."""
        run_id = str(uuid4())
        handler.on_llm_start(
            serialized={"kwargs": {"model_name": "claude-3-sonnet"}},
            prompts=["Test"],
            run_id=run_id,
        )
        assert handler._llm_calls[run_id].model == "claude-3-sonnet"

    def test_model_from_invocation_params(self, handler):
        """Test extracting model from invocation_params."""
        run_id = str(uuid4())
        handler.on_llm_start(
            serialized={},
            prompts=["Test"],
            run_id=run_id,
            invocation_params={"model": "gpt-4-turbo"},
        )
        assert handler._llm_calls[run_id].model == "gpt-4-turbo"


class TestCostEstimation:
    """Test cost estimation methods including fallback logic."""

    @pytest.fixture
    def handler(self):
        """Create a handler for testing."""
        pytest.importorskip("langchain_core")
        from traigent.integrations.langchain.handler import TraigentHandler

        return TraigentHandler(trace_id="test-trace")

    def test_fallback_cost_estimate_gpt4o(self, handler):
        """Test fallback cost estimation for GPT-4o model."""
        cost = handler._fallback_cost_estimate("gpt-4o", 1000, 500)
        # GPT-4o: $2.50/1M input, $10.00/1M output
        expected = (1000 * 2.50 + 500 * 10.00) / 1_000_000
        assert cost == pytest.approx(expected)

    def test_fallback_cost_estimate_gpt35_turbo(self, handler):
        """Test fallback cost estimation for GPT-3.5-turbo model."""
        cost = handler._fallback_cost_estimate("gpt-3.5-turbo", 1000, 500)
        # GPT-3.5-turbo: $0.50/1M input, $1.50/1M output
        expected = (1000 * 0.50 + 500 * 1.50) / 1_000_000
        assert cost == pytest.approx(expected)

    def test_fallback_cost_estimate_claude(self, handler):
        """Test fallback cost estimation for Claude models."""
        cost = handler._fallback_cost_estimate("claude-3-sonnet", 1000, 500)
        # Claude-3-sonnet: $3.00/1M input, $15.00/1M output
        expected = (1000 * 3.00 + 500 * 15.00) / 1_000_000
        assert cost == pytest.approx(expected)

    def test_fallback_cost_estimate_claude_haiku(self, handler):
        """Test fallback cost estimation for Claude Haiku."""
        cost = handler._fallback_cost_estimate("claude-3-haiku", 1000, 500)
        # Claude-3-haiku: $0.25/1M input, $1.25/1M output
        expected = (1000 * 0.25 + 500 * 1.25) / 1_000_000
        assert cost == pytest.approx(expected)

    def test_fallback_cost_estimate_unknown_model(self, handler):
        """Test fallback cost estimation for unknown model."""
        cost = handler._fallback_cost_estimate("some-unknown-model", 1000, 500)
        # Default: $1.00/1M input, $3.00/1M output
        expected = (1000 * 1.0 + 500 * 3.0) / 1_000_000
        assert cost == pytest.approx(expected)

    def test_fallback_cost_estimate_case_insensitive(self, handler):
        """Test that model matching is case-insensitive."""
        cost_lower = handler._fallback_cost_estimate("gpt-4o", 1000, 500)
        cost_upper = handler._fallback_cost_estimate("GPT-4O", 1000, 500)
        assert cost_lower == cost_upper

    def test_estimate_cost_with_litellm_available(self, handler):
        """Test _estimate_cost when litellm is available and works."""
        # litellm IS available in test env, so this tests the happy path
        # For gpt-4o, litellm should return a valid cost
        cost = handler._estimate_cost("gpt-4o", 1000, 500)
        assert cost > 0, "Should return non-zero cost for known model"

    def test_estimate_cost_falls_back_for_unknown_model(self, handler):
        """Test _estimate_cost falls back for unknown model (litellm returns 0)."""
        # Use a model name litellm doesn't know - forces fallback path
        cost = handler._estimate_cost("unknown-model-xyz-123", 1000, 500)
        # Should fall back to hardcoded default estimates
        expected = (1000 * 1.0 + 500 * 3.0) / 1_000_000
        assert cost == pytest.approx(expected)

    def test_estimate_cost_exception_handling(self, handler):
        """Test _estimate_cost handles exceptions gracefully."""
        # Patch calculate_llm_cost to raise an exception - covers except branch
        with patch(
            "traigent.utils.cost_calculator.calculate_llm_cost",
            side_effect=Exception("Test error"),
        ):
            cost = handler._estimate_cost("gpt-4o", 1000, 500)
            # Should fall back to hardcoded estimates via except handler
            expected = (1000 * 2.50 + 500 * 10.00) / 1_000_000
            assert cost == pytest.approx(expected)
