"""Tests for the PydanticAI handler."""

from __future__ import annotations

import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from traigent.integrations.pydantic_ai._types import (
    AgentRunMetrics,
    PydanticAIHandlerMetrics,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_usage(
    input_tokens: int = 100,
    output_tokens: int = 50,
    total_tokens: int = 150,
    requests: int = 1,
) -> SimpleNamespace:
    return SimpleNamespace(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        requests=requests,
    )


def _make_result(usage=None):
    usage = usage or _make_usage()
    result = MagicMock()
    result.usage.return_value = usage
    result.data = "Test response"
    return result


def _make_stream_result(usage=None):
    """Create a mock stream result with usage available after context exit."""
    usage = usage or _make_usage()
    result = MagicMock()
    result.usage.return_value = usage
    result.data = "Streamed response"
    return result


def _make_agent(model="openai:gpt-4o"):
    agent = MagicMock()
    agent.model = model
    agent.run = AsyncMock(return_value=_make_result())
    agent.run_sync = MagicMock(return_value=_make_result())

    # Set up stream context managers
    stream_result = _make_stream_result()
    agent.run_stream = MagicMock(return_value=AsyncContextManagerMock(stream_result))
    agent.run_stream_sync = MagicMock(return_value=ContextManagerMock(stream_result))
    return agent


class AsyncContextManagerMock:
    """Mock async context manager for run_stream."""

    def __init__(self, result):
        self._result = result

    async def __aenter__(self):
        return self._result

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        return False


class ContextManagerMock:
    """Mock sync context manager for run_stream_sync."""

    def __init__(self, result):
        self._result = result

    def __enter__(self):
        return self._result

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False


# ---------------------------------------------------------------------------
# Data types tests
# ---------------------------------------------------------------------------


class TestAgentRunMetrics:
    def test_latency_ms_with_end_time(self) -> None:
        m = AgentRunMetrics(start_time=1.0, end_time=1.5)
        assert m.latency_ms == pytest.approx(500.0)

    def test_latency_ms_without_end_time(self) -> None:
        m = AgentRunMetrics(start_time=1.0)
        assert m.latency_ms == 0.0

    def test_defaults(self) -> None:
        m = AgentRunMetrics()
        assert m.model is None
        assert m.input_tokens == 0
        assert m.output_tokens == 0
        assert m.cost == 0.0


class TestPydanticAIHandlerMetrics:
    def test_empty_runs(self) -> None:
        metrics = PydanticAIHandlerMetrics()
        assert metrics.total_cost == 0.0
        assert metrics.run_count == 0

    def test_aggregation(self) -> None:
        runs = [
            AgentRunMetrics(
                input_tokens=100, output_tokens=50, cost=0.01, start_time=0, end_time=1
            ),
            AgentRunMetrics(
                input_tokens=200, output_tokens=100, cost=0.02, start_time=1, end_time=3
            ),
        ]
        metrics = PydanticAIHandlerMetrics(runs=runs)
        assert metrics.total_input_tokens == 300
        assert metrics.total_output_tokens == 150
        assert metrics.total_cost == pytest.approx(0.03)
        assert metrics.run_count == 2
        assert metrics.total_latency_ms == pytest.approx(3000.0)

    def test_to_measures_dict(self) -> None:
        runs = [AgentRunMetrics(input_tokens=100, output_tokens=50, cost=0.01)]
        metrics = PydanticAIHandlerMetrics(runs=runs)
        d = metrics.to_measures_dict(prefix="pai_")
        assert "pai_total_cost" in d
        assert "pai_total_input_tokens" in d
        assert "pai_run_count" in d
        assert d["pai_total_input_tokens"] == 100

    def test_to_measures_dict_empty_prefix(self) -> None:
        metrics = PydanticAIHandlerMetrics()
        d = metrics.to_measures_dict()
        assert "total_cost" in d


# ---------------------------------------------------------------------------
# Handler tests
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _mock_pydantic_ai_available(monkeypatch):
    """Ensure handler module thinks pydantic_ai is installed."""
    import traigent.integrations.pydantic_ai.handler as handler_mod

    monkeypatch.setattr(handler_mod, "PYDANTICAI_AVAILABLE", True)


class TestPydanticAIHandler:
    def _make_handler(self, agent=None, **kwargs):
        from traigent.integrations.pydantic_ai.handler import PydanticAIHandler

        agent = agent or _make_agent()
        return PydanticAIHandler(agent, **kwargs)

    def test_init_extracts_model_name(self) -> None:
        handler = self._make_handler(_make_agent("openai:gpt-4o"))
        assert handler._model_name == "gpt-4o"

    def test_init_model_name_no_colon(self) -> None:
        handler = self._make_handler(_make_agent("gpt-4o"))
        assert handler._model_name == "gpt-4o"

    def test_init_model_name_none(self) -> None:
        agent = MagicMock()
        agent.model = None
        handler = self._make_handler(agent)
        assert handler._model_name == "unknown"

    @pytest.mark.asyncio
    async def test_run_captures_metrics(self) -> None:
        handler = self._make_handler()
        result = await handler.run("Hello")

        assert result.data == "Test response"
        metrics = handler.get_metrics()
        assert metrics.run_count == 1
        assert metrics.total_input_tokens == 100
        assert metrics.total_output_tokens == 50

    @pytest.mark.asyncio
    async def test_run_passes_model_settings(self) -> None:
        agent = _make_agent()
        handler = self._make_handler(agent, traigent_config={"temperature": 0.5})
        await handler.run("Hello")

        call_kwargs = agent.run.call_args
        ms = call_kwargs.kwargs.get("model_settings", {})
        assert ms["temperature"] == 0.5

    @pytest.mark.asyncio
    async def test_run_user_settings_take_precedence(self) -> None:
        agent = _make_agent()
        handler = self._make_handler(agent, traigent_config={"temperature": 0.5})
        await handler.run("Hello", model_settings={"temperature": 0.1})

        call_kwargs = agent.run.call_args
        ms = call_kwargs.kwargs.get("model_settings", {})
        assert ms["temperature"] == 0.1

    def test_run_sync_captures_metrics(self) -> None:
        handler = self._make_handler()
        result = handler.run_sync("Hello")

        assert result.data == "Test response"
        metrics = handler.get_metrics()
        assert metrics.run_count == 1
        assert metrics.total_input_tokens == 100

    @pytest.mark.asyncio
    async def test_run_error_records_failed_run(self) -> None:
        agent = _make_agent()
        agent.run = AsyncMock(side_effect=ValueError("test error"))

        handler = self._make_handler(agent)
        with pytest.raises(ValueError, match="test error"):
            await handler.run("Hello")

        metrics = handler.get_metrics()
        assert metrics.run_count == 1
        assert metrics.total_input_tokens == 0  # Error → no usage

    def test_get_measures_dict_format(self) -> None:
        handler = self._make_handler()
        handler.run_sync("Hello")

        d = handler.get_measures_dict()
        assert "pydantic_ai_total_cost" in d
        assert "pydantic_ai_total_input_tokens" in d
        assert "pydantic_ai_run_count" in d
        assert d["pydantic_ai_run_count"] == 1

    def test_get_measures_dict_custom_prefix(self) -> None:
        handler = self._make_handler(metric_prefix="custom_")
        handler.run_sync("Hello")

        d = handler.get_measures_dict()
        assert "custom_total_cost" in d

    def test_multi_run_aggregation(self) -> None:
        handler = self._make_handler()
        handler.run_sync("Hello")
        handler.run_sync("World")

        metrics = handler.get_metrics()
        assert metrics.run_count == 2
        assert metrics.total_input_tokens == 200  # 100 * 2

    def test_reset_clears_state(self) -> None:
        handler = self._make_handler()
        handler.run_sync("Hello")
        assert handler.get_metrics().run_count == 1

        handler.reset()
        assert handler.get_metrics().run_count == 0

    def test_thread_safety(self) -> None:
        """Multiple threads can safely call run_sync concurrently."""
        handler = self._make_handler()
        errors = []

        def worker():
            try:
                handler.run_sync("Hello")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)

        assert not errors
        assert handler.get_metrics().run_count == 10

    def test_merge_model_settings_empty(self) -> None:
        handler = self._make_handler()
        merged = handler._merge_model_settings(None)
        assert merged == {}

    def test_merge_model_settings_traigent_config(self) -> None:
        handler = self._make_handler(
            traigent_config={"temperature": 0.7, "max_tokens": 500}
        )
        merged = handler._merge_model_settings(None)
        assert merged["temperature"] == 0.7
        assert merged["max_tokens"] == 500

    def test_merge_model_settings_user_overrides(self) -> None:
        handler = self._make_handler(traigent_config={"temperature": 0.7})
        merged = handler._merge_model_settings({"temperature": 0.1, "top_p": 0.9})
        assert merged["temperature"] == 0.1  # User wins
        assert merged["top_p"] == 0.9

    def test_estimate_cost_zero_tokens(self) -> None:
        handler = self._make_handler()
        cost = handler._estimate_cost(0, 0)
        assert cost == 0.0

    def test_estimate_cost_with_tokens(self) -> None:
        """_estimate_cost calls cost_from_tokens for non-zero tokens."""
        handler = self._make_handler()
        cost = handler._estimate_cost(100, 50)
        # Should return a float (may be 0.0 for unknown models)
        assert isinstance(cost, float)

    def test_estimate_cost_unknown_model_raises_in_strict_mode(self) -> None:
        """Strict cost accounting raises for unknown model pricing."""
        from traigent.utils.cost_calculator import UnknownModelError

        handler = self._make_handler(_make_agent("openai:unknown-model-xyz-123"))
        with patch.dict(
            "os.environ", {"TRAIGENT_STRICT_COST_ACCOUNTING": "true"}, clear=False
        ):
            with pytest.raises(UnknownModelError):
                handler._estimate_cost(100, 50)

    @patch(
        "traigent.utils.cost_calculator.cost_from_tokens",
        side_effect=RuntimeError("cost calc failed"),
    )
    def test_estimate_cost_handles_failure(self, _mock) -> None:
        """_estimate_cost returns 0.0 when cost_from_tokens raises."""
        handler = self._make_handler()
        cost = handler._estimate_cost(100, 50)
        assert cost == 0.0

    @patch(
        "traigent.integrations.pydantic_ai.handler.PydanticAIHandler._estimate_cost",
        return_value=0.005,
    )
    def test_cost_recorded_in_metrics(self, mock_cost) -> None:
        handler = self._make_handler()
        handler.run_sync("Hello")

        metrics = handler.get_metrics()
        assert metrics.total_cost == pytest.approx(0.005)

    def test_run_sync_error_records_failed_run(self) -> None:
        """run_sync should record a failed run on exception."""
        agent = _make_agent()
        agent.run_sync = MagicMock(side_effect=ValueError("sync error"))

        handler = self._make_handler(agent)
        with pytest.raises(ValueError, match="sync error"):
            handler.run_sync("Hello")

        metrics = handler.get_metrics()
        assert metrics.run_count == 1
        assert metrics.total_input_tokens == 0

    def test_record_usage_fallback_on_bad_result(self) -> None:
        """_record_usage should still append metrics when usage() raises."""
        handler = self._make_handler()
        bad_result = MagicMock()
        bad_result.usage.side_effect = AttributeError("no usage")

        handler._record_usage(bad_result, 1.0)

        metrics = handler.get_metrics()
        assert metrics.run_count == 1
        assert metrics.total_input_tokens == 0  # Fallback metrics


class TestStreamingMethods:
    """Tests for run_stream and run_stream_sync."""

    def _make_handler(self, agent=None, **kwargs):
        from traigent.integrations.pydantic_ai.handler import PydanticAIHandler

        agent = agent or _make_agent()
        return PydanticAIHandler(agent, **kwargs)

    @pytest.mark.asyncio
    async def test_run_stream_captures_metrics(self) -> None:
        handler = self._make_handler()
        async with handler.run_stream("Hello") as stream:
            _ = stream.data  # consume stream

        metrics = handler.get_metrics()
        assert metrics.run_count == 1
        assert metrics.total_input_tokens == 100

    @pytest.mark.asyncio
    async def test_run_stream_error_records_failed_run(self) -> None:
        handler = self._make_handler()
        with pytest.raises(ValueError, match="stream error"):
            async with handler.run_stream("Hello"):
                raise ValueError("stream error")

        metrics = handler.get_metrics()
        assert metrics.run_count == 1
        assert metrics.total_input_tokens == 0  # Error → no usage

    @pytest.mark.asyncio
    async def test_run_stream_passes_model_settings(self) -> None:
        agent = _make_agent()
        handler = self._make_handler(agent, traigent_config={"temperature": 0.5})
        async with handler.run_stream("Hello") as stream:
            _ = stream.data

        call_kwargs = agent.run_stream.call_args
        assert call_kwargs.kwargs.get("model_settings", {})[
            "temperature"
        ] == pytest.approx(0.5)

    def test_run_stream_sync_captures_metrics(self) -> None:
        handler = self._make_handler()
        with handler.run_stream_sync("Hello") as stream:
            _ = stream.data

        metrics = handler.get_metrics()
        assert metrics.run_count == 1
        assert metrics.total_input_tokens == 100

    def test_run_stream_sync_error_records_failed_run(self) -> None:
        handler = self._make_handler()
        with pytest.raises(ValueError, match="stream error"):
            with handler.run_stream_sync("Hello"):
                raise ValueError("stream error")

        metrics = handler.get_metrics()
        assert metrics.run_count == 1
        assert metrics.total_input_tokens == 0  # Error → no usage

    def test_run_stream_sync_passes_model_settings(self) -> None:
        agent = _make_agent()
        handler = self._make_handler(agent, traigent_config={"temperature": 0.3})
        with handler.run_stream_sync("Hello") as stream:
            _ = stream.data

        call_kwargs = agent.run_stream_sync.call_args
        assert call_kwargs.kwargs.get("model_settings", {})[
            "temperature"
        ] == pytest.approx(0.3)


class TestExtractModelName:
    def _extract(self, model_value):
        from traigent.integrations.pydantic_ai.handler import PydanticAIHandler

        agent = MagicMock()
        agent.model = model_value
        return PydanticAIHandler._extract_model_name(agent)

    def test_string_with_colon(self) -> None:
        assert self._extract("openai:gpt-4o") == "gpt-4o"

    def test_string_without_colon(self) -> None:
        assert self._extract("gpt-4o") == "gpt-4o"

    def test_none(self) -> None:
        assert self._extract(None) == "unknown"

    def test_model_object_with_model_name(self) -> None:
        model = SimpleNamespace(model_name="claude-3-5-sonnet-20241022")
        assert self._extract(model) == "claude-3-5-sonnet-20241022"

    def test_model_object_str_fallback(self) -> None:
        model = MagicMock()
        model.model_name = None
        model.model_id = None
        model.name = None
        model.__str__ = lambda self: "anthropic:claude-3-opus"
        assert self._extract(model) == "claude-3-opus"

    def test_model_object_str_no_colon(self) -> None:
        model = MagicMock()
        model.model_name = None
        model.model_id = None
        model.name = None
        model.__str__ = lambda self: "gpt-4o-mini"
        assert self._extract(model) == "gpt-4o-mini"


class TestFactoryFunction:
    """Test create_pydantic_ai_handler factory."""

    def test_create_pydantic_ai_handler(self) -> None:
        from traigent.integrations.pydantic_ai.handler import create_pydantic_ai_handler

        agent = _make_agent()
        handler = create_pydantic_ai_handler(agent, metric_prefix="test_")
        assert handler._metric_prefix == "test_"
        assert handler._model_name == "gpt-4o"

    def test_create_with_traigent_config(self) -> None:
        from traigent.integrations.pydantic_ai.handler import create_pydantic_ai_handler

        agent = _make_agent()
        handler = create_pydantic_ai_handler(
            agent, traigent_config={"temperature": 0.5}
        )
        assert handler._traigent_config == {"temperature": 0.5}
