"""Unit tests for HybridAPIEvaluator.

Tests cover:
- __init__ and configuration storage
- _get_transport (lazy creation and reuse)
- _get_capabilities (lazy caching)
- discover_config_space
- _ensure_lifecycle_manager
- evaluate (normal, sample_lease, empty dataset, progress_callback)
- _execute_batch (combined, two-phase, execute-only, error handling)
- _extract_input and _extract_expected helper methods
- _process_combined_response, _evaluate_outputs, _process_execute_only_response
- _compute_aggregated_metrics
- close, __aenter__, __aexit__
- HybridExampleResult.success property
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from traigent.cloud.dtos import MeasuresDict
from traigent.evaluators.base import Dataset, EvaluationExample, EvaluationResult
from traigent.evaluators.hybrid_api import HybridAPIEvaluator, HybridExampleResult
from traigent.hybrid.protocol import (
    BenchmarkEntry,
    BenchmarksResponse,
    HybridEvaluateResponse,
    HybridExecuteResponse,
    ServiceCapabilities,
)
from traigent.hybrid.transport import TransportError

# ---------------------------------------------------------------------------
# Helpers / Fixtures
# ---------------------------------------------------------------------------


def _make_dataset(examples: list[dict[str, Any]] | None = None) -> Dataset:
    """Create a Dataset from dicts for testing."""
    if examples is None:
        examples = [
            {"input_data": {"question": "What is 2+2?"}, "expected_output": "4"},
            {"input_data": {"question": "What is 3+3?"}, "expected_output": "6"},
        ]
    eval_examples = [
        EvaluationExample(
            input_data=e["input_data"],
            expected_output=e.get("expected_output"),
        )
        for e in examples
    ]
    return Dataset(examples=eval_examples, name="test")


def _make_execute_response(
    *,
    outputs: list[dict[str, Any]] | None = None,
    operational_metrics: dict[str, float] | None = None,
    quality_metrics: dict[str, float] | None = None,
    session_id: str | None = None,
    execution_id: str = "exec-1",
) -> HybridExecuteResponse:
    """Create a mock HybridExecuteResponse."""
    return HybridExecuteResponse(
        request_id="req-1",
        execution_id=execution_id,
        status="completed",
        outputs=outputs or [],
        operational_metrics=operational_metrics
        or {"cost_usd": 0.01, "latency_ms": 100.0},
        quality_metrics=quality_metrics,
        session_id=session_id,
    )


def _make_evaluate_response(
    results: list[dict[str, Any]] | None = None,
) -> HybridEvaluateResponse:
    """Create a mock HybridEvaluateResponse."""
    return HybridEvaluateResponse(
        request_id="req-1",
        status="completed",
        results=results or [],
        aggregate_metrics={},
    )


def _default_capabilities(
    supports_evaluate: bool = True,
    supports_keep_alive: bool = False,
) -> ServiceCapabilities:
    return ServiceCapabilities(
        version="1.0",
        supports_evaluate=supports_evaluate,
        supports_keep_alive=supports_keep_alive,
    )


@pytest.fixture
def mock_transport() -> MagicMock:
    """Create a fully mocked HybridTransport."""
    transport = MagicMock()
    transport.execute = AsyncMock(
        return_value=_make_execute_response(
            outputs=[
                {"example_id": "ex_0", "output": "result_0"},
                {"example_id": "ex_1", "output": "result_1"},
            ],
        )
    )
    transport.evaluate = AsyncMock(
        return_value=_make_evaluate_response(
            results=[
                {"example_id": "ex_0", "metrics": {"accuracy": 1.0}},
                {"example_id": "ex_1", "metrics": {"accuracy": 0.5}},
            ]
        )
    )
    transport.capabilities = AsyncMock(return_value=_default_capabilities())
    transport.benchmarks = AsyncMock(
        return_value=BenchmarksResponse(
            benchmarks=[
                BenchmarkEntry(
                    benchmark_id="test-bench",
                    tunable_ids=["test_cap"],
                    example_ids=["ex_0", "ex_1"],
                )
            ]
        )
    )
    transport.close = AsyncMock()
    transport.keep_alive = AsyncMock(return_value=True)
    return transport


@pytest.fixture
def evaluator(mock_transport: MagicMock) -> HybridAPIEvaluator:
    """Create a HybridAPIEvaluator with an injected mock transport."""
    return HybridAPIEvaluator(
        transport=mock_transport,
        tunable_id="test_cap",
        batch_size=10,
        keep_alive=False,
    )


# ---------------------------------------------------------------------------
# HybridExampleResult tests
# ---------------------------------------------------------------------------


class TestHybridExampleResult:
    """Tests for HybridExampleResult dataclass."""

    def test_success_when_no_error(self) -> None:
        """success property returns True when error is None."""
        result = HybridExampleResult(example_id="ex_0", error=None)
        assert result.success is True

    def test_not_success_when_error(self) -> None:
        """success property returns False when error is set."""
        result = HybridExampleResult(example_id="ex_0", error="something went wrong")
        assert result.success is False

    def test_default_values(self) -> None:
        """Default field values are sensible."""
        result = HybridExampleResult(example_id="x")
        assert result.actual_output is None
        assert result.expected_output is None
        assert result.metrics == {}
        assert result.cost_usd == 0.0
        assert result.latency_ms == 0.0
        assert result.error is None


# ---------------------------------------------------------------------------
# __init__ tests
# ---------------------------------------------------------------------------


class TestHybridAPIEvaluatorInit:
    """Tests for HybridAPIEvaluator.__init__."""

    def test_init_with_transport(self, mock_transport: MagicMock) -> None:
        """When transport is provided, it is stored and not owned."""
        ev = HybridAPIEvaluator(transport=mock_transport)
        assert ev._transport is mock_transport
        assert ev._owns_transport is False

    def test_init_without_transport_owns_transport(self) -> None:
        """When no transport is provided, owns_transport is True."""
        ev = HybridAPIEvaluator(api_endpoint="http://localhost:8080")
        assert ev._transport is None
        assert ev._owns_transport is True

    def test_init_stores_configuration(self) -> None:
        """All configuration arguments are stored correctly."""
        ev = HybridAPIEvaluator(
            api_endpoint="http://example.com",
            transport_type="http",
            tunable_id="my_cap",
            auto_discover_tvars=False,
            batch_size=5,
            batch_parallelism=3,
            keep_alive=True,
            heartbeat_interval=15.0,
            timeout=120.0,
            auth_header="Bearer tok",
        )
        assert ev._api_endpoint == "http://example.com"
        assert ev._transport_type == "http"
        assert ev._tunable_id == "my_cap"
        assert ev._auto_discover is False
        assert ev._batch_size == 5
        assert ev._batch_parallelism == 3
        assert ev._keep_alive_enabled is True
        assert ev._heartbeat_interval == 15.0
        assert ev._timeout == 120.0
        assert ev._auth_header == "Bearer tok"

    def test_init_batch_size_minimum_clamp(self) -> None:
        """batch_size is clamped to minimum of 1."""
        ev = HybridAPIEvaluator(transport=MagicMock(), batch_size=0)
        assert ev._batch_size == 1
        ev2 = HybridAPIEvaluator(transport=MagicMock(), batch_size=-5)
        assert ev2._batch_size == 1

    def test_init_batch_parallelism_minimum_clamp(self) -> None:
        """batch_parallelism is clamped to minimum of 1."""
        ev = HybridAPIEvaluator(transport=MagicMock(), batch_parallelism=0)
        assert ev._batch_parallelism == 1

    def test_init_lazy_components_none(self) -> None:
        """Lazy components are None after init."""
        ev = HybridAPIEvaluator(transport=MagicMock())
        assert ev._lifecycle_manager is None
        assert ev._discovery is None
        assert ev._capabilities is None
        assert ev._session_id is None


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------


class TestProperties:
    """Tests for simple property accessors."""

    def test_lifecycle_manager_property(self, evaluator: HybridAPIEvaluator) -> None:
        """lifecycle_manager returns internal _lifecycle_manager."""
        assert evaluator.lifecycle_manager is None
        sentinel = MagicMock()
        evaluator._lifecycle_manager = sentinel
        assert evaluator.lifecycle_manager is sentinel

    def test_tunable_id_property(self, evaluator: HybridAPIEvaluator) -> None:
        """tunable_id returns internal _tunable_id."""
        assert evaluator.tunable_id == "test_cap"


# ---------------------------------------------------------------------------
# _get_transport tests
# ---------------------------------------------------------------------------


class TestGetTransport:
    """Tests for _get_transport method."""

    @pytest.mark.asyncio
    async def test_returns_existing_transport(
        self, evaluator: HybridAPIEvaluator, mock_transport: MagicMock
    ) -> None:
        """Returns injected transport without creating a new one."""
        transport = await evaluator._get_transport()
        assert transport is mock_transport

    @pytest.mark.asyncio
    async def test_creates_transport_when_none(self) -> None:
        """Creates transport via create_transport when none is provided."""
        ev = HybridAPIEvaluator(
            api_endpoint="http://localhost:8080",
            transport_type="http",
            auth_header="Bearer tok",
            timeout=60.0,
        )
        mock_created = MagicMock()
        with patch(
            "traigent.evaluators.hybrid_api.create_transport",
            return_value=mock_created,
        ) as mock_factory:
            transport = await ev._get_transport()
            assert transport is mock_created
            mock_factory.assert_called_once_with(
                transport_type="http",
                base_url="http://localhost:8080",
                auth_header="Bearer tok",
                timeout=60.0,
                mcp_client=None,
                mcp_config=None,
            )
            assert ev._owns_transport is True

    @pytest.mark.asyncio
    async def test_cached_after_creation(self) -> None:
        """Once created, subsequent calls return the same transport."""
        ev = HybridAPIEvaluator(api_endpoint="http://localhost:8080")
        mock_created = MagicMock()
        with patch(
            "traigent.evaluators.hybrid_api.create_transport",
            return_value=mock_created,
        ) as mock_factory:
            t1 = await ev._get_transport()
            t2 = await ev._get_transport()
            assert t1 is t2
            mock_factory.assert_called_once()


# ---------------------------------------------------------------------------
# _get_capabilities tests
# ---------------------------------------------------------------------------


class TestGetCapabilities:
    """Tests for _get_capabilities method."""

    @pytest.mark.asyncio
    async def test_fetches_and_caches(
        self, evaluator: HybridAPIEvaluator, mock_transport: MagicMock
    ) -> None:
        """Fetches capabilities on first call and caches."""
        caps = await evaluator._get_capabilities()
        assert caps.version == "1.0"
        mock_transport.capabilities.assert_awaited_once()

        # Second call should use cache
        caps2 = await evaluator._get_capabilities()
        assert caps2 is caps
        assert mock_transport.capabilities.await_count == 1


# ---------------------------------------------------------------------------
# discover_config_space tests
# ---------------------------------------------------------------------------


class TestDiscoverConfigSpace:
    """Tests for discover_config_space method."""

    @pytest.mark.asyncio
    async def test_discover_config_space(
        self, evaluator: HybridAPIEvaluator, mock_transport: MagicMock
    ) -> None:
        """Fetches config space via ConfigSpaceDiscovery."""
        mock_discovery = MagicMock()
        mock_discovery.fetch_and_normalize = AsyncMock(
            return_value={
                "model": ["gpt-4", "claude-3"],
                "temperature": {"low": 0, "high": 1},
            }
        )
        mock_discovery.get_tunable_id = MagicMock(return_value="discovered_cap")
        mock_discovery.build_optimization_spec = AsyncMock(
            return_value={"runtime_overrides": {"max_trials": 50}}
        )

        with patch(
            "traigent.evaluators.hybrid_api.ConfigSpaceDiscovery",
            return_value=mock_discovery,
        ):
            result = await evaluator.discover_config_space()

        assert result == {
            "model": ["gpt-4", "claude-3"],
            "temperature": {"low": 0, "high": 1},
        }
        assert evaluator.optimization_spec == {"runtime_overrides": {"max_trials": 50}}

    @pytest.mark.asyncio
    async def test_discover_updates_tunable_id_when_none(
        self, mock_transport: MagicMock
    ) -> None:
        """If tunable_id is None, discovery updates it."""
        ev = HybridAPIEvaluator(
            transport=mock_transport,
            tunable_id=None,
            keep_alive=False,
        )
        mock_discovery = MagicMock()
        mock_discovery.fetch_and_normalize = AsyncMock(return_value={})
        mock_discovery.get_tunable_id = MagicMock(return_value="auto_cap")
        mock_discovery.build_optimization_spec = AsyncMock(return_value={})

        with patch(
            "traigent.evaluators.hybrid_api.ConfigSpaceDiscovery",
            return_value=mock_discovery,
        ):
            await ev.discover_config_space()

        assert ev._tunable_id == "auto_cap"

    @pytest.mark.asyncio
    async def test_discover_keeps_existing_tunable_id(
        self, evaluator: HybridAPIEvaluator, mock_transport: MagicMock
    ) -> None:
        """If tunable_id is set, discovery does not overwrite it."""
        mock_discovery = MagicMock()
        mock_discovery.fetch_and_normalize = AsyncMock(return_value={})
        mock_discovery.get_tunable_id = MagicMock(return_value="other")
        mock_discovery.build_optimization_spec = AsyncMock(return_value={})

        with patch(
            "traigent.evaluators.hybrid_api.ConfigSpaceDiscovery",
            return_value=mock_discovery,
        ):
            await evaluator.discover_config_space()

        assert evaluator._tunable_id == "test_cap"

    @pytest.mark.asyncio
    async def test_discover_reuses_discovery_instance(
        self, evaluator: HybridAPIEvaluator, mock_transport: MagicMock
    ) -> None:
        """ConfigSpaceDiscovery is created once and reused."""
        mock_discovery = MagicMock()
        mock_discovery.fetch_and_normalize = AsyncMock(return_value={})
        mock_discovery.get_tunable_id = MagicMock(return_value="cap")
        mock_discovery.build_optimization_spec = AsyncMock(return_value={})

        with patch(
            "traigent.evaluators.hybrid_api.ConfigSpaceDiscovery",
            return_value=mock_discovery,
        ) as mock_cls:
            await evaluator.discover_config_space()
            await evaluator.discover_config_space()
            # Constructor called only once
            mock_cls.assert_called_once()


# ---------------------------------------------------------------------------
# discover_example_ids tests
# ---------------------------------------------------------------------------


class TestDiscoverExampleIds:
    """Tests for discover_example_ids method."""

    @pytest.mark.asyncio
    async def test_discover_example_ids_single_benchmark(
        self, evaluator: HybridAPIEvaluator, mock_transport: MagicMock
    ) -> None:
        """Returns all example IDs from a single benchmark."""
        from traigent.hybrid.protocol import BenchmarkEntry, BenchmarksResponse

        mock_transport.benchmarks = AsyncMock(
            return_value=BenchmarksResponse(
                benchmarks=[
                    BenchmarkEntry(
                        benchmark_id="bench_001",
                        tunable_ids=["test_cap"],
                        example_ids=["case_001", "case_002", "case_003"],
                        name="Test Benchmark",
                    )
                ],
                benchmarks_revision=None,
            )
        )

        ids = await evaluator.discover_example_ids()

        assert ids == ["case_001", "case_002", "case_003"]
        mock_transport.benchmarks.assert_called_once_with(tunable_id="test_cap")

    @pytest.mark.asyncio
    async def test_discover_example_ids_multiple_benchmarks_raises(
        self, evaluator: HybridAPIEvaluator, mock_transport: MagicMock
    ) -> None:
        """Raises ValueError when multiple benchmarks match the tunable."""
        from traigent.hybrid.protocol import BenchmarkEntry, BenchmarksResponse

        mock_transport.benchmarks = AsyncMock(
            return_value=BenchmarksResponse(
                benchmarks=[
                    BenchmarkEntry(
                        benchmark_id="bench_001",
                        tunable_ids=["test_cap"],
                        example_ids=["a", "b"],
                    ),
                    BenchmarkEntry(
                        benchmark_id="bench_002",
                        tunable_ids=["test_cap"],
                        example_ids=["c", "d", "e"],
                    ),
                ],
                benchmarks_revision=None,
            )
        )

        with pytest.raises(ValueError, match="Multiple benchmarks match"):
            await evaluator.discover_example_ids()

        assert mock_transport.benchmarks.call_count == 1

    @pytest.mark.asyncio
    async def test_discover_example_ids_multiple_benchmarks_with_explicit_id(
        self, evaluator: HybridAPIEvaluator, mock_transport: MagicMock
    ) -> None:
        """Selects correct benchmark when benchmark_id is provided explicitly."""
        from traigent.hybrid.protocol import BenchmarkEntry, BenchmarksResponse

        mock_transport.benchmarks = AsyncMock(
            return_value=BenchmarksResponse(
                benchmarks=[
                    BenchmarkEntry(
                        benchmark_id="bench_001",
                        tunable_ids=["test_cap"],
                        example_ids=["a", "b"],
                    ),
                    BenchmarkEntry(
                        benchmark_id="bench_002",
                        tunable_ids=["test_cap"],
                        example_ids=["c", "d", "e"],
                    ),
                ],
                benchmarks_revision=None,
            )
        )

        ids = await evaluator.discover_example_ids(benchmark_id="bench_002")

        assert ids == ["c", "d", "e"]
        assert evaluator._benchmark_id == "bench_002"
        assert mock_transport.benchmarks.call_count == 1

    @pytest.mark.asyncio
    async def test_discover_example_ids_empty_benchmarks_raises(
        self, evaluator: HybridAPIEvaluator, mock_transport: MagicMock
    ) -> None:
        """Raises ValueError when no benchmarks are available."""
        from traigent.hybrid.protocol import BenchmarksResponse

        mock_transport.benchmarks = AsyncMock(
            return_value=BenchmarksResponse(
                benchmarks=[],
                benchmarks_revision=None,
            )
        )

        with pytest.raises(ValueError, match="No benchmarks found"):
            await evaluator.discover_example_ids()

        assert mock_transport.benchmarks.call_count == 1

    @pytest.mark.asyncio
    async def test_discover_example_ids_explicit_tunable(
        self, evaluator: HybridAPIEvaluator, mock_transport: MagicMock
    ) -> None:
        """Uses explicit tunable_id when provided."""
        from traigent.hybrid.protocol import BenchmarkEntry, BenchmarksResponse

        mock_transport.benchmarks = AsyncMock(
            return_value=BenchmarksResponse(
                benchmarks=[
                    BenchmarkEntry(
                        benchmark_id="bench_001",
                        tunable_ids=["other"],
                        example_ids=["x"],
                    )
                ],
                benchmarks_revision=None,
            )
        )

        ids = await evaluator.discover_example_ids(tunable_id="other")

        assert ids == ["x"]
        mock_transport.benchmarks.assert_called_once_with(tunable_id="other")

    @pytest.mark.asyncio
    async def test_discover_example_ids_no_tunable_raises(
        self, mock_transport: MagicMock
    ) -> None:
        """Raises ValueError when no tunable_id is available."""
        ev = HybridAPIEvaluator(
            transport=mock_transport,
            tunable_id=None,
            keep_alive=False,
        )

        with pytest.raises(ValueError, match="tunable_id is required"):
            await ev.discover_example_ids()


# ---------------------------------------------------------------------------
# _ensure_lifecycle_manager tests
# ---------------------------------------------------------------------------


class TestEnsureLifecycleManager:
    """Tests for _ensure_lifecycle_manager."""

    @pytest.mark.asyncio
    async def test_noop_when_keep_alive_disabled(
        self, evaluator: HybridAPIEvaluator
    ) -> None:
        """Does nothing when keep_alive is disabled."""
        evaluator._keep_alive_enabled = False
        await evaluator._ensure_lifecycle_manager()
        assert evaluator._lifecycle_manager is None

    @pytest.mark.asyncio
    async def test_noop_when_service_does_not_support(
        self, mock_transport: MagicMock
    ) -> None:
        """Does nothing when service does not support keep-alive."""
        mock_transport.capabilities = AsyncMock(
            return_value=_default_capabilities(supports_keep_alive=False)
        )
        ev = HybridAPIEvaluator(transport=mock_transport, keep_alive=True)
        await ev._ensure_lifecycle_manager()
        assert ev._lifecycle_manager is None

    @pytest.mark.asyncio
    async def test_creates_lifecycle_manager(self, mock_transport: MagicMock) -> None:
        """Creates lifecycle manager when keep-alive is supported."""
        mock_transport.capabilities = AsyncMock(
            return_value=_default_capabilities(supports_keep_alive=True)
        )

        mock_lm = MagicMock()
        mock_lm.register = AsyncMock()

        ev = HybridAPIEvaluator(
            transport=mock_transport,
            keep_alive=True,
            heartbeat_interval=10.0,
        )

        with patch(
            "traigent.evaluators.hybrid_api.AgentLifecycleManager",
            return_value=mock_lm,
        ):
            await ev._ensure_lifecycle_manager()

        assert ev._lifecycle_manager is mock_lm
        assert ev._session_id is None
        mock_lm.register.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_idempotent(self, mock_transport: MagicMock) -> None:
        """Calling multiple times does not re-create."""
        mock_transport.capabilities = AsyncMock(
            return_value=_default_capabilities(supports_keep_alive=True)
        )
        mock_lm = MagicMock()
        mock_lm.register = AsyncMock()

        ev = HybridAPIEvaluator(transport=mock_transport, keep_alive=True)

        with patch(
            "traigent.evaluators.hybrid_api.AgentLifecycleManager",
            return_value=mock_lm,
        ) as mock_cls:
            await ev._ensure_lifecycle_manager()
            await ev._ensure_lifecycle_manager()
            mock_cls.assert_called_once()

    @pytest.mark.asyncio
    async def test_registers_existing_session_id(
        self, mock_transport: MagicMock
    ) -> None:
        """Registers keep-alive only after a real session_id is known."""
        mock_transport.capabilities = AsyncMock(
            return_value=_default_capabilities(supports_keep_alive=True)
        )
        mock_lm = MagicMock()
        mock_lm.register = AsyncMock()
        ev = HybridAPIEvaluator(transport=mock_transport, keep_alive=True)
        ev._session_id = "session-abc"

        with patch(
            "traigent.evaluators.hybrid_api.AgentLifecycleManager",
            return_value=mock_lm,
        ):
            await ev._ensure_lifecycle_manager()

        mock_lm.register.assert_awaited_once_with("session-abc")


# ---------------------------------------------------------------------------
# _extract_input tests
# ---------------------------------------------------------------------------


class TestExtractInput:
    """Tests for _extract_input helper method."""

    @pytest.fixture
    def ev(self, mock_transport: MagicMock) -> HybridAPIEvaluator:
        return HybridAPIEvaluator(transport=mock_transport, keep_alive=False)

    def test_extract_from_evaluation_example_dict(self, ev: HybridAPIEvaluator) -> None:
        """Handles EvaluationExample with dict input_data."""
        example = EvaluationExample(input_data={"question": "hi"})
        result = ev._extract_input(example)
        assert result == {"question": "hi"}

    def test_extract_from_evaluation_example_non_dict(
        self, ev: HybridAPIEvaluator
    ) -> None:
        """Handles EvaluationExample with non-dict input_data."""
        example = MagicMock()
        example.input_data = "plain text"
        result = ev._extract_input(example)
        assert result == {"input": "plain text"}

    def test_extract_from_dict_with_input_key(self, ev: HybridAPIEvaluator) -> None:
        """Handles dict with 'input' key."""
        example = {"input": "hello", "extra": "data"}
        result = ev._extract_input(example)
        assert result == {"input": "hello", "extra": "data"}

    def test_extract_from_dict_with_question_key(self, ev: HybridAPIEvaluator) -> None:
        """Handles dict with 'question' key."""
        example = {"question": "What?", "context": "none"}
        result = ev._extract_input(example)
        assert result == {"input": "What?", "context": "none"}

    def test_extract_from_dict_with_query_key(self, ev: HybridAPIEvaluator) -> None:
        """Handles dict with 'query' key."""
        example = {"query": "search term"}
        result = ev._extract_input(example)
        assert result == {"input": "search term"}

    def test_extract_from_dict_with_text_key(self, ev: HybridAPIEvaluator) -> None:
        """Handles dict with 'text' key."""
        example = {"text": "document text"}
        result = ev._extract_input(example)
        assert result == {"input": "document text"}

    def test_extract_from_dict_with_data_key(self, ev: HybridAPIEvaluator) -> None:
        """Handles dict with 'data' key."""
        example = {"data": {"nested": True}}
        result = ev._extract_input(example)
        assert result == {"input": {"nested": True}}

    def test_extract_from_dict_without_known_key(self, ev: HybridAPIEvaluator) -> None:
        """Handles dict without a recognized input key."""
        example = {"custom_field": "value", "other": 42}
        result = ev._extract_input(example)
        assert result == {"custom_field": "value", "other": 42}

    def test_extract_from_non_dict_non_example(self, ev: HybridAPIEvaluator) -> None:
        """Handles raw non-dict, non-EvaluationExample values."""
        result = ev._extract_input("just a string")
        assert result == {"input": "just a string"}

    def test_extract_from_integer(self, ev: HybridAPIEvaluator) -> None:
        """Handles raw integer input."""
        result = ev._extract_input(42)
        assert result == {"input": 42}


# ---------------------------------------------------------------------------
# _extract_expected tests
# ---------------------------------------------------------------------------


class TestExtractExpected:
    """Tests for _extract_expected helper method."""

    @pytest.fixture
    def ev(self, mock_transport: MagicMock) -> HybridAPIEvaluator:
        return HybridAPIEvaluator(transport=mock_transport, keep_alive=False)

    def test_extract_from_evaluation_example(self, ev: HybridAPIEvaluator) -> None:
        """Handles EvaluationExample with expected_output."""
        example = EvaluationExample(input_data={"q": "?"}, expected_output="answer")
        result = ev._extract_expected(example)
        assert result == "answer"

    def test_extract_from_evaluation_example_none(self, ev: HybridAPIEvaluator) -> None:
        """Handles EvaluationExample with None expected_output."""
        example = EvaluationExample(input_data={"q": "?"}, expected_output=None)
        result = ev._extract_expected(example)
        assert result is None

    def test_extract_from_dict_expected_output(self, ev: HybridAPIEvaluator) -> None:
        """Handles dict with 'expected_output' key."""
        result = ev._extract_expected({"expected_output": "yes"})
        assert result == "yes"

    def test_extract_from_dict_output(self, ev: HybridAPIEvaluator) -> None:
        """Handles dict with 'output' key."""
        result = ev._extract_expected({"output": "hello"})
        assert result == "hello"

    def test_extract_from_dict_answer(self, ev: HybridAPIEvaluator) -> None:
        """Handles dict with 'answer' key."""
        result = ev._extract_expected({"answer": "42"})
        assert result == "42"

    def test_extract_from_dict_target(self, ev: HybridAPIEvaluator) -> None:
        """Handles dict with 'target' key."""
        result = ev._extract_expected({"target": [1, 2]})
        assert result == [1, 2]

    def test_extract_from_dict_label(self, ev: HybridAPIEvaluator) -> None:
        """Handles dict with 'label' key."""
        result = ev._extract_expected({"label": "positive"})
        assert result == "positive"

    def test_extract_from_dict_no_known_key(self, ev: HybridAPIEvaluator) -> None:
        """Returns None for dict without recognized expected key."""
        result = ev._extract_expected({"custom": "data"})
        assert result is None

    def test_extract_from_non_dict(self, ev: HybridAPIEvaluator) -> None:
        """Returns None for non-dict, non-EvaluationExample."""
        result = ev._extract_expected("just a string")
        assert result is None

    def test_extract_from_integer(self, ev: HybridAPIEvaluator) -> None:
        """Returns None for integer."""
        result = ev._extract_expected(123)
        assert result is None


# ---------------------------------------------------------------------------
# _compute_aggregated_metrics tests
# ---------------------------------------------------------------------------


class TestComputeAggregatedMetrics:
    """Tests for _compute_aggregated_metrics method."""

    @pytest.fixture
    def ev(self, mock_transport: MagicMock) -> HybridAPIEvaluator:
        return HybridAPIEvaluator(transport=mock_transport, keep_alive=False)

    def test_empty_results(self, ev: HybridAPIEvaluator) -> None:
        """Returns empty dict for empty results."""
        result = ev._compute_aggregated_metrics([], 0.0)
        assert result == {}

    def test_basic_metrics(self, ev: HybridAPIEvaluator) -> None:
        """Computes cost and success_rate."""
        results = [
            HybridExampleResult(example_id="1", metrics={}),
            HybridExampleResult(example_id="2", error="fail", metrics={}),
        ]
        agg = ev._compute_aggregated_metrics(results, total_cost=0.05)
        assert agg["cost"] == 0.05
        assert agg["total_cost"] == 0.05
        assert agg["success_rate"] == 0.5

    def test_all_successful(self, ev: HybridAPIEvaluator) -> None:
        """success_rate is 1.0 when all succeed."""
        results = [
            HybridExampleResult(example_id="1"),
            HybridExampleResult(example_id="2"),
        ]
        agg = ev._compute_aggregated_metrics(results, total_cost=0.0)
        assert agg["success_rate"] == 1.0

    def test_per_example_metrics_averaged(self, ev: HybridAPIEvaluator) -> None:
        """Per-example metrics are averaged across results."""
        results = [
            HybridExampleResult(example_id="1", metrics={"accuracy": 1.0, "f1": 0.8}),
            HybridExampleResult(example_id="2", metrics={"accuracy": 0.5, "f1": 0.6}),
        ]
        agg = ev._compute_aggregated_metrics(results, total_cost=0.0)
        assert agg["accuracy"] == pytest.approx(0.75)
        assert agg["f1"] == pytest.approx(0.7)

    def test_latency_averaged(self, ev: HybridAPIEvaluator) -> None:
        """Average latency is computed from positive latency values."""
        results = [
            HybridExampleResult(example_id="1", latency_ms=100.0),
            HybridExampleResult(example_id="2", latency_ms=200.0),
            HybridExampleResult(example_id="3", latency_ms=0.0),  # skipped
        ]
        agg = ev._compute_aggregated_metrics(results, total_cost=0.0)
        assert agg["latency"] == pytest.approx(150.0)

    def test_no_positive_latency(self, ev: HybridAPIEvaluator) -> None:
        """No 'latency' key when all latencies are zero."""
        results = [
            HybridExampleResult(example_id="1", latency_ms=0.0),
        ]
        agg = ev._compute_aggregated_metrics(results, total_cost=0.0)
        assert "latency" not in agg

    def test_partial_metrics(self, ev: HybridAPIEvaluator) -> None:
        """Metrics only present in some results are averaged over their count."""
        results = [
            HybridExampleResult(example_id="1", metrics={"accuracy": 1.0}),
            HybridExampleResult(example_id="2", metrics={}),
        ]
        agg = ev._compute_aggregated_metrics(results, total_cost=0.0)
        # accuracy only has 1 data point
        assert agg["accuracy"] == pytest.approx(1.0)

    def test_infers_accuracy_from_overall_accuracy(
        self, ev: HybridAPIEvaluator
    ) -> None:
        """Derives canonical accuracy from overall_accuracy when missing."""
        results = [
            HybridExampleResult(example_id="1", metrics={"overall_accuracy": 0.8}),
            HybridExampleResult(example_id="2", metrics={"overall_accuracy": 0.6}),
        ]
        agg = ev._compute_aggregated_metrics(results, total_cost=0.0)
        assert agg["overall_accuracy"] == pytest.approx(0.7)
        assert agg["accuracy"] == pytest.approx(0.7)
        assert agg["score"] == pytest.approx(0.7)

    def test_infers_accuracy_from_split_accuracy_metrics(
        self, ev: HybridAPIEvaluator
    ) -> None:
        """Derives canonical accuracy from split *_accuracy metrics."""
        results = [
            HybridExampleResult(
                example_id="1",
                metrics={"text_accuracy": 0.6, "tool_accuracy": 1.0},
                latency_ms=123.0,
            ),
        ]
        agg = ev._compute_aggregated_metrics(results, total_cost=0.0)
        assert agg["accuracy"] == pytest.approx(0.8)
        assert agg["score"] == pytest.approx(0.8)
        assert agg["response_time_ms"] == pytest.approx(123.0)

    def test_preserves_explicit_accuracy_over_fallback(
        self, ev: HybridAPIEvaluator
    ) -> None:
        """Explicit accuracy remains the source of truth."""
        results = [
            HybridExampleResult(
                example_id="1",
                metrics={
                    "accuracy": 0.9,
                    "overall_accuracy": 0.1,
                    "text_accuracy": 0.2,
                },
            ),
        ]
        agg = ev._compute_aggregated_metrics(results, total_cost=0.0)
        assert agg["accuracy"] == pytest.approx(0.9)
        assert agg["score"] == pytest.approx(0.9)

    def test_comparability_payload_full_coverage(self, ev: HybridAPIEvaluator) -> None:
        """Comparability metadata is emitted with full coverage details."""
        results = [
            HybridExampleResult(
                input_id="ex1",
                metrics={"accuracy": 1.0},
                latency_ms=10,
                metadata={"evaluation_mode": "evaluated"},
            ),
            HybridExampleResult(
                input_id="ex2",
                metrics={"accuracy": 0.5},
                latency_ms=15,
                metadata={"evaluation_mode": "evaluated"},
            ),
        ]
        _agg, comparability = ev._compute_aggregated_metrics_with_comparability(
            results, total_cost=0.02
        )

        assert comparability["total_examples"] == 2
        assert comparability["examples_with_primary_metric"] == 2
        assert comparability["coverage_ratio"] == pytest.approx(1.0)
        assert comparability["ranking_eligible"] is True
        assert comparability["evaluation_mode"] == "evaluated"
        assert comparability["per_metric_coverage"]["accuracy"][
            "ratio"
        ] == pytest.approx(1.0)
        assert comparability["per_metric_coverage"]["total_cost"][
            "ratio"
        ] == pytest.approx(1.0)

    def test_comparability_payload_partial_coverage(
        self, ev: HybridAPIEvaluator
    ) -> None:
        """Missing primary objective on examples is flagged as partial coverage."""
        results = [
            HybridExampleResult(
                input_id="ex1",
                metrics={"accuracy": 1.0},
                metadata={"evaluation_mode": "evaluated"},
            ),
            HybridExampleResult(
                input_id="ex2",
                metrics={},
                metadata={"evaluation_mode": "evaluated"},
            ),
        ]
        _agg, comparability = ev._compute_aggregated_metrics_with_comparability(
            results, total_cost=0.01
        )

        assert comparability["coverage_ratio"] == pytest.approx(0.5)
        assert comparability["ranking_eligible"] is False
        assert "MCI-002" in comparability["warning_codes"]

    def test_comparability_infers_operational_primary_objective(
        self, ev: HybridAPIEvaluator
    ) -> None:
        """Execute-only operational metrics are rankable when fully covered."""
        results = [
            HybridExampleResult(
                input_id="ex1",
                metrics={},
                cost_usd=0.01,
                latency_ms=12.0,
                metadata={"evaluation_mode": "execute_only"},
            ),
            HybridExampleResult(
                input_id="ex2",
                metrics={},
                cost_usd=0.02,
                latency_ms=20.0,
                metadata={"evaluation_mode": "execute_only"},
            ),
        ]
        _agg, comparability = ev._compute_aggregated_metrics_with_comparability(
            results, total_cost=0.03
        )

        assert comparability["primary_objective"] == "total_cost"
        assert comparability["coverage_ratio"] == pytest.approx(1.0)
        assert comparability["evaluation_mode"] == "execute_only"
        assert comparability["ranking_eligible"] is True


class TestMetricNormalizationHelpers:
    """Coverage for low-level numeric normalization helpers."""

    def test_compute_describe_stats_single_value(self) -> None:
        """Single-value input should produce stable percentile values."""
        stats = HybridAPIEvaluator._compute_describe_stats([0.8])
        assert stats == {
            "count": 1.0,
            "mean": 0.8,
            "std": 0.0,
            "min": 0.8,
            "25%": 0.8,
            "50%": 0.8,
            "75%": 0.8,
            "max": 0.8,
        }

    def test_compute_describe_stats_two_values_interpolates(self) -> None:
        """Percentiles should linearly interpolate with two points."""
        stats = HybridAPIEvaluator._compute_describe_stats([0.0, 1.0])
        assert stats["count"] == pytest.approx(2.0)
        assert stats["mean"] == pytest.approx(0.5)
        assert stats["std"] == pytest.approx(0.70710678)
        assert stats["25%"] == pytest.approx(0.25)
        assert stats["50%"] == pytest.approx(0.5)
        assert stats["75%"] == pytest.approx(0.75)

    def test_derive_accuracy_preserves_explicit_zero(self) -> None:
        """Explicit accuracy=0.0 must not be treated as missing."""
        metrics = {"accuracy": 0.0, "overall_accuracy": 0.9, "text_accuracy": 1.0}
        assert HybridAPIEvaluator._derive_accuracy_from_metrics(
            metrics
        ) == pytest.approx(0.0)

    def test_derive_accuracy_ignores_bool_accuracy_values(self) -> None:
        """Bool-valued keys must not be interpreted as numeric accuracy."""
        metrics = {"text_accuracy": True, "tool_accuracy": False, "f1": 0.8}
        assert HybridAPIEvaluator._derive_accuracy_from_metrics(metrics) is None

    def test_derive_accuracy_uses_numeric_split_accuracy(self) -> None:
        """Derive mean of numeric split accuracy keys only."""
        metrics = {"text_accuracy": 0.6, "tool_accuracy": 1.0, "aux_accuracy": "n/a"}
        assert HybridAPIEvaluator._derive_accuracy_from_metrics(
            metrics
        ) == pytest.approx(0.8)


# ---------------------------------------------------------------------------
# _execute_batch tests
# ---------------------------------------------------------------------------


class TestExecuteBatch:
    """Tests for _execute_batch method."""

    @pytest.mark.asyncio
    async def test_combined_mode(self, mock_transport: MagicMock) -> None:
        """When execute response has quality_metrics, uses combined mode."""
        ev = HybridAPIEvaluator(
            transport=mock_transport,
            tunable_id="cap",
            keep_alive=False,
        )
        ev._benchmark_id = "test-bench"
        exec_response = _make_execute_response(
            outputs=[
                {"example_id": "ex_0", "output": "out0", "metrics": {"accuracy": 0.9}},
            ],
            quality_metrics={"overall_accuracy": 0.9},
        )
        mock_transport.execute = AsyncMock(return_value=exec_response)
        caps = _default_capabilities(supports_evaluate=True)

        dataset = _make_dataset([{"input_data": {"q": "?"}, "expected_output": "a"}])
        batch = list(dataset)

        results = await ev._execute_batch(
            mock_transport, caps, {"model": "gpt-4"}, batch
        )

        assert len(results) == 1
        assert results[0].example_id == "ex_0"
        assert results[0].actual_output == "out0"
        assert results[0].metrics == {"accuracy": 0.9}
        # evaluate should NOT be called in combined mode
        mock_transport.evaluate.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_two_phase_mode(self, mock_transport: MagicMock) -> None:
        """When no quality_metrics and supports_evaluate, calls evaluate endpoint."""
        ev = HybridAPIEvaluator(
            transport=mock_transport,
            tunable_id="cap",
            keep_alive=False,
        )
        ev._benchmark_id = "test-bench"
        exec_response = _make_execute_response(
            outputs=[{"example_id": "ex_0", "output": "out0"}],
            quality_metrics=None,
            operational_metrics={"cost_usd": 0.02, "latency_ms": 50.0},
        )
        eval_response = _make_evaluate_response(
            results=[{"example_id": "ex_0", "metrics": {"accuracy": 0.8}}]
        )
        mock_transport.execute = AsyncMock(return_value=exec_response)
        mock_transport.evaluate = AsyncMock(return_value=eval_response)
        caps = _default_capabilities(supports_evaluate=True)

        dataset = _make_dataset([{"input_data": {"q": "?"}, "expected_output": "a"}])
        batch = list(dataset)

        results = await ev._execute_batch(
            mock_transport, caps, {"model": "gpt-4"}, batch
        )

        assert len(results) == 1
        assert results[0].metrics == {"accuracy": 0.8}
        mock_transport.evaluate.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_execute_only_mode(self, mock_transport: MagicMock) -> None:
        """When no quality_metrics and no evaluate support, returns outputs only."""
        ev = HybridAPIEvaluator(
            transport=mock_transport,
            tunable_id="cap",
            keep_alive=False,
        )
        ev._benchmark_id = "test-bench"
        exec_response = _make_execute_response(
            outputs=[{"example_id": "ex_0", "output": "out0"}],
            quality_metrics=None,
            operational_metrics={"cost_usd": 0.01, "latency_ms": 30.0},
        )
        mock_transport.execute = AsyncMock(return_value=exec_response)
        caps = _default_capabilities(supports_evaluate=False)

        dataset = _make_dataset([{"input_data": {"q": "?"}, "expected_output": "a"}])
        batch = list(dataset)

        results = await ev._execute_batch(mock_transport, caps, {}, batch)

        assert len(results) == 1
        assert results[0].metrics == {}
        assert results[0].actual_output == "out0"
        assert results[0].cost_usd == pytest.approx(0.01)
        assert results[0].latency_ms == pytest.approx(30.0)
        mock_transport.evaluate.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_transport_error_returns_error_results(
        self, mock_transport: MagicMock
    ) -> None:
        """TransportError produces error HybridExampleResults for all examples."""
        ev = HybridAPIEvaluator(
            transport=mock_transport,
            tunable_id="cap",
            keep_alive=False,
        )
        ev._benchmark_id = "test-bench"
        mock_transport.execute = AsyncMock(
            side_effect=TransportError("connection lost", status_code=500)
        )
        caps = _default_capabilities()

        dataset = _make_dataset(
            [
                {"input_data": {"q": "1"}, "expected_output": "a1"},
                {"input_data": {"q": "2"}, "expected_output": "a2"},
            ]
        )
        batch = list(dataset)

        results = await ev._execute_batch(mock_transport, caps, {}, batch)

        assert len(results) == 2
        for r in results:
            assert r.success is False
            assert "connection lost" in r.error
        assert results[0].expected_output == "a1"
        assert results[1].expected_output == "a2"

    @pytest.mark.asyncio
    async def test_session_id_updated_from_response(
        self, mock_transport: MagicMock
    ) -> None:
        """Session ID is updated when response returns a new one."""
        ev = HybridAPIEvaluator(
            transport=mock_transport,
            tunable_id="cap",
            keep_alive=False,
        )
        ev._benchmark_id = "test-bench"
        ev._session_id = "old_session"
        exec_response = _make_execute_response(
            outputs=[{"example_id": "ex_0", "output": "out"}],
            session_id="new_session",
        )
        mock_transport.execute = AsyncMock(return_value=exec_response)
        caps = _default_capabilities(supports_evaluate=False)

        dataset = _make_dataset([{"input_data": {"q": "?"}}])
        batch = list(dataset)

        await ev._execute_batch(mock_transport, caps, {}, batch)
        assert ev._session_id == "new_session"

    @pytest.mark.asyncio
    async def test_session_id_update_re_registers_lifecycle_manager(
        self, mock_transport: MagicMock
    ) -> None:
        """When session_id changes and lifecycle_manager exists, re-registers."""
        ev = HybridAPIEvaluator(
            transport=mock_transport,
            tunable_id="cap",
            keep_alive=False,
        )
        ev._benchmark_id = "test-bench"
        ev._session_id = "old"
        mock_lm = MagicMock()
        mock_lm.register = AsyncMock()
        ev._lifecycle_manager = mock_lm

        exec_response = _make_execute_response(
            outputs=[{"example_id": "ex_0", "output": "out"}],
            session_id="new",
        )
        mock_transport.execute = AsyncMock(return_value=exec_response)
        caps = _default_capabilities(supports_evaluate=False)

        dataset = _make_dataset([{"input_data": {"q": "?"}}])
        batch = list(dataset)

        await ev._execute_batch(mock_transport, caps, {}, batch)
        mock_lm.register.assert_awaited_once_with("new")

    @pytest.mark.asyncio
    async def test_uses_example_id_from_input_data(
        self, mock_transport: MagicMock
    ) -> None:
        """Uses example_id from the input data if present."""
        ev = HybridAPIEvaluator(
            transport=mock_transport,
            tunable_id="cap",
            keep_alive=False,
        )
        ev._benchmark_id = "test-bench"
        exec_response = _make_execute_response(
            outputs=[{"example_id": "custom_id", "output": "out"}],
            quality_metrics=None,
        )
        mock_transport.execute = AsyncMock(return_value=exec_response)
        caps = _default_capabilities(supports_evaluate=False)

        # EvaluationExample with example_id in input_data
        example = EvaluationExample(
            input_data={"example_id": "custom_id", "question": "?"}
        )
        dataset = Dataset(examples=[example], name="test")
        batch = list(dataset)

        results = await ev._execute_batch(mock_transport, caps, {}, batch)
        assert results[0].example_id == "custom_id"


# ---------------------------------------------------------------------------
# _evaluate_outputs tests
# ---------------------------------------------------------------------------


class TestEvaluateOutputs:
    """Tests for _evaluate_outputs (two-phase evaluation)."""

    @pytest.mark.asyncio
    async def test_evaluate_outputs_merges_results(
        self, mock_transport: MagicMock
    ) -> None:
        """Merges execute outputs with evaluate metrics."""
        ev = HybridAPIEvaluator(
            transport=mock_transport,
            tunable_id="cap",
            keep_alive=False,
        )
        exec_response = _make_execute_response(
            outputs=[
                {"example_id": "ex_0", "output": "result_0"},
                {"example_id": "ex_1", "output": "result_1"},
            ],
            operational_metrics={"cost_usd": 0.04, "latency_ms": 200.0},
        )
        eval_response = _make_evaluate_response(
            results=[
                {"example_id": "ex_0", "metrics": {"accuracy": 1.0}},
                {"example_id": "ex_1", "metrics": {"accuracy": 0.5}},
            ]
        )
        mock_transport.evaluate = AsyncMock(return_value=eval_response)

        dataset = _make_dataset()
        batch = list(dataset)
        inputs = [
            {"example_id": "ex_0", "data": {"question": "What is 2+2?"}},
            {"example_id": "ex_1", "data": {"question": "What is 3+3?"}},
        ]

        results = await ev._evaluate_outputs(
            mock_transport, batch, inputs, exec_response
        )

        assert len(results) == 2
        assert results[0].actual_output == "result_0"
        assert results[0].metrics == {"accuracy": 1.0}
        assert results[1].metrics == {"accuracy": 0.5}
        assert results[0].cost_usd == pytest.approx(0.02)  # 0.04 / 2

    @pytest.mark.asyncio
    async def test_evaluate_outputs_fallback_on_error(
        self, mock_transport: MagicMock
    ) -> None:
        """Falls back to execute-only when evaluate call fails."""
        ev = HybridAPIEvaluator(
            transport=mock_transport,
            tunable_id="cap",
            keep_alive=False,
        )
        exec_response = _make_execute_response(
            outputs=[{"example_id": "ex_0", "output": "out_0"}],
            operational_metrics={"cost_usd": 0.01, "latency_ms": 50.0},
        )
        mock_transport.evaluate = AsyncMock(side_effect=Exception("evaluate failed"))

        dataset = _make_dataset([{"input_data": {"q": "?"}, "expected_output": "a"}])
        batch = list(dataset)
        inputs = [{"example_id": "ex_0", "data": {"q": "?"}}]

        results = await ev._evaluate_outputs(
            mock_transport, batch, inputs, exec_response
        )

        # Should fall back to execute-only (no quality metrics)
        assert len(results) == 1
        assert results[0].metrics == {}
        assert results[0].actual_output == "out_0"

    @pytest.mark.asyncio
    async def test_evaluate_outputs_no_matching_output(
        self, mock_transport: MagicMock
    ) -> None:
        """Handles case where output example_id doesn't match."""
        ev = HybridAPIEvaluator(
            transport=mock_transport,
            tunable_id="cap",
            keep_alive=False,
        )
        # outputs have different example_id than expected
        exec_response = _make_execute_response(
            outputs=[{"example_id": "other_id", "output": "something"}],
            operational_metrics={"cost_usd": 0.01, "latency_ms": 50.0},
        )
        eval_response = _make_evaluate_response(results=[])
        mock_transport.evaluate = AsyncMock(return_value=eval_response)

        dataset = _make_dataset([{"input_data": {"q": "?"}, "expected_output": "a"}])
        batch = list(dataset)
        inputs = [{"example_id": "ex_0", "data": {"q": "?"}}]

        results = await ev._evaluate_outputs(
            mock_transport, batch, inputs, exec_response
        )

        assert len(results) == 1
        assert results[0].actual_output is None
        assert results[0].metrics == {}

    @pytest.mark.asyncio
    async def test_evaluate_outputs_passes_timeout_budget(
        self, mock_transport: MagicMock
    ) -> None:
        """Evaluate request propagates evaluator timeout via timeout_ms."""
        ev = HybridAPIEvaluator(
            transport=mock_transport,
            tunable_id="cap",
            keep_alive=False,
            timeout=12.5,
        )
        exec_response = _make_execute_response(
            outputs=[{"example_id": "ex_0", "output": "result_0"}],
            operational_metrics={"cost_usd": 0.01, "latency_ms": 100.0},
        )
        mock_transport.evaluate = AsyncMock(return_value=_make_evaluate_response())

        dataset = _make_dataset([{"input_data": {"q": "?"}, "expected_output": "a"}])
        batch = list(dataset)
        inputs = [{"example_id": "ex_0", "data": {"q": "?"}}]

        await ev._evaluate_outputs(mock_transport, batch, inputs, exec_response)

        eval_request = mock_transport.evaluate.await_args.args[0]
        assert eval_request.timeout_ms == 12500


# ---------------------------------------------------------------------------
# _process_combined_response tests
# ---------------------------------------------------------------------------


class TestProcessCombinedResponse:
    """Tests for _process_combined_response."""

    def test_basic_combined_processing(self, mock_transport: MagicMock) -> None:
        """Processes combined response with metrics per output."""
        ev = HybridAPIEvaluator(
            transport=mock_transport,
            tunable_id="cap",
            keep_alive=False,
        )
        response = _make_execute_response(
            outputs=[
                {"example_id": "ex_0", "output": "out0", "metrics": {"score": 0.9}},
                {"example_id": "ex_1", "output": "out1", "metrics": {"score": 0.7}},
            ],
            operational_metrics={"cost_usd": 0.10, "latency_ms": 300.0},
            quality_metrics={"overall": 0.8},
        )
        dataset = _make_dataset()
        batch = list(dataset)
        inputs = [
            {"example_id": "ex_0", "data": {}},
            {"example_id": "ex_1", "data": {}},
        ]

        results = ev._process_combined_response(batch, inputs, response)

        assert len(results) == 2
        assert results[0].actual_output == "out0"
        assert results[0].metrics == {"score": 0.9}
        assert results[0].cost_usd == pytest.approx(0.05)  # 0.10 / 2
        assert results[0].latency_ms == 300.0
        assert results[1].actual_output == "out1"
        assert results[1].expected_output == "6"  # From dataset

    def test_combined_no_matching_output(self, mock_transport: MagicMock) -> None:
        """When output example_id doesn't match, output is None."""
        ev = HybridAPIEvaluator(
            transport=mock_transport,
            tunable_id="cap",
            keep_alive=False,
        )
        response = _make_execute_response(
            outputs=[{"example_id": "no_match", "output": "x"}],
            operational_metrics={"cost_usd": 0.01, "latency_ms": 10.0},
            quality_metrics={"overall": 0.5},
        )
        dataset = _make_dataset([{"input_data": {"q": "?"}, "expected_output": "a"}])
        batch = list(dataset)
        inputs = [{"example_id": "ex_0", "data": {}}]

        results = ev._process_combined_response(batch, inputs, response)

        assert len(results) == 1
        assert results[0].actual_output is None
        assert results[0].metrics == {}


# ---------------------------------------------------------------------------
# _process_execute_only_response tests
# ---------------------------------------------------------------------------


class TestProcessExecuteOnlyResponse:
    """Tests for _process_execute_only_response."""

    def test_basic_execute_only(self, mock_transport: MagicMock) -> None:
        """Processes execute-only response with empty metrics."""
        ev = HybridAPIEvaluator(
            transport=mock_transport,
            tunable_id="cap",
            keep_alive=False,
        )
        response = _make_execute_response(
            outputs=[{"example_id": "ex_0", "output": "result"}],
            operational_metrics={"cost_usd": 0.02, "latency_ms": 150.0},
        )
        dataset = _make_dataset([{"input_data": {"q": "?"}, "expected_output": "ans"}])
        batch = list(dataset)
        inputs = [{"example_id": "ex_0", "data": {}}]

        results = ev._process_execute_only_response(batch, inputs, response)

        assert len(results) == 1
        assert results[0].actual_output == "result"
        assert results[0].expected_output == "ans"
        assert results[0].metrics == {}
        assert results[0].cost_usd == pytest.approx(0.02)
        assert results[0].latency_ms == pytest.approx(150.0)

    def test_multiple_examples(self, mock_transport: MagicMock) -> None:
        """Cost is divided among examples."""
        ev = HybridAPIEvaluator(
            transport=mock_transport,
            tunable_id="cap",
            keep_alive=False,
        )
        response = _make_execute_response(
            outputs=[
                {"example_id": "ex_0", "output": "r0"},
                {"example_id": "ex_1", "output": "r1"},
            ],
            operational_metrics={"cost_usd": 0.10, "latency_ms": 100.0},
        )
        dataset = _make_dataset()
        batch = list(dataset)
        inputs = [
            {"example_id": "ex_0", "data": {}},
            {"example_id": "ex_1", "data": {}},
        ]

        results = ev._process_execute_only_response(batch, inputs, response)

        assert len(results) == 2
        assert results[0].cost_usd == pytest.approx(0.05)
        assert results[1].cost_usd == pytest.approx(0.05)


# ---------------------------------------------------------------------------
# evaluate tests
# ---------------------------------------------------------------------------


class TestEvaluate:
    """Tests for the main evaluate method."""

    @pytest.mark.asyncio
    async def test_evaluate_normal(
        self, evaluator: HybridAPIEvaluator, mock_transport: MagicMock
    ) -> None:
        """Normal evaluation processes all examples and returns EvaluationResult."""
        mock_transport.capabilities = AsyncMock(
            return_value=_default_capabilities(supports_evaluate=False)
        )
        exec_response = _make_execute_response(
            outputs=[
                {"example_id": "ex_0", "output": "r0"},
                {"example_id": "ex_1", "output": "r1"},
            ],
            operational_metrics={"cost_usd": 0.04, "latency_ms": 100.0},
        )
        mock_transport.execute = AsyncMock(return_value=exec_response)

        dataset = _make_dataset()
        config = {"model": "gpt-4"}

        result = await evaluator.evaluate(
            func=lambda: None,
            config=config,
            dataset=dataset,
        )

        assert isinstance(result, EvaluationResult)
        assert result.config == config
        assert result.total_examples == 2
        assert result.successful_examples == 2
        assert result.duration > 0
        assert result.sample_budget_exhausted is False

    @pytest.mark.asyncio
    async def test_evaluate_empty_dataset(
        self, evaluator: HybridAPIEvaluator, mock_transport: MagicMock
    ) -> None:
        """Returns empty result for empty dataset."""
        mock_transport.capabilities = AsyncMock(return_value=_default_capabilities())

        dataset = _make_dataset([])
        config = {"model": "gpt-4"}

        result = await evaluator.evaluate(
            func=lambda: None,
            config=config,
            dataset=dataset,
        )

        assert result.total_examples == 0
        assert result.successful_examples == 0
        assert result.aggregated_metrics == {}
        assert result.example_results == []

    @pytest.mark.asyncio
    async def test_evaluate_with_sample_lease(
        self, evaluator: HybridAPIEvaluator, mock_transport: MagicMock
    ) -> None:
        """Sample lease limits examples and tracks exhaustion."""
        mock_transport.capabilities = AsyncMock(
            return_value=_default_capabilities(supports_evaluate=False)
        )
        exec_response = _make_execute_response(
            outputs=[{"example_id": "ex_0", "output": "r0"}],
            operational_metrics={"cost_usd": 0.01, "latency_ms": 50.0},
        )
        mock_transport.execute = AsyncMock(return_value=exec_response)

        # Create a mock sample_lease that limits to 1 example
        sample_lease = MagicMock()
        sample_lease.remaining = MagicMock(return_value=1)
        sample_lease.try_take = MagicMock(return_value=True)
        sample_lease.exhausted = True

        dataset = _make_dataset()  # 2 examples
        config = {"model": "gpt-4"}

        result = await evaluator.evaluate(
            func=lambda: None,
            config=config,
            dataset=dataset,
            sample_lease=sample_lease,
        )

        assert result.total_examples == 1
        assert result.sample_budget_exhausted is True

    @pytest.mark.asyncio
    async def test_evaluate_sample_lease_exhausted_during_batch(
        self, evaluator: HybridAPIEvaluator, mock_transport: MagicMock
    ) -> None:
        """Stops processing when sample_lease.try_take returns False."""
        mock_transport.capabilities = AsyncMock(
            return_value=_default_capabilities(supports_evaluate=False)
        )
        evaluator._batch_size = 1  # Process one at a time

        exec_response = _make_execute_response(
            outputs=[{"example_id": "ex_0", "output": "r0"}],
            operational_metrics={"cost_usd": 0.01, "latency_ms": 50.0},
        )
        mock_transport.execute = AsyncMock(return_value=exec_response)

        sample_lease = MagicMock()
        sample_lease.remaining = MagicMock(return_value=10)
        # First call succeeds, second returns False (budget exhausted)
        sample_lease.try_take = MagicMock(side_effect=[True, False])
        sample_lease.exhausted = True

        dataset = _make_dataset()  # 2 examples
        config = {"model": "gpt-4"}

        result = await evaluator.evaluate(
            func=lambda: None,
            config=config,
            dataset=dataset,
            sample_lease=sample_lease,
        )

        # Only first batch was processed
        assert result.total_examples == 1

    @pytest.mark.asyncio
    async def test_evaluate_sample_lease_zero_remaining(
        self, evaluator: HybridAPIEvaluator, mock_transport: MagicMock
    ) -> None:
        """Returns empty result when sample_lease has zero remaining."""
        mock_transport.capabilities = AsyncMock(return_value=_default_capabilities())

        sample_lease = MagicMock()
        sample_lease.remaining = MagicMock(return_value=0)
        sample_lease.exhausted = True

        dataset = _make_dataset()
        config = {"model": "gpt-4"}

        result = await evaluator.evaluate(
            func=lambda: None,
            config=config,
            dataset=dataset,
            sample_lease=sample_lease,
        )

        assert result.total_examples == 0
        assert result.sample_budget_exhausted is True

    @pytest.mark.asyncio
    async def test_evaluate_with_progress_callback(
        self, evaluator: HybridAPIEvaluator, mock_transport: MagicMock
    ) -> None:
        """Progress callback is called with batch info."""
        mock_transport.capabilities = AsyncMock(
            return_value=_default_capabilities(supports_evaluate=False)
        )
        exec_response = _make_execute_response(
            outputs=[
                {"example_id": "ex_0", "output": "r0"},
                {"example_id": "ex_1", "output": "r1"},
            ],
            operational_metrics={"cost_usd": 0.02, "latency_ms": 100.0},
        )
        mock_transport.execute = AsyncMock(return_value=exec_response)

        callback = MagicMock()
        dataset = _make_dataset()
        config = {"model": "gpt-4"}

        await evaluator.evaluate(
            func=lambda: None,
            config=config,
            dataset=dataset,
            progress_callback=callback,
        )

        callback.assert_called_once()
        call_args = callback.call_args
        # First arg is number of examples processed
        assert call_args[0][0] == 2
        # Second arg is dict with batch info
        info = call_args[0][1]
        assert "batch_size" in info
        assert "total_cost" in info
        assert "successful" in info

    @pytest.mark.asyncio
    async def test_evaluate_progress_callback_error_handled(
        self, evaluator: HybridAPIEvaluator, mock_transport: MagicMock
    ) -> None:
        """Progress callback errors are caught and logged, not propagated."""
        mock_transport.capabilities = AsyncMock(
            return_value=_default_capabilities(supports_evaluate=False)
        )
        exec_response = _make_execute_response(
            outputs=[{"example_id": "ex_0", "output": "r0"}],
            operational_metrics={"cost_usd": 0.01, "latency_ms": 50.0},
        )
        mock_transport.execute = AsyncMock(return_value=exec_response)

        callback = MagicMock(side_effect=RuntimeError("callback boom"))
        dataset = _make_dataset([{"input_data": {"q": "?"}}])

        # Should not raise despite callback error
        result = await evaluator.evaluate(
            func=lambda: None,
            config={},
            dataset=dataset,
            progress_callback=callback,
        )
        assert result.total_examples == 1

    @pytest.mark.asyncio
    async def test_evaluate_batching(self, mock_transport: MagicMock) -> None:
        """Examples are processed in batches of batch_size."""
        ev = HybridAPIEvaluator(
            transport=mock_transport,
            tunable_id="cap",
            batch_size=1,  # one example per batch
            keep_alive=False,
        )
        ev._benchmark_id = "test-bench"
        mock_transport.capabilities = AsyncMock(
            return_value=_default_capabilities(supports_evaluate=False)
        )

        # Two separate calls for two batches
        responses = [
            _make_execute_response(
                outputs=[{"example_id": "ex_0", "output": "r0"}],
                operational_metrics={"cost_usd": 0.01, "latency_ms": 50.0},
            ),
            _make_execute_response(
                outputs=[{"example_id": "ex_0", "output": "r1"}],
                operational_metrics={"cost_usd": 0.02, "latency_ms": 60.0},
            ),
        ]
        mock_transport.execute = AsyncMock(side_effect=responses)

        dataset = _make_dataset()
        result = await ev.evaluate(func=lambda: None, config={}, dataset=dataset)

        assert result.total_examples == 2
        assert mock_transport.execute.await_count == 2

    @pytest.mark.asyncio
    async def test_evaluate_examples_consumed(
        self, evaluator: HybridAPIEvaluator, mock_transport: MagicMock
    ) -> None:
        """examples_consumed is set in the result."""
        mock_transport.capabilities = AsyncMock(
            return_value=_default_capabilities(supports_evaluate=False)
        )
        exec_response = _make_execute_response(
            outputs=[
                {"example_id": "ex_0", "output": "r0"},
                {"example_id": "ex_1", "output": "r1"},
            ],
        )
        mock_transport.execute = AsyncMock(return_value=exec_response)

        dataset = _make_dataset()
        result = await evaluator.evaluate(func=lambda: None, config={}, dataset=dataset)

        assert result.examples_consumed == 2

    @pytest.mark.asyncio
    async def test_evaluate_populates_summary_stats_for_hybrid(
        self, evaluator: HybridAPIEvaluator, mock_transport: MagicMock
    ) -> None:
        """Hybrid evaluator should produce describe-style summary_stats."""
        mock_transport.capabilities = AsyncMock(
            return_value=_default_capabilities(supports_evaluate=False)
        )
        exec_response = _make_execute_response(
            outputs=[
                {
                    "example_id": "ex_0",
                    "output": "r0",
                    "metrics": {"text_accuracy": 1.0, "tool_accuracy": 0.5},
                },
                {
                    "example_id": "ex_1",
                    "output": "r1",
                    "metrics": {"text_accuracy": 0.5, "tool_accuracy": 1.0},
                },
            ],
            operational_metrics={"cost_usd": 0.04, "latency_ms": 100.0},
        )
        mock_transport.execute = AsyncMock(return_value=exec_response)

        dataset = _make_dataset()
        result = await evaluator.evaluate(func=lambda: None, config={}, dataset=dataset)

        assert result.summary_stats is not None
        assert "metrics" in result.summary_stats
        assert "accuracy" in result.summary_stats["metrics"]
        assert "response_time_ms" in result.summary_stats["metrics"]
        assert "score" in result.summary_stats["metrics"]
        assert "success_rate" in result.summary_stats["metrics"]

        accuracy_stats = result.summary_stats["metrics"]["accuracy"]
        assert accuracy_stats["count"] == pytest.approx(2.0)
        assert accuracy_stats["mean"] == pytest.approx(0.75)

    @pytest.mark.asyncio
    async def test_hybrid_aggregates_are_measuresdict_compatible(
        self, evaluator: HybridAPIEvaluator, mock_transport: MagicMock
    ) -> None:
        """Cross-module safeguard: hybrid aggregate metrics fit MeasuresDict."""
        mock_transport.capabilities = AsyncMock(
            return_value=_default_capabilities(supports_evaluate=False)
        )
        exec_response = _make_execute_response(
            outputs=[
                {
                    "example_id": "ex_0",
                    "output": "r0",
                    "metrics": {"text_accuracy": 0.8, "tool_accuracy": 0.9},
                },
                {
                    "example_id": "ex_1",
                    "output": "r1",
                    "metrics": {"text_accuracy": 0.6, "tool_accuracy": 1.0},
                },
            ],
            operational_metrics={"cost_usd": 0.02, "latency_ms": 90.0},
        )
        mock_transport.execute = AsyncMock(return_value=exec_response)

        dataset = _make_dataset()
        result = await evaluator.evaluate(func=lambda: None, config={}, dataset=dataset)

        validated = MeasuresDict(result.aggregated_metrics)
        assert "accuracy" in validated
        assert len(validated) <= MeasuresDict.MAX_KEYS


# ---------------------------------------------------------------------------
# close / context manager tests
# ---------------------------------------------------------------------------


class TestCloseAndContextManager:
    """Tests for close(), __aenter__, __aexit__."""

    @pytest.mark.asyncio
    async def test_close_releases_lifecycle_and_transport(
        self, mock_transport: MagicMock
    ) -> None:
        """close() releases lifecycle manager and closes owned transport."""
        ev = HybridAPIEvaluator(
            transport=mock_transport,
            keep_alive=False,
        )
        # Simulate that evaluator owns the transport
        ev._owns_transport = True

        mock_lm = MagicMock()
        mock_lm.release = AsyncMock()
        ev._lifecycle_manager = mock_lm

        await ev.close()

        mock_lm.release.assert_awaited_once()
        mock_transport.close.assert_awaited_once()
        assert ev._lifecycle_manager is None
        assert ev._transport is None

    @pytest.mark.asyncio
    async def test_close_does_not_close_non_owned_transport(
        self, mock_transport: MagicMock
    ) -> None:
        """close() does not close transport it doesn't own."""
        ev = HybridAPIEvaluator(transport=mock_transport, keep_alive=False)
        assert ev._owns_transport is False

        await ev.close()

        mock_transport.close.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_close_without_lifecycle_manager(
        self, mock_transport: MagicMock
    ) -> None:
        """close() is safe when no lifecycle manager is set."""
        ev = HybridAPIEvaluator(transport=mock_transport, keep_alive=False)
        ev._owns_transport = True
        await ev.close()
        # Should not raise

    @pytest.mark.asyncio
    async def test_close_without_transport(self) -> None:
        """close() is safe when transport is None."""
        ev = HybridAPIEvaluator(api_endpoint="http://localhost:8080")
        await ev.close()
        # Should not raise

    @pytest.mark.asyncio
    async def test_async_context_manager_enter(self, mock_transport: MagicMock) -> None:
        """__aenter__ returns self."""
        ev = HybridAPIEvaluator(transport=mock_transport, keep_alive=False)
        result = await ev.__aenter__()
        assert result is ev

    @pytest.mark.asyncio
    async def test_async_context_manager_exit_calls_close(
        self, mock_transport: MagicMock
    ) -> None:
        """__aexit__ calls close()."""
        ev = HybridAPIEvaluator(transport=mock_transport, keep_alive=False)
        ev._owns_transport = True

        async with ev as evaluator:
            assert evaluator is ev

        mock_transport.close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_async_context_manager_full_lifecycle(
        self, mock_transport: MagicMock
    ) -> None:
        """Full lifecycle via context manager with lifecycle manager."""
        mock_lm = MagicMock()
        mock_lm.release = AsyncMock()

        ev = HybridAPIEvaluator(transport=mock_transport, keep_alive=False)
        ev._owns_transport = True
        ev._lifecycle_manager = mock_lm

        async with ev:
            pass

        mock_lm.release.assert_awaited_once()
        mock_transport.close.assert_awaited_once()
        assert ev._lifecycle_manager is None
        assert ev._transport is None
