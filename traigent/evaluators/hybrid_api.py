"""Hybrid API evaluator for external agentic service optimization.

Provides evaluation via external HTTP or MCP endpoints, enabling
optimization of any agentic service that implements the Traigent
hybrid API protocol.
"""

# Traceability: HYBRID-MODE-OPTIMIZATION EVALUATOR-INTEGRATION

from __future__ import annotations

import asyncio
import os
import statistics
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from traigent._version import get_version
from traigent.api.types import ComparabilityInfo, MetricCoverage
from traigent.evaluators.base import BaseEvaluator, Dataset, EvaluationResult
from traigent.hybrid import (
    AgentLifecycleManager,
    BatchOptions,
    BenchmarkEntry,
    BenchmarksResponse,
    ConfigSpaceDiscovery,
    HybridEvaluateRequest,
    HybridExecuteRequest,
    HybridTransport,
    ServiceCapabilities,
    TransportError,
    create_transport,
)
from traigent.utils.logging import get_logger
from traigent.utils.objectives import classify_objective

if TYPE_CHECKING:
    from traigent.cloud.production_mcp_client import (
        MCPServerConfig,
        ProductionMCPClient,
    )
    from traigent.core.sample_budget import SampleBudgetLease

logger = get_logger(__name__)


@dataclass
class HybridExampleResult:
    """Result for a single example in hybrid evaluation.

    Attributes:
        example_id: Identifier for the example
        actual_output: Output produced by the agent
        expected_output: Expected output (from dataset)
        metrics: Per-example quality metrics
        cost_usd: Cost for this example
        latency_ms: Latency for this example
        error: Error message if failed
    """

    example_id: str = ""
    actual_output: Any = None
    expected_output: Any = None
    metrics: dict[str, float] = field(default_factory=dict)
    cost_usd: float = 0.0
    latency_ms: float = 0.0
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    input_id: str | None = None

    def __post_init__(self) -> None:
        """Normalize legacy input_id and canonical example_id fields."""
        resolved_id = self.example_id or self.input_id or ""
        if not resolved_id:
            raise ValueError("HybridExampleResult requires example_id or input_id")
        self.example_id = resolved_id
        self.input_id = resolved_id

    @property
    def success(self) -> bool:
        """Whether this example was processed successfully."""
        return self.error is None


class HybridAPIEvaluator(BaseEvaluator):
    """Evaluator that executes trials via external API endpoints.

    This evaluator bridges the Traigent optimization loop with external
    agentic services, supporting both HTTP REST and MCP transports.

    Key features:
        - Batch execution with configurable batch size
        - Two-phase evaluation (execute → evaluate) or combined mode
        - Keep-alive management for stateful agents
        - Auto-discovery of configuration space from external service
        - Cost tracking from operational metrics

    Example:
        evaluator = HybridAPIEvaluator(
            api_endpoint="http://agent-service:8080",
            tunable_id="financial_qa",
            batch_size=10,
        )

        result = await evaluator.evaluate(
            func=lambda: None,  # Not used in hybrid mode
            config={"temperature": 0.7, "model": "gpt-4"},
            dataset=my_dataset,
        )
    """

    # Public constructor intentionally exposes transport and runtime knobs.
    def __init__(  # NOSONAR(S107)
        self,
        # Transport options (one required)
        api_endpoint: str | None = None,
        transport: HybridTransport | None = None,
        transport_type: Literal["http", "mcp", "auto"] = "auto",
        # MCP options
        mcp_client: ProductionMCPClient | None = None,
        mcp_config: MCPServerConfig | None = None,
        # Tunable options
        tunable_id: str | None = None,
        auto_discover_tvars: bool = True,
        # Execution options
        batch_size: int = 1,
        batch_parallelism: int = 1,
        keep_alive: bool = True,
        heartbeat_interval: float = 30.0,
        timeout: float = 300.0,
        evaluation_kwargs: dict[str, str | int | float | bool] | None = None,
        # Auth options
        auth_header: str | None = None,
        # Base evaluator options
        **kwargs: Any,
    ) -> None:
        """Initialize HybridAPIEvaluator.

        Args:
            api_endpoint: Base URL for HTTP transport.
            transport: Pre-configured HybridTransport instance.
            transport_type: Transport type ("http", "mcp", "auto").

            mcp_client: Existing MCP client for MCP transport.
            mcp_config: MCP server config for MCP transport.

            tunable_id: Identifier for the tunable.
                Auto-discovered if not provided.
            auto_discover_tvars: Whether to auto-discover config space.

            batch_size: Number of examples per execute request (1-N).
            batch_parallelism: Concurrent executions within batch.
            keep_alive: Enable keep-alive for stateful agents.
            heartbeat_interval: Seconds between heartbeats.
            timeout: Request timeout in seconds.
            evaluation_kwargs: Optional static evaluator kwargs sent to /evaluate.

            auth_header: Authorization header for HTTP transport.

            **kwargs: Additional args passed to BaseEvaluator.
        """
        super().__init__(**kwargs)

        # Store configuration
        self._api_endpoint = api_endpoint
        self._transport_type = transport_type
        self._mcp_client = mcp_client
        self._mcp_config = mcp_config
        self._auth_header = auth_header
        self._timeout = timeout

        self._tunable_id = tunable_id
        self._auto_discover = auto_discover_tvars
        self._batch_size = max(1, batch_size)
        self._batch_parallelism = max(1, batch_parallelism)
        self._keep_alive_enabled = keep_alive
        self._heartbeat_interval = heartbeat_interval
        self._evaluation_kwargs = (
            dict(evaluation_kwargs) if evaluation_kwargs is not None else None
        )

        # Initialize transport (may be provided or created on demand)
        self._transport: HybridTransport | None = transport
        self._owns_transport = transport is None
        self._transport_lock = asyncio.Lock()

        # Lazy-initialized components
        self._lifecycle_manager: AgentLifecycleManager | None = None
        self._discovery: ConfigSpaceDiscovery | None = None
        self._capabilities: ServiceCapabilities | None = None
        self._session_id: str | None = None
        self._optimization_spec: dict[str, Any] | None = None
        self._benchmark_id: str | None = None
        self._benchmark_lock = asyncio.Lock()

    @property
    def lifecycle_manager(self) -> AgentLifecycleManager | None:
        """Get the lifecycle manager (for orchestrator cleanup)."""
        return self._lifecycle_manager

    @property
    def tunable_id(self) -> str | None:
        """Get the tunable ID (may be auto-discovered)."""
        return self._tunable_id

    @property
    def optimization_spec(self) -> dict[str, Any] | None:
        """Optimization spec discovered from external config-space metadata."""
        return self._optimization_spec

    async def _get_transport(self) -> HybridTransport:
        """Get or create the transport."""
        if self._transport is not None:
            return self._transport

        async with self._transport_lock:
            if self._transport is None:
                self._transport = create_transport(
                    transport_type=self._transport_type,
                    base_url=self._api_endpoint,
                    auth_header=self._auth_header,
                    timeout=self._timeout,
                    mcp_client=self._mcp_client,
                    mcp_config=self._mcp_config,
                )
                self._owns_transport = True

        return self._transport

    async def _get_capabilities(self) -> ServiceCapabilities:
        """Get service capabilities."""
        if self._capabilities is None:
            transport = await self._get_transport()
            self._capabilities = await transport.capabilities()
        return self._capabilities

    async def discover_config_space(self) -> dict[str, Any]:
        """Fetch TVARs from external service and normalize.

        Returns:
            Configuration space dictionary for Traigent optimizer.

        Raises:
            TransportError: If discovery fails.
        """
        transport = await self._get_transport()

        if self._discovery is None:
            self._discovery = ConfigSpaceDiscovery(transport)

        config_space = await self._discovery.fetch_and_normalize()
        self._optimization_spec = await self._discovery.build_optimization_spec()

        # Update tunable_id if not set
        if self._tunable_id is None:
            self._tunable_id = self._discovery.get_tunable_id()

        return config_space

    async def discover_example_ids(
        self,
        tunable_id: str | None = None,
        *,
        benchmark_id: str | None = None,
    ) -> list[str]:
        """Discover available example IDs from the external service.

        Fetches benchmarks via the /benchmarks endpoint and selects the
        appropriate benchmark for the given tunable.

        Args:
            tunable_id: Tunable to list examples for.
                Falls back to self._tunable_id if not provided.
            benchmark_id: Explicit benchmark to use. Required when
                multiple benchmarks match the tunable.

        Returns:
            List of example IDs from the selected benchmark.

        Raises:
            TransportError: If discovery fails.
            ValueError: If no tunable_id is available, no benchmarks match,
                or multiple benchmarks match without an explicit benchmark_id.
        """
        tid = tunable_id or self._tunable_id
        if not tid:
            raise ValueError(
                "tunable_id is required for example discovery. "
                "Set it explicitly or call discover_config_space() first."
            )

        transport = await self._get_transport()
        resp: BenchmarksResponse = await transport.benchmarks(tunable_id=tid)

        # Filter benchmarks where this tunable is linked
        matching: list[BenchmarkEntry] = [
            entry for entry in resp.benchmarks if tid in entry.tunable_ids
        ]

        # Large payload guard — check only benchmarks matched to this tunable
        total_example_ids = sum(len(e.example_ids) for e in matching)
        if total_example_ids > 10000:
            logger.warning(
                "Matched benchmarks contain %d total example_ids across %d "
                "benchmarks for tunable %r; consider limiting dataset size",
                total_example_ids,
                len(matching),
                tid,
            )

        # Benchmark selection logic
        if len(matching) == 0:
            available = [e.benchmark_id for e in resp.benchmarks]
            raise ValueError(
                f"No benchmarks found for tunable_id={tid!r}. "
                f"Available benchmarks: {available}"
            )
        elif len(matching) == 1:
            selected = matching[0]
        else:
            # Multiple matches
            if benchmark_id is not None:
                candidates = [e for e in matching if e.benchmark_id == benchmark_id]
                if not candidates:
                    available = [e.benchmark_id for e in matching]
                    raise ValueError(
                        f"benchmark_id={benchmark_id!r} not found among "
                        f"benchmarks for tunable_id={tid!r}. "
                        f"Available: {available}"
                    )
                selected = candidates[0]
            else:
                available = [e.benchmark_id for e in matching]
                raise ValueError(
                    f"Multiple benchmarks match tunable_id={tid!r}: {available}. "
                    f"Pass benchmark_id= to select one."
                )

        self._benchmark_id = selected.benchmark_id

        logger.info(
            "Selected benchmark %s with %d example IDs for tunable %s",
            selected.benchmark_id,
            len(selected.example_ids),
            tid,
        )
        return selected.example_ids

    async def _ensure_benchmark_id(self) -> None:
        """Ensure ``_benchmark_id`` is populated, discovering it if needed.

        Uses a lock with double-check to avoid duplicate discovery calls
        when parallel trials start simultaneously.
        """
        if self._benchmark_id:
            return
        async with self._benchmark_lock:
            if self._benchmark_id:
                return  # another coroutine resolved it while we waited
            await self.discover_example_ids()

    async def _ensure_lifecycle_manager(self) -> None:
        """Ensure lifecycle manager is initialized if keep-alive enabled."""
        if not self._keep_alive_enabled:
            return

        caps = await self._get_capabilities()
        if not caps.supports_keep_alive:
            logger.debug("External service does not support keep-alive")
            return

        if self._lifecycle_manager is None:
            transport = await self._get_transport()
            self._lifecycle_manager = AgentLifecycleManager(
                transport=transport,
                heartbeat_interval=self._heartbeat_interval,
            )
            logger.info("Started lifecycle manager for hybrid API evaluator")

        # Register only when the external service has issued a real session_id.
        # This avoids heartbeat failures against servers that do not pre-create
        # sessions from client-generated identifiers.
        if self._session_id:
            await self._lifecycle_manager.register(self._session_id)

    @staticmethod
    def _coerce_metric(value: Any, default: float = 0.0) -> float:
        """Coerce a numeric metric value to float with a safe fallback."""
        if isinstance(value, bool):
            return default
        if isinstance(value, (int, float)):
            return float(value)
        return default

    @staticmethod
    def _find_output_item(
        outputs: list[dict[str, Any]],
        example_id: str,
    ) -> dict[str, Any]:
        """Find output item for an example_id/input_id."""
        for out in outputs:
            if out.get("example_id") == example_id or out.get("input_id") == example_id:
                return out
        return {}

    @staticmethod
    def _extract_target_fields(expected: Any) -> tuple[str, Any]:
        """Return ('target'|'target_id', value) from expected output payload."""
        if (
            isinstance(expected, dict)
            and "target_id" in expected
            and "target" not in expected
        ):
            return "target_id", expected["target_id"]
        return "target", expected

    @staticmethod
    def _compute_describe_stats(values: list[float]) -> dict[str, float]:
        """Compute pandas.describe()-style statistics for numeric values."""
        if not values:
            return {
                "count": 0.0,
                "mean": 0.0,
                "std": 0.0,
                "min": 0.0,
                "25%": 0.0,
                "50%": 0.0,
                "75%": 0.0,
                "max": 0.0,
            }

        sorted_values = sorted(values)
        n = len(sorted_values)

        def percentile(p: float) -> float:
            k = (n - 1) * p
            low = int(k)
            frac = k - low
            if low + 1 < n:
                return float(
                    sorted_values[low] * (1 - frac) + sorted_values[low + 1] * frac
                )
            return float(sorted_values[low])

        return {
            "count": float(n),
            "mean": float(statistics.mean(sorted_values)),
            "std": float(statistics.stdev(sorted_values)) if n > 1 else 0.0,
            "min": float(sorted_values[0]),
            "25%": percentile(0.25),
            "50%": percentile(0.50),
            "75%": percentile(0.75),
            "max": float(sorted_values[-1]),
        }

    @staticmethod
    def _get_comparability_mode() -> Literal["legacy", "warn", "strict"]:
        """Resolve comparability mode from environment.

        Hybrid evaluators do not currently receive full TraigentConfig directly.
        This uses the shared env contract to keep behavior aligned with orchestrator.
        """
        mode = os.getenv("TRAIGENT_COMPARABILITY_MODE", "warn").strip().lower()
        if mode in {"legacy", "warn", "strict"}:
            return mode  # type: ignore[return-value]
        return "warn"

    @classmethod
    def _derive_accuracy_with_path(
        cls, metrics: dict[str, Any]
    ) -> tuple[float | None, str]:
        """Derive canonical accuracy and return derivation path."""
        explicit = cls._coerce_metric(metrics.get("accuracy"), default=-1.0)
        if explicit >= 0.0:
            return explicit, "explicit"

        overall = cls._coerce_metric(metrics.get("overall_accuracy"), default=-1.0)
        if overall >= 0.0:
            return overall, "overall_accuracy"

        split_accuracies = [
            float(v)
            for k, v in metrics.items()
            if k.endswith("_accuracy")
            and isinstance(v, (int, float))
            and not isinstance(v, bool)
        ]
        if split_accuracies:
            return float(sum(split_accuracies) / len(split_accuracies)), "split_mean"
        return None, "none"

    @classmethod
    def _derive_accuracy_from_metrics(cls, metrics: dict[str, Any]) -> float | None:
        """Derive canonical accuracy from a metric dictionary."""
        derived, _ = cls._derive_accuracy_with_path(metrics)
        return derived

    @classmethod
    def _normalize_example_metrics_with_path(
        cls, metrics: Any
    ) -> tuple[dict[str, float], str]:
        """Normalize per-example metrics and return derivation path."""
        if not isinstance(metrics, dict):
            return {}, "none"

        normalized: dict[str, float] = {}
        for key, value in metrics.items():
            if isinstance(value, bool):
                continue
            if isinstance(value, (int, float)):
                normalized[key] = float(value)

        derived_accuracy, path = cls._derive_accuracy_with_path(normalized)
        if "accuracy" not in normalized and derived_accuracy is not None:
            normalized["accuracy"] = derived_accuracy

        return normalized, path

    @classmethod
    def _normalize_example_metrics(cls, metrics: Any) -> dict[str, float]:
        """Normalize per-example metrics to numeric values and derive accuracy."""
        normalized, _ = cls._normalize_example_metrics_with_path(metrics)
        return normalized

    def _build_summary_stats(
        self,
        results: list[HybridExampleResult],
        duration: float,
        comparability: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        """Build summary_stats from per-example hybrid results."""
        if not results:
            return None

        metric_values: dict[str, list[float]] = {}
        accuracy_values: list[float] = []
        cost_values: list[float] = []
        latency_values: list[float] = []
        success_values: list[float] = []

        for result in results:
            success_values.append(1.0 if result.success else 0.0)

            if result.cost_usd > 0:
                cost_values.append(float(result.cost_usd))

            if result.latency_ms > 0:
                latency_values.append(float(result.latency_ms))

            per_example_accuracy = self._derive_accuracy_from_metrics(result.metrics)
            if per_example_accuracy is not None:
                accuracy_values.append(float(per_example_accuracy))

            for metric_name, value in result.metrics.items():
                if isinstance(value, bool):
                    continue
                if not isinstance(value, (int, float)):
                    continue
                metric_values.setdefault(metric_name, []).append(float(value))

        if accuracy_values and "accuracy" not in metric_values:
            metric_values["accuracy"] = accuracy_values
        if cost_values:
            metric_values.setdefault("cost", cost_values)
            metric_values.setdefault("total_cost", cost_values)
        if latency_values:
            metric_values.setdefault("latency", latency_values)
            metric_values.setdefault("response_time_ms", latency_values)
        if success_values:
            metric_values.setdefault("success_rate", success_values)
        if "score" not in metric_values and "accuracy" in metric_values:
            metric_values["score"] = list(metric_values["accuracy"])

        summary_metrics = {
            metric_name: self._compute_describe_stats(values)
            for metric_name, values in metric_values.items()
            if values
        }
        if not summary_metrics:
            return None

        return {
            "metrics": summary_metrics,
            "execution_time": float(duration),
            "total_examples": len(results),
            "metadata": {
                "sdk_version": get_version(),
                "aggregation_method": "pandas.describe",
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "source": "hybrid_api",
                **(
                    {"comparability": comparability}
                    if isinstance(comparability, dict)
                    else {}
                ),
            },
        }

    async def evaluate(
        self,
        func: Callable[..., Any],
        config: dict[str, Any],
        dataset: Dataset,
        *,
        sample_lease: SampleBudgetLease | None = None,
        progress_callback: Callable[[int, dict[str, Any]], Any] | None = None,
    ) -> EvaluationResult:
        """Execute trial via external API with batching support.

        Args:
            func: Function to evaluate (not used in hybrid mode).
            config: Configuration parameters (TVAR values).
            dataset: Evaluation dataset.
            sample_lease: Optional sample budget lease.
            progress_callback: Optional progress callback.

        Returns:
            EvaluationResult with metrics and outputs.

        Raises:
            EvaluationError: If evaluation fails.
        """
        start_time = time.time()
        transport = await self._get_transport()
        caps = await self._get_capabilities()

        # Initialize lifecycle manager if needed
        await self._ensure_lifecycle_manager()

        # Prepare examples
        examples = list(dataset)
        total_examples = len(examples)

        # Apply sample budget limits
        if sample_lease is not None:
            available = int(sample_lease.remaining())
            if available < total_examples:
                examples = examples[:available]
                logger.debug(
                    f"Sample budget limited examples to {len(examples)}/{total_examples}"
                )

        if not examples:
            return EvaluationResult(
                config=config,
                example_results=[],
                aggregated_metrics={},
                total_examples=0,
                successful_examples=0,
                duration=0.0,
                sample_budget_exhausted=sample_lease is not None
                and sample_lease.exhausted,
            )

        # Process in batches
        example_results: list[HybridExampleResult] = []
        total_cost = 0.0

        for batch_start in range(0, len(examples), self._batch_size):
            batch_end = min(batch_start + self._batch_size, len(examples))
            batch = examples[batch_start:batch_end]

            # Consume from sample lease
            if sample_lease is not None:
                if not sample_lease.try_take(len(batch)):
                    logger.warning("Sample budget exhausted during batch processing")
                    break

            # Execute batch
            batch_results = await self._execute_batch(transport, caps, config, batch)
            example_results.extend(batch_results)

            # Track cost
            batch_cost = sum(r.cost_usd for r in batch_results)
            total_cost += batch_cost

            # Progress callback
            if progress_callback is not None:
                try:
                    progress_callback(
                        len(example_results),
                        {
                            "batch_size": len(batch),
                            "total_cost": total_cost,
                            "successful": sum(1 for r in example_results if r.success),
                        },
                    )
                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    logger.warning(f"Progress callback error: {e}")

        # Compute aggregated metrics
        duration = time.time() - start_time
        aggregated_metrics, comparability = (
            self._compute_aggregated_metrics_with_comparability(
                example_results, total_cost
            )
        )
        summary_stats = self._build_summary_stats(
            example_results, duration, comparability=comparability
        )

        evaluation_result = EvaluationResult(
            config=config,
            example_results=example_results,
            aggregated_metrics=aggregated_metrics,
            total_examples=len(example_results),
            successful_examples=sum(1 for r in example_results if r.success),
            duration=duration,
            summary_stats=summary_stats,
            sample_budget_exhausted=sample_lease is not None and sample_lease.exhausted,
            examples_consumed=len(example_results),
        )
        # Compatibility-safe dynamic field for downstream trial metadata mapping.
        evaluation_result.comparability = comparability  # type: ignore[attr-defined]
        return evaluation_result

    async def _execute_batch(
        self,
        transport: HybridTransport,
        caps: ServiceCapabilities,
        config: dict[str, Any],
        batch: list[Any],
    ) -> list[HybridExampleResult]:
        """Execute a batch of examples.

        Args:
            transport: Transport to use.
            caps: Service capabilities.
            config: Configuration for this trial.
            batch: Batch of examples to process.

        Returns:
            List of results for each example.
        """
        # Prepare examples list
        inputs = []
        for i, example in enumerate(batch):
            input_data = self._extract_input(example)
            example_id = (
                input_data.get("example_id")
                or input_data.get("input_id")
                or f"ex_{i}"
            )
            inputs.append(
                {
                    "example_id": example_id,
                    "data": input_data,
                }
            )

        request = HybridExecuteRequest(
            tunable_id=self._tunable_id or "default",
            config=config,
            examples=inputs,
            session_id=self._session_id,
            batch_options=BatchOptions(
                parallelism=self._batch_parallelism,
            ),
            timeout_ms=int(self._timeout * 1000),
        )

        try:
            # Execute
            execute_response = await transport.execute(request)

            # Log raw execute response for observability
            logger.info(
                "Hybrid execute response: status=%s, outputs=%d, "
                "operational_metrics=%s",
                execute_response.status,
                len(execute_response.outputs),
                execute_response.operational_metrics,
            )

            # Update session ID if returned
            if execute_response.session_id:
                if self._session_id != execute_response.session_id:
                    self._session_id = execute_response.session_id
                    if self._lifecycle_manager:
                        await self._lifecycle_manager.register(self._session_id)

            # Check if combined mode (quality metrics included)
            if execute_response.quality_metrics:
                # Combined mode - use quality metrics directly
                return self._process_combined_response(batch, inputs, execute_response)
            elif caps.supports_evaluate:
                # Two-phase mode - need separate evaluate call
                return await self._evaluate_outputs(
                    transport, config, batch, inputs, execute_response
                )
            else:
                # No evaluation - return outputs only
                return self._process_execute_only_response(
                    batch, inputs, execute_response
                )

        except TransportError as e:
            logger.error(f"Batch execution failed: {e}")
            # Return error results for all examples
            return [
                HybridExampleResult(
                    example_id=inp["example_id"],
                    expected_output=self._extract_expected(batch[i]),
                    error=str(e),
                )
                for i, inp in enumerate(inputs)
            ]

    def _extract_input(self, example: Any) -> dict[str, Any]:
        """Extract input data from dataset example."""
        # Handle EvaluationExample dataclass
        if hasattr(example, "input_data"):
            input_data = example.input_data
            if isinstance(input_data, dict):
                return input_data
            return {"input": input_data}

        if isinstance(example, dict):
            # Check for common input key patterns
            for key in ["input", "question", "query", "text", "data"]:
                if key in example:
                    return {
                        "input": example[key],
                        **{k: v for k, v in example.items() if k != key},
                    }
            return example
        return {"input": example}

    def _extract_expected(self, example: Any) -> Any:
        """Extract expected output from dataset example."""
        # Handle EvaluationExample dataclass
        if hasattr(example, "expected_output"):
            return example.expected_output

        if isinstance(example, dict):
            for key in ["expected_output", "output", "answer", "target", "label"]:
                if key in example:
                    return example[key]
        return None

    def _process_combined_response(
        self,
        batch: list[Any],
        inputs: list[dict[str, Any]],
        response: Any,  # HybridExecuteResponse
    ) -> list[HybridExampleResult]:
        """Process response in combined mode (quality metrics included)."""
        results: list[HybridExampleResult] = []
        fallback_cost = self._coerce_metric(response.get_total_cost()) / max(
            len(inputs), 1
        )
        fallback_latency = self._coerce_metric(
            response.operational_metrics.get("latency_ms", 0.0)
        )

        for i, inp in enumerate(inputs):
            example_id = inp["example_id"]
            expected = self._extract_expected(batch[i])

            # Find matching output
            output_item = self._find_output_item(response.outputs, example_id)
            output_data = output_item.get("output")
            if output_data is None and "output_id" in output_item:
                output_data = {"output_id": output_item["output_id"]}
            if isinstance(output_item.get("metrics"), dict):
                per_example_metrics, derivation_path = (
                    self._normalize_example_metrics_with_path(
                        output_item.get("metrics", {})
                    )
                )
            else:
                per_example_metrics, derivation_path = ({}, "none")
            error = output_item.get("error")

            # Prefer per-item metrics, fall back to aggregate execution metrics.
            cost = self._coerce_metric(output_item.get("cost_usd"), fallback_cost)
            latency = self._coerce_metric(
                output_item.get("latency_ms"), fallback_latency
            )

            results.append(
                HybridExampleResult(
                    example_id=example_id,
                    actual_output=output_data,
                    expected_output=expected,
                    metrics=per_example_metrics,
                    cost_usd=cost,
                    latency_ms=latency,
                    error=error if isinstance(error, str) else None,
                    metadata={
                        "evaluation_mode": "evaluated",
                        "accuracy_derivation_path": derivation_path,
                    },
                )
            )

        return results

    async def _evaluate_outputs(
        self,
        transport: HybridTransport,
        config: dict[str, Any],
        batch: list[Any],
        inputs: list[dict[str, Any]],
        execute_response: Any,  # HybridExecuteResponse
    ) -> list[HybridExampleResult]:
        """Call evaluate endpoint for quality metrics."""
        # Build evaluations with output + target pairs
        evaluations = []
        for i, inp in enumerate(inputs):
            example_id = inp["example_id"]
            expected = self._extract_expected(batch[i])

            # Find matching output
            output_item = self._find_output_item(execute_response.outputs, example_id)

            evaluation: dict[str, Any] = {
                "example_id": example_id,
                # Legacy compatibility: some services expect input_id.
                "input_id": example_id,
            }
            if "output" in output_item:
                evaluation["output"] = output_item.get("output")
            elif "output_id" in output_item:
                evaluation["output_id"] = output_item.get("output_id")
            else:
                evaluation["output"] = None

            target_key, target_value = self._extract_target_fields(expected)
            evaluation[target_key] = target_value
            evaluations.append(evaluation)

        eval_request = HybridEvaluateRequest(
            tunable_id=self._tunable_id or "default",
            execution_id=execute_response.execution_id,
            evaluations=evaluations,
            kwargs=self._evaluation_kwargs,
            session_id=self._session_id,
            timeout_ms=int(self._timeout * 1000),
        )

        try:
            eval_response = await transport.evaluate(eval_request)

            # Log raw evaluate response for observability
            logger.info(
                "Hybrid evaluate response: results=%s, aggregate=%s",
                [
                    {
                        "example_id": r.get("example_id") or r.get("input_id"),
                        "metrics": r.get("metrics"),
                    }
                    for r in eval_response.results
                ],
                getattr(eval_response, "aggregate_metrics", None),
            )

            # Merge execute and evaluate results
            results: list[HybridExampleResult] = []
            fallback_cost = self._coerce_metric(
                execute_response.get_total_cost()
            ) / max(len(inputs), 1)
            fallback_latency = self._coerce_metric(
                execute_response.operational_metrics.get("latency_ms", 0.0)
            )
            for i, inp in enumerate(inputs):
                example_id = inp["example_id"]
                expected = self._extract_expected(batch[i])

                # Find output from execute
                output_item = self._find_output_item(
                    execute_response.outputs, example_id
                )
                output_data = output_item.get("output")
                if output_data is None and "output_id" in output_item:
                    output_data = {"output_id": output_item["output_id"]}
                execute_error = output_item.get("error")

                # Find metrics from evaluate
                per_example_metrics: dict[str, float] = {}
                eval_error: str | None = None
                eval_expected: str | None = None
                for result in eval_response.results:
                    result_id = result.get("example_id") or result.get("input_id")
                    if result_id == example_id:
                        per_example_metrics, derivation_path = (
                            self._normalize_example_metrics_with_path(
                                result.get("metrics", {})
                            )
                        )
                        if isinstance(result.get("error"), str):
                            eval_error = result.get("error")
                        if result.get("expected_behavior"):
                            eval_expected = result["expected_behavior"]
                        break
                else:
                    derivation_path = "none"

                # Prefer per-item operational metrics when present.
                cost = self._coerce_metric(output_item.get("cost_usd"), fallback_cost)
                latency = self._coerce_metric(
                    output_item.get("latency_ms"), fallback_latency
                )

                # Log raw per-example result for observability
                logger.info(
                    "Hybrid example result: example_id=%s, "
                    "output_id=%s, accuracy=%s, cost=%.6f, "
                    "latency=%.0fms, error=%s",
                    example_id,
                    output_item.get("output_id"),
                    per_example_metrics.get("accuracy"),
                    cost,
                    latency,
                    eval_error or execute_error,
                )

                results.append(
                    HybridExampleResult(
                        example_id=example_id,
                        actual_output=output_data,
                        expected_output=expected or eval_expected,
                        metrics=per_example_metrics,
                        cost_usd=cost,
                        latency_ms=latency,
                        error=eval_error
                        or (execute_error if isinstance(execute_error, str) else None),
                        metadata={
                            "evaluation_mode": "evaluated",
                            "accuracy_derivation_path": derivation_path,
                        },
                    )
                )

            return results

        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.warning(f"Evaluate call failed, using execute-only: {e}")
            return self._process_execute_only_response(
                batch,
                inputs,
                execute_response,
                evaluation_mode="evaluate_fallback",
            )

    def _process_execute_only_response(
        self,
        batch: list[Any],
        inputs: list[dict[str, Any]],
        response: Any,  # HybridExecuteResponse
        *,
        evaluation_mode: str = "execute_only",
    ) -> list[HybridExampleResult]:
        """Process response when no evaluation is available."""
        results: list[HybridExampleResult] = []
        fallback_cost = self._coerce_metric(response.get_total_cost()) / max(
            len(inputs), 1
        )
        fallback_latency = self._coerce_metric(
            response.operational_metrics.get("latency_ms", 0.0)
        )

        for i, inp in enumerate(inputs):
            example_id = inp["example_id"]
            expected = self._extract_expected(batch[i])

            # Find matching output
            output_item = self._find_output_item(response.outputs, example_id)
            output_data = output_item.get("output")
            if output_data is None and "output_id" in output_item:
                output_data = {"output_id": output_item["output_id"]}
            error = output_item.get("error")
            per_example_metrics, derivation_path = (
                self._normalize_example_metrics_with_path(
                    output_item.get("metrics", {})
                )
            )
            cost = self._coerce_metric(output_item.get("cost_usd"), fallback_cost)
            latency = self._coerce_metric(
                output_item.get("latency_ms"), fallback_latency
            )

            results.append(
                HybridExampleResult(
                    example_id=example_id,
                    actual_output=output_data,
                    expected_output=expected,
                    metrics=per_example_metrics,
                    cost_usd=cost,
                    latency_ms=latency,
                    error=error if isinstance(error, str) else None,
                    metadata={
                        "evaluation_mode": evaluation_mode,
                        "accuracy_derivation_path": derivation_path,
                    },
                )
            )

        return results

    def _build_comparability_info(
        self,
        results: list[HybridExampleResult],
        metric_counts: dict[str, int],
        *,
        primary_objective: str = "accuracy",
    ) -> dict[str, Any]:
        """Build comparability metadata from per-example results."""
        total_examples = len(results)
        mode = self._get_comparability_mode()
        if total_examples <= 0:
            return ComparabilityInfo(
                primary_objective=primary_objective,
                warning_codes=["MCI-001"],
                ranking_eligible=False,
            ).to_dict()

        successful_results = [result for result in results if result.success]
        evaluation_modes = {
            str(result.metadata.get("evaluation_mode", "unknown"))
            for result in successful_results
        }
        if not evaluation_modes:
            evaluation_mode = "unknown"
        elif len(evaluation_modes) == 1:
            evaluation_mode = next(iter(evaluation_modes))
        else:
            evaluation_mode = "mixed"

        examples_with_primary_metric = 0
        missing_example_ids: list[str] = []
        derivation_paths: list[str] = []
        split_key_sets: set[tuple[str, ...]] = set()

        for result in results:
            if not result.success:
                missing_example_ids.append(result.example_id)
                continue

            derivation_path = str(
                result.metadata.get("accuracy_derivation_path", "none")
            ).strip()
            if not derivation_path or derivation_path == "none":
                _, derivation_path = self._derive_accuracy_with_path(result.metrics)

            split_keys = tuple(
                sorted(k for k in result.metrics.keys() if k.endswith("_accuracy"))
            )
            if split_keys:
                split_key_sets.add(split_keys)

            primary_metric_value = self._resolve_primary_metric_value(
                result, primary_objective
            )
            if primary_metric_value is not None:
                examples_with_primary_metric += 1
                derivation_paths.append(derivation_path or "none")
            else:
                missing_example_ids.append(result.example_id)

        coverage_ratio = examples_with_primary_metric / total_examples
        per_metric_coverage = {
            metric_name: MetricCoverage(
                present=count,
                total=total_examples,
                ratio=(count / total_examples),
            )
            for metric_name, count in metric_counts.items()
        }

        warning_codes: list[str] = []
        if examples_with_primary_metric == 0:
            warning_codes.append("MCI-004")
        elif coverage_ratio < 1.0:
            warning_codes.append("MCI-002")

        objective_class = classify_objective(primary_objective)
        if objective_class == "quality" and evaluation_mode != "evaluated":
            warning_codes.append("MCI-004")

        non_none_paths = [p for p in derivation_paths if p and p != "none"]
        if len(set(non_none_paths)) > 1:
            warning_codes.append("MCI-003")
        if "split_mean" in non_none_paths:
            warning_codes.append("MCI-005")
        if len(split_key_sets) > 1:
            warning_codes.append("MCI-006")
        if not all(result.success for result in results):
            warning_codes.append("MCI-007")
        warning_codes = list(dict.fromkeys(warning_codes))

        # In warn/strict modes we default to conservative ranking-eligibility.
        has_full_primary_coverage = (
            examples_with_primary_metric == total_examples and total_examples > 0
        )
        ranking_eligible = has_full_primary_coverage
        if objective_class == "quality":
            ranking_eligible = (
                has_full_primary_coverage and evaluation_mode == "evaluated"
            )
        if mode == "legacy":
            ranking_eligible = True

        dominant_path = "none"
        if non_none_paths:
            dominant_path = Counter(non_none_paths).most_common(1)[0][0]

        comparability = ComparabilityInfo(
            primary_objective=primary_objective,
            evaluation_mode=evaluation_mode,
            total_examples=total_examples,
            examples_with_primary_metric=examples_with_primary_metric,
            coverage_ratio=coverage_ratio,
            derivation_path=dominant_path,
            ranking_eligible=ranking_eligible,
            warning_codes=warning_codes,
            per_metric_coverage=per_metric_coverage,
            missing_example_ids=missing_example_ids,
        )
        return comparability.to_dict()

    @staticmethod
    def _resolve_primary_metric_value(
        result: HybridExampleResult,
        primary_objective: str,
    ) -> float | None:
        """Resolve primary metric value from per-example quality or operational fields."""
        if primary_objective in result.metrics:
            value = result.metrics.get(primary_objective)
            if value is not None:
                return float(value)

        lowered = primary_objective.strip().lower()
        if lowered in {"cost", "total_cost"}:
            return float(result.cost_usd)
        if lowered in {"latency", "response_time_ms"}:
            return float(result.latency_ms)
        return None

    @staticmethod
    def _infer_primary_objective(
        aggregated: dict[str, float],
        metric_counts: dict[str, int],
    ) -> str:
        """Infer objective key used for comparability coverage semantics."""
        keys = set(aggregated.keys()) | set(metric_counts.keys())
        for preferred in ("accuracy", "score"):
            if preferred in keys:
                return preferred

        accuracy_like = sorted(
            key for key in keys if isinstance(key, str) and key.endswith("_accuracy")
        )
        if accuracy_like:
            return accuracy_like[0]

        for preferred in ("total_cost", "cost", "latency", "response_time_ms"):
            if preferred in keys:
                return preferred

        if keys:
            return sorted(str(key) for key in keys)[0]
        return "accuracy"

    def _compute_aggregated_metrics_with_comparability(
        self,
        results: list[HybridExampleResult],
        total_cost: float,
    ) -> tuple[dict[str, float], dict[str, Any]]:
        """Compute aggregated metrics and comparability metadata."""
        if not results:
            return {}, self._build_comparability_info([], {})

        aggregated: dict[str, float] = {
            "cost": total_cost,
            # Keep canonical trial-level key for downstream extractors.
            "total_cost": total_cost,
            "success_rate": sum(1 for r in results if r.success) / len(results),
        }

        # Aggregate per-example metrics
        metric_sums: dict[str, float] = {}
        metric_counts: dict[str, int] = {}

        for result in results:
            for metric_name, value in result.metrics.items():
                if metric_name not in metric_sums:
                    metric_sums[metric_name] = 0.0
                    metric_counts[metric_name] = 0
                metric_sums[metric_name] += value
                metric_counts[metric_name] += 1
            if result.success:
                for metric_name in (
                    "cost",
                    "total_cost",
                    "latency",
                    "response_time_ms",
                ):
                    metric_counts[metric_name] = metric_counts.get(metric_name, 0) + 1

        # Compute means
        for metric_name, total in metric_sums.items():
            count = metric_counts[metric_name]
            if count > 0:
                aggregated[metric_name] = total / count

        # Average latency
        latencies = [r.latency_ms for r in results if r.latency_ms > 0]
        if latencies:
            aggregated["latency"] = sum(latencies) / len(latencies)

        if "accuracy" not in aggregated:
            derived_accuracy = self._derive_accuracy_from_metrics(aggregated)
            if derived_accuracy is not None:
                aggregated["accuracy"] = derived_accuracy

        if "score" not in aggregated and "accuracy" in aggregated:
            aggregated["score"] = aggregated["accuracy"]
        if "response_time_ms" not in aggregated and "latency" in aggregated:
            aggregated["response_time_ms"] = aggregated["latency"]

        primary_objective = self._infer_primary_objective(aggregated, metric_counts)
        comparability = self._build_comparability_info(
            results,
            metric_counts,
            primary_objective=primary_objective,
        )
        return aggregated, comparability

    def _compute_aggregated_metrics(
        self,
        results: list[HybridExampleResult],
        total_cost: float,
    ) -> dict[str, float]:
        """Compute aggregated metrics from example results."""
        aggregated, _ = self._compute_aggregated_metrics_with_comparability(
            results, total_cost
        )
        return aggregated

    async def close(self) -> None:
        """Close evaluator and release resources."""
        if self._lifecycle_manager is not None:
            await self._lifecycle_manager.release()
            self._lifecycle_manager = None

        if self._transport is not None and self._owns_transport:
            await self._transport.close()
            self._transport = None

    async def __aenter__(self) -> HybridAPIEvaluator:
        """Async context manager entry."""
        return self

    async def __aexit__(self, *args: Any) -> None:
        """Async context manager exit."""
        await self.close()
