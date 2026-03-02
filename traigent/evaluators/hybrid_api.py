"""Hybrid API evaluator for external agentic service optimization.

Provides evaluation via external HTTP or MCP endpoints, enabling
optimization of any agentic service that implements the Traigent
hybrid API protocol.
"""

# Traceability: HYBRID-MODE-OPTIMIZATION EVALUATOR-INTEGRATION

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from traigent.evaluators.base import BaseEvaluator, Dataset, EvaluationResult
from traigent.hybrid import (
    AgentLifecycleManager,
    BatchOptions,
    ConfigSpaceDiscovery,
    HybridEvaluateRequest,
    HybridExecuteRequest,
    HybridTransport,
    ServiceCapabilities,
    TransportError,
    create_transport,
)
from traigent.utils.logging import get_logger

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
        input_id: Identifier for the input example
        actual_output: Output produced by the agent
        expected_output: Expected output (from dataset)
        metrics: Per-example quality metrics
        cost_usd: Cost for this example
        latency_ms: Latency for this example
        error: Error message if failed
    """

    input_id: str
    actual_output: Any = None
    expected_output: Any = None
    metrics: dict[str, float] = field(default_factory=dict)
    cost_usd: float = 0.0
    latency_ms: float = 0.0
    error: str | None = None

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

    # NOSONAR - public constructor intentionally exposes transport/runtime knobs.
    def __init__(
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
        input_id: str,
    ) -> dict[str, Any]:
        """Find output item for an input_id."""
        for out in outputs:
            if out.get("input_id") == input_id:
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
        aggregated_metrics = self._compute_aggregated_metrics(
            example_results, total_cost
        )

        return EvaluationResult(
            config=config,
            example_results=example_results,
            aggregated_metrics=aggregated_metrics,
            total_examples=len(example_results),
            successful_examples=sum(1 for r in example_results if r.success),
            duration=duration,
            sample_budget_exhausted=sample_lease is not None and sample_lease.exhausted,
            examples_consumed=len(example_results),
        )

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
        # Prepare inputs
        inputs = []
        for i, example in enumerate(batch):
            input_data = self._extract_input(example)
            input_id = input_data.get("input_id", f"ex_{i}")
            inputs.append(
                {
                    "input_id": input_id,
                    "data": input_data,
                }
            )

        # Build execute request
        request = HybridExecuteRequest(
            tunable_id=self._tunable_id or "default",
            config=config,
            inputs=inputs,
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
                    transport, batch, inputs, execute_response
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
                    input_id=inp["input_id"],
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
            input_id = inp["input_id"]
            expected = self._extract_expected(batch[i])

            # Find matching output
            output_item = self._find_output_item(response.outputs, input_id)
            output_data = output_item.get("output")
            if output_data is None and "output_id" in output_item:
                output_data = {"output_id": output_item["output_id"]}
            per_example_metrics = (
                output_item.get("metrics", {})
                if isinstance(output_item.get("metrics"), dict)
                else {}
            )
            error = output_item.get("error")

            # Prefer per-item metrics, fall back to aggregate execution metrics.
            cost = self._coerce_metric(output_item.get("cost_usd"), fallback_cost)
            latency = self._coerce_metric(
                output_item.get("latency_ms"), fallback_latency
            )

            results.append(
                HybridExampleResult(
                    input_id=input_id,
                    actual_output=output_data,
                    expected_output=expected,
                    metrics=per_example_metrics,
                    cost_usd=cost,
                    latency_ms=latency,
                    error=error if isinstance(error, str) else None,
                )
            )

        return results

    async def _evaluate_outputs(
        self,
        transport: HybridTransport,
        batch: list[Any],
        inputs: list[dict[str, Any]],
        execute_response: Any,  # HybridExecuteResponse
    ) -> list[HybridExampleResult]:
        """Call evaluate endpoint for quality metrics."""
        # Build evaluations with output + target pairs
        evaluations = []
        for i, inp in enumerate(inputs):
            input_id = inp["input_id"]
            expected = self._extract_expected(batch[i])

            # Find matching output
            output_item = self._find_output_item(execute_response.outputs, input_id)

            evaluation: dict[str, Any] = {"input_id": input_id}
            if "output" in output_item:
                evaluation["output"] = output_item.get("output")
            elif "output_id" in output_item:
                evaluation["output_id"] = output_item.get("output_id")
            else:
                evaluation["output"] = None

            target_key, target_value = self._extract_target_fields(expected)
            evaluation[target_key] = target_value
            evaluations.append(evaluation)

        # Call evaluate
        eval_request = HybridEvaluateRequest(
            tunable_id=self._tunable_id or "default",
            execution_id=execute_response.execution_id,
            evaluations=evaluations,
            session_id=self._session_id,
        )

        try:
            eval_response = await transport.evaluate(eval_request)

            # Log raw evaluate response for observability
            logger.info(
                "Hybrid evaluate response: results=%s, aggregate=%s",
                [
                    {"input_id": r.get("input_id"), "metrics": r.get("metrics")}
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
                input_id = inp["input_id"]
                expected = self._extract_expected(batch[i])

                # Find output from execute
                output_item = self._find_output_item(execute_response.outputs, input_id)
                output_data = output_item.get("output")
                if output_data is None and "output_id" in output_item:
                    output_data = {"output_id": output_item["output_id"]}
                execute_error = output_item.get("error")

                # Find metrics from evaluate
                per_example_metrics: dict[str, float] = {}
                eval_error: str | None = None
                eval_expected: str | None = None
                for result in eval_response.results:
                    if result.get("input_id") == input_id:
                        per_example_metrics = result.get("metrics", {})
                        if isinstance(result.get("error"), str):
                            eval_error = result.get("error")
                        if result.get("expected_behavior"):
                            eval_expected = result["expected_behavior"]
                        break

                # Prefer per-item operational metrics when present.
                cost = self._coerce_metric(output_item.get("cost_usd"), fallback_cost)
                latency = self._coerce_metric(
                    output_item.get("latency_ms"), fallback_latency
                )

                # Log raw per-example result for observability
                logger.info(
                    "Hybrid example result: input_id=%s, "
                    "output_id=%s, accuracy=%s, cost=%.6f, "
                    "latency=%.0fms, error=%s",
                    input_id,
                    output_item.get("output_id"),
                    per_example_metrics.get("accuracy"),
                    cost,
                    latency,
                    eval_error or execute_error,
                )

                results.append(
                    HybridExampleResult(
                        input_id=input_id,
                        actual_output=output_data,
                        expected_output=expected or eval_expected,
                        metrics=per_example_metrics,
                        cost_usd=cost,
                        latency_ms=latency,
                        error=eval_error
                        or (execute_error if isinstance(execute_error, str) else None),
                    )
                )

            return results

        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.warning(f"Evaluate call failed, using execute-only: {e}")
            return self._process_execute_only_response(batch, inputs, execute_response)

    def _process_execute_only_response(
        self,
        batch: list[Any],
        inputs: list[dict[str, Any]],
        response: Any,  # HybridExecuteResponse
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
            input_id = inp["input_id"]
            expected = self._extract_expected(batch[i])

            # Find matching output
            output_item = self._find_output_item(response.outputs, input_id)
            output_data = output_item.get("output")
            if output_data is None and "output_id" in output_item:
                output_data = {"output_id": output_item["output_id"]}
            error = output_item.get("error")
            cost = self._coerce_metric(output_item.get("cost_usd"), fallback_cost)
            latency = self._coerce_metric(
                output_item.get("latency_ms"), fallback_latency
            )

            results.append(
                HybridExampleResult(
                    input_id=input_id,
                    actual_output=output_data,
                    expected_output=expected,
                    metrics={},  # No quality metrics
                    cost_usd=cost,
                    latency_ms=latency,
                    error=error if isinstance(error, str) else None,
                )
            )

        return results

    def _compute_aggregated_metrics(
        self,
        results: list[HybridExampleResult],
        total_cost: float,
    ) -> dict[str, float]:
        """Compute aggregated metrics from example results."""
        if not results:
            return {}

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

        # Compute means
        for metric_name, total in metric_sums.items():
            count = metric_counts[metric_name]
            if count > 0:
                aggregated[metric_name] = total / count

        # Average latency
        latencies = [r.latency_ms for r in results if r.latency_ms > 0]
        if latencies:
            aggregated["latency"] = sum(latencies) / len(latencies)

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
