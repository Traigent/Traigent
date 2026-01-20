"""Langfuse API client for reading traces and extracting metrics.

This client reads traces from Langfuse to extract metrics for Traigent optimization.
It supports both sync and async operations.

The client extracts:
- Total cost, latency, and token counts
- Per-agent metrics (using langgraph_node metadata or observation names)
- OpenInference-compatible attributes

Usage:
    client = LangfuseClient(
        public_key="pk-xxx",
        secret_key="sk-xxx",
    )

    # Get metrics for optimization
    metrics = await client.get_trace_metrics(trace_id="trace_123")

    # Use in Traigent measures
    measures = metrics.to_measures_dict()
    # {"total_cost": 0.006, "total_latency_ms": 1200, "grader_cost": 0.001, ...}
"""

# Traceability: CONC-Layer-Integration FUNC-INTEGRATIONS REQ-INT-008

from __future__ import annotations

import os
import threading
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any

from traigent.utils.logging import get_logger

logger = get_logger(__name__)

# Check for optional dependencies
try:
    import aiohttp

    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False
    aiohttp = None  # type: ignore[assignment]

try:
    import requests

    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    requests = None  # type: ignore[assignment]

try:
    from langfuse import Langfuse

    LANGFUSE_SDK_AVAILABLE = True
except ImportError:
    LANGFUSE_SDK_AVAILABLE = False
    Langfuse = None  # type: ignore[assignment, misc]

if TYPE_CHECKING:
    from langfuse import Langfuse as LangfuseType


# =============================================================================
# Data Models
# =============================================================================


@dataclass
class LangfuseObservation:
    """Represents a single observation from Langfuse (span, generation, etc.)."""

    id: str
    name: str
    observation_type: str  # "span", "generation", "event"
    start_time: datetime | None = None
    end_time: datetime | None = None
    model: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cost: float = 0.0
    latency_ms: float = 0.0
    status: str = "success"
    parent_observation_id: str | None = None

    # Agent identification (from metadata)
    langgraph_node: str | None = None
    langgraph_step: int | None = None
    openinference_node_id: str | None = None

    # Raw metadata for custom extraction
    metadata: dict[str, Any] = field(default_factory=dict)

    def get_agent_identifier(self) -> str | None:
        """Get agent identifier using priority: OpenInference > langgraph_node > name."""
        # Priority 1: OpenInference graph.node.id
        if self.openinference_node_id:
            return self.openinference_node_id

        # Priority 2: LangGraph metadata
        if self.langgraph_node:
            return self.langgraph_node

        # Priority 3: Fall back to observation name (heuristic)
        # Only if name looks like an agent name (not generic like "LLMChain")
        if self.name and not self.name.startswith(("LLM", "Chat", "Chain")):
            return self.name

        return None


@dataclass
class LangfuseTraceMetrics:
    """Aggregated metrics extracted from a Langfuse trace."""

    trace_id: str
    total_cost: float = 0.0
    total_latency_ms: float = 0.0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_tokens: int = 0

    # Per-agent metrics (keyed by agent identifier with underscores)
    per_agent_costs: dict[str, float] = field(default_factory=dict)
    per_agent_latencies: dict[str, float] = field(default_factory=dict)
    per_agent_tokens: dict[str, int] = field(default_factory=dict)

    # All observations for detailed analysis
    observations: list[LangfuseObservation] = field(default_factory=list)

    # Trace metadata
    trace_name: str | None = None
    trace_metadata: dict[str, Any] = field(default_factory=dict)
    session_id: str | None = None
    user_id: str | None = None

    def to_measures_dict(
        self,
        *,
        prefix: str = "",
        include_per_agent: bool = True,
    ) -> dict[str, float | int]:
        """Convert to MeasuresDict-compatible format (underscore keys, numeric values).

        Args:
            prefix: Prefix for all metric keys (e.g., "langfuse_")
            include_per_agent: Include per-agent breakdown metrics (default: True)

        Returns:
            Dictionary with keys like "total_cost", "grader_cost", "generator_latency_ms"
            (or "langfuse_total_cost" etc. if prefix is set)
        """
        measures: dict[str, float | int] = {
            f"{prefix}total_cost": self.total_cost,
            f"{prefix}total_latency_ms": self.total_latency_ms,
            f"{prefix}total_input_tokens": self.total_input_tokens,
            f"{prefix}total_output_tokens": self.total_output_tokens,
            f"{prefix}total_tokens": self.total_tokens,
        }

        if include_per_agent:
            # Add per-agent metrics with underscore naming
            for agent, cost in self.per_agent_costs.items():
                # Sanitize agent name: replace dots/dashes with underscores
                safe_agent = agent.replace(".", "_").replace("-", "_").replace(" ", "_")
                measures[f"{prefix}{safe_agent}_cost"] = cost

            for agent, latency in self.per_agent_latencies.items():
                safe_agent = agent.replace(".", "_").replace("-", "_").replace(" ", "_")
                measures[f"{prefix}{safe_agent}_latency_ms"] = latency

            for agent, tokens in self.per_agent_tokens.items():
                safe_agent = agent.replace(".", "_").replace("-", "_").replace(" ", "_")
                measures[f"{prefix}{safe_agent}_tokens"] = tokens

        return measures


# =============================================================================
# Langfuse Client
# =============================================================================


class LangfuseClient:
    """Client for reading traces from Langfuse API.

    This client supports two modes:
    1. Using the official Langfuse SDK (recommended if installed)
    2. Direct HTTP API calls (fallback)

    The client reads traces and extracts metrics for Traigent optimization,
    including per-agent cost attribution using OpenInference attributes
    and LangGraph metadata.

    Args:
        public_key: Langfuse public key (or env LANGFUSE_PUBLIC_KEY)
        secret_key: Langfuse secret key (or env LANGFUSE_SECRET_KEY)
        host: Langfuse host URL (default: https://cloud.langfuse.com)
        timeout: Request timeout in seconds

    Example:
        client = LangfuseClient(
            public_key="pk-xxx",
            secret_key="sk-xxx",
        )

        # Get metrics for a trace
        metrics = client.get_trace_metrics("trace-id-123")
        print(metrics.total_cost)
        print(metrics.per_agent_costs)
    """

    def __init__(
        self,
        public_key: str | None = None,
        secret_key: str | None = None,
        host: str | None = None,
        timeout: float = 30.0,
    ) -> None:
        """Initialize the Langfuse client."""
        self.public_key = public_key or os.environ.get("LANGFUSE_PUBLIC_KEY")
        self.secret_key = secret_key or os.environ.get("LANGFUSE_SECRET_KEY")
        self.host = (
            host or os.environ.get("LANGFUSE_HOST", "https://cloud.langfuse.com")
        ).rstrip("/")
        self.timeout = timeout
        self._lock = threading.Lock()

        # Initialize SDK client if available
        self._sdk_client: LangfuseType | None = None
        if LANGFUSE_SDK_AVAILABLE and self.public_key and self.secret_key:
            try:
                self._sdk_client = Langfuse(
                    public_key=self.public_key,
                    secret_key=self.secret_key,
                    host=self.host,
                )
                logger.debug("Initialized Langfuse SDK client")
            except Exception as e:
                logger.warning(f"Failed to initialize Langfuse SDK: {e}")
                self._sdk_client = None

    def _get_auth_header(self) -> dict[str, str]:
        """Get HTTP Basic Auth header for direct API calls."""
        if not self.public_key or not self.secret_key:
            raise ValueError("Langfuse public_key and secret_key are required")

        import base64

        credentials = f"{self.public_key}:{self.secret_key}"
        encoded = base64.b64encode(credentials.encode()).decode()
        return {
            "Authorization": f"Basic {encoded}",
            "Content-Type": "application/json",
        }

    # =========================================================================
    # Sync API
    # =========================================================================

    def get_trace(self, trace_id: str) -> dict[str, Any] | None:
        """Get a trace by ID.

        Args:
            trace_id: The trace ID to fetch

        Returns:
            Trace data dict or None if not found
        """
        # Try SDK first
        if self._sdk_client:
            try:
                trace = self._sdk_client.get_trace(trace_id)
                if trace:
                    # SDK returns a Trace object, convert to dict
                    return self._trace_to_dict(trace)
            except Exception as e:
                logger.warning(f"SDK get_trace failed, falling back to HTTP: {e}")

        # Fall back to HTTP API
        return self._get_trace_http(trace_id)

    def _get_trace_http(self, trace_id: str) -> dict[str, Any] | None:
        """Get trace via HTTP API."""
        if not REQUESTS_AVAILABLE:
            raise ImportError(
                "requests is required. Install with: pip install requests"
            )

        try:
            response = requests.get(
                f"{self.host}/api/public/traces/{trace_id}",
                headers=self._get_auth_header(),
                timeout=self.timeout,
            )

            if response.status_code == 404:
                return None

            response.raise_for_status()
            result: dict[str, Any] = response.json()
            return result
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to get trace {trace_id}: {e}")
            return None

    def get_trace_metrics(self, trace_id: str) -> LangfuseTraceMetrics | None:
        """Get aggregated metrics for a trace.

        This is the main method for extracting optimization metrics from Langfuse.
        It fetches the trace and all its observations, then aggregates:
        - Total cost, latency, tokens
        - Per-agent costs (using OpenInference/LangGraph metadata)

        Args:
            trace_id: The trace ID to analyze

        Returns:
            LangfuseTraceMetrics with aggregated data, or None if trace not found
        """
        trace_data = self.get_trace(trace_id)
        if not trace_data:
            return None

        return self._extract_metrics_from_trace(trace_data)

    def get_observations_for_trace(self, trace_id: str) -> list[LangfuseObservation]:
        """Get all observations for a trace.

        Args:
            trace_id: The trace ID

        Returns:
            List of LangfuseObservation objects
        """
        # Try SDK first
        if self._sdk_client:
            try:
                # SDK method to get observations
                observations = self._sdk_client.get_observations(trace_id=trace_id)
                if observations:
                    return [
                        self._observation_to_model(obs) for obs in observations.data
                    ]
            except Exception as e:
                logger.warning(f"SDK get_observations failed: {e}")

        # Fall back to HTTP API
        return self._get_observations_http(trace_id)

    def _get_observations_http(
        self, trace_id: str, *, max_pages: int = 100
    ) -> list[LangfuseObservation]:
        """Get observations via HTTP API with pagination.

        Args:
            trace_id: The trace ID to fetch observations for
            max_pages: Maximum number of pages to fetch (safety limit)

        Returns:
            List of all observations for the trace
        """
        if not REQUESTS_AVAILABLE:
            raise ImportError("requests is required")

        observations: list[LangfuseObservation] = []
        page = 1

        try:
            while page <= max_pages:
                response = requests.get(
                    f"{self.host}/api/public/observations",
                    params={"traceId": trace_id, "limit": "1000", "page": str(page)},
                    headers=self._get_auth_header(),
                    timeout=self.timeout,
                )
                response.raise_for_status()
                data = response.json()

                page_data = data.get("data", [])
                if not page_data:
                    # No more data
                    break

                for obs_data in page_data:
                    observations.append(self._dict_to_observation(obs_data))

                # Check if there are more pages
                meta = data.get("meta", {})
                total_items = meta.get("totalItems", 0)
                if len(observations) >= total_items:
                    break

                page += 1

            if page > max_pages:
                logger.warning(
                    f"Hit max_pages limit ({max_pages}) fetching observations "
                    f"for trace {trace_id}. Some observations may be missing."
                )

            return observations
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to get observations for trace {trace_id}: {e}")
            return observations  # Return what we have so far

    def wait_for_trace(
        self,
        trace_id: str,
        timeout_seconds: float = 60.0,
        poll_interval: float = 1.0,
    ) -> bool:
        """Wait for a trace to be available and fully processed.

        Langfuse ingestion is async, so traces may not be immediately available
        after a workflow completes. This method polls until the trace is ready.

        Args:
            trace_id: The trace ID to wait for
            timeout_seconds: Maximum time to wait
            poll_interval: Time between polls

        Returns:
            True if trace is available, False if timeout
        """
        import time

        start = time.time()
        while time.time() - start < timeout_seconds:
            trace = self.get_trace(trace_id)
            if trace:
                # Check if trace has observations (indicates processing complete)
                obs = self.get_observations_for_trace(trace_id)
                if obs:
                    return True
            time.sleep(poll_interval)

        logger.warning(f"Timeout waiting for trace {trace_id}")
        return False

    # =========================================================================
    # Async API
    # =========================================================================

    async def get_trace_async(self, trace_id: str) -> dict[str, Any] | None:
        """Async version of get_trace."""
        if not AIOHTTP_AVAILABLE:
            raise ImportError(
                "aiohttp is required for async. Install with: pip install aiohttp"
            )

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.host}/api/public/traces/{trace_id}",
                    headers=self._get_auth_header(),
                    timeout=aiohttp.ClientTimeout(total=self.timeout),
                ) as response:
                    if response.status == 404:
                        return None
                    response.raise_for_status()
                    result: dict[str, Any] = await response.json()
                    return result
        except aiohttp.ClientError as e:
            logger.error(f"Failed to get trace {trace_id}: {e}")
            return None

    async def get_trace_metrics_async(
        self, trace_id: str
    ) -> LangfuseTraceMetrics | None:
        """Async version of get_trace_metrics."""
        trace_data = await self.get_trace_async(trace_id)
        if not trace_data:
            return None

        return self._extract_metrics_from_trace(trace_data)

    async def get_observations_for_trace_async(
        self, trace_id: str, *, max_pages: int = 100
    ) -> list[LangfuseObservation]:
        """Async version of get_observations_for_trace with pagination.

        Args:
            trace_id: The trace ID to fetch observations for
            max_pages: Maximum number of pages to fetch (safety limit)

        Returns:
            List of all observations for the trace
        """
        if not AIOHTTP_AVAILABLE:
            raise ImportError("aiohttp is required for async")

        observations: list[LangfuseObservation] = []
        page = 1

        try:
            async with aiohttp.ClientSession() as session:
                while page <= max_pages:
                    async with session.get(
                        f"{self.host}/api/public/observations",
                        params={
                            "traceId": trace_id,
                            "limit": "1000",
                            "page": str(page),
                        },
                        headers=self._get_auth_header(),
                        timeout=aiohttp.ClientTimeout(total=self.timeout),
                    ) as response:
                        response.raise_for_status()
                        data = await response.json()

                        page_data = data.get("data", [])
                        if not page_data:
                            # No more data
                            break

                        for obs_data in page_data:
                            observations.append(self._dict_to_observation(obs_data))

                        # Check if there are more pages
                        meta = data.get("meta", {})
                        total_items = meta.get("totalItems", 0)
                        if len(observations) >= total_items:
                            break

                        page += 1

            if page > max_pages:
                logger.warning(
                    f"Hit max_pages limit ({max_pages}) fetching observations "
                    f"for trace {trace_id}. Some observations may be missing."
                )

            return observations
        except aiohttp.ClientError as e:
            logger.error(f"Failed to get observations for trace {trace_id}: {e}")
            return observations  # Return what we have so far

    async def wait_for_trace_async(
        self,
        trace_id: str,
        timeout_seconds: float = 60.0,
        poll_interval: float = 1.0,
    ) -> bool:
        """Async version of wait_for_trace."""
        import asyncio

        start = asyncio.get_event_loop().time()
        while asyncio.get_event_loop().time() - start < timeout_seconds:
            trace = await self.get_trace_async(trace_id)
            if trace:
                obs = await self.get_observations_for_trace_async(trace_id)
                if obs:
                    return True
            await asyncio.sleep(poll_interval)

        logger.warning(f"Timeout waiting for trace {trace_id}")
        return False

    # =========================================================================
    # Internal Conversion Methods
    # =========================================================================

    def _trace_to_dict(self, trace: Any) -> dict[str, Any]:
        """Convert SDK Trace object to dict."""
        # Handle both SDK object and dict
        if isinstance(trace, dict):
            return trace

        # SDK Trace object - extract attributes
        result: dict[str, Any] = {
            "id": getattr(trace, "id", None),
            "name": getattr(trace, "name", None),
            "metadata": getattr(trace, "metadata", {}),
            "sessionId": getattr(trace, "session_id", None),
            "userId": getattr(trace, "user_id", None),
            "input": getattr(trace, "input", None),
            "output": getattr(trace, "output", None),
            "observations": [],
        }

        # Add observations if present
        if hasattr(trace, "observations"):
            result["observations"] = [
                self._observation_to_dict(obs) for obs in trace.observations
            ]

        return result

    def _observation_to_dict(self, obs: Any) -> dict[str, Any]:
        """Convert SDK Observation object to dict."""
        if isinstance(obs, dict):
            return obs

        return {
            "id": getattr(obs, "id", None),
            "name": getattr(obs, "name", None),
            "type": getattr(obs, "type", "span"),
            "startTime": getattr(obs, "start_time", None),
            "endTime": getattr(obs, "end_time", None),
            "model": getattr(obs, "model", None),
            "modelParameters": getattr(obs, "model_parameters", {}),
            "usage": getattr(obs, "usage", {}),
            "metadata": getattr(obs, "metadata", {}),
            "parentObservationId": getattr(obs, "parent_observation_id", None),
            "level": getattr(obs, "level", "DEFAULT"),
            "statusMessage": getattr(obs, "status_message", None),
            "calculatedTotalCost": getattr(obs, "calculated_total_cost", None),
            "calculatedInputCost": getattr(obs, "calculated_input_cost", None),
            "calculatedOutputCost": getattr(obs, "calculated_output_cost", None),
            "latency": getattr(obs, "latency", None),
        }

    def _observation_to_model(self, obs: Any) -> LangfuseObservation:
        """Convert SDK Observation to LangfuseObservation model."""
        obs_dict = self._observation_to_dict(obs)
        return self._dict_to_observation(obs_dict)

    def _dict_to_observation(self, obs_data: dict[str, Any]) -> LangfuseObservation:
        """Convert dict to LangfuseObservation model."""
        metadata = obs_data.get("metadata") or {}
        usage = obs_data.get("usage") or {}

        # Extract token counts
        input_tokens = usage.get("input", 0) or usage.get("promptTokens", 0) or 0
        output_tokens = usage.get("output", 0) or usage.get("completionTokens", 0) or 0
        total_tokens = (
            usage.get("total", 0)
            or usage.get("totalTokens", 0)
            or (input_tokens + output_tokens)
        )

        # Extract cost
        cost = obs_data.get("calculatedTotalCost") or 0.0

        # Calculate latency from timestamps
        latency_ms = 0.0
        if obs_data.get("latency"):
            # Langfuse returns latency in seconds
            latency_ms = float(obs_data["latency"]) * 1000
        elif obs_data.get("startTime") and obs_data.get("endTime"):
            start = self._parse_timestamp(obs_data["startTime"])
            end = self._parse_timestamp(obs_data["endTime"])
            if start and end:
                latency_ms = (end - start).total_seconds() * 1000

        # Extract agent identifiers from metadata
        # Priority: OpenInference > langgraph_node > name
        langgraph_node = metadata.get("langgraph_node")
        langgraph_step = metadata.get("langgraph_step")
        openinference_node_id = (
            metadata.get("graph.node.id")
            or metadata.get("openinference.node.id")
            or metadata.get("node_id")
        )

        return LangfuseObservation(
            id=obs_data.get("id", ""),
            name=obs_data.get("name", ""),
            observation_type=obs_data.get("type", "span"),
            start_time=self._parse_timestamp(obs_data.get("startTime")),
            end_time=self._parse_timestamp(obs_data.get("endTime")),
            model=obs_data.get("model"),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            cost=cost,
            latency_ms=latency_ms,
            status="error" if obs_data.get("level") == "ERROR" else "success",
            parent_observation_id=obs_data.get("parentObservationId"),
            langgraph_node=langgraph_node,
            langgraph_step=langgraph_step,
            openinference_node_id=openinference_node_id,
            metadata=metadata,
        )

    def _parse_timestamp(self, ts: Any) -> datetime | None:
        """Parse timestamp from various formats."""
        if ts is None:
            return None

        if isinstance(ts, datetime):
            return ts

        if isinstance(ts, str):
            # Try ISO format
            try:
                return datetime.fromisoformat(ts.replace("Z", "+00:00"))
            except ValueError:
                pass

        return None

    def _extract_metrics_from_trace(
        self, trace_data: dict[str, Any]
    ) -> LangfuseTraceMetrics:
        """Extract aggregated metrics from trace data.

        Args:
            trace_data: Raw trace data from API

        Returns:
            LangfuseTraceMetrics with aggregated metrics
        """
        trace_id = trace_data.get("id", "")
        trace_name = trace_data.get("name")
        trace_metadata = trace_data.get("metadata") or {}
        session_id = trace_data.get("sessionId")
        user_id = trace_data.get("userId")

        # Get observations (may be embedded or need separate fetch)
        observations: list[LangfuseObservation] = []
        if "observations" in trace_data:
            for obs_data in trace_data["observations"]:
                observations.append(self._dict_to_observation(obs_data))
        else:
            # Fetch observations separately
            observations = self.get_observations_for_trace(trace_id)

        # Aggregate metrics
        total_cost = 0.0
        total_latency_ms = 0.0
        total_input_tokens = 0
        total_output_tokens = 0
        total_tokens = 0

        per_agent_costs: dict[str, float] = {}
        per_agent_latencies: dict[str, float] = {}
        per_agent_tokens: dict[str, int] = {}

        for obs in observations:
            total_cost += obs.cost
            total_latency_ms += obs.latency_ms
            total_input_tokens += obs.input_tokens
            total_output_tokens += obs.output_tokens
            total_tokens += obs.total_tokens

            # Per-agent attribution
            agent_id = obs.get_agent_identifier()
            if agent_id:
                # Sanitize for MeasuresDict compatibility
                safe_agent = (
                    agent_id.replace(".", "_").replace("-", "_").replace(" ", "_")
                )

                per_agent_costs[safe_agent] = (
                    per_agent_costs.get(safe_agent, 0.0) + obs.cost
                )
                per_agent_latencies[safe_agent] = (
                    per_agent_latencies.get(safe_agent, 0.0) + obs.latency_ms
                )
                per_agent_tokens[safe_agent] = (
                    per_agent_tokens.get(safe_agent, 0) + obs.total_tokens
                )

        return LangfuseTraceMetrics(
            trace_id=trace_id,
            total_cost=total_cost,
            total_latency_ms=total_latency_ms,
            total_input_tokens=total_input_tokens,
            total_output_tokens=total_output_tokens,
            total_tokens=total_tokens,
            per_agent_costs=per_agent_costs,
            per_agent_latencies=per_agent_latencies,
            per_agent_tokens=per_agent_tokens,
            observations=observations,
            trace_name=trace_name,
            trace_metadata=trace_metadata,
            session_id=session_id,
            user_id=user_id,
        )


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "LangfuseClient",
    "LangfuseTraceMetrics",
    "LangfuseObservation",
    "LANGFUSE_SDK_AVAILABLE",
    "AIOHTTP_AVAILABLE",
    "REQUESTS_AVAILABLE",
]
