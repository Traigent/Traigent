"""Langfuse API client for reading traces and extracting metrics.

This client reads traces from Langfuse to extract metrics for Traigent optimization.
It supports both sync and async operations.

The client extracts:
- Total cost, latency, and token counts
- Per-agent metrics (using langgraph_node metadata or observation names)
- OpenInference-compatible attributes

Usage:
    client = LangfuseClient(
        public_key="pk-xxx",  # pragma: allowlist secret
        secret_key="sk-xxx",  # pragma: allowlist secret
    )

    # Get metrics for optimization
    metrics = await client.get_trace_metrics(trace_id="trace_123")

    # Use in Traigent measures
    measures = metrics.to_measures_dict()
    # {"total_cost": 0.006, "total_latency_ms": 1200, "grader_cost": 0.001, ...}
"""

# Traceability: CONC-Layer-Integration FUNC-INTEGRATIONS REQ-INT-008

from __future__ import annotations

import asyncio
import json
import os
import threading
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
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


# Observations API v2 (Langfuse v4 data model). See
# https://langfuse.com/faq/all/deprecated-api-migration
_V2_OBSERVATIONS_PATH = "/api/public/v2/observations"
_V2_FIELDS = "core,basic,metadata,model,usage,metrics,trace_context"
_V2_IO_FIELDS = _V2_FIELDS + ",io"  # adds input/output payloads (get_trace only)


def _root_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Root observations among v2 rows.

    Langfuse logical roots (``isRootObservation`` true, ``basic`` field group)
    may carry a non-null ``parentObservationId`` (distributed/linked traces),
    so they take precedence. Physical roots (``parentObservationId`` present
    and None) are the fallback when no row is flagged.
    """
    logical = [r for r in rows if r.get("isRootObservation") is True]
    if logical:
        return logical
    return [
        r
        for r in rows
        if "parentObservationId" in r and r["parentObservationId"] is None
    ]


_V2_PAGE_LIMIT = 1000  # v2 max (v1 was 100)
# Legacy v1 /api/public/observations rejects limits above 100.
_LEGACY_PAGE_LIMIT = 100
# v2 requires a bounded start-time window; traces are looked up shortly after
# they are produced, so a 30 day look-back is generous. Default for the
# ``v2_lookback_days`` constructor argument.
_V2_LOOKBACK_DAYS = 30
_V2_MAX_LOOKBACK_DAYS = 3650
# A 404 from v2 that is not yet backed by any v2 success downgrades the client
# to the legacy API; the downgrade expires after this many seconds so a
# transient 404 (proxy, rolling deploy) cannot pin the client to v1 forever.
_LEGACY_REPROBE_SECONDS = 600


class _V2Unavailable(Exception):
    """The v2 observations endpoint does not exist (Langfuse v3 self-hosted)."""


def _to_float(value: Any) -> float:
    """Coerce numbers or decimal strings (v2 returns costs as strings) to float."""
    try:
        return float(value) if value is not None else 0.0
    except (TypeError, ValueError):
        return 0.0


def _to_int(value: Any) -> int:
    return int(_to_float(value))


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
    observations_partial: bool = False

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
            f"{prefix}observations_partial": int(self.observations_partial),
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
        v2_lookback_days: How far back (days) the Langfuse v4 Observations API v2
            searches for a trace's observations (default 30). The window is
            ``[now - v2_lookback_days, now + 1h]`` computed once per fetch; the
            +1h upper bound tolerates client clock skew. Traces whose
            observations started before the window are NOT found. A fetched
            trace whose rows contain no root observation (its root started
            before the window) is reported partial; a trace with a root
            inside the window is complete.
            Must be an int (not bool) with ``1 <= v2_lookback_days <= 3650``,
            otherwise ``ValueError`` is raised.

    Example:
        client = LangfuseClient(
            public_key="pk-xxx",
            secret_key="sk-xxx",  # pragma: allowlist secret
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
        v2_lookback_days: int = _V2_LOOKBACK_DAYS,
    ) -> None:
        """Initialize the Langfuse client."""
        self.public_key = public_key or os.environ.get("LANGFUSE_PUBLIC_KEY")
        self.secret_key = secret_key or os.environ.get("LANGFUSE_SECRET_KEY")
        resolved_host: str = host or os.environ.get(  # type: ignore[assignment]
            "LANGFUSE_HOST", "https://cloud.langfuse.com"
        )
        self.host = resolved_host.rstrip("/")
        self.timeout = timeout
        if (
            isinstance(v2_lookback_days, bool)
            or not isinstance(v2_lookback_days, int)
            or not 1 <= v2_lookback_days <= _V2_MAX_LOOKBACK_DAYS
        ):
            raise ValueError(
                f"v2_lookback_days must be an int between 1 and {_V2_MAX_LOOKBACK_DAYS}"
            )
        self.v2_lookback_days = v2_lookback_days
        self._observations_partial_by_trace: dict[str, bool] = {}
        # Set once the v2 endpoint answers 404 (Langfuse v3 self-hosted): use
        # the legacy v1 endpoints from then on.
        self._legacy_api = False
        # Monotonic time of the (unconfirmed) downgrade; injectable for tests.
        self._clock = time.monotonic
        self._legacy_since = 0.0
        # Set once any v2 request succeeded: from then on a 404 is an ordinary
        # failure, never evidence that the endpoint is missing.
        self._v2_confirmed = False
        self._api_flag_lock = threading.Lock()

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

    def _confirm_v2(self) -> None:
        """Record a v2 success: v2 is confirmed and any downgrade is undone."""
        with self._api_flag_lock:
            self._v2_confirmed = True
            self._legacy_api = False

    def _switch_to_legacy(self) -> bool:
        """Flip to the v1 API unless a v2 request has already succeeded.

        Returns True if the client is now on the legacy API; False if the switch
        was refused (v2 already confirmed), in which case the caller must treat
        the request as a failed v2 request and not use the legacy path.
        """
        with self._api_flag_lock:
            if self._v2_confirmed:
                return False
            self._legacy_api = True
            self._legacy_since = self._clock()
            return True

    def _use_legacy(self) -> bool:
        """Whether to use the legacy API; an unconfirmed downgrade expires."""
        with self._api_flag_lock:
            if (
                self._legacy_api
                and not self._v2_confirmed
                and self._clock() - self._legacy_since >= _LEGACY_REPROBE_SECONDS
            ):
                self._legacy_api = False  # re-probe v2
            return self._legacy_api

    @staticmethod
    def _strip_io(trace: dict[str, Any]) -> dict[str, Any]:
        """Drop ``input``/``output`` from a legacy trace and its observations."""
        out = {k: v for k, v in trace.items() if k not in ("input", "output")}
        obs = out.get("observations")
        if isinstance(obs, list):
            out["observations"] = [
                (
                    {k: v for k, v in o.items() if k not in ("input", "output")}
                    if isinstance(o, dict)
                    else o
                )
                for o in obs
            ]
        return out

    def get_trace(
        self, trace_id: str, *, include_io: bool = True
    ) -> dict[str, Any] | None:
        """Get a trace by ID.

        Args:
            trace_id: The trace ID to fetch
            include_io: Also fetch trace ``input``/``output`` (v2 ``io`` field
                group, taken from the root observation). Defaults to True
                (unchanged public behaviour). On the legacy (self-hosted v3)
                fallback, I/O is still downloaded and is only stripped from
                the returned dicts when False. Callers that only need metrics
                should use ``get_trace_metrics``, which never requests io on the
                v2 path (on the legacy fallback io is downloaded, then stripped).

        Returns:
            Trace data dict or None if not found
        """
        # The Langfuse SDK's read helpers (get_trace/get_observations) were
        # removed in SDK v3+ and the endpoints they call are removed in Langfuse
        # v4, so reads always go through the HTTP API (v2 with v1 fallback).
        if include_io:
            return self._get_trace_http(trace_id)
        return self._get_trace_http(trace_id, include_io=False)

    def _get_trace_http(
        self, trace_id: str, *, include_io: bool = True
    ) -> dict[str, Any] | None:
        """Get trace via HTTP API (v2 observations, legacy v1 on Langfuse v3)."""
        if not REQUESTS_AVAILABLE:
            raise ImportError(
                "requests is required. Install with: pip install requests"
            )

        if not self._use_legacy():
            try:
                rows, partial = self._fetch_observations_v2(
                    trace_id, include_io=include_io
                )
            except _V2Unavailable:
                if not self._switch_to_legacy():
                    logger.error(f"Failed to get trace {trace_id}: v2 404")
                    return None
            except requests.exceptions.RequestException as e:
                logger.error(f"Failed to get trace {trace_id}: {e}")
                return None
            else:
                return self._trace_from_observation_rows(
                    trace_id, rows, partial, include_io=include_io
                )

        return self._get_trace_http_legacy(trace_id, include_io=include_io)

    def _get_trace_http_legacy(
        self, trace_id: str, *, include_io: bool = True
    ) -> dict[str, Any] | None:
        """Legacy v1 ``GET /api/public/traces/{id}`` (Langfuse v3 self-hosted only)."""
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
            return result if include_io else self._strip_io(result)
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to get trace {trace_id}: {e}")
            return None

    # ---- v2 helpers -------------------------------------------------------

    def _v2_window(self) -> tuple[str, str]:
        """Start-time window for one fetch (computed once, reused on every page)."""
        now = datetime.now(UTC)
        return (
            (now - timedelta(days=self.v2_lookback_days)).isoformat(),
            (now + timedelta(hours=1)).isoformat(),
        )

    @staticmethod
    def _v2_params(
        trace_id: str,
        cursor: str | None,
        window: tuple[str, str],
        include_io: bool = False,
    ) -> dict[str, str]:
        params = {
            "traceId": trace_id,
            "fields": _V2_IO_FIELDS if include_io else _V2_FIELDS,
            "limit": str(_V2_PAGE_LIMIT),
            "fromStartTime": window[0],
            "toStartTime": window[1],
        }
        if cursor:
            params["cursor"] = cursor
        return params

    @staticmethod
    def _v2_page(payload: Any) -> tuple[list[dict[str, Any]], str | None]:
        """Split a v2 response into (rows, next cursor)."""
        if not isinstance(payload, dict):
            return [], None
        rows = payload.get("data") or []
        meta = payload.get("meta") or {}
        cursor = meta.get("cursor") if isinstance(meta, dict) else None
        return [r for r in rows if isinstance(r, dict)], cursor or None

    @staticmethod
    def _merge_rows(
        rows: list[dict[str, Any]],
        seen_ids: set[str],
        page: list[dict[str, Any]],
    ) -> int:
        """Append page rows, de-duplicating observations by ``id``.

        Rows whose ``id`` is missing, not a string or empty cannot be
        de-duplicated and are excluded; returns how many were dropped.
        """
        invalid = 0
        for row in page:
            row_id = row.get("id")
            if not isinstance(row_id, str) or not row_id:
                invalid += 1
                continue
            if row_id in seen_ids:
                continue
            seen_ids.add(row_id)
            rows.append(row)
        return invalid

    def _finish_v2(
        self,
        rows: list[dict[str, Any]],
        invalid: int,
        window: tuple[str, str],
    ) -> tuple[list[dict[str, Any]], bool]:
        """Final (rows, partial) once pagination ended cleanly.

        Partial if rows were dropped for invalid ids, or if rows were fetched
        but none is a root observation (``isRootObservation`` true, or
        ``parentObservationId`` present and None): the root started before the window's lower bound, so only
        later children were found.
        """
        partial = False
        if invalid:
            logger.warning(f"Dropped {invalid} observation row(s) with invalid id.")
            partial = True
        has_root = bool(_root_rows(rows))
        if rows and not has_root:
            logger.warning(
                "No root observation in the lookback window; "
                "trace started before it and is partial."
            )
            partial = True
        return rows, partial

    def _fetch_observations_v2(
        self,
        trace_id: str,
        *,
        max_pages: int = 100,
        include_io: bool = False,
    ) -> tuple[list[dict[str, Any]], bool]:
        """Fetch all observation rows of a trace via cursor pagination (sync).

        Returns (rows, partial). Raises _V2Unavailable only when the FIRST page
        answers 404 and no v2 request has ever succeeded. A failure after at
        least one page returns the accumulated rows with partial=True. With no
        rows yet, requests exceptions propagate. A repeated cursor stops the
        loop (partial=True).
        """
        rows: list[dict[str, Any]] = []
        seen_ids: set[str] = set()
        seen_cursors: set[str] = set()
        window = self._v2_window()
        invalid = 0
        cursor: str | None = None
        for page_no in range(max_pages):
            try:
                response = requests.get(
                    f"{self.host}{_V2_OBSERVATIONS_PATH}",
                    params=self._v2_params(trace_id, cursor, window, include_io),
                    headers=self._get_auth_header(),
                    timeout=self.timeout,
                )
                if response.status_code == 404:
                    if page_no == 0 and not self._v2_confirmed:
                        raise _V2Unavailable()
                    raise requests.exceptions.HTTPError("404 from v2 observations")
                response.raise_for_status()
                page, cursor = self._v2_page(response.json())
            except requests.exceptions.RequestException:
                if not rows:
                    raise
                return rows, True
            self._confirm_v2()
            invalid += self._merge_rows(rows, seen_ids, page)
            if not cursor:
                return self._finish_v2(rows, invalid, window)
            if cursor in seen_cursors:
                logger.warning(f"Cursor cycle fetching observations for {trace_id}.")
                return rows, True
            seen_cursors.add(cursor)
        logger.warning(
            f"Hit max_pages limit ({max_pages}) fetching observations "
            f"for trace {trace_id}. Some observations may be missing."
        )
        return rows, True

    async def _fetch_observations_v2_async(
        self,
        trace_id: str,
        *,
        max_pages: int = 100,
        include_io: bool = False,
    ) -> tuple[list[dict[str, Any]], bool]:
        """Async version of _fetch_observations_v2."""
        rows: list[dict[str, Any]] = []
        seen_ids: set[str] = set()
        seen_cursors: set[str] = set()
        window = self._v2_window()
        invalid = 0
        cursor: str | None = None
        async with aiohttp.ClientSession(trust_env=True) as session:
            for page_no in range(max_pages):
                try:
                    async with session.get(
                        f"{self.host}{_V2_OBSERVATIONS_PATH}",
                        params=self._v2_params(trace_id, cursor, window, include_io),
                        headers=self._get_auth_header(),
                        timeout=aiohttp.ClientTimeout(total=self.timeout),
                    ) as response:
                        if response.status == 404:
                            if page_no == 0 and not self._v2_confirmed:
                                raise _V2Unavailable()
                            raise aiohttp.ClientError("404 from v2 observations")
                        response.raise_for_status()
                        page, cursor = self._v2_page(await response.json())
                except (TimeoutError, aiohttp.ClientError):
                    if not rows:
                        raise
                    return rows, True
                self._confirm_v2()
                invalid += self._merge_rows(rows, seen_ids, page)
                if not cursor:
                    return self._finish_v2(rows, invalid, window)
                if cursor in seen_cursors:
                    logger.warning(
                        f"Cursor cycle fetching observations for {trace_id}."
                    )
                    return rows, True
                seen_cursors.add(cursor)
        logger.warning(
            f"Hit max_pages limit ({max_pages}) fetching observations "
            f"for trace {trace_id}. Some observations may be missing."
        )
        return rows, True

    @staticmethod
    def _decode_io(value: Any) -> Any:
        """JSON-decode a string payload; keep the raw string if it is not JSON."""
        if isinstance(value, str):
            try:
                return json.loads(value)
            except ValueError:
                return value
        return value

    @staticmethod
    def _trace_from_observation_rows(
        trace_id: str,
        rows: list[dict[str, Any]],
        partial: bool,
        *,
        include_io: bool = False,
    ) -> dict[str, Any] | None:
        """Rebuild a v1-shaped trace dict from v2 observation rows.

        v4 has no trace objects; trace-level fields come from the root
        observation (``isRootObservation`` true, else ``parentObservationId``
        present and null; explicit logical roots win; the earliest
        ``startTime`` wins, ties broken by ``id``). With no root, root-derived fields stay
        unset instead of borrowing a child's. Returns None when the trace has no
        observations (same as a v1 404).
        """
        if not rows:
            return None
        roots = _root_rows(rows)
        root: dict[str, Any] | None = None
        if roots:
            root = min(
                roots,
                key=lambda r: (
                    r.get("startTime") is None,
                    r.get("startTime") or "",
                    str(r.get("id") or ""),
                ),
            )
            if len(roots) > 1:
                logger.warning(
                    f"Trace {trace_id} has {len(roots)} root observations; "
                    f"using the earliest ({root.get('id')})."
                )
        else:
            logger.warning(f"Trace {trace_id} has no root observation.")
        trace: dict[str, Any] = {
            "id": trace_id,
            "name": (root.get("traceName") or root.get("name")) if root else None,
            "metadata": (root.get("metadata") or {}) if root else None,
            "sessionId": root.get("sessionId") if root else None,
            "userId": root.get("userId") if root else None,
            "observations": rows,
            "observationsPartial": partial,
        }
        if include_io:
            trace["input"] = (
                LangfuseClient._decode_io(root.get("input")) if root else None
            )
            trace["output"] = (
                LangfuseClient._decode_io(root.get("output")) if root else None
            )
        return trace

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
        # v2 never requests io here; the legacy fallback downloads the full
        # response and strips io afterwards.
        trace_data = self.get_trace(trace_id, include_io=False)
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
        return self._get_observations_http(trace_id)

    def _get_observations_http(
        self, trace_id: str, *, max_pages: int = 100
    ) -> list[LangfuseObservation]:
        """Get observations via HTTP API with pagination.

        Uses v2 cursor pagination; falls back to legacy v1 page pagination on
        Langfuse v3 self-hosted (v2 answers 404).
        """
        if not REQUESTS_AVAILABLE:
            raise ImportError("requests is required")

        if not self._use_legacy():
            self._observations_partial_by_trace[trace_id] = False
            try:
                rows, partial = self._fetch_observations_v2(
                    trace_id, max_pages=max_pages
                )
            except _V2Unavailable:
                if not self._switch_to_legacy():
                    self._observations_partial_by_trace[trace_id] = True
                    return []
            except requests.exceptions.RequestException as e:
                logger.error(f"Failed to get observations for trace {trace_id}: {e}")
                self._observations_partial_by_trace[trace_id] = True
                return []
            else:
                self._observations_partial_by_trace[trace_id] = partial
                return [self._dict_to_observation(r) for r in rows]

        return self._get_observations_http_legacy(trace_id, max_pages=max_pages)

    def _get_observations_http_legacy(
        self, trace_id: str, *, max_pages: int = 100
    ) -> list[LangfuseObservation]:
        """Legacy v1 ``GET /api/public/observations`` (Langfuse v3 self-hosted)."""

        observations: list[LangfuseObservation] = []
        page = 1
        self._observations_partial_by_trace[trace_id] = False

        try:
            while page <= max_pages:
                response = requests.get(
                    f"{self.host}/api/public/observations",
                    params={
                        "traceId": trace_id,
                        "limit": str(_LEGACY_PAGE_LIMIT),
                        "page": str(page),
                    },
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
                self._observations_partial_by_trace[trace_id] = True

            return observations
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to get observations for trace {trace_id}: {e}")
            self._observations_partial_by_trace[trace_id] = True
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
            trace = self.get_trace(trace_id, include_io=False)
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

    async def get_trace_async(
        self, trace_id: str, *, include_io: bool = True
    ) -> dict[str, Any] | None:
        """Async version of get_trace."""
        if not AIOHTTP_AVAILABLE:
            raise ImportError(
                "aiohttp is required for async. Install with: pip install aiohttp"
            )

        if not self._use_legacy():
            try:
                rows, partial = await self._fetch_observations_v2_async(
                    trace_id, include_io=include_io
                )
            except _V2Unavailable:
                if not self._switch_to_legacy():
                    logger.error(f"Failed to get trace {trace_id}: v2 404")
                    return None
            except (TimeoutError, aiohttp.ClientError) as e:
                logger.error(f"Failed to get trace {trace_id}: {e}")
                return None
            else:
                return self._trace_from_observation_rows(
                    trace_id, rows, partial, include_io=include_io
                )

        return await self._get_trace_async_legacy(trace_id, include_io=include_io)

    async def _get_trace_async_legacy(
        self, trace_id: str, *, include_io: bool = True
    ) -> dict[str, Any] | None:
        """Legacy v1 trace endpoint (Langfuse v3 self-hosted only)."""
        try:
            async with aiohttp.ClientSession(trust_env=True) as session:
                async with session.get(
                    f"{self.host}/api/public/traces/{trace_id}",
                    headers=self._get_auth_header(),
                    timeout=aiohttp.ClientTimeout(total=self.timeout),
                ) as response:
                    if response.status == 404:
                        return None
                    response.raise_for_status()
                    result: dict[str, Any] = await response.json()
                    return result if include_io else self._strip_io(result)
        except (TimeoutError, aiohttp.ClientError) as e:
            logger.error(f"Failed to get trace {trace_id}: {e}")
            return None

    async def get_trace_metrics_async(
        self, trace_id: str
    ) -> LangfuseTraceMetrics | None:
        """Async version of get_trace_metrics."""
        trace_data = await self.get_trace_async(trace_id, include_io=False)
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

        if not self._use_legacy():
            self._observations_partial_by_trace[trace_id] = False
            try:
                rows, partial = await self._fetch_observations_v2_async(
                    trace_id, max_pages=max_pages
                )
            except _V2Unavailable:
                if not self._switch_to_legacy():
                    self._observations_partial_by_trace[trace_id] = True
                    return []
            except (TimeoutError, aiohttp.ClientError) as e:
                logger.error(f"Failed to get observations for trace {trace_id}: {e}")
                self._observations_partial_by_trace[trace_id] = True
                return []
            else:
                self._observations_partial_by_trace[trace_id] = partial
                return [self._dict_to_observation(r) for r in rows]

        return await self._get_observations_async_legacy(trace_id, max_pages=max_pages)

    async def _get_observations_async_legacy(
        self, trace_id: str, *, max_pages: int = 100
    ) -> list[LangfuseObservation]:
        """Legacy v1 observations endpoint (Langfuse v3 self-hosted only)."""
        observations: list[LangfuseObservation] = []
        page = 1
        self._observations_partial_by_trace[trace_id] = False

        try:
            async with aiohttp.ClientSession(trust_env=True) as session:
                while page <= max_pages:
                    async with session.get(
                        f"{self.host}/api/public/observations",
                        params={
                            "traceId": trace_id,
                            "limit": str(_LEGACY_PAGE_LIMIT),
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
                self._observations_partial_by_trace[trace_id] = True

            return observations
        except (TimeoutError, aiohttp.ClientError) as e:
            logger.error(f"Failed to get observations for trace {trace_id}: {e}")
            self._observations_partial_by_trace[trace_id] = True
            return observations  # Return what we have so far

    async def wait_for_trace_async(
        self,
        trace_id: str,
        timeout_seconds: float = 60.0,
        poll_interval: float = 1.0,
    ) -> bool:
        """Async version of wait_for_trace."""
        loop = asyncio.get_running_loop()
        start = loop.time()
        while loop.time() - start < timeout_seconds:
            trace = await self.get_trace_async(trace_id, include_io=False)
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

        # Extract token counts (v1: ``usage`` dict; v2: flat *Usage fields)
        input_tokens = (
            usage.get("input", 0)
            or usage.get("promptTokens", 0)
            or _to_int(obs_data.get("inputUsage"))
        )
        output_tokens = (
            usage.get("output", 0)
            or usage.get("completionTokens", 0)
            or _to_int(obs_data.get("outputUsage"))
        )
        total_tokens = (
            usage.get("total", 0)
            or usage.get("totalTokens", 0)
            or _to_int(obs_data.get("totalUsage"))
            or (input_tokens + output_tokens)
        )

        # Extract cost (v1: calculatedTotalCost number; v2: totalCost string)
        cost = _to_float(obs_data.get("calculatedTotalCost")) or _to_float(
            obs_data.get("totalCost")
        )

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
        observations_partial = bool(
            trace_data.get("observations_partial")
            or trace_data.get("observationsPartial")
        )
        if "observations" in trace_data:
            for obs_data in trace_data["observations"]:
                observations.append(self._dict_to_observation(obs_data))
        else:
            # Fetch observations separately
            observations = self.get_observations_for_trace(trace_id)
            observations_partial = self._observations_partial_by_trace.get(
                trace_id, False
            )

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
            observations_partial=observations_partial,
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
