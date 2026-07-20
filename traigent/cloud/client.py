"""Traigent backend client with reserved cloud remote-execution methods."""

# Traceability: CONC-Layer-Infra CONC-Quality-Reliability FUNC-CLOUD-HYBRID FUNC-AGENTS REQ-CLOUD-009 REQ-AGNT-013

from __future__ import annotations

import asyncio
import inspect
import os
import time
import weakref
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from importlib import import_module
from typing import Any, NoReturn, cast
from urllib.parse import urlparse

from traigent.cloud._aiohttp_compat import AIOHTTP_AVAILABLE, aiohttp

# Capture the genuine, unpatched ClientSession class at module-load time — before
# any test can monkeypatch the mutable ``aiohttp.ClientSession`` module attribute.
# The submodule attribute (``aiohttp.client.ClientSession``) is never the patch
# target, so ``isinstance`` checks against this reference stay correct even when
# ``aiohttp.ClientSession`` has been replaced with a MagicMock by another test.
if AIOHTTP_AVAILABLE:
    try:
        from aiohttp.client import ClientSession as _REAL_AIOHTTP_CLIENT_SESSION
    except ImportError:  # pragma: no cover - unexpected aiohttp layout
        _REAL_AIOHTTP_CLIENT_SESSION = aiohttp.ClientSession
else:  # pragma: no cover - minimal environments without aiohttp
    _REAL_AIOHTTP_CLIENT_SESSION = None

from traigent.config.backend_config import BackendConfig
from traigent.evaluators.base import Dataset
from traigent.utils.artifact_fingerprints import (
    artifact_fingerprints_to_wire,
    fingerprint_meta_to_wire,
)
from traigent.utils.env_config import is_backend_offline
from traigent.utils.error_handler import OfflineModeError
from traigent.utils.error_handler import TraigentError as HelpfulTraigentError
from traigent.utils.exceptions import ValidationError as ValidationException
from traigent.utils.logging import get_logger
from traigent.utils.retry import NetworkError, RateLimitError, retry_http_request
from traigent.utils.validation import CoreValidators, validate_or_raise

from .auth import AuthenticationError, AuthManager, _strip_trace_context_headers
from .billing import UsageTracker
from .models import (
    AgentExecutionRequest,
    AgentExecutionResponse,
    AgentOptimizationRequest,
    AgentOptimizationResponse,
    AgentOptimizationStatus,
    AgentSpecification,
    NextTrialRequest,
    NextTrialResponse,
    OptimizationFinalizationRequest,
    OptimizationFinalizationResponse,
    OptimizationSessionStatus,
    SessionCreationRequest,
    SessionCreationResponse,
    SessionObjectiveDefinition,
    SessionSummary,
    TrialResultSubmission,
    TrialSuggestion,
)
from .session_budgets import (
    ensure_cost_metric_for_budgeted_completed_submission,
    is_cost_budget_armed_session,
    remember_cost_budget_armed_session,
)
from .session_objectives import normalize_typed_objectives, session_objective_to_wire
from .smart_pruning import normalize_smart_pruning_options
from .subset_selection import SmartSubsetSelector
from .url_security import validate_cloud_base_url

logger = get_logger(__name__)

# Error messages for session state validation
_SESSION_NOT_INITIALIZED = "Session not initialized"
_CLIENT_SESSION_NOT_INITIALIZED = "Client session not initialized"
_AGENT_SPEC_REQUIRED = "agent_spec is required"
CLOUD_REMOTE_EXECUTION_UNAVAILABLE = (
    "Cloud remote execution is not available yet; use hybrid for "
    "portal-tracked optimization. Supported modes are local/edge_analytics "
    "and hybrid."
)


class CloudEgressBlockedError(OfflineModeError):
    """Raised before backend transport when offline/no-egress is active."""

    def __init__(self, operation: str) -> None:
        self.operation = operation
        HelpfulTraigentError.__init__(
            self,
            message=f"backend egress is disabled (offline); not sending {operation}",
            fix=(
                "Unset TRAIGENT_OFFLINE/TRAIGENT_OFFLINE_MODE or clear the "
                "runtime no_egress policy before using backend communication."
            ),
            docs_link="https://github.com/Traigent/Traigent#offline-mode",
        )


def no_egress_flag_enabled(value: Any) -> bool:
    """Return True only for an explicit boolean no-egress policy flag."""

    return value is True


def cloud_backend_egress_disabled(no_egress: Any = False) -> bool:
    """Return whether cloud/backend transport must be blocked."""

    if no_egress_flag_enabled(no_egress):
        return True
    return bool(is_backend_offline())


def raise_if_cloud_egress_disabled(
    operation: str,
    *,
    no_egress: Any = False,
) -> None:
    """Fail closed before backend transport when offline/no-egress is active."""

    if cloud_backend_egress_disabled(no_egress):
        raise CloudEgressBlockedError(operation)


def _session_is_closed(session: Any) -> bool:
    """Robustly determine whether an aiohttp session is closed."""
    closed_flag = getattr(session, "closed", False)
    if isinstance(closed_flag, bool):
        return closed_flag
    return False


def _log_finalizer_close_task_result(task: asyncio.Future[Any], owner: str) -> None:
    try:
        task.result()
    except asyncio.CancelledError:
        logger.debug("%s aiohttp session close task was cancelled", owner)
    except Exception as exc:  # pragma: no cover - defensive finalizer callback
        logger.debug("Error closing %s aiohttp session: %s", owner, exc)


def _close_awaitable(thing: Any) -> None:
    close_method = getattr(thing, "close", None)
    if callable(close_method):
        try:
            close_method()
        except Exception:
            pass


def _schedule_aiohttp_session_close(
    session: Any,
    loop: asyncio.AbstractEventLoop,
    owner: str,
) -> bool:
    close_method = getattr(session, "close", None)
    if not callable(close_method):
        return False

    def create_task() -> None:
        close_result: Any = None
        try:
            close_result = close_method()
            if inspect.isawaitable(close_result):
                task: asyncio.Future[Any] = asyncio.ensure_future(
                    cast(Awaitable[Any], close_result),
                    loop=loop,
                )
                task.add_done_callback(
                    lambda done_task: _log_finalizer_close_task_result(done_task, owner)
                )
        except Exception as exc:  # pragma: no cover - defensive finalizer path
            _close_awaitable(close_result)
            logger.debug("Error scheduling %s aiohttp session close: %s", owner, exc)

    try:
        current_loop = asyncio.get_running_loop()
    except RuntimeError:
        current_loop = None

    try:
        if current_loop is loop:
            create_task()
        else:
            loop.call_soon_threadsafe(create_task)
    except Exception as exc:  # pragma: no cover - defensive finalizer path
        logger.debug("Error scheduling %s aiohttp session close: %s", owner, exc)
        return False
    return True


def _run_aiohttp_session_close(
    session: Any,
    loop: asyncio.AbstractEventLoop,
    owner: str,
) -> bool:
    close_method = getattr(session, "close", None)
    if not callable(close_method):
        return False

    close_result: Any = None
    try:
        close_result = close_method()
        if inspect.isawaitable(close_result):
            loop.run_until_complete(close_result)
        return True
    except Exception as exc:  # pragma: no cover - defensive finalizer path
        _close_awaitable(close_result)
        logger.debug("Error closing %s aiohttp session on its loop: %s", owner, exc)
        return False


def _iter_connector_protocols(connector: Any) -> list[Any]:
    protocols: list[Any] = []
    seen: set[int] = set()

    def add(protocol: Any) -> None:
        if protocol is None:
            return
        identifier = id(protocol)
        if identifier in seen:
            return
        seen.add(identifier)
        protocols.append(protocol)

    conns = getattr(connector, "_conns", None)
    values = getattr(conns, "values", None)
    if callable(values):
        for pooled_connections in list(values()):
            for connection_entry in list(pooled_connections):
                protocol = connection_entry
                if isinstance(connection_entry, (tuple, list)) and connection_entry:
                    protocol = connection_entry[0]
                add(getattr(protocol, "protocol", protocol))

    for protocol in list(getattr(connector, "_acquired", ()) or ()):
        add(getattr(protocol, "protocol", protocol))

    return protocols


def _raw_socket_from_transport(transport: Any) -> Any:
    get_extra_info = getattr(transport, "get_extra_info", None)
    if callable(get_extra_info):
        try:
            socket_info = get_extra_info("socket")
        except Exception:
            socket_info = None
        if socket_info is not None:
            return getattr(socket_info, "_sock", socket_info)
    return getattr(transport, "_sock", None)


def _close_transport_sync(transport: Any, seen: set[int] | None = None) -> None:
    if transport is None:
        return

    if seen is None:
        seen = set()
    identifier = id(transport)
    if identifier in seen:
        return
    seen.add(identifier)

    for method_name in ("abort", "close"):
        method = getattr(transport, method_name, None)
        if callable(method):
            try:
                method()
            except Exception:
                pass

    call_connection_lost = getattr(transport, "_call_connection_lost", None)
    if callable(call_connection_lost):
        try:
            call_connection_lost(None)
        except Exception:
            pass

    raw_socket = _raw_socket_from_transport(transport)
    if raw_socket is not None:
        close_socket = getattr(raw_socket, "close", None)
        if callable(close_socket):
            try:
                close_socket()
            except Exception:
                pass

    for nested_attr in ("_transport",):
        nested_transport = getattr(transport, nested_attr, None)
        if nested_transport is not None:
            _close_transport_sync(nested_transport, seen)

    ssl_protocol = getattr(transport, "_ssl_protocol", None)
    nested_transport = getattr(ssl_protocol, "_transport", None)
    if nested_transport is not None:
        _close_transport_sync(nested_transport, seen)


def _close_protocol_sync(protocol: Any) -> None:
    transport = getattr(protocol, "transport", None)

    for method_name in ("abort", "close"):
        method = getattr(protocol, method_name, None)
        if callable(method):
            try:
                method()
                break
            except Exception:
                pass

    _close_transport_sync(transport)


def _cancel_connector_handle(connector: Any, name: str) -> None:
    handle = getattr(connector, name, None)
    cancel = getattr(handle, "cancel", None)
    if callable(cancel):
        try:
            cancel()
        except Exception:
            pass
    if hasattr(connector, name):
        try:
            setattr(connector, name, None)
        except Exception:
            pass


def _clear_connector_waiters(connector: Any) -> None:
    waiters = getattr(connector, "_waiters", None)
    values = getattr(waiters, "values", None)
    if callable(values):
        for keyed_waiters in list(values()):
            for waiter in list(keyed_waiters):
                cancel = getattr(waiter, "cancel", None)
                if callable(cancel):
                    try:
                        cancel()
                    except Exception:
                        pass
    clear = getattr(waiters, "clear", None)
    if callable(clear):
        try:
            clear()
        except Exception:
            pass


def _clear_connector_collection(connector: Any, name: str) -> None:
    collection = getattr(connector, name, None)
    clear = getattr(collection, "clear", None)
    if callable(clear):
        try:
            clear()
        except Exception:
            pass


def _close_connector_transports_sync(connector: Any) -> None:
    for protocol in _iter_connector_protocols(connector):
        _close_protocol_sync(protocol)

    for transport in list(getattr(connector, "_cleanup_closed_transports", ()) or ()):
        _close_transport_sync(transport)

    for handle_name in ("_cleanup_handle", "_cleanup_closed_handle"):
        _cancel_connector_handle(connector, handle_name)

    _clear_connector_waiters(connector)
    for collection_name in (
        "_conns",
        "_acquired",
        "_acquired_per_host",
        "_cleanup_closed_transports",
    ):
        _clear_connector_collection(connector, collection_name)

    if hasattr(connector, "_closed"):
        try:
            connector._closed = True
        except Exception:
            pass


def _close_aiohttp_session_transports_sync(session: Any) -> None:
    connector = getattr(session, "_connector", None)
    owns_connector = bool(getattr(session, "_connector_owner", True))

    if connector is not None and owns_connector:
        _close_connector_transports_sync(connector)

    if hasattr(session, "_connector"):
        try:
            session._connector = None
        except Exception:
            pass


def _is_real_aiohttp_session(session: Any) -> bool:
    """Return True only when *session* is a genuine aiohttp ClientSession.

    The ``isinstance`` check uses ``_REAL_AIOHTTP_CLIENT_SESSION`` — the real
    class captured from ``aiohttp.client`` at module-load time — rather than the
    mutable ``aiohttp.ClientSession`` module attribute. Tests routinely
    monkeypatch that attribute to a MagicMock; checking against it would both
    raise ``TypeError`` (a non-type as the second isinstance arg) AND wrongly
    reject a *genuine* session whenever the global is polluted, which would skip
    finalizer registration on a real session. The captured class is always a
    real type, so the check stays correct under such pollution.
    """
    return (
        AIOHTTP_AVAILABLE
        and _REAL_AIOHTTP_CLIENT_SESSION is not None
        and isinstance(session, _REAL_AIOHTTP_CLIENT_SESSION)
    )


def _finalize_aiohttp_session(session: Any, owner: str) -> None:
    """Synchronously close an aiohttp session from a weakref finalizer."""

    if not _is_real_aiohttp_session(session):
        return

    try:
        if session is None or _session_is_closed(session):
            return

        session_loop = getattr(session, "_loop", None)
        if session_loop is not None and not session_loop.is_closed():
            if session_loop.is_running():
                if _schedule_aiohttp_session_close(session, session_loop, owner):
                    return
            elif _run_aiohttp_session_close(session, session_loop, owner):
                return

        _close_aiohttp_session_transports_sync(session)
    except Exception as exc:  # pragma: no cover - defensive finalizer
        logger.debug("Error finalizing %s aiohttp session: %s", owner, exc)


def _coerce_enum(enum_cls: Any, raw: Any, *, fallback: Any) -> Any:
    """Case/format-tolerant enum coercion for inbound backend status fields.

    The backend may send a status in upper case (``RUNNING``), lower case
    (``running``), or — for forward/backward compatibility — a value not in the
    SDK's enum at all. The strict ``EnumCls(raw)`` construction raises
    ``ValueError`` on any of these, which previously crashed deserialization of
    an otherwise-valid response (issue #1302, AC3).

    This helper tries, in order:
      1. the raw value as-is (handles already-correct lowercase enum values),
      2. the lower-cased value (handles backend UPPER -> SDK lowercase enums),
      3. matching by member *name* (handles UPPER name like ``RUNNING``),
    and otherwise returns ``fallback`` — a neutral default, never silently
    ``COMPLETED``, so a failed/running run is never reported as succeeded.
    """
    if isinstance(raw, enum_cls):
        return raw
    for candidate in (raw, str(raw).lower() if raw is not None else raw):
        try:
            return enum_cls(candidate)
        except (ValueError, KeyError):
            continue
    # Try matching by member name (e.g. backend "RUNNING" -> EnumCls.RUNNING).
    if raw is not None:
        name = str(raw).upper()
        member = enum_cls.__members__.get(name)
        if member is not None:
            return member
    logger.warning(
        "Unrecognized %s value %r from backend; using neutral fallback %r",
        getattr(enum_cls, "__name__", "status"),
        raw,
        fallback,
    )
    return fallback


class BaseTraigentClient(ABC):
    """Base interface for all Traigent clients.

    This provides a standardized interface for session-based optimization
    across different execution modes (local, cloud, hybrid).
    """

    @abstractmethod
    async def create_session(
        self,
        function_name: str,
        configuration_space: dict[str, Any],
        objectives: list[str],
        max_trials: int = 50,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Create optimization session.

        Args:
            function_name: Name of function being optimized
            configuration_space: Parameter search space
            objectives: Optimization objectives
            max_trials: Maximum number of trials
            metadata: Additional session metadata

        Returns:
            Session ID
        """
        raise NotImplementedError

    @abstractmethod
    async def get_next_trial(
        self, session_id: str, previous_results: list[Any] | None = None
    ) -> Any | None:
        """Get next trial suggestion.

        Args:
            session_id: Session ID
            previous_results: Previous trial results

        Returns:
            Next trial suggestion or None if optimization complete

        Note: This method can be implemented differently by each client.
        Some may return raw suggestion objects, others may return responses.
        """
        raise NotImplementedError

    @abstractmethod
    async def submit_trial_result(
        self,
        session_id: str,
        trial_id: str,
        metrics: dict[str, float],
        duration: float,
        status: str = "completed",
        outputs_sample: list[Any] | None = None,
        error_message: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Submit trial results.

        Args:
            session_id: Session ID
            trial_id: Trial ID
            metrics: Evaluation metrics
            duration: Execution duration
            status: Trial status (completed, failed, skipped)
            outputs_sample: Optional sample of outputs
            error_message: Error message if failed
            metadata: Additional metadata

        Note: This method can be implemented differently by each client.
        Implementations may choose to ignore optional fields.
        """
        raise NotImplementedError

    @abstractmethod
    async def finalize_session(
        self, session_id: str, include_full_history: bool = False
    ) -> dict[str, Any]:
        """Finalize session and get results.

        Args:
            session_id: Session ID
            include_full_history: Include full trial history

        Returns:
            Final optimization results
        """
        raise NotImplementedError

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        return False


class StandardizedClientError(Exception):
    """Standardized error class for all Traigent clients."""

    def __init__(
        self,
        message: str,
        error_type: str = "unknown",
        original_error: Exception | None = None,
    ) -> None:
        super().__init__(message)
        self.error_type = error_type
        self.original_error = original_error
        self.timestamp = time.time()


@dataclass
class CloudOptimizationResult:
    """Result shape for future remote cloud optimization."""

    best_config: dict[str, Any]
    best_metrics: dict[str, float]
    trials_count: int
    cost_reduction: float
    optimization_time: float
    subset_used: bool
    subset_size: int | None = None


@dataclass(frozen=True)
class RecommendationBundle:
    """TVAR recommendation hints returned by ``GET /optimization/recommendations``."""

    value_recommendations: tuple[dict[str, Any], ...] = ()
    correlations: tuple[dict[str, Any], ...] = ()
    generated_at: str | None = None

    @classmethod
    def empty(cls) -> RecommendationBundle:
        """Return an empty, offline-safe recommendation bundle."""

        return cls()

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> RecommendationBundle:
        """Parse the backend response shape into immutable tuple fields."""

        return cls(
            value_recommendations=_dict_tuple(payload.get("value_recommendations")),
            correlations=_dict_tuple(payload.get("correlations")),
            generated_at=(
                str(payload["generated_at"])
                if payload.get("generated_at") is not None
                else None
            ),
        )

    @property
    def is_empty(self) -> bool:
        """Whether the bundle carries no recommendation hints."""

        return not self.value_recommendations and not self.correlations


def _dict_tuple(value: Any) -> tuple[dict[str, Any], ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return ()
    return tuple(dict(item) for item in value if isinstance(item, Mapping))


def _tvar_recommendations_params(
    agent_type: str | None,
    metric: str | None,
    tvar_names: Sequence[str] | None,
) -> dict[str, str]:
    params: dict[str, str] = {}
    if agent_type:
        params["agent_type"] = str(agent_type)
    if metric:
        params["metric"] = str(metric)
    names = [str(name).strip() for name in tvar_names or () if str(name).strip()]
    if names:
        params["tvar_names"] = ",".join(names)
    return params


class _DirectResponseContext:
    """Wrapper for direct response objects to support async context manager protocol."""

    def __init__(self, response: Any) -> None:
        self._response = response

    async def __aenter__(self) -> Any:
        return self._response

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        return False


def _get_status_code(response: Any) -> int:
    """Extract status code from response object."""
    raw_status = getattr(response, "status", 200)
    if isinstance(raw_status, (int, float)):
        return int(raw_status)
    return 200


async def _get_json_response(response: Any) -> dict[str, Any] | None:
    """Extract JSON from response if available."""
    json_method = getattr(response, "json", None)
    if callable(json_method):
        result = json_method()
        return cast(
            dict[str, Any],
            await result if asyncio.iscoroutine(result) else result,
        )
    return None


async def _get_error_text(response: Any) -> str:
    """Extract error text from response."""
    text_method = getattr(response, "text", None)
    if callable(text_method):
        result = text_method()
        if asyncio.iscoroutine(result):
            text: str = await result
            return text
        return str(result)
    return ""


def _get_retry_delay(response: Any) -> float:
    """Get retry delay from Retry-After header."""
    headers = getattr(response, "headers", {}) or {}
    if not isinstance(headers, dict):
        return 0.0
    retry_after = headers.get("Retry-After")
    if retry_after is not None:
        try:
            return float(retry_after)
        except (TypeError, ValueError):
            return 0.0
    return 0.0


class TraigentCloudClient(BaseTraigentClient):
    """Client for backend integration APIs and reserved cloud APIs.

    Portal-tracked SDK runs should use hybrid mode. The SDK
    ``execution_mode="cloud"`` product path fails closed until remote cloud
    execution is implemented.
    """

    _AUTH_FAILURE_MESSAGE = "Not authenticated with Traigent Cloud Service"

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        enable_fallback: bool = True,
        max_retries: int = 3,
        timeout: float = 30.0,
        no_egress: bool = False,
    ) -> None:
        """Initialize cloud client.

        Args:
            api_key: Traigent Cloud API key
            base_url: Cloud service base URL
            enable_fallback: Reserved compatibility setting for future cloud behavior
            max_retries: Maximum retry attempts for cloud requests
            timeout: Request timeout in seconds
            no_egress: Runtime policy flag that forbids backend transport
        """
        if not AIOHTTP_AVAILABLE:
            logger.warning(
                "aiohttp not available, cloud client will use fallback mode only"
            )

        if api_key is not None:
            validate_or_raise(CoreValidators.validate_type(api_key, str, "api_key"))
            if not api_key.strip():
                raise ValidationException("api_key must be a non-empty string")

        if base_url:
            validate_or_raise(CoreValidators.validate_type(base_url, str, "base_url"))
            if not base_url.strip():
                raise ValidationException("base_url must be a non-empty string")

            origin, path = BackendConfig.split_api_url(base_url)
            if not origin:
                origin = BackendConfig.normalize_backend_origin(base_url)
                if not origin:
                    raise ValidationException("Invalid base_url provided")

            parsed_origin = urlparse(origin)
            if parsed_origin.hostname is None:
                raise ValidationException("Invalid base_url provided")

            resolved_origin = origin
            api_base_candidate = (
                f"{origin}{path or BackendConfig.get_default_api_path()}"
            )
        else:
            resolved_origin = BackendConfig.get_cloud_backend_url()
            api_base_candidate = BackendConfig.get_cloud_api_url()

        self.base_url = validate_cloud_base_url(
            resolved_origin.rstrip("/"), purpose="cloud client"
        )
        _api_origin, api_path = BackendConfig.split_api_url(api_base_candidate)
        self.api_base_url = (
            f"{self.base_url}{api_path or BackendConfig.get_default_api_path()}"
        ).rstrip("/")
        self.enable_fallback = enable_fallback
        self.max_retries = max_retries
        self.timeout = timeout
        self.no_egress = bool(no_egress)

        # Initialize components
        self.auth = AuthManager(api_key=api_key, no_egress=self.no_egress)
        self.auth_manager = self.auth
        self.usage_tracker = UsageTracker()
        self.subset_selector = SmartSubsetSelector()

        # Configuration for agent optimization
        self.config: dict[str, Any] = {"user_id": None, "billing_tier": "standard"}

        # Session management
        self._aio_session: aiohttp.ClientSession | None = None
        self._session_lock: asyncio.Lock = asyncio.Lock()
        self._session_finalizer: weakref.finalize | None = None
        # Track session ownership fingerprints for clearer error reporting
        self._session_owners: dict[str, dict[str, Any]] = {}
        self._cost_budget_armed_sessions: set[str] = set()

    @property
    def _session(self) -> aiohttp.ClientSession | None:
        """Expose the underlying HTTP session for tests."""

        return self._aio_session

    @_session.setter
    def _session(self, value: aiohttp.ClientSession | None) -> None:
        self._aio_session = value

    async def __aenter__(self):
        """Async context manager entry."""
        if cloud_backend_egress_disabled(self.no_egress):
            return self
        if not AIOHTTP_AVAILABLE:
            logger.warning("aiohttp not available, will use fallback optimization")
            return self

        self._aio_session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.timeout),
            # Session-default headers are long-lived: never freeze a
            # traceparent/tracestate into them (it would stamp every later
            # request with a stale span context). Trace context travels only
            # on per-request headers (see _get_headers / _inject_trace_context).
            headers=_strip_trace_context_headers(await self.auth.get_headers()),
            trust_env=True,
        )
        self._register_session_finalizer(self._aio_session)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> bool:
        """Async context manager exit."""
        await self.close(_reason="context-exit")
        return False

    async def close(self, *, _reason: str = "shutdown") -> None:
        """Close and discard the shared HTTP session."""
        await self._close_http_session(reason=_reason)

    async def _ensure_session(self):
        """Ensure session exists with current auth headers.

        This method creates a session if it doesn't exist, ensuring
        authentication headers are always included.
        """
        self._raise_if_backend_egress_disabled("backend request")
        existing = self._aio_session
        if existing is not None and not _session_is_closed(existing):
            return existing

        async with self._session_lock:
            existing = self._aio_session
            if existing is not None and not _session_is_closed(existing):
                return existing

            try:
                headers = await self.auth.get_headers()
            except AuthenticationError as exc:
                raise AuthenticationError(self._AUTH_FAILURE_MESSAGE) from exc
            except Exception as exc:
                raise AuthenticationError(
                    f"{self._AUTH_FAILURE_MESSAGE}: {exc}"
                ) from exc

            if "Authorization" not in headers and "X-API-Key" not in headers:
                raise AuthenticationError(self._AUTH_FAILURE_MESSAGE)

            if not AIOHTTP_AVAILABLE:
                raise CloudServiceError(
                    "Client session not initialized and aiohttp not available"
                )

            self._aio_session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.timeout),
                # Same rule as __aenter__: no stale trace context in
                # long-lived session-default headers.
                headers=_strip_trace_context_headers(headers),
                trust_env=True,
            )
            self._register_session_finalizer(self._aio_session)
            return self._aio_session

    async def _get_headers(self) -> dict[str, str]:
        """Get headers for HTTP requests."""
        headers = await self.auth.get_headers()
        if "Authorization" not in headers and "X-API-Key" not in headers:
            raise AuthenticationError(self._AUTH_FAILURE_MESSAGE)
        return cast(dict[str, str], headers)

    def _raise_if_backend_egress_disabled(self, operation: str) -> None:
        """Fail closed before any backend HTTP request."""

        raise_if_cloud_egress_disabled(operation, no_egress=self.no_egress)

    async def _reset_http_session(self, reason: str | None = None) -> None:
        """Close and discard the shared aiohttp session after failures."""
        await self._close_http_session(reason=reason)

    def _register_session_finalizer(self, session: aiohttp.ClientSession) -> None:
        self._detach_session_finalizer()
        if _is_real_aiohttp_session(session):
            self._session_finalizer = weakref.finalize(
                self,
                _finalize_aiohttp_session,
                session,
                self.__class__.__name__,
            )

    def _detach_session_finalizer(self) -> None:
        finalizer = getattr(self, "_session_finalizer", None)
        if finalizer is not None:
            finalizer.detach()
            self._session_finalizer = None

    async def _close_http_session(self, reason: str | None = None) -> None:
        """Best-effort close for the shared HTTP session."""
        if not self._aio_session:
            return

        session = self._aio_session
        if session is None:
            return
        self._aio_session = None
        self._detach_session_finalizer()

        close_method = getattr(session, "close", None)
        if not callable(close_method):
            return

        try:
            if not session.__class__.__module__.startswith(
                "unittest.mock"
            ) and _session_is_closed(session):
                return

            close_result = close_method()
            if inspect.isawaitable(close_result):
                await close_result
            # On shutdown/context-exit paths, give the event loop time to run
            # the connector's cleanup callbacks (especially for HTTPS/SSL).
            # Without this, asyncio.run() may tear down the loop before the
            # underlying TCP transport is fully closed, producing the
            # "Unclosed client session" ResourceWarning.
            # Skip the delay on retry/error paths to avoid adding 250ms
            # on every transient reset.
            if reason in ("shutdown", "context-exit"):
                await asyncio.sleep(0.25)
        except Exception as exc:  # pragma: no cover - defensive cleanup
            logger.debug(
                "Error closing cloud session%s: %s",
                f" ({reason})" if reason else "",
                exc,
            )

    def _safe_owner_fingerprint(self) -> dict[str, Any]:
        """Best-effort fingerprint fetch that tolerates async mocks and errors.

        Returns an empty dict when the auth backend does not expose synchronous
        fingerprint metadata. Example structure when available::

            {
                "owner_user_id": "user-123",
                "owner_api_key_id": "key-456",
                "created_by": "user-123",
                "owner_scope": ["optimize"],
                "credential_source": "environment",
            }
        """

        get_fingerprint = getattr(self.auth, "get_owner_fingerprint", None)
        if not callable(get_fingerprint):
            return {}

        # Avoid awaiting AsyncMock or coroutine functions in sync context
        if inspect.iscoroutinefunction(get_fingerprint):
            logger.debug(
                "Owner fingerprint callable is async; skipping lookup in sync context"
            )
            return {}

        try:
            fingerprint = get_fingerprint()
        except Exception as exc:  # pragma: no cover - defensive safety
            logger.debug("Owner fingerprint retrieval failed: %s", exc)
            return {}

        if inspect.isawaitable(fingerprint):
            logger.debug("Owner fingerprint returned coroutine; skipping")
            return {}
        if isinstance(fingerprint, dict):
            return fingerprint

        logger.debug(
            "Owner fingerprint returned unsupported type %s; ignoring",
            type(fingerprint).__name__,
        )
        return {}

    def _ensure_owner_metadata(self, metadata: dict[str, Any] | None) -> dict[str, Any]:
        """Attach ownership fingerprint so backend can enforce access control."""

        if isinstance(metadata, Mapping):
            sanitized = dict(metadata)
        else:
            sanitized = {}
        fingerprint = self._safe_owner_fingerprint()

        for key in ("owner_user_id", "owner_api_key_id", "created_by"):
            value = fingerprint.get(key)
            if value and not sanitized.get(key):
                sanitized[key] = value

        return sanitized

    def _remember_session_owner(
        self,
        session_id: str,
        request_metadata: dict[str, Any] | None,
        response_metadata: dict[str, Any] | None,
    ) -> None:
        """Cache ownership metadata for diagnostics and remediation hints."""

        owner_info: dict[str, Any] = {}
        for source in (response_metadata or {}, request_metadata or {}):
            if not source:
                continue
            for key in (
                "owner_user_id",
                "owner_api_key_id",
                "created_by",
                "owner_scope",
                "credential_source",
                "owner_api_key_preview",
            ):
                value = source.get(key)
                if value and key not in owner_info:
                    owner_info[key] = value

        if not owner_info:
            fingerprint = self._safe_owner_fingerprint()
            for key, value in fingerprint.items():
                if value and key not in owner_info:
                    owner_info[key] = value

        if owner_info:
            self._session_owners[session_id] = owner_info

    def _remember_cost_budget_armed_session(
        self, session_id: str, budget: dict[str, Any] | None
    ) -> None:
        """Track sessions whose typed create armed a positive cost budget."""

        remember_cost_budget_armed_session(self, session_id, budget)

    def _is_cost_budget_armed_session(self, session_id: str) -> bool:
        """Return whether this client created the session with a positive budget."""

        return is_cost_budget_armed_session(self, session_id)

    @staticmethod
    def _summarize_actor(info: dict[str, Any] | None) -> str:
        """Create a human-readable summary of a session owner or caller."""

        if not info:
            return "unknown principal"

        parts: list[str] = []
        if info.get("owner_user_id"):
            parts.append(f"user '{info['owner_user_id']}'")
        if info.get("owner_api_key_id"):
            parts.append(f"api-key '{info['owner_api_key_id']}'")
        if info.get("created_by") and info.get("created_by") not in (
            info.get("owner_user_id"),
            None,
        ):
            parts.append(f"created_by '{info['created_by']}'")
        if info.get("owner_api_key_preview"):
            parts.append(f"token {info['owner_api_key_preview']}")
        if info.get("credential_source"):
            parts.append(f"source={info['credential_source']}")

        return ", ".join(parts) if parts else "unknown principal"

    def _raise_ownership_error(
        self,
        session_id: str,
        action: str,
        status: int,
        error_text: str,
    ) -> NoReturn:
        """Raise a CloudServiceError with ownership remediation guidance."""

        owner_info = self._session_owners.get(session_id)
        current_actor = self._safe_owner_fingerprint()

        summary_owner = self._summarize_actor(owner_info)
        summary_current = self._summarize_actor(current_actor)

        excerpt = self._first_error_line(error_text)

        message = (
            f"{action} failed for session '{session_id}': HTTP {status} Forbidden. "
            "Traigent Cloud now enforces session ownership. "
            f"Session owner: {summary_owner}. Calling credentials: {summary_current}. "
            "Re-authenticate with the owning token or an admin-scoped key, or recreate the session "
            "under the currently active credentials."
        )

        if excerpt:
            message = f"{message} Backend response: {excerpt}"

        raise CloudServiceError(message)

    @staticmethod
    def _first_error_line(error_text: str | None) -> str:
        """Return a trimmed first line for diagnostics without raising errors."""

        if not error_text:
            return ""

        lines = error_text.strip().splitlines()
        if not lines:
            return ""

        excerpt = lines[0].strip()
        if len(excerpt) > 200:
            return f"{excerpt[:197]}..."
        return excerpt

    async def optimize_function(
        self,
        function_name: str,
        dataset: Dataset,
        configuration_space: dict[str, Any],
        objectives: list[str],
        max_trials: int = 50,
        target_cost_reduction: float = 0.65,
        *,
        local_function: Callable[..., Any] | None = None,
    ) -> CloudOptimizationResult:
        """Fail closed for remote cloud optimization.

        Remote cloud execution is reserved for a future implementation. The
        working portal-visible path today is hybrid mode: trials execute
        locally and metrics are submitted to the backend session API.
        """
        _ = (
            function_name,
            dataset,
            configuration_space,
            objectives,
            max_trials,
            target_cost_reduction,
            local_function,
        )
        raise CloudRemoteExecutionUnavailableError()

    async def _submit_optimization(
        self, request_data: dict[str, Any]
    ) -> dict[str, Any]:
        """Submit optimization request to the reserved remote service."""
        self._raise_if_backend_egress_disabled("submit optimization")
        await self._ensure_session()
        if self._aio_session is None:
            raise CloudServiceError(_SESSION_NOT_INITIALIZED)

        url = f"{self.api_base_url}/optimize"
        session = self._aio_session  # Capture for use in nested function

        async def submit_request():
            """Internal function to submit request with proper error handling."""
            try:
                async with session.post(
                    url, json=request_data, headers=await self._get_headers()
                ) as response:
                    if response.status == 200:
                        return await response.json()
                    elif response.status == 429:
                        # Rate limited - convert to retryable error
                        retry_after = response.headers.get("Retry-After")
                        raise RateLimitError(
                            "Rate limited by backend service",
                            retry_after=int(retry_after) if retry_after else None,
                        )
                    else:
                        error_text = await response.text()
                        raise CloudServiceError(f"HTTP {response.status}: {error_text}")

            except aiohttp.ClientError as e:
                # Convert to retryable network error
                raise NetworkError(
                    f"Network error during optimization request: {e}"
                ) from None

        # Apply retry decorator to the submit request function
        @retry_http_request
        async def submit_with_retry():
            return await submit_request()

        # Use retry handler for robust HTTP requests
        try:
            result = await submit_with_retry()
            return cast(dict[str, Any], result)
        except Exception as e:
            if hasattr(e, "last_exception") and isinstance(
                e.last_exception, CloudServiceError
            ):
                # Re-raise CloudServiceError directly (non-retryable errors)
                raise e.last_exception from None
            raise CloudServiceError(
                f"Optimization request failed after retries: {e}"
            ) from e

    async def _fallback_optimization(
        self,
        function_name: str,
        dataset: Dataset,
        configuration_space: dict[str, Any],
        objectives: list[str],
        max_trials: int,
        *,
        local_function: Callable[..., Any] | None = None,
    ) -> CloudOptimizationResult:
        """Compatibility local fallback used by older remote-service flows."""
        from traigent.core.orchestrator import OptimizationOrchestrator
        from traigent.evaluators.local import LocalEvaluator
        from traigent.optimizers.registry import get_optimizer

        def _as_dict(value: Any) -> dict[str, Any]:
            if isinstance(value, Mapping):
                return {str(k): v for k, v in value.items()}
            return {}

        def _as_metrics(value: Any) -> dict[str, float]:
            if not isinstance(value, Mapping):
                return {}
            metrics: dict[str, float] = {}
            for key, raw in value.items():
                try:
                    metrics[str(key)] = float(raw)
                except (TypeError, ValueError):
                    continue
            return metrics

        def _resolve_local_function() -> Callable[..., Any]:
            if callable(local_function):
                return local_function

            module_name, separator, attr_path = function_name.rpartition(".")
            if not separator:
                raise CloudServiceError(
                    "Local fallback requires local_function when function_name "
                    "is not an importable dotted path"
                )

            try:
                resolved: Any = import_module(module_name)
                for attr in attr_path.split("."):
                    resolved = getattr(resolved, attr)
            except (ImportError, AttributeError) as exc:
                raise CloudServiceError(
                    f"Unable to resolve local fallback function '{function_name}'"
                ) from exc

            if not callable(resolved):
                raise CloudServiceError(
                    f"Resolved fallback target '{function_name}' is not callable"
                )

            return cast(Callable[..., Any], resolved)

        # Use random optimizer for fallback (faster than grid).
        fallback_max_trials = max(1, min(max_trials, 20))
        fallback_function = _resolve_local_function()
        optimizer = get_optimizer(
            "random",
            configuration_space,
            objectives,
            max_trials=fallback_max_trials,
        )
        evaluator = LocalEvaluator(metrics=objectives)

        # Run local optimization
        start_time = time.time()
        orchestrator = OptimizationOrchestrator(
            optimizer=optimizer,
            evaluator=evaluator,
            max_trials=fallback_max_trials,
            objectives=objectives,
        )
        optimization_result = await orchestrator.optimize(
            func=fallback_function,
            dataset=dataset,
            function_name=function_name,
        )
        best_config = _as_dict(getattr(optimization_result, "best_config", {}))
        best_metrics = _as_metrics(getattr(optimization_result, "best_metrics", {}))
        if not best_metrics and objectives:
            best_score_raw = getattr(optimization_result, "best_score", None)
            if best_score_raw is None:
                best_metrics = {}
            else:
                try:
                    best_metrics = {objectives[0]: float(best_score_raw)}
                except (TypeError, ValueError):
                    best_metrics = {}
        trials = optimization_result.trials
        trials_count = len(trials)

        return CloudOptimizationResult(
            best_config=best_config,
            best_metrics=best_metrics,
            trials_count=trials_count,
            cost_reduction=0.0,  # No cost reduction in Edge Analytics mode
            optimization_time=time.time() - start_time,
            subset_used=False,
        )

    def _serialize_dataset(self, dataset: Dataset) -> dict[str, Any]:
        """Serialize dataset for cloud transmission."""
        return {
            "name": dataset.name,
            "examples": [
                {
                    "input_data": example.input_data,
                    "expected_output": example.expected_output,
                    "metadata": example.metadata,
                }
                for example in dataset.examples
            ],
        }

    async def get_usage_stats(self) -> dict[str, Any]:
        """Get usage statistics for current billing period."""
        return cast(dict[str, Any], await self.usage_tracker.get_usage_stats())

    async def fetch_tvar_recommendations(
        self,
        agent_type: str | None = None,
        metric: str | None = None,
        tvar_names: Sequence[str] | None = None,
    ) -> RecommendationBundle:
        """Fetch TVAR recommendation hints without blocking local recommendations.

        The recommendation system treats backend responses as optional hints. Any
        offline mode, missing credentials, transport issue, auth failure, or
        malformed response therefore returns an empty bundle instead of raising.
        """

        if self._should_skip_tvar_recommendations_fetch():
            return RecommendationBundle.empty()

        try:
            await self._ensure_session()
            if self._aio_session is None:
                return RecommendationBundle.empty()

            url = f"{self.api_base_url}/optimization/recommendations"
            params = _tvar_recommendations_params(agent_type, metric, tvar_names)
            request = self._aio_session.get(
                url,
                params=params,
                headers=await self._get_headers(),
            )
            response_candidate = (
                await request if asyncio.iscoroutine(request) else request
            )
            manager = (
                response_candidate
                if hasattr(response_candidate, "__aenter__")
                else _DirectResponseContext(response_candidate)
            )

            async with manager as response:
                if _get_status_code(response) != 200:
                    return RecommendationBundle.empty()
                payload = await _get_json_response(response)
                if not isinstance(payload, Mapping):
                    return RecommendationBundle.empty()
                return RecommendationBundle.from_payload(payload)
        except aiohttp.ClientError as exc:
            await self._reset_http_session("tvar_recommendations network error")
            logger.debug("TVAR recommendations fetch failed: %s", exc)
            return RecommendationBundle.empty()
        except Exception as exc:
            logger.debug("TVAR recommendations fetch skipped after error: %s", exc)
            return RecommendationBundle.empty()

    def _should_skip_tvar_recommendations_fetch(self) -> bool:
        """Return True when recommendation fetches must stay local/offline."""

        if cloud_backend_egress_disabled(self.no_egress):
            return True

        edge_env_values = {
            os.getenv("TRAIGENT_EDGE_ANALYTICS_MODE", "").strip().lower(),
            os.getenv("TRAIGENT_EXECUTION_MODE", "").strip().lower(),
        }
        if edge_env_values & {"true", "1", "yes", "local", "edge_analytics"}:
            return True

        try:
            return not (self.auth.has_api_key() or BackendConfig.has_auth_credentials())
        except Exception:
            return not self.auth.has_api_key()

    async def check_service_status(self) -> dict[str, Any]:
        """Check Traigent backend service status."""
        self._raise_if_backend_egress_disabled("check service status")
        await self._ensure_session()
        if self._aio_session is None:
            raise CloudServiceError(_SESSION_NOT_INITIALIZED)

        url = f"{self.base_url}/health"
        attempts = max(1, self.max_retries)
        last_error: Exception | None = None

        for attempt in range(1, attempts + 1):
            try:
                result = await self._check_service_status_attempt(
                    url, attempt, attempts
                )
                if result is not None:
                    return result
                # result is None means retry (continue loop)
            except CloudEgressBlockedError:
                raise
            except CloudServiceError as exc:
                last_error = exc
                if attempt == attempts:
                    return {"status": "unavailable", "error": str(exc)}
            except Exception as exc:
                last_error = exc
                if attempt == attempts:
                    return {"status": "unavailable", "error": str(exc)}

        return {
            "status": "unavailable",
            "error": str(last_error) if last_error else "Unknown error",
        }

    async def _check_service_status_attempt(
        self, url: str, attempt: int, attempts: int
    ) -> dict[str, Any] | None:
        """Execute a single service status check attempt.

        Returns:
            dict with status on success/final failure, or None to signal retry.
        """
        self._raise_if_backend_egress_disabled("check service status")
        request = self._aio_session.get(url, headers=await self._get_headers())  # type: ignore[union-attr]
        response_candidate = await request if asyncio.iscoroutine(request) else request

        if hasattr(response_candidate, "__aenter__"):
            manager = response_candidate
        else:
            manager = _DirectResponseContext(response_candidate)

        async with manager as response:
            status_code = _get_status_code(response)

            if status_code == 200:
                json_result = await _get_json_response(response)
                return json_result if json_result is not None else {"status": "ok"}

            error_text = await _get_error_text(response)

            if status_code in {429, 503} and attempt < attempts:
                delay = _get_retry_delay(response)
                if delay > 0:
                    await asyncio.sleep(min(delay, 1.0))
                return None  # Signal retry

            raise CloudServiceError(
                f"Service status check failed: HTTP {status_code} {error_text}"
            )

    # Standard interface implementation (BaseTraigentClient)
    # Note: The existing methods already provide the interface functionality

    async def create_session(
        self,
        function_name: str,
        configuration_space: dict[str, Any],
        objectives: list[str],
        max_trials: int = 50,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Standard interface: Create optimization session.

        This implements the BaseTraigentClient interface for compatibility.
        """
        try:
            response = await self.create_optimization_session(
                function_name,
                configuration_space=configuration_space,
                objectives=objectives,
                dataset_metadata=metadata or {},
                max_trials=max_trials,
            )
            return cast(str, response.session_id)
        except Exception as e:
            raise StandardizedClientError(
                f"Failed to create session: {e}", "session_creation", e
            ) from e

    async def finalize_session(
        self, session_id: str, include_full_history: bool = False
    ) -> dict[str, Any]:
        """Standard interface: Finalize session and get results."""
        try:
            response = await self.finalize_optimization(
                session_id, include_full_history
            )
            return {
                "session_id": response.session_id,
                "best_config": response.best_config,
                "best_metrics": response.best_metrics,
                "total_trials": response.total_trials,
                "successful_trials": response.successful_trials,
                "total_duration": response.total_duration,
                "cost_savings": response.cost_savings,
            }
        except Exception as e:
            raise StandardizedClientError(
                f"Failed to finalize session: {e}", "session_finalization", e
            ) from e

    async def delete_session(self, session_id: str, cascade: bool = False) -> bool:
        """Delete a session via ``DELETE /api/v1/sessions/<id>``.

        Non-destructive by default (``cascade=False``), matching the backend
        default: only the session and its run artifacts are removed. Pass
        ``cascade=True`` to additionally hard-delete the parent experiment when
        the deleted session was its last run. Use this with
        :meth:`list_sessions` to enumerate and clean up orphaned / zombie
        "running" sessions from the client.
        """
        if not session_id:
            raise ValueError("session_id is required")

        self._raise_if_backend_egress_disabled("delete session")
        await self._ensure_session()
        if self._aio_session is None:
            raise CloudServiceError(_SESSION_NOT_INITIALIZED)

        url = f"{self.api_base_url}/sessions/{session_id}"
        params = {"cascade": "true" if cascade else "false"}

        try:
            async with self._aio_session.delete(url, params=params) as response:
                if response.status in {200, 202, 204}:
                    self._session_owners.pop(session_id, None)
                    return True
                if response.status == 404:
                    logger.info(
                        "Session %s already absent when attempting cleanup", session_id
                    )
                    return False

                error_text = await response.text()
                logger.warning(
                    "Failed to delete session %s: status=%s body=%s",
                    session_id,
                    response.status,
                    error_text[:200],
                )
                return False

        except aiohttp.ClientError as exc:
            await self._reset_http_session("delete_session network error")
            raise StandardizedClientError(
                f"Failed to delete session: {exc}", "session_delete", exc
            ) from exc

    async def list_sessions(
        self,
        status: str | None = None,
        pattern: str | None = None,
        limit: int | None = None,
    ) -> list[SessionSummary]:
        """List sessions via ``GET /api/v1/sessions``.

        Enumerates the caller's optimization sessions so orphaned / zombie
        "running" sessions can be found and cleaned up with
        :meth:`delete_session`. Returns typed :class:`SessionSummary` objects.

        Args:
            status: Optional lifecycle filter (server-side ``status`` query param).
            pattern: Optional session-id match (server-side ``pattern`` query
                param).
            limit: Optional cap on the number of summaries returned. The backend
                does not support a server-side limit, so this is applied
                client-side after the response is received.

        Returns:
            A list of :class:`SessionSummary` (empty when there are none).
        """
        self._raise_if_backend_egress_disabled("list sessions")
        await self._ensure_session()
        if self._aio_session is None:
            raise CloudServiceError(_SESSION_NOT_INITIALIZED)

        params: dict[str, str] = {}
        if status is not None:
            params["status"] = str(status)
        if pattern is not None:
            params["pattern"] = str(pattern)

        url = f"{self.api_base_url}/sessions"
        try:
            async with self._aio_session.get(url, params=params) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise CloudServiceError(
                        f"Failed to list sessions: status={response.status} "
                        f"body={error_text[:200]}"
                    )
                data = await response.json()
        except aiohttp.ClientError as exc:
            await self._reset_http_session("list_sessions network error")
            raise StandardizedClientError(
                f"Failed to list sessions: {exc}", "session_list", exc
            ) from exc

        raw_sessions = data.get("sessions") if isinstance(data, dict) else None
        if not isinstance(raw_sessions, list):
            raw_sessions = []
        summaries = [
            SessionSummary.from_dict(item)
            for item in raw_sessions
            if isinstance(item, dict)
        ]
        if limit is not None and limit >= 0:
            summaries = summaries[:limit]
        return summaries

    # Stateful optimization methods for interactive model

    async def create_optimization_session(
        self,
        request_or_function_name,
        configuration_space: dict[str, Any] | None = None,
        objectives: (
            Sequence[str | SessionObjectiveDefinition | dict[str, Any]] | None
        ) = None,
        dataset_metadata: dict[str, Any] | None = None,
        max_trials: int = 50,
        budget: dict[str, Any] | None = None,
        constraints: dict[str, Any] | None = None,
        default_config: dict[str, Any] | None = None,
        promotion_policy: dict[str, Any] | None = None,
        optimization_strategy: dict[str, Any] | None = None,
        user_id: str | None = None,
        billing_tier: str = "standard",
        artifact_fingerprints: dict[str, str | None] | None = None,
        fingerprint_meta: dict[str, Any] | None = None,
        evaluator_id: str | None = None,
        evaluator_definition_id: str | None = None,
    ) -> SessionCreationResponse:
        """Create a new optimization session for interactive optimization.

        Args:
            request_or_function_name: Either a SessionCreationRequest object or function name string
            configuration_space: Parameter search space (ignored if first arg is SessionCreationRequest)
            objectives: Optimization objectives (ignored if first arg is SessionCreationRequest)
            dataset_metadata: Metadata about the dataset (ignored if first arg is SessionCreationRequest)
            max_trials: Maximum number of trials (ignored if first arg is SessionCreationRequest)
            optimization_strategy: Optional optimization strategy (ignored if first arg is SessionCreationRequest)
            user_id: Optional user identifier (ignored if first arg is SessionCreationRequest)
            billing_tier: User's billing tier (ignored if first arg is SessionCreationRequest)

        Returns:
            SessionCreationResponse with session details

        Raises:
            CloudServiceError: If session creation fails
        """
        self._raise_if_backend_egress_disabled("create session")
        await self._ensure_session()

        # Handle both calling patterns: with SessionCreationRequest object or separate params
        if isinstance(request_or_function_name, SessionCreationRequest):
            # It's a SessionCreationRequest object
            request = request_or_function_name
        else:
            # It's the old signature with separate parameters
            request = SessionCreationRequest(
                function_name=request_or_function_name,
                configuration_space=configuration_space,
                objectives=objectives,
                dataset_metadata=dataset_metadata,
                max_trials=max_trials,
                budget=budget,
                constraints=constraints,
                default_config=default_config,
                promotion_policy=promotion_policy,
                optimization_strategy=optimization_strategy,
                user_id=user_id,
                billing_tier=billing_tier,
                artifact_fingerprints=artifact_fingerprints,
                fingerprint_meta=fingerprint_meta,
                evaluator_id=evaluator_id,
                evaluator_definition_id=evaluator_definition_id,
            )

        if self._aio_session is None:
            raise CloudServiceError(_CLIENT_SESSION_NOT_INITIALIZED)
        try:
            url = f"{self.api_base_url}/sessions"
            request_data = self._serialize_session_request(request)

            async with self._aio_session.post(
                url, json=request_data, headers=await self._get_headers()
            ) as response:
                if response.status == 201:
                    data = await response.json()
                    result = self._deserialize_session_response(data)
                    self._remember_session_owner(
                        result.session_id,
                        dict(request.metadata or {}),
                        data.get("metadata", {}),
                    )
                    self._remember_cost_budget_armed_session(
                        result.session_id, request_data.get("budget")
                    )
                    return result
                else:
                    error_text = await response.text()
                    raise CloudServiceError(
                        f"Failed to create session: HTTP {response.status}: {error_text}"
                    )

        except aiohttp.ClientError as e:
            await self._reset_http_session("create_session network error")
            raise CloudServiceError(f"Network error creating session: {e}") from None

    async def get_next_trial(
        self,
        session_id: str,
        previous_results: list[TrialResultSubmission] | None = None,
    ) -> NextTrialResponse:
        """Get next trial suggestion from the backend service.

        Args:
            session_id: Optimization session ID
            previous_results: Optional list of recent trial results

        Returns:
            NextTrialResponse with trial suggestion

        Raises:
            CloudServiceError: If request fails
        """
        self._raise_if_backend_egress_disabled("get next trial")
        await self._ensure_session()

        request = NextTrialRequest(
            session_id=session_id, previous_results=previous_results
        )

        if self._aio_session is None:
            raise CloudServiceError(_CLIENT_SESSION_NOT_INITIALIZED)
        try:
            url = f"{self.api_base_url}/sessions/{session_id}/next-trial"
            request_data = self._serialize_next_trial_request(request)

            async with self._aio_session.post(
                url, json=request_data, headers=await self._get_headers()
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return self._deserialize_next_trial_response(data)
                elif response.status == 403:
                    error_text = await response.text()
                    self._raise_ownership_error(
                        session_id,
                        "Retrieving the next trial",
                        response.status,
                        error_text,
                    )
                else:
                    error_text = await response.text()
                    raise CloudServiceError(
                        f"Failed to get next trial: HTTP {response.status}: {error_text}"
                    )

        except aiohttp.ClientError as e:
            await self._reset_http_session("next_trial network error")
            raise CloudServiceError(f"Network error getting next trial: {e}") from None

    async def submit_trial_result(
        self,
        session_id: str,
        trial_id: str,
        metrics: dict[str, float],
        duration: float,
        status: str = "completed",
        outputs_sample: list[Any] | None = None,
        error_message: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Submit trial results to the backend service.

        Args:
            session_id: Optimization session ID
            trial_id: Trial identifier
            metrics: Computed metrics
            duration: Execution duration in seconds
            status: Trial status (completed, failed, skipped)
            outputs_sample: Optional sample of outputs
            error_message: Error message if failed
            metadata: Additional metadata

        Raises:
            CloudServiceError: If submission fails
        """
        self._raise_if_backend_egress_disabled("submit trial result")
        await self._ensure_session()

        from traigent.cloud.models import TrialStatus

        # Convert string status to enum, tolerating case / out-of-set values.
        # NEVER silently default to COMPLETED — that would assert success on a
        # failed/running trial (issue #1302, AC3). Fall back to UNKNOWN instead.
        trial_status = _coerce_enum(TrialStatus, status, fallback=TrialStatus.UNKNOWN)

        result = TrialResultSubmission(
            session_id=session_id,
            trial_id=trial_id,
            metrics=metrics,
            duration=duration,
            status=trial_status,
            outputs_sample=outputs_sample,
            error_message=error_message,
            metadata=metadata or {},
        )

        if self._aio_session is None:
            raise CloudServiceError(_CLIENT_SESSION_NOT_INITIALIZED)
        try:
            url = f"{self.api_base_url}/sessions/{session_id}/results"
            request_data = self._serialize_trial_result(result)
            request_data["metrics"] = dict(request_data.get("metrics") or {})
            ensure_cost_metric_for_budgeted_completed_submission(
                client=self,
                session_id=session_id,
                metrics=request_data["metrics"],
                status=request_data.get("status"),
                telemetry_sources=(result.metadata, request_data.get("metadata")),
                logger=logger,
            )

            async with self._aio_session.post(
                url, json=request_data, headers=await self._get_headers()
            ) as response:
                if response.status == 403:
                    error_text = await response.text()
                    self._raise_ownership_error(
                        session_id,
                        "Submitting trial results",
                        response.status,
                        error_text,
                    )
                elif response.status not in [200, 201, 204]:
                    error_text = await response.text()
                    raise CloudServiceError(
                        f"Failed to submit result: HTTP {response.status}: {error_text}"
                    )

        except aiohttp.ClientError as e:
            await self._reset_http_session("submit_trial_result network error")
            raise CloudServiceError(f"Network error submitting result: {e}") from None

    async def finalize_optimization(
        self, session_id: str, include_full_history: bool = False
    ) -> OptimizationFinalizationResponse:
        """Finalize an optimization session and get results.

        Args:
            session_id: Optimization session ID
            include_full_history: Whether to include full trial history

        Returns:
            OptimizationFinalizationResponse with final results

        Raises:
            CloudServiceError: If finalization fails
        """
        self._raise_if_backend_egress_disabled("finalize session")
        await self._ensure_session()

        request = OptimizationFinalizationRequest(
            session_id=session_id, include_full_history=include_full_history
        )

        if self._aio_session is None:
            raise CloudServiceError(_CLIENT_SESSION_NOT_INITIALIZED)
        try:
            url = f"{self.api_base_url}/sessions/{session_id}/finalize"
            request_data = self._serialize_finalization_request(request)

            async with self._aio_session.post(
                url, json=request_data, headers=await self._get_headers()
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    result = self._deserialize_finalization_response(data)
                    # Optimization session is complete; ownership cache can be released
                    self._session_owners.pop(session_id, None)
                    return result
                elif response.status == 403:
                    error_text = await response.text()
                    self._raise_ownership_error(
                        session_id,
                        "Finalizing the session",
                        response.status,
                        error_text,
                    )
                else:
                    error_text = await response.text()
                    raise CloudServiceError(
                        f"Failed to finalize session: HTTP {response.status}: {error_text}"
                    )

        except aiohttp.ClientError as e:
            await self._reset_http_session("finalize_session network error")
            raise CloudServiceError(f"Network error finalizing session: {e}") from None

    # Serialization/deserialization helpers

    def _serialize_session_request(
        self, request: SessionCreationRequest
    ) -> dict[str, Any]:
        """Serialize session creation request."""
        metadata = self._ensure_owner_metadata(request.metadata)
        request.metadata = metadata
        objectives_payload = normalize_typed_objectives(request.objectives)
        payload: dict[str, Any] = {
            "function_name": request.function_name,
            "configuration_space": request.configuration_space,
            "objectives": objectives_payload,
            "dataset_metadata": request.dataset_metadata,
            "max_trials": request.max_trials,
            "optimization_strategy": request.optimization_strategy,
            "user_id": request.user_id,
            "billing_tier": request.billing_tier,
            "metadata": metadata,
        }
        if request.budget is not None:
            payload["budget"] = request.budget
        if request.constraints is not None:
            payload["constraints"] = request.constraints
        if request.default_config is not None:
            payload["default_config"] = request.default_config
        if request.promotion_policy is not None:
            payload["promotion_policy"] = request.promotion_policy
        # Warm-start: prior experiment id (empty string treated as absent).
        if request.warm_start_from:
            payload["warm_start_from"] = request.warm_start_from
        smart_pruning = normalize_smart_pruning_options(request.smart_pruning)
        if smart_pruning:
            payload["smart_pruning"] = smart_pruning
        artifact_fingerprints = artifact_fingerprints_to_wire(
            getattr(request, "artifact_fingerprints", None)
        )
        if artifact_fingerprints is not None:
            payload["artifact_fingerprints"] = artifact_fingerprints
        fingerprint_meta = fingerprint_meta_to_wire(
            getattr(request, "fingerprint_meta", None)
        )
        if fingerprint_meta is not None:
            payload["fingerprint_meta"] = fingerprint_meta
        evaluator_id = getattr(request, "evaluator_id", None)
        if isinstance(evaluator_id, str) and evaluator_id.strip():
            payload["evaluator_id"] = evaluator_id.strip()
        evaluator_definition_id = getattr(request, "evaluator_definition_id", None)
        if isinstance(evaluator_definition_id, str) and evaluator_definition_id.strip():
            payload["evaluator_definition_id"] = evaluator_definition_id.strip()
        return payload

    @staticmethod
    def _serialize_session_objective(
        objective: str | SessionObjectiveDefinition | dict[str, Any],
    ) -> str | dict[str, Any]:
        """Serialize a typed objective while preserving string shorthand."""
        return session_objective_to_wire(objective)

    def _deserialize_session_response(
        self, data: dict[str, Any]
    ) -> SessionCreationResponse:
        """Deserialize session creation response."""
        return SessionCreationResponse(
            session_id=data["session_id"],
            status=_coerce_enum(
                OptimizationSessionStatus,
                data["status"],
                fallback=OptimizationSessionStatus.UNKNOWN,
            ),
            optimization_strategy=data.get("optimization_strategy", {}),
            estimated_duration=data.get("estimated_duration"),
            billing_estimate=data.get("billing_estimate"),
            metadata=data.get("metadata", {}),
        )

    def _serialize_next_trial_request(
        self, request: NextTrialRequest
    ) -> dict[str, Any]:
        """Serialize next trial request."""
        return {
            "session_id": request.session_id,
            "previous_results": [
                self._serialize_trial_result(r)
                for r in (request.previous_results or [])
            ],
            "request_metadata": request.request_metadata,
        }

    def _deserialize_next_trial_response(
        self, data: dict[str, Any]
    ) -> NextTrialResponse:
        """Deserialize next trial response."""
        from traigent.cloud.models import DatasetSubsetIndices

        suggestion = None
        if data.get("suggestion"):
            sugg_data = data["suggestion"]
            suggestion = TrialSuggestion(
                trial_id=sugg_data["trial_id"],
                session_id=sugg_data["session_id"],
                trial_number=sugg_data["trial_number"],
                config=sugg_data["config"],
                dataset_subset=DatasetSubsetIndices(
                    indices=sugg_data["dataset_subset"]["indices"],
                    selection_strategy=sugg_data["dataset_subset"][
                        "selection_strategy"
                    ],
                    confidence_level=sugg_data["dataset_subset"]["confidence_level"],
                    estimated_representativeness=sugg_data["dataset_subset"][
                        "estimated_representativeness"
                    ],
                    metadata=sugg_data["dataset_subset"].get("metadata", {}),
                ),
                exploration_type=sugg_data["exploration_type"],
                priority=sugg_data.get("priority", 1),
                estimated_duration=sugg_data.get("estimated_duration"),
                metadata=sugg_data.get("metadata", {}),
            )

        return NextTrialResponse(
            suggestion=suggestion,
            should_continue=data["should_continue"],
            reason=data.get("reason"),
            stop_reason=data.get("stop_reason"),
            session_status=_coerce_enum(
                OptimizationSessionStatus,
                data.get("session_status", "active"),
                fallback=OptimizationSessionStatus.UNKNOWN,
            ),
            metadata=data.get("metadata", {}),
        )

    def _serialize_trial_result(self, result: TrialResultSubmission) -> dict[str, Any]:
        """Serialize trial result submission."""
        return {
            "session_id": result.session_id,
            "trial_id": result.trial_id,
            "metrics": result.metrics,
            "duration": result.duration,
            "status": result.status.value,
            "outputs_sample": result.outputs_sample,
            "error_message": result.error_message,
            "metadata": result.metadata,
        }

    def _serialize_finalization_request(
        self, request: OptimizationFinalizationRequest
    ) -> dict[str, Any]:
        """Serialize finalization request."""
        return {
            "session_id": request.session_id,
            "include_full_history": request.include_full_history,
            "metadata": request.metadata,
        }

    def _deserialize_finalization_response(
        self, data: dict[str, Any]
    ) -> OptimizationFinalizationResponse:
        """Deserialize finalization response."""
        return OptimizationFinalizationResponse(
            session_id=data["session_id"],
            best_config=data["best_config"],
            best_metrics=data["best_metrics"],
            total_trials=data["total_trials"],
            successful_trials=data["successful_trials"],
            total_duration=data["total_duration"],
            cost_savings=data["cost_savings"],
            stop_reason=data.get("stop_reason")
            or (data.get("metadata", {}) or {}).get("stop_reason"),
            convergence_history=data.get("convergence_history"),
            full_history=(
                [
                    self._deserialize_trial_result(r)
                    for r in data.get("full_history", [])
                ]
                if data.get("full_history")
                else None
            ),
            metadata=data.get("metadata", {}),
        )

    def _deserialize_trial_result(self, data: dict[str, Any]) -> TrialResultSubmission:
        """Deserialize trial result from server data."""
        from traigent.cloud.models import TrialStatus

        return TrialResultSubmission(
            session_id=data["session_id"],
            trial_id=data["trial_id"],
            metrics=data["metrics"],
            duration=data["duration"],
            status=_coerce_enum(
                TrialStatus, data["status"], fallback=TrialStatus.UNKNOWN
            ),
            outputs_sample=data.get("outputs_sample"),
            error_message=data.get("error_message"),
            metadata=data.get("metadata", {}),
        )

    # Agent-based optimization methods (Model 2: Agent Specification-Based Execution)

    async def start_agent_optimization(
        self, request: AgentOptimizationRequest
    ) -> AgentOptimizationResponse:
        """Start agent optimization through the low-level managed endpoint.

        This is not the supported SDK ``execution_mode="cloud"`` path. SDK
        users wanting portal-visible runs should use ``execution_mode="hybrid"``.

        Args:
            request: Agent optimization request

        Returns:
            AgentOptimizationResponse with optimization details

        Raises:
            CloudServiceError: If optimization start fails
        """
        if request.agent_spec is None:
            raise ValueError(_AGENT_SPEC_REQUIRED)
        if request.dataset is None:
            raise ValueError("dataset is required")
        if request.configuration_space is None:
            raise ValueError("configuration_space is required")

        return await self.optimize_agent(
            agent_spec=request.agent_spec,
            dataset=request.dataset,
            configuration_space=request.configuration_space,
            objectives=request.objectives,
            max_trials=request.max_trials,
            target_cost_reduction=request.target_cost_reduction,
            optimization_strategy=request.metadata,
        )

    async def optimize_agent(
        self,
        agent_spec: AgentSpecification,
        dataset: Dataset,
        configuration_space: dict[str, Any],
        objectives: list[str] | None = None,
        max_trials: int = 50,
        target_cost_reduction: float = 0.65,
        optimization_strategy: dict[str, Any] | None = None,
    ) -> AgentOptimizationResponse:
        """Start agent optimization through the low-level managed endpoint.

        This low-level client method is for backends that implement the managed
        agent endpoint. It is not the supported SDK ``execution_mode="cloud"``
        path; use hybrid for portal-tracked SDK optimization today.

        Args:
            agent_spec: Agent specification to optimize
            dataset: Evaluation dataset
            configuration_space: Parameter space to optimize
            objectives: Optimization objectives (defaults to ["accuracy"])
            max_trials: Maximum number of trials
            target_cost_reduction: Target cost reduction ratio
            optimization_strategy: Strategy configuration

        Returns:
            AgentOptimizationResponse with optimization details

        Raises:
            CloudServiceError: If optimization start fails
        """
        self._raise_if_backend_egress_disabled("start agent optimization")
        await self._ensure_session()

        # Default objectives
        if objectives is None:
            objectives = ["accuracy"]

        # Create optimization request
        request = AgentOptimizationRequest(
            agent_spec=agent_spec,
            dataset=dataset,
            configuration_space=configuration_space,
            objectives=objectives,
            max_trials=max_trials,
            target_cost_reduction=target_cost_reduction,
            user_id=self.config.get("user_id"),
            billing_tier=self.config.get("billing_tier", "standard"),
            metadata=optimization_strategy or {},
        )

        if self._aio_session is None:
            raise CloudServiceError(_CLIENT_SESSION_NOT_INITIALIZED)
        try:
            url = f"{self.api_base_url}/agent/optimize"
            payload = self._serialize_agent_optimization_request(request)

            async with self._aio_session.post(
                url,
                json=payload,
                headers=await self._get_headers(),
                timeout=aiohttp.ClientTimeout(total=300),  # 5 minute timeout
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return self._deserialize_agent_optimization_response(data)
                elif response.status in {401, 403}:
                    raise AuthenticationError(self._AUTH_FAILURE_MESSAGE)
                else:
                    error_text = await response.text()
                    raise CloudServiceError(
                        f"Failed to start agent optimization: HTTP {response.status}: {error_text}"
                    )

        except AuthenticationError:
            await self._reset_http_session("agent_optimize auth failure")
            raise
        except CloudServiceError:
            raise
        except aiohttp.ClientError as e:
            await self._reset_http_session("agent_optimize network error")
            raise CloudServiceError(
                f"Network error starting agent optimization: {e}"
            ) from None
        except Exception as e:
            await self._reset_http_session("agent_optimize failure")
            raise CloudServiceError(
                f"Network error starting agent optimization: {e}"
            ) from None

    async def execute_agent(
        self,
        agent_spec_or_request,
        input_data: dict[str, Any] | None = None,
        config_overrides: dict[str, Any] | None = None,
        execution_context: dict[str, Any] | None = None,
    ) -> AgentExecutionResponse:
        """Execute agent through the low-level managed endpoint.

        This method is separate from SDK ``execution_mode="cloud"``. Use hybrid
        for supported portal-tracked SDK optimization today.

        Args:
            agent_spec_or_request: Either AgentExecutionRequest object or agent specification
            input_data: Input data for agent (ignored if first arg is AgentExecutionRequest)
            config_overrides: Configuration overrides (ignored if first arg is AgentExecutionRequest)
            execution_context: Additional execution context (ignored if first arg is AgentExecutionRequest)

        Returns:
            AgentExecutionResponse with execution results

        Raises:
            CloudServiceError: If execution fails
        """
        self._raise_if_backend_egress_disabled("execute agent")
        await self._ensure_session()

        # Handle both calling patterns
        if hasattr(agent_spec_or_request, "agent_spec"):
            # It's an AgentExecutionRequest object
            request = agent_spec_or_request
        else:
            # It's the old signature with separate parameters
            request = AgentExecutionRequest(
                agent_spec=agent_spec_or_request,
                input_data=input_data,
                config_overrides=config_overrides,
                execution_context=execution_context,
            )

        if self._aio_session is None:
            raise CloudServiceError(_CLIENT_SESSION_NOT_INITIALIZED)
        try:
            url = f"{self.api_base_url}/agent/execute"
            payload = self._serialize_agent_execution_request(request)

            async with self._aio_session.post(
                url,
                json=payload,
                headers=await self._get_headers(),
                timeout=aiohttp.ClientTimeout(total=120),  # 2 minute timeout
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return self._deserialize_agent_execution_response(data)
                else:
                    error_text = await response.text()
                    raise CloudServiceError(
                        f"Failed to execute agent: HTTP {response.status}: {error_text}"
                    )

        except CloudServiceError:
            raise
        except aiohttp.ClientError as e:
            await self._reset_http_session("agent_execute network error")
            raise CloudServiceError(f"Network error executing agent: {e}") from None
        except Exception as e:
            await self._reset_http_session("agent_execute failure")
            raise CloudServiceError(f"Network error executing agent: {e}") from None

    async def get_agent_optimization_status(
        self, optimization_id: str
    ) -> AgentOptimizationStatus:
        """Get status of agent optimization.

        Args:
            optimization_id: Optimization identifier

        Returns:
            AgentOptimizationStatus with current status

        Raises:
            CloudServiceError: If status retrieval fails
        """
        self._raise_if_backend_egress_disabled("get agent optimization status")
        await self._ensure_session()
        if self._aio_session is None:
            raise CloudServiceError(_SESSION_NOT_INITIALIZED)

        try:
            url = f"{self.api_base_url}/agent/optimize/{optimization_id}/status"

            async with self._aio_session.get(
                url,
                headers=await self._get_headers(),
                timeout=aiohttp.ClientTimeout(total=30),
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return self._deserialize_agent_optimization_status(data)
                else:
                    error_text = await response.text()
                    raise CloudServiceError(
                        f"Failed to get optimization status: HTTP {response.status}: {error_text}"
                    )

        except aiohttp.ClientError as e:
            await self._reset_http_session("agent_status network error")
            raise CloudServiceError(
                f"Network error getting optimization status: {e}"
            ) from None
        except Exception as e:
            await self._reset_http_session("agent_status failure")
            raise CloudServiceError(
                f"Network error getting optimization status: {e}"
            ) from None

    async def cancel_agent_optimization(self, optimization_id: str) -> dict[str, Any]:
        """Cancel running agent optimization.

        Args:
            optimization_id: Optimization identifier

        Returns:
            Cancellation confirmation

        Raises:
            CloudServiceError: If cancellation fails
        """
        self._raise_if_backend_egress_disabled("cancel agent optimization")
        await self._ensure_session()

        if self._aio_session is None:
            raise CloudServiceError(_CLIENT_SESSION_NOT_INITIALIZED)
        try:
            url = f"{self.api_base_url}/agent/optimize/{optimization_id}/cancel"

            async with self._aio_session.post(
                url,
                headers=await self._get_headers(),
                timeout=aiohttp.ClientTimeout(total=30),
            ) as response:
                if response.status == 200:
                    return cast(dict[str, Any], await response.json())
                else:
                    error_text = await response.text()
                    raise CloudServiceError(
                        f"Failed to cancel optimization: HTTP {response.status}: {error_text}"
                    )

        except aiohttp.ClientError as e:
            await self._reset_http_session("agent_cancel network error")
            raise CloudServiceError(
                f"Network error canceling optimization: {e}"
            ) from None
        except Exception as e:
            await self._reset_http_session("agent_cancel failure")
            raise CloudServiceError(
                f"Network error canceling optimization: {e}"
            ) from None

    # Serialization methods for agent requests

    def _serialize_agent_optimization_request(
        self, request: AgentOptimizationRequest
    ) -> dict[str, Any]:
        """Serialize agent optimization request."""
        if request.agent_spec is None:
            raise ValueError(_AGENT_SPEC_REQUIRED)
        if request.dataset is None:
            raise ValueError("dataset is required")

        return {
            "agent_spec": self._serialize_agent_spec(request.agent_spec),
            "dataset": self._serialize_dataset(request.dataset),
            "configuration_space": request.configuration_space,
            "objectives": request.objectives,
            "max_trials": request.max_trials,
            "target_cost_reduction": request.target_cost_reduction,
            "user_id": request.user_id,
            "billing_tier": request.billing_tier,
            "metadata": request.metadata,
        }

    def _serialize_agent_execution_request(
        self, request: AgentExecutionRequest
    ) -> dict[str, Any]:
        """Serialize agent execution request."""
        if request.agent_spec is None:
            raise ValueError(_AGENT_SPEC_REQUIRED)

        return {
            "agent_spec": self._serialize_agent_spec(request.agent_spec),
            "input_data": request.input_data,
            "config_overrides": request.config_overrides,
            "execution_context": request.execution_context,
        }

    def _serialize_agent_spec(self, agent_spec: AgentSpecification) -> dict[str, Any]:
        """Serialize agent specification."""
        return {
            "id": agent_spec.id,
            "name": agent_spec.name,
            "agent_type": agent_spec.agent_type,
            "agent_platform": agent_spec.agent_platform,
            "prompt_template": agent_spec.prompt_template,
            "model_parameters": agent_spec.model_parameters,
            "reasoning": agent_spec.reasoning,
            "style": agent_spec.style,
            "tone": agent_spec.tone,
            "format": agent_spec.format,
            "persona": agent_spec.persona,
            "guidelines": agent_spec.guidelines,
            "response_validation": agent_spec.response_validation,
            "custom_tools": agent_spec.custom_tools,
            "metadata": agent_spec.metadata,
        }

    # Deserialization methods for agent responses

    def _deserialize_agent_optimization_response(
        self, data: dict[str, Any]
    ) -> AgentOptimizationResponse:
        """Deserialize agent optimization response."""
        return AgentOptimizationResponse(
            session_id=data["session_id"],
            optimization_id=data["optimization_id"],
            status=data["status"],
            estimated_cost=data.get("estimated_cost"),
            estimated_duration=data.get("estimated_duration"),
            next_steps=data.get("next_steps", []),
        )

    def _deserialize_agent_execution_response(
        self, data: dict[str, Any]
    ) -> AgentExecutionResponse:
        """Deserialize agent execution response."""
        return AgentExecutionResponse(
            output=data["output"],
            duration=data["duration"],
            tokens_used=data.get("tokens_used"),
            cost=data.get("cost"),
            metadata=data.get("metadata", {}),
            error=data.get("error"),
        )

    def _deserialize_agent_optimization_status(
        self, data: dict[str, Any]
    ) -> AgentOptimizationStatus:
        """Deserialize agent optimization status."""
        from traigent.cloud.models import AgentOptimizationStatus

        return AgentOptimizationStatus(
            optimization_id=data["optimization_id"],
            status=_coerce_enum(
                OptimizationSessionStatus,
                data["status"],
                fallback=OptimizationSessionStatus.UNKNOWN,
            ),
            progress=data["progress"],
            completed_trials=data["completed_trials"],
            total_trials=data["total_trials"],
            current_best_metrics=data.get("current_best_metrics"),
            estimated_time_remaining=data.get("estimated_time_remaining"),
            metadata=data.get("metadata", {}),
        )


class CloudServiceError(StandardizedClientError):
    """Exception raised for cloud service errors."""

    def __init__(self, message: str, original_error: Exception | None = None) -> None:
        super().__init__(message, "cloud_service", original_error)


class SessionContractError(CloudServiceError):
    """A session-create CONTRACT failure (Phase 8): an invalid
    TRAIGENT_SESSION_CONTRACT value or a governed/strict session forced onto
    the legacy shape. NEVER absorbed into the local-fallback path — silently
    degrading a contract refusal would launder strict mode (RFC 0001 P7)."""


class CloudRemoteExecutionUnavailableError(CloudServiceError):
    """Raised when remote cloud execution endpoints are intentionally unavailable."""

    def __init__(
        self,
        operation: str | None = None,
        original_error: Exception | None = None,
    ) -> None:
        message = CLOUD_REMOTE_EXECUTION_UNAVAILABLE
        if operation:
            message = f"{message} ({operation})"
        super().__init__(message, original_error)


# Backward compatibility aliases
TraigentClientError = StandardizedClientError
