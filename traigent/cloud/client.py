"""Traigent Cloud Service Client for commercial optimization."""

# Traceability: CONC-Layer-Infra CONC-Quality-Reliability FUNC-CLOUD-HYBRID FUNC-AGENTS REQ-CLOUD-009 REQ-AGNT-013

from __future__ import annotations

import asyncio
import inspect
import time
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from importlib import import_module
from typing import TYPE_CHECKING, Any, NoReturn, cast
from urllib.parse import urlparse

from traigent.cloud._aiohttp_compat import AIOHTTP_AVAILABLE, aiohttp

if TYPE_CHECKING:
    from traigent.arena.dtos import ArenaExecutionRef
from traigent.config.backend_config import BackendConfig
from traigent.config.project import read_optional_project_env
from traigent.config.tenant import TENANT_HEADER_NAME, read_optional_env
from traigent.evaluators.base import Dataset
from traigent.utils.exceptions import ValidationError as ValidationException
from traigent.utils.logging import get_logger
from traigent.utils.retry import NetworkError, RateLimitError, retry_http_request
from traigent.utils.validation import CoreValidators, validate_or_raise

from .auth import AuthenticationError, AuthManager
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
    TrialResultSubmission,
    TrialSuggestion,
)
from .subset_selection import SmartSubsetSelector

logger = get_logger(__name__)
PROJECT_HEADER_NAME = "X-Project-Id"

# Error messages for session state validation
_SESSION_NOT_INITIALIZED = "Session not initialized"
_CLIENT_SESSION_NOT_INITIALIZED = "Client session not initialized"
_AGENT_SPEC_REQUIRED = "agent_spec is required"


def _session_is_closed(session: Any) -> bool:
    """Robustly determine whether an aiohttp session is closed."""
    closed_flag = getattr(session, "closed", False)
    if isinstance(closed_flag, bool):
        return closed_flag
    return False


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
    """Result from cloud optimization service."""

    best_config: dict[str, Any]
    best_metrics: dict[str, float]
    trials_count: int
    cost_reduction: float
    optimization_time: float
    subset_used: bool
    subset_size: int | None = None


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
    """Client for Traigent Cloud Service - enables commercial optimization features."""

    _AUTH_FAILURE_MESSAGE = "Not authenticated with Traigent Cloud Service"

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        enable_fallback: bool = True,
        max_retries: int = 3,
        timeout: float = 30.0,
    ) -> None:
        """Initialize cloud client.

        Args:
            api_key: Traigent Cloud API key
            base_url: Cloud service base URL
            enable_fallback: Fall back to local optimization if cloud fails
            max_retries: Maximum retry attempts for cloud requests
            timeout: Request timeout in seconds
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

        self.base_url = resolved_origin.rstrip("/")
        self.api_base_url = api_base_candidate.rstrip("/")
        self.enable_fallback = enable_fallback
        self.max_retries = max_retries
        self.timeout = timeout

        # Initialize components
        self.auth = AuthManager(api_key=api_key)
        self.auth_manager = self.auth
        self.usage_tracker = UsageTracker()
        self.subset_selector = SmartSubsetSelector()

        # Configuration for agent optimization
        self.config: dict[str, Any] = {"user_id": None, "billing_tier": "standard"}

        # Session management
        self._aio_session: aiohttp.ClientSession | None = None
        self._session_lock: asyncio.Lock = asyncio.Lock()
        # Track session ownership fingerprints for clearer error reporting
        self._session_owners: dict[str, dict[str, Any]] = {}

    @property
    def _session(self) -> aiohttp.ClientSession | None:
        """Expose the underlying HTTP session for tests."""

        return self._aio_session

    @_session.setter
    def _session(self, value: aiohttp.ClientSession | None) -> None:
        self._aio_session = value

    async def __aenter__(self):
        """Async context manager entry."""
        if not AIOHTTP_AVAILABLE:
            logger.warning("aiohttp not available, will use fallback optimization")
            return self

        self._aio_session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.timeout),
            headers=await self.auth.get_headers(),
        )
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
            headers = self._apply_workspace_headers(cast(dict[str, str], headers))

            if not AIOHTTP_AVAILABLE:
                raise CloudServiceError(
                    "Client session not initialized and aiohttp not available"
                )

            self._aio_session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.timeout),
                headers=headers,
            )
            return self._aio_session

    async def _get_headers(self) -> dict[str, str]:
        """Get headers for HTTP requests."""
        headers = await self.auth.get_headers()
        if "Authorization" not in headers and "X-API-Key" not in headers:
            raise AuthenticationError(self._AUTH_FAILURE_MESSAGE)
        return self._apply_workspace_headers(cast(dict[str, str], headers))

    def _apply_workspace_headers(self, headers: dict[str, str]) -> dict[str, str]:
        tenant_id = read_optional_env("TRAIGENT_TENANT_ID")
        project_id = read_optional_project_env()
        resolved_headers = dict(headers)
        if tenant_id:
            resolved_headers[TENANT_HEADER_NAME] = tenant_id
        if project_id:
            resolved_headers[PROJECT_HEADER_NAME] = project_id
        return resolved_headers

    async def _reset_http_session(self, reason: str | None = None) -> None:
        """Close and discard the shared aiohttp session after failures."""
        await self._close_http_session(reason=reason)

    async def _close_http_session(self, reason: str | None = None) -> None:
        """Best-effort close for the shared HTTP session."""
        if not self._aio_session:
            return

        session = self._aio_session
        if session is None:
            return
        self._aio_session = None

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
        """Run optimization using Traigent Cloud Service.

        Args:
            function_name: Name of function being optimized
            dataset: Evaluation dataset
            configuration_space: Parameter search space
            objectives: Optimization objectives
            max_trials: Maximum optimization trials
            target_cost_reduction: Target cost reduction (0.0-1.0)
            local_function: Callable to use for local fallback optimization.

        Returns:
            CloudOptimizationResult with optimization results

        Raises:
            CloudServiceError: If cloud optimization fails and fallback disabled
        """
        start_time = time.time()

        try:
            # Check if aiohttp is available
            if not AIOHTTP_AVAILABLE:
                raise CloudServiceError("aiohttp not available, using fallback")

            # Check authentication
            auth_status = self.auth.is_authenticated()
            if inspect.isawaitable(auth_status):
                auth_status = await auth_status

            if not auth_status:
                raise CloudServiceError("Not authenticated with Traigent Cloud Service")

            # Smart dataset subset selection for cost optimization
            original_size = len(dataset.examples)
            subset_dataset = await self.subset_selector.select_optimal_subset(
                dataset, target_reduction=target_cost_reduction
            )
            subset_size = len(subset_dataset.examples)

            logger.info(
                f"Smart subset selection: {original_size} → {subset_size} examples "
                f"({(1 - subset_size / original_size) * 100:.1f}% reduction)"
            )

            # Prepare optimization request
            request_data = {
                "function_name": function_name,
                "dataset": self._serialize_dataset(subset_dataset),
                "configuration_space": configuration_space,
                "objectives": objectives,
                "max_trials": max_trials,
                "target_cost_reduction": target_cost_reduction,
                "client_version": "0.1.0",
            }

            # Submit optimization to cloud
            result = await self._submit_optimization(request_data)

            # Track usage for billing
            await self.usage_tracker.record_optimization(
                function_name=function_name,
                trials_count=result["trials_count"],
                dataset_size=subset_size,
                optimization_time=time.time() - start_time,
            )

            # Calculate actual cost reduction
            cost_reduction = 1 - (subset_size / original_size)

            return CloudOptimizationResult(
                best_config=result["best_config"],
                best_metrics=result["best_metrics"],
                trials_count=result["trials_count"],
                cost_reduction=cost_reduction,
                optimization_time=time.time() - start_time,
                subset_used=True,
                subset_size=subset_size,
            )

        except Exception as e:
            await self._reset_http_session("cloud optimization failure")
            logger.warning(f"Cloud optimization failed: {e}")

            if self.enable_fallback:
                logger.info("Falling back to local optimization")
                return await self._fallback_optimization(
                    function_name,
                    dataset,
                    configuration_space,
                    objectives,
                    max_trials,
                    local_function=local_function,
                )
            else:
                raise CloudServiceError(f"Cloud optimization failed: {e}") from None

    async def _submit_optimization(
        self, request_data: dict[str, Any]
    ) -> dict[str, Any]:
        """Submit optimization request to cloud service."""
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
                            "Rate limited by cloud service",
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
        """Fallback to local optimization when cloud service unavailable."""
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
        trials = getattr(optimization_result, "trials", [])
        trials_count = len(trials) if isinstance(trials, list) else 0

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

    async def check_service_status(self) -> dict[str, Any]:
        """Check Traigent Cloud Service status."""
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

    async def delete_session(self, session_id: str, cascade: bool = True) -> bool:
        """Delete a session using the backend cleanup endpoint."""
        if not session_id:
            raise ValueError("session_id is required")

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
        arena: ArenaExecutionRef | dict[str, Any] | None = None,
        user_id: str | None = None,
        billing_tier: str = "standard",
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
                arena=arena,
                user_id=user_id,
                billing_tier=billing_tier,
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
        """Get next trial suggestion from the cloud service.

        Args:
            session_id: Optimization session ID
            previous_results: Optional list of recent trial results

        Returns:
            NextTrialResponse with trial suggestion

        Raises:
            CloudServiceError: If request fails
        """
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
        """Submit trial results to the cloud service.

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
        await self._ensure_session()

        from traigent.cloud.models import TrialStatus

        # Convert string status to enum
        try:
            trial_status = TrialStatus(status)
        except ValueError:
            trial_status = TrialStatus.COMPLETED

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
        payload: dict[str, Any] = {
            "function_name": request.function_name,
            "configuration_space": request.configuration_space,
            "objectives": [
                self._serialize_session_objective(objective)
                for objective in (request.objectives or [])
            ],
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
        if request.arena is not None:
            if hasattr(request.arena, "to_dict"):
                payload["arena"] = request.arena.to_dict()
            else:
                payload["arena"] = dict(request.arena)
        return payload

    @staticmethod
    def _serialize_session_objective(
        objective: str | SessionObjectiveDefinition | dict[str, Any],
    ) -> str | dict[str, Any]:
        """Serialize a typed objective while preserving string shorthand."""
        if isinstance(objective, str):
            return objective
        if isinstance(objective, SessionObjectiveDefinition):
            payload: dict[str, Any] = {
                "metric": objective.metric,
            }
            if objective.band is not None:
                payload["band"] = dict(objective.band)
                if objective.test is not None:
                    payload["test"] = objective.test
                if objective.alpha is not None:
                    payload["alpha"] = objective.alpha
            elif objective.direction is not None:
                payload["direction"] = objective.direction
            if objective.weight is not None:
                payload["weight"] = objective.weight
            return payload
        if isinstance(objective, dict):
            return dict(objective)
        raise TypeError(
            "Session objectives must be strings, dicts, or SessionObjectiveDefinition objects"
        )

    def _deserialize_session_response(
        self, data: dict[str, Any]
    ) -> SessionCreationResponse:
        """Deserialize session creation response."""
        return SessionCreationResponse(
            session_id=data["session_id"],
            status=OptimizationSessionStatus(data["status"]),
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
            session_status=OptimizationSessionStatus(
                data.get("session_status", "active")
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
            status=TrialStatus(data["status"]),
            outputs_sample=data.get("outputs_sample"),
            error_message=data.get("error_message"),
            metadata=data.get("metadata", {}),
        )

    # Agent-based optimization methods (Model 2: Agent Specification-Based Execution)

    async def start_agent_optimization(
        self, request: AgentOptimizationRequest
    ) -> AgentOptimizationResponse:
        """Start agent optimization using cloud service (alternative entry point).

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
        """Start agent optimization using cloud service.

        This method implements Model 2: Agent Specification-Based Execution where
        the agent specification and dataset are sent to the cloud service for
        remote execution and optimization.

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
        """Execute agent on cloud service.

        This method allows direct agent execution on the cloud service
        without optimization, useful for testing or production inference.

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
            status=OptimizationSessionStatus(data["status"]),
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


# Backward compatibility aliases
TraigentClientError = StandardizedClientError
