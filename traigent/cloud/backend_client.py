"""Enhanced Traigent Cloud Client with Backend Integration (Refactored).

This is the refactored version of backend_client.py using modular sub-components
for better maintainability and adherence to software engineering principles.
"""

# Traceability: CONC-Layer-Infra CONC-Quality-Reliability CONC-Security FUNC-CLOUD-HYBRID FUNC-AGENTS FUNC-SECURITY REQ-CLOUD-009 REQ-AGNT-013 REQ-SEC-010

import asyncio
import concurrent.futures
import json
import os
import secrets
import sys
import time
import weakref
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from pathlib import Path
from threading import Lock
from typing import TYPE_CHECKING, Any, cast
from urllib.parse import quote, unquote, urlparse, urlunparse

# Import and re-export BackendClientConfig for backward compatibility
from traigent.cloud._aiohttp_compat import AIOHTTP_AVAILABLE, aiohttp
from traigent.cloud.api_operations import ApiOperations, TraigentSessionApiResult
from traigent.cloud.auth import (
    AuthenticationError,
    _build_api_key_auth_headers,
    _inject_trace_context,
    _strip_trace_context_headers,
)
from traigent.cloud.backend_bridges import SDKBackendBridge, SessionExperimentMapping
from traigent.cloud.backend_bridges import bridge as _bridge
from traigent.cloud.backend_components import (
    BackendAuthManager,
    BackendClientConfig,
    BackendSessionManager,
    BackendTrialManager,
)
from traigent.cloud.client import (
    CloudEgressBlockedError,
    CloudServiceError,
    _finalize_aiohttp_session,
    _is_real_aiohttp_session,
    cloud_backend_egress_disabled,
    raise_if_cloud_egress_disabled,
)
from traigent.cloud.cloud_operations import CloudOperations
from traigent.cloud.models import (
    AgentExecutionRequest,
    AgentExecutionResponse,
    AgentOptimizationRequest,
    AgentOptimizationResponse,
    AgentSpecification,
    NextTrialRequest,
    NextTrialResponse,
    OptimizationFinalizationResponse,
    OptimizationSession,
    SessionCreationRequest,
    SessionCreationResponse,
    TrialResultSubmission,
    TrialSuggestion,
)
from traigent.cloud.session_budgets import (
    is_cost_budget_armed_session,
    remember_cost_budget_armed_session,
)

# Import refactored sub-modules
from traigent.cloud.privacy_operations import PrivacyOperations
from traigent.cloud.session_operations import SessionOperations
from traigent.cloud.session_types import SessionCreationResult
from traigent.cloud.subset_selection import SmartSubsetSelector
from traigent.cloud.trial_operations import (
    TrialOperations,
    TrialSlotResult,
    TrialSubmissionResult,
)
from traigent.cloud.url_security import (
    CloudUrlUnreachableError,
    validate_cloud_base_url,
)
from traigent.cloud.user_agent import get_sdk_user_agent
from traigent.config.backend_config import BackendConfig
from traigent.config.project import read_optional_project_env
from traigent.evaluators.base import Dataset
from traigent.security.session_manager import SessionManager as SecuritySessionManager
from traigent.utils.exceptions import RetryableError
from traigent.utils.logging import get_logger
from traigent.utils.retry import RetryConfig, RetryHandler

# Type alias for aiohttp session, falling back to Any at runtime if unavailable
if TYPE_CHECKING:
    from aiohttp import ClientSession as AioClientSession
else:
    AioClientSession = Any  # type: ignore[assignment]

# Import local storage for fallback when backend is unavailable
LOCAL_STORAGE_AVAILABLE = False
try:
    from traigent.storage.local_storage import (
        LocalStorageManager as LocalStorageBackendManager,
    )

    LOCAL_STORAGE_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency for offline mode
    LOCAL_STORAGE_AVAILABLE = False

    class LocalStorageBackendManager:  # type: ignore[no-redef]
        """Fallback stub when local storage support is not installed."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ImportError(
                "LocalStorageManager is unavailable. Install optional storage dependencies."
            ) from None

        def add_trial_result(
            self,
            session_id: str,
            config: dict[str, Any],
            score: float | None,
            metadata: dict[str, Any] | None = None,
            error: str | None = None,
        ) -> None:
            raise ImportError(
                "Local storage functionality requires optional dependencies."
            ) from None


logger = get_logger(__name__)
_ALLOWED_EXAMPLE_FEATURE_KINDS = frozenset({"simhash_v1"})
_JSON_CONTENT_TYPE = "application/json"
_SDK_USER_AGENT = get_sdk_user_agent()
_SYNC_BACKEND_TRANSIENT_STATUSES = frozenset({408, 429, *range(500, 600)})
_INTERACTION_POLICY_REQUIRED_KEYS = frozenset(
    {
        "schema_version",
        "profile",
        "policy_text",
        "question_budget",
        "options_max",
        "jargon_level",
        "next_skill_hint",
        "fallback_policy",
    }
)
_INTERACTION_POLICY_PROFILE_REQUIRED_KEYS = frozenset(
    {
        "schema_version",
        "control",
        "expertise",
        "pace",
        "source",
        "confidence",
    }
)
_INTERACTION_POLICY_RESPONSE_SCHEMA_VERSION = "traigent.agent_interaction.response.v1"
_INTERACTION_POLICY_PROFILE_SCHEMA_VERSION = "traigent.interaction_policy.v1"

STATIC_POLICY_TEXT = (
    "Adapt to the user's interaction profile — persona "
    "(control=delegate|guided|inspect, expertise=se|ds|unknown) and session "
    "mood (pace=execute|balanced|explore); default guided,se,balanced; infer "
    "from explicit statements first then behavior, corrections win. Always be "
    "concise. Match terminology to expertise: plain words and define each term "
    "once for se; compact statistics/optimization terms for ds. When asking, "
    "show at most 3 options, mark one Recommended, give one short trade-off "
    "each. For delegate/execute, proceed with the recommended reversible "
    "action and ask only at hard gates (paid/provider calls, data egress, "
    "destructive edits, service-owned decisions, missing required facts); for "
    "inspect/explore, give brief rationale before asking. Always recommend "
    "the next skill or action. Never weaken safety (dry-run before paid runs; "
    "approval before cost or egress; service-returned plans are authoritative) "
    "and never put persona or private content into telemetry, metadata, names, "
    "logs, or provenance."
)


def _require_project_id(project_id: str) -> str:
    text = (project_id or "").strip()
    if not text:
        raise ValueError("project_id must be a non-empty string.")
    return text


class _SyncBackendTransientError(RetryableError):
    """Retryable wrapper for transient sync backend transport failures."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        retry_after: float | None = None,
    ) -> None:
        super().__init__(message, retry_after=retry_after)
        self.status_code = status_code


_SYNC_BACKEND_RETRY_CONFIG = RetryConfig(
    max_attempts=3,
    initial_delay=0.25,
    max_delay=2.0,
    jitter=False,
    retry_on_exception={_SyncBackendTransientError},
    retry_on_status=set(_SYNC_BACKEND_TRANSIENT_STATUSES),
    respect_retry_after=True,
)


def _parse_retry_after(headers: Any) -> float | None:
    if not headers:
        return None
    try:
        value = headers.get("Retry-After")
    except AttributeError:
        return None
    if value is None:
        return None

    try:
        return max(0.0, float(value))
    except (TypeError, ValueError):
        pass

    try:
        retry_at = parsedate_to_datetime(str(value))
    except (TypeError, ValueError, IndexError, OverflowError):
        return None
    if retry_at.tzinfo is None:
        retry_at = retry_at.replace(tzinfo=UTC)
    return max(0.0, (retry_at - datetime.now(UTC)).total_seconds())


def _raise_for_transient_backend_response(response: Any, operation: str) -> None:
    status_code = getattr(response, "status_code", None)
    if status_code not in _SYNC_BACKEND_TRANSIENT_STATUSES:
        return

    response_text = getattr(response, "text", "")
    raise _SyncBackendTransientError(
        f"{operation} returned transient HTTP {status_code}: {str(response_text)[:200]}",
        status_code=status_code,
        retry_after=_parse_retry_after(getattr(response, "headers", None)),
    )


def _run_sync_backend_request_with_retry(
    operation: str,
    request_call: Callable[[], Any],
) -> Any:
    retry = RetryHandler(_SYNC_BACKEND_RETRY_CONFIG)
    result = retry.execute_with_result(request_call)
    if result.success:
        return result.result

    error = result.error
    if isinstance(error, AuthenticationError):
        raise error
    raise CloudServiceError(f"{operation} failed: {error}") from error


def _session_is_closed(session: Any) -> bool:
    """Return True when the aiohttp session reports being closed."""
    closed_flag = getattr(session, "closed", False)
    if isinstance(closed_flag, bool):
        return closed_flag
    return False


# Export public API
__all__ = [
    "AnalyticsNamespace",
    "BackendClientConfig",
    "BackendIntegratedClient",
    "ExperimentGroupsNamespace",
]


class AnalyticsNamespace:
    """Read-only ``client.analytics`` accessor for backend analytics.

    A thin facade over :class:`traigent.cloud.analytics_client.BackendAnalyticsClient`.
    Each call opens and closes a short-lived async read client that reuses the
    owning client's resolved backend URL and the SDK's existing credential
    resolution. It is READ-only: there are no write methods here, and tenancy is
    owned by the backend (no caller-supplied ``tenant_id``).
    """

    def __init__(self, client: "BackendIntegratedClient") -> None:
        self._client = client

    async def _new_read_client(self) -> Any:
        from traigent.cloud.analytics_auth import (
            resolve_analytics_read_client_credentials,
        )
        from traigent.cloud.analytics_client import BackendAnalyticsClient

        credential_kwargs = await resolve_analytics_read_client_credentials(
            self._client.auth_manager.auth,
            api_key_fallback=self._client._api_key_fallback,
        )
        return BackendAnalyticsClient(
            backend_url=self._client.base_url,
            timeout=self._client.timeout,
            **credential_kwargs,
        )

    async def get_run_report(self, project_id: str, run_id: str) -> dict[str, Any]:
        """Return the backend's full analytics report for one run."""
        async with await self._new_read_client() as reader:
            return cast(dict[str, Any], await reader.get_run_report(project_id, run_id))

    async def get_project_overview(self, project_id: str) -> dict[str, Any]:
        """Return the backend's cross-run overview for a project."""
        async with await self._new_read_client() as reader:
            return cast(dict[str, Any], await reader.get_project_overview(project_id))

    async def compare_runs(self, project_id: str, run_ids: list[str]) -> dict[str, Any]:
        """Compare two or more runs within a project."""
        async with await self._new_read_client() as reader:
            return cast(dict[str, Any], await reader.compare_runs(project_id, run_ids))

    async def get_run_decision_brief(
        self,
        project_id: str,
        run_id: str,
        intent: str = "iterate",
    ) -> dict[str, Any]:
        """Return the backend's decision brief (decision_payload v0) for a run."""
        async with await self._new_read_client() as reader:
            return cast(
                dict[str, Any],
                await reader.get_run_decision_brief(project_id, run_id, intent),
            )

    async def get_single_run_pareto(
        self,
        project_id: str,
        run_id: str,
        *,
        x_measure: str = "cost",
        y_measure: str = "quality",
        request_count: int = 1,
    ) -> dict[str, Any]:
        """Return the backend's Pareto frontier for one run."""
        async with await self._new_read_client() as reader:
            return cast(
                dict[str, Any],
                await reader.get_single_run_pareto(
                    project_id,
                    run_id,
                    x_measure=x_measure,
                    y_measure=y_measure,
                    request_count=request_count,
                ),
            )

    async def get_correlation_matrix(
        self,
        project_id: str,
        run_id: str,
        *,
        method: str = "pearson",
        min_sample: int = 3,
    ) -> dict[str, Any]:
        """Return the backend's correlation matrix for one run."""
        async with await self._new_read_client() as reader:
            return cast(
                dict[str, Any],
                await reader.get_correlation_matrix(
                    project_id,
                    run_id,
                    method=method,
                    min_sample=min_sample,
                ),
            )

    async def get_run_leaderboard(
        self,
        project_id: str,
        run_id: str,
        *,
        objective: str = "weighted",
        weights: Mapping[str, object] | str | None = None,
        constraints: Mapping[str, object] | str | None = None,
        request_count: int = 1,
        limit: int = 50,
    ) -> dict[str, Any]:
        """Return the backend's ranked configuration leaderboard for one run."""
        async with await self._new_read_client() as reader:
            return cast(
                dict[str, Any],
                await reader.get_run_leaderboard(
                    project_id,
                    run_id,
                    objective=objective,
                    weights=weights,
                    constraints=constraints,
                    request_count=request_count,
                    limit=limit,
                ),
            )

    async def get_parameter_insights(
        self,
        project_id: str,
        run_id: str,
        *,
        target_measure: str = "quality",
        min_trials: int = 10,
        top_k: int = 10,
    ) -> dict[str, Any]:
        """Return the backend's parameter-importance insights for one run."""
        async with await self._new_read_client() as reader:
            return cast(
                dict[str, Any],
                await reader.get_parameter_insights(
                    project_id,
                    run_id,
                    target_measure=target_measure,
                    min_trials=min_trials,
                    top_k=top_k,
                ),
            )

    async def get_example_insights(
        self, project_id: str, run_id: str
    ) -> dict[str, Any]:
        """Return the backend's privacy-bounded example insights for one run."""
        async with await self._new_read_client() as reader:
            return cast(
                dict[str, Any],
                await reader.get_example_insights(project_id, run_id),
            )

    async def search_observability_traces(
        self, project_id: str, **filters: Any
    ) -> dict[str, Any]:
        """Return bounded content-free trace summaries for an explicit window."""
        async with await self._new_read_client() as reader:
            return cast(
                dict[str, Any],
                await reader.search_observability_traces(project_id, **filters),
            )

    async def list_observability_issues(
        self, project_id: str, **filters: Any
    ) -> dict[str, Any]:
        """Return recurring observability issues for a project."""
        async with await self._new_read_client() as reader:
            return cast(
                dict[str, Any],
                await reader.list_observability_issues(project_id, **filters),
            )

    async def get_observability_issue(
        self, project_id: str, issue_id: str, **pagination: Any
    ) -> dict[str, Any]:
        """Return one issue with bounded content-free occurrence evidence."""
        async with await self._new_read_client() as reader:
            return cast(
                dict[str, Any],
                await reader.get_observability_issue(
                    project_id, issue_id, **pagination
                ),
            )

    async def list_observability_variants(
        self, project_id: str, **filters: Any
    ) -> dict[str, Any]:
        """Return exact structural trace variants for a project."""
        async with await self._new_read_client() as reader:
            return cast(
                dict[str, Any],
                await reader.list_observability_variants(project_id, **filters),
            )

    async def get_observability_variant(
        self, project_id: str, variant_id: str, **pagination: Any
    ) -> dict[str, Any]:
        """Return one structural variant and bounded trace references."""
        async with await self._new_read_client() as reader:
            return cast(
                dict[str, Any],
                await reader.get_observability_variant(
                    project_id, variant_id, **pagination
                ),
            )

    async def get_observability_trace_analysis(
        self, project_id: str, trace_id: str
    ) -> dict[str, Any]:
        """Return server-derived structural analysis for one trace."""
        async with await self._new_read_client() as reader:
            return cast(
                dict[str, Any],
                await reader.get_observability_trace_analysis(project_id, trace_id),
            )

    async def get_observability_trace_slice(
        self, project_id: str, trace_id: str, **projection: Any
    ) -> dict[str, Any]:
        """Return a bounded content-free slice of one trace."""
        async with await self._new_read_client() as reader:
            return cast(
                dict[str, Any],
                await reader.get_observability_trace_slice(
                    project_id, trace_id, **projection
                ),
            )

    async def get_observability_tool_analysis(
        self, project_id: str, **query: Any
    ) -> dict[str, Any]:
        """Return content-free aggregate tool execution analysis."""
        async with await self._new_read_client() as reader:
            return cast(
                dict[str, Any],
                await reader.get_observability_tool_analysis(project_id, **query),
            )

    async def compare_observability_cohorts(
        self, project_id: str, **comparison: Any
    ) -> dict[str, Any]:
        """Compare bounded reference and comparison trace cohorts."""
        async with await self._new_read_client() as reader:
            return cast(
                dict[str, Any],
                await reader.compare_observability_cohorts(project_id, **comparison),
            )

    async def get_observability_related_changes(
        self, project_id: str, trace_id: str
    ) -> dict[str, Any]:
        """Return lineage links related to a trace without causal attribution."""
        async with await self._new_read_client() as reader:
            return cast(
                dict[str, Any],
                await reader.get_observability_related_changes(project_id, trace_id),
            )

    async def list_experiment_groups(
        self,
        project_id: str,
        *,
        agent_id: str | None = None,
        dataset_id: str | None = None,
        page: int = 1,
        page_size: int = 50,
    ) -> Any:
        """Compatibility alias for ``client.experiment_groups.list``."""
        project_id = _require_project_id(project_id)
        async with await self._new_read_client() as reader:
            return await reader.list_experiment_groups(
                project_id,
                agent_id=agent_id,
                dataset_id=dataset_id,
                page=page,
                page_size=page_size,
            )

    async def get_experiment_group(
        self,
        group_id: str,
        project_id: str,
    ) -> Any:
        """Compatibility alias for ``client.experiment_groups.get``."""
        project_id = _require_project_id(project_id)
        async with await self._new_read_client() as reader:
            return await reader.get_experiment_group(group_id, project_id)

    async def list_experiment_group_configuration_runs(
        self,
        group_id: str,
        project_id: str,
        *,
        page: int = 1,
        page_size: int = 50,
    ) -> Any:
        """Compatibility alias for ``client.experiment_groups.configuration_runs``."""
        project_id = _require_project_id(project_id)
        async with await self._new_read_client() as reader:
            return await reader.list_experiment_group_configuration_runs(
                group_id,
                project_id,
                page=page,
                page_size=page_size,
            )


class ExperimentGroupsNamespace:
    """Read-only ``client.experiment_groups`` accessor for group/cohort reads."""

    def __init__(self, client: "BackendIntegratedClient") -> None:
        self._client = client

    async def _new_read_client(self) -> Any:
        from traigent.cloud.analytics_auth import (
            resolve_analytics_read_client_credentials,
        )
        from traigent.cloud.analytics_client import BackendAnalyticsClient

        credential_kwargs = await resolve_analytics_read_client_credentials(
            self._client.auth_manager.auth,
            api_key_fallback=self._client._api_key_fallback,
        )
        return BackendAnalyticsClient(
            backend_url=self._client.base_url,
            timeout=self._client.timeout,
            **credential_kwargs,
        )

    async def list(
        self,
        project_id: str,
        *,
        agent_id: str | None = None,
        dataset_id: str | None = None,
        page: int = 1,
        page_size: int = 50,
    ) -> Any:
        """List experiment groups/cohorts."""
        project_id = _require_project_id(project_id)
        async with await self._new_read_client() as reader:
            return await reader.list_experiment_groups(
                project_id,
                agent_id=agent_id,
                dataset_id=dataset_id,
                page=page,
                page_size=page_size,
            )

    async def get(self, group_id: str, project_id: str) -> Any:
        """Read one experiment group/cohort."""
        project_id = _require_project_id(project_id)
        async with await self._new_read_client() as reader:
            return await reader.get_experiment_group(group_id, project_id)

    async def configuration_runs(
        self,
        group_id: str,
        project_id: str,
        *,
        page: int = 1,
        page_size: int = 50,
    ) -> Any:
        """List source-preserving configuration rows for a group/cohort."""
        project_id = _require_project_id(project_id)
        async with await self._new_read_client() as reader:
            return await reader.list_experiment_group_configuration_runs(
                group_id,
                project_id,
                page=page,
                page_size=page_size,
            )


# Backwards compatibility: expose shared bridge instance at this module scope so
# existing tests patching ``traigent.cloud.backend_client.bridge`` continue to
# work after the refactor.
bridge = _bridge


class BackendIntegratedClient:
    """Enhanced backend client with Traigent Backend integration (Refactored).

    This client supports both execution models:
    - Hybrid: client-side trial execution with backend session/result tracking
    - Reserved cloud: future backend agent execution paths that fail closed today

    The refactored version delegates specific responsibilities to sub-modules:
    - validators: Data validation functions
    - privacy_operations: Privacy-first optimization operations
    - cloud_operations: Reserved cloud remote-execution operations
    - session_operations: Session lifecycle management
    - trial_operations: Trial management and result submission
    - api_operations: Backend API integration methods
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        backend_config: BackendClientConfig | None = None,
        enable_fallback: bool = True,
        max_retries: int = 3,
        timeout: float = 30.0,
        enable_rate_limiting: bool = True,
        rate_limit_calls: int = 100,
        rate_limit_period: float = 60.0,
        local_storage_path: str | None = None,
        no_egress: bool = False,
    ) -> None:
        """Initialize backend integrated client.

        Args:
            api_key: Traigent backend API key
            base_url: Backend service base URL
            backend_config: Backend integration configuration
            enable_fallback: Reserved compatibility flag for future cloud behavior
            max_retries: Maximum retry attempts for requests
            timeout: Request timeout in seconds
            enable_rate_limiting: Enable rate limiting
            rate_limit_calls: Maximum calls within rate limit period
            rate_limit_period: Rate limit period in seconds
            no_egress: Runtime policy flag that forbids backend transport
        """
        if isinstance(api_key, BackendClientConfig) and backend_config is None:
            backend_config = api_key
            api_key = None
        if not AIOHTTP_AVAILABLE:
            logger.warning("aiohttp not available, client will use fallback mode only")

        # Initialize API operations helper
        self._api_ops = ApiOperations(self)

        self.backend_config = backend_config or BackendClientConfig()

        explicit_base_origin = None
        explicit_api_base = None
        if base_url:
            parsed_origin, parsed_path = BackendConfig.split_api_url(base_url)
            explicit_base_origin = (
                parsed_origin or BackendConfig.normalize_backend_origin(base_url)
            )
            if explicit_base_origin:
                explicit_api_base = (
                    f"{explicit_base_origin}"
                    f"{parsed_path or BackendConfig.get_default_api_path()}"
                )

        effective_base_url = (
            explicit_base_origin or base_url or self.backend_config.backend_base_url
        )
        if effective_base_url is None:
            effective_base_url = BackendConfig.get_backend_url()

        # Validate and sanitize URLs.  If the supplied URL is simply
        # *unreachable* (host cannot be resolved), mark _url_invalid so
        # get_interaction_policy() can fall back to the static policy without
        # raising.  UNSAFE-origin rejections (private/loopback/metadata IPs,
        # bad scheme, embedded credentials, path traversal) are NOT caught here
        # — they keep failing loud to preserve SSRF protection.  The live
        # validation path is otherwise entirely unchanged.
        self._url_invalid = False
        try:
            effective_base_url = validate_cloud_base_url(
                effective_base_url, purpose="backend client"
            )
            self.base_url = self._api_ops.validate_and_sanitize_url(effective_base_url)

            backend_base_url = self.base_url
            if not explicit_base_origin:
                backend_base_url = self.backend_config.backend_base_url or self.base_url
            backend_base_url = validate_cloud_base_url(
                backend_base_url, purpose="backend client"
            )
            self.backend_config.backend_base_url = (
                self._api_ops.validate_and_sanitize_url(backend_base_url)
            )

            if explicit_api_base and not getattr(
                self.backend_config, "api_explicitly_set", False
            ):
                api_base_candidate = explicit_api_base
            else:
                api_base_candidate = (
                    self.backend_config.api_base_url
                    or BackendConfig.build_api_base(
                        self.backend_config.backend_base_url
                    )
                )
            api_origin, api_path = BackendConfig.split_api_url(api_base_candidate)
            if api_origin is None:
                raise ValueError("backend client API base URL must include an origin")
            api_origin = validate_cloud_base_url(api_origin, purpose="backend client")
            # validate_cloud_base_url only saw the origin; the API path is
            # reattached below, so re-check the (decoded) path for traversal
            # here — otherwise an explicit api_base_url like ".../../admin" would
            # bypass the traversal guard that validate_cloud_base_url applies to
            # full URLs.
            # Decode to a fixed point (bounded) so multiply-encoded traversal
            # (e.g. %25252e) cannot survive a fixed two-pass decode.
            _decoded_api_path = api_path or ""
            for _ in range(8):
                _next = unquote(_decoded_api_path)
                if _next == _decoded_api_path:
                    break
                _decoded_api_path = _next
            else:
                # Still changing after 8 decodes — pathologically encoded; reject.
                raise ValueError(
                    "backend client API base URL must not contain path traversal"
                )
            if any(seg in {".", ".."} for seg in _decoded_api_path.split("/") if seg):
                raise ValueError(
                    "backend client API base URL must not contain path traversal"
                )
            api_base_candidate = (
                f"{api_origin}{api_path or BackendConfig.get_default_api_path()}"
            )
            self.api_base_url = self._api_ops.validate_and_sanitize_url(
                api_base_candidate
            )
            self.backend_config.api_base_url = self.api_base_url
        except CloudUrlUnreachableError:
            # The backend host could not be resolved (genuinely unreachable).
            # Store a safe inert placeholder so the rest of __init__ completes;
            # get_interaction_policy() will short-circuit to _static_interaction_policy,
            # while every cloud op fails closed via _raise_if_backend_egress_disabled.
            # NOTE: unsafe-origin ValueErrors are intentionally NOT caught here —
            # they propagate so SSRF/credentialed-URL rejections still fail loud.
            self._url_invalid = True
            _placeholder = "https://backend.invalid"
            self.base_url = _placeholder
            self.backend_config.backend_base_url = _placeholder
            self.api_base_url = _placeholder
            self.backend_config.api_base_url = _placeholder
            logger.debug(
                "BackendIntegratedClient: URL validation failed; "
                "falling back to static interaction policy"
            )

        self.enable_fallback = enable_fallback
        self.no_egress = bool(no_egress)
        self.cloud_egress_intent = False
        self._api_key_fallback = api_key
        self.enable_rate_limiting = enable_rate_limiting
        self.rate_limit_calls = rate_limit_calls
        self.rate_limit_period = rate_limit_period
        self._rate_limit_timestamps: list[float] = []

        # Initialize extracted components
        effective_rate_limit = rate_limit_calls if enable_rate_limiting else sys.maxsize
        self.auth_manager: BackendAuthManager = BackendAuthManager(
            api_key,
            effective_rate_limit,
            rate_limit_period,
            no_egress=self.no_egress,
        )
        self.auth = self.auth_manager.auth
        self.session_manager = BackendSessionManager(
            self.auth_manager, self.backend_config
        )
        self.trial_manager = BackendTrialManager(self.auth_manager, self.backend_config)
        self.max_retries = min(max_retries, 10)  # Cap max retries for safety
        self.timeout = min(timeout, 300.0)  # Cap timeout at 5 minutes

        # Initialize session
        self._session: AioClientSession | None = None
        self._session_lock: asyncio.Lock = asyncio.Lock()
        self._session_finalizer: weakref.finalize | None = None

        # Initialize remaining components
        self.subset_selector = SmartSubsetSelector()

        # Request tracking for security
        self._request_nonces: set[str] = set()
        self._max_nonces = 10000  # Prevent unbounded growth

        # Initialize local storage for fallback when backend is unavailable
        self._local_storage_root = self._resolve_local_storage_path(local_storage_path)
        self.local_storage: LocalStorageBackendManager | None = None
        if LOCAL_STORAGE_AVAILABLE and enable_fallback:
            try:
                self.local_storage = LocalStorageBackendManager(
                    str(self._local_storage_root)
                )
                logger.info(
                    "Local storage fallback initialized at %s",
                    self._local_storage_root,
                )
            except Exception as e:
                logger.warning(f"Failed to initialize local storage fallback: {e}")

        # Session management
        self._session = None
        self._active_sessions: dict[str, OptimizationSession] = {}
        self._active_sessions_lock = Lock()
        self._cost_budget_armed_sessions: set[str] = set()

        # Memory bounds to prevent unbounded growth
        self._max_active_sessions = 100  # Maximum concurrent sessions

        # Security session manager for token issuance and validation
        redis_url = os.getenv("TRAIGENT_SESSION_REDIS_URL")
        session_ttl = int(os.getenv("TRAIGENT_SESSION_TTL", "3600"))
        max_sessions_per_user = int(os.getenv("TRAIGENT_MAX_SESSIONS_PER_USER", "10"))
        self._security_sessions = SecuritySessionManager(
            redis_url=redis_url,
            session_ttl=session_ttl,
            max_sessions_per_user=max_sessions_per_user,
        )
        self._session_credentials: dict[str, str] = {}

        # Initialize operation handlers (delegating to sub-modules)
        self._privacy_ops = PrivacyOperations(self)
        self._cloud_ops = CloudOperations(self)
        self._session_ops = SessionOperations(self)
        self._trial_ops = TrialOperations(self)

        # Read-only analytics accessor (lazily exposed via the ``analytics``
        # property). Kept separate from the write paths above.
        self._analytics_ns: AnalyticsNamespace | None = None
        self._experiment_groups_ns: ExperimentGroupsNamespace | None = None
        self._interaction_policy_cache: dict[
            tuple[str | None, str | None], dict[str, Any]
        ] = {}
        self._interaction_policy_fallback_logged = False

    @property
    def analytics(self) -> "AnalyticsNamespace":
        """Read-only ``client.analytics`` namespace for backend analytics.

        Provides cloud-READ access to optimization-results analytics
        (run reports, project overviews, run comparisons, decision briefs)
        using this client's resolved backend URL and the SDK's existing
        credentials. This namespace performs no writes.
        """
        if self._analytics_ns is None:
            self._analytics_ns = AnalyticsNamespace(self)
        return self._analytics_ns

    @property
    def experiment_groups(self) -> "ExperimentGroupsNamespace":
        """Read-only ``client.experiment_groups`` namespace.

        Provides backend reads for experiment groups/cohorts and their source
        configuration rows. This namespace performs no writes and does not reuse
        or mutate active optimization sessions.
        """
        if self._experiment_groups_ns is None:
            self._experiment_groups_ns = ExperimentGroupsNamespace(self)
        return self._experiment_groups_ns

    def _log_local_interaction_policy_once(self) -> None:
        if self._interaction_policy_fallback_logged:
            return
        logger.info("using local interaction policy")
        self._interaction_policy_fallback_logged = True

    def _remember_cost_budget_armed_session(
        self, session_id: str, budget: Mapping[str, Any] | None
    ) -> None:
        """Track sessions whose typed create armed a positive cost budget."""

        remember_cost_budget_armed_session(self, session_id, budget)

    def _is_cost_budget_armed_session(self, session_id: str) -> bool:
        """Return whether this client created the session with a positive budget."""

        return is_cost_budget_armed_session(self, session_id)

    def _static_interaction_policy(self) -> dict[str, Any]:
        self._log_local_interaction_policy_once()
        return {
            "schema_version": _INTERACTION_POLICY_RESPONSE_SCHEMA_VERSION,
            "profile": {
                "control": "guided",
                "expertise": "se",
                "pace": "balanced",
                "source": "default",
                "confidence": 0.0,
                "schema_version": _INTERACTION_POLICY_PROFILE_SCHEMA_VERSION,
            },
            "policy_text": STATIC_POLICY_TEXT,
            "question_budget": 2,
            "options_max": 3,
            "jargon_level": "plain",
            "next_skill_hint": None,
            "fallback_policy": "static_v1",
        }

    @staticmethod
    def _interaction_policy_cache_key(
        *,
        harness: str | None,
        skill: str | None,
    ) -> tuple[str | None, str | None]:
        def _normalize(value: str | None) -> str | None:
            if not isinstance(value, str):
                return value
            cleaned = value.strip()
            return cleaned or None

        return (_normalize(harness), _normalize(skill))

    @staticmethod
    def _build_interaction_policy_params(
        *,
        harness: str | None,
        skill: str | None,
        signals: Any,
    ) -> dict[str, str] | None:
        params: dict[str, str] = {}
        if isinstance(harness, str) and harness.strip():
            params["harness"] = harness.strip()
        if isinstance(skill, str) and skill.strip():
            params["skill"] = skill.strip()
        if signals is not None:
            params["signals"] = json.dumps(
                signals, separators=(",", ":"), sort_keys=True
            )
        return params or None

    @staticmethod
    def _normalize_interaction_policy_payload(payload: Any) -> dict[str, Any]:
        if not isinstance(payload, dict):
            raise CloudServiceError(
                "Interaction policy endpoint returned a non-JSON object response"
            )

        data = payload.get("data", payload)
        if not isinstance(data, dict):
            raise CloudServiceError(
                "Interaction policy endpoint returned invalid response data"
            )

        missing = sorted(_INTERACTION_POLICY_REQUIRED_KEYS - data.keys())
        if missing:
            raise CloudServiceError(
                "Interaction policy endpoint returned malformed response: "
                f"missing {', '.join(missing)}"
            )

        if data.get("schema_version") != _INTERACTION_POLICY_RESPONSE_SCHEMA_VERSION:
            raise CloudServiceError(
                "Interaction policy endpoint returned unsupported response schema"
            )

        profile = data.get("profile")
        if not isinstance(profile, dict):
            raise CloudServiceError(
                "Interaction policy endpoint returned malformed profile"
            )

        missing_profile = sorted(
            _INTERACTION_POLICY_PROFILE_REQUIRED_KEYS - profile.keys()
        )
        if missing_profile:
            raise CloudServiceError(
                "Interaction policy endpoint returned malformed profile: "
                f"missing {', '.join(missing_profile)}"
            )

        if profile.get("schema_version") != _INTERACTION_POLICY_PROFILE_SCHEMA_VERSION:
            raise CloudServiceError(
                "Interaction policy endpoint returned unsupported profile schema"
            )

        return {
            "schema_version": data["schema_version"],
            "profile": {
                "control": profile["control"],
                "expertise": profile["expertise"],
                "pace": profile["pace"],
                "source": profile["source"],
                "confidence": profile["confidence"],
                "schema_version": profile["schema_version"],
            },
            "policy_text": data["policy_text"],
            "question_budget": data["question_budget"],
            "options_max": data["options_max"],
            "jargon_level": data["jargon_level"],
            "next_skill_hint": data["next_skill_hint"],
            "fallback_policy": data["fallback_policy"],
        }

    def _cache_interaction_policy(
        self,
        *,
        cache_key: tuple[str | None, str | None],
        policy: Mapping[str, Any],
        etag: str | None,
    ) -> None:
        profile = policy.get("profile")
        cached_profile: dict[str, Any] | None = None
        if isinstance(profile, Mapping):
            cached_profile = {
                "control": profile.get("control"),
                "expertise": profile.get("expertise"),
                "pace": profile.get("pace"),
                "source": profile.get("source"),
                "confidence": profile.get("confidence"),
                "schema_version": profile.get("schema_version"),
            }
        self._interaction_policy_cache[cache_key] = {
            "etag": etag,
            "policy_text": policy["policy_text"],
            "profile": cached_profile,
        }

    def _cached_interaction_policy(
        self, cache_key: tuple[str | None, str | None]
    ) -> dict[str, Any] | None:
        cached = self._interaction_policy_cache.get(cache_key)
        if not isinstance(cached, dict):
            return None

        policy_text = cached.get("policy_text")
        if not isinstance(policy_text, str):
            return None

        profile = cached.get("profile")
        if not isinstance(profile, dict):
            return None

        restored = {
            "schema_version": _INTERACTION_POLICY_RESPONSE_SCHEMA_VERSION,
            "profile": {
                "control": profile.get("control"),
                "expertise": profile.get("expertise"),
                "pace": profile.get("pace"),
                "source": profile.get("source"),
                "confidence": profile.get("confidence"),
                "schema_version": profile.get("schema_version"),
            },
            "policy_text": policy_text,
            "question_budget": 2,
            "options_max": 3,
            "jargon_level": "plain",
            "next_skill_hint": None,
            "fallback_policy": "static_v1",
        }
        return restored

    @property
    def session_bridge(self) -> "SDKBackendBridge":
        """Return the shared session bridge (exposed for test patching)."""

        return bridge

    async def get_interaction_policy(
        self,
        *,
        harness: str | None = None,
        skill: str | None = None,
        signals: Any = None,
    ) -> dict[str, Any]:
        """Return the backend persona-interaction policy seed for this session.

        The backend response is a seed for the agent's starting interaction
        profile only. Explicit user corrections made in-conversation remain
        authoritative and must override this seed.
        """
        from traigent.utils.env_config import is_mock_llm

        if cloud_backend_egress_disabled(self.no_egress):
            return self._static_interaction_policy()

        if getattr(self, "_url_invalid", False) is True:
            return self._static_interaction_policy()

        api_key = os.getenv("TRAIGENT_API_KEY") or self._api_key_fallback
        if not api_key or is_mock_llm() or not AIOHTTP_AVAILABLE:
            return self._static_interaction_policy()

        cache_key = self._interaction_policy_cache_key(
            harness=harness,
            skill=skill,
        )
        cached = self._interaction_policy_cache.get(cache_key)

        headers = _build_api_key_auth_headers(api_key)
        headers.setdefault("User-Agent", _SDK_USER_AGENT)
        # This path bypasses AuthManager._add_common_headers, so inject the
        # active trace context directly (no-op when tracing is unavailable).
        _inject_trace_context(headers)
        cached_etag = cached.get("etag") if isinstance(cached, dict) else None
        if isinstance(cached_etag, str) and cached_etag:
            headers["If-None-Match"] = cached_etag

        try:
            params = self._build_interaction_policy_params(
                harness=harness,
                skill=skill,
                signals=signals,
            )
            backend_origin = (
                self.backend_config.backend_base_url or BackendConfig.get_backend_url()
            ).rstrip("/")
            url = f"{backend_origin}/api/v1/auth/me/interaction-policy"

            async with aiohttp.ClientSession(trust_env=True) as session:
                async with session.get(
                    url,
                    headers=headers,
                    params=params,
                    timeout=aiohttp.ClientTimeout(total=self.timeout),
                ) as response:
                    if response.status == 304:
                        cached_policy = self._cached_interaction_policy(cache_key)
                        if cached_policy is not None:
                            return cached_policy
                        raise CloudServiceError(
                            "Interaction policy cache miss for backend 304 response"
                        )

                    if response.status >= 400:
                        error_text = await response.text()
                        raise CloudServiceError(
                            "Interaction policy request failed with HTTP "
                            f"{response.status}: {error_text[:200]}"
                        )

                    try:
                        payload = await response.json()
                    except Exception as exc:
                        raise CloudServiceError(
                            "Interaction policy endpoint returned non-JSON response"
                        ) from exc

                    policy = self._normalize_interaction_policy_payload(payload)
                    self._cache_interaction_policy(
                        cache_key=cache_key,
                        policy=policy,
                        etag=response.headers.get("ETag"),
                    )
                    return policy
        except (
            aiohttp.ClientError,
            TimeoutError,
            CloudServiceError,
            AuthenticationError,
            TypeError,
            ValueError,
        ) as exc:
            logger.debug(
                "Interaction policy request failed; using local fallback: %s",
                exc,
            )
            return self._static_interaction_policy()

    def _resolve_local_storage_path(self, override: str | None) -> Path:
        """Determine the root path for local fallback storage."""

        candidates: list[str | None] = [
            override,
            os.getenv("TRAIGENT_RESULTS_FOLDER"),
        ]

        for candidate in candidates:
            if candidate:
                return Path(candidate).expanduser().resolve()

        return (Path.home() / ".traigent").expanduser().resolve()

    # === Rate Limiting & Security ===

    async def _check_rate_limit(self) -> None:
        """Check and enforce rate limiting.

        Raises:
            CloudServiceError: If rate limit is exceeded
        """
        if not self.enable_rate_limiting:
            return

        current_time = time.time()

        # Clean old timestamps
        self._rate_limit_timestamps = [
            t
            for t in self._rate_limit_timestamps
            if current_time - t < self.rate_limit_period
        ]

        # Check if rate limit exceeded
        if len(self._rate_limit_timestamps) >= self.rate_limit_calls:
            wait_time = self.rate_limit_period - (
                current_time - self._rate_limit_timestamps[0]
            )
            raise CloudServiceError(
                f"Rate limit exceeded. Please wait {wait_time:.1f} seconds before making another request."
            )

        # Add current timestamp
        self._rate_limit_timestamps.append(current_time)

    def _generate_request_nonce(self) -> str:
        """Generate a unique nonce for request tracking.

        Returns:
            Unique nonce string
        """
        # Clean old nonces if too many
        if len(self._request_nonces) >= self._max_nonces:
            # Keep only recent half
            to_remove = len(self._request_nonces) // 2
            for _ in range(to_remove):
                self._request_nonces.pop()

        nonce = secrets.token_urlsafe(32)
        self._request_nonces.add(nonce)
        return nonce

    def _register_security_session(
        self,
        session_id: str,
        user_id: str | None,
        metadata: dict[str, Any],
    ) -> str:
        """Register session with security manager and return issued token."""
        try:
            _, session_token = self._security_sessions.create_session(
                user_id=user_id or "anonymous",
                metadata=metadata,
                session_id=session_id,
            )
        except Exception as exc:
            logger.warning(
                "Security session registration failed for %s: %s", session_id, exc
            )
            session_token = secrets.token_urlsafe(32)

        self._session_credentials[session_id] = session_token
        return cast(str, session_token)

    def _revoke_security_session(self, session_id: str) -> None:
        """Revoke a session from the security manager and clear cached token."""
        try:
            self._security_sessions.revoke_session(session_id)
        except Exception as exc:
            logger.debug("Security session revoke failed for %s: %s", session_id, exc)
        finally:
            self._session_credentials.pop(session_id, None)

    def get_session_token(self, session_id: str) -> str | None:
        """Return the cached session token for the given session, if available."""
        return self._session_credentials.get(session_id)

    def validate_session_token(self, session_id: str, token: str) -> bool:
        """Validate a session token using the security manager."""
        try:
            return (
                self._security_sessions.validate_session(session_id, token) is not None
            )
        except Exception as exc:
            logger.debug(
                "Security session validation failed for %s: %s", session_id, exc
            )
            return False

    # === Context Manager Support ===

    async def __aenter__(self) -> "BackendIntegratedClient":
        """Async context manager entry."""
        if cloud_backend_egress_disabled(self.no_egress):
            return self
        # Fail closed on an unusable backend URL: the placeholder origin set in
        # __init__ must never be dialed, and auth.get_headers() below may POST
        # to /keys/validate. Return an inert client with no transport.
        if getattr(self, "_url_invalid", False) is True:
            return self
        if AIOHTTP_AVAILABLE:
            # Try to get headers, but don't fail if authentication is not available
            # B4 ROUND 4: ``AuthenticationError`` must propagate -- a
            # backend-rejected key must NEVER produce an authenticated session,
            # even one with empty auth headers (which could be paired with a
            # raw key elsewhere in the stack). Other exceptions (e.g. Edge
            # Analytics mode without a key) continue with empty headers.
            try:
                headers = await self.auth_manager.auth.get_headers()
            except AuthenticationError:
                raise
            except Exception as e:
                # In Edge Analytics mode or when no API key is available, continue without auth headers
                logger.debug(f"No authentication available for context manager: {e}")
                headers = {}

            # Add security headers
            headers.update(
                {
                    "X-Request-ID": self._generate_request_nonce(),
                    "X-Client-Version": "2.0.0",
                }
            )
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.timeout),
                # Session-default headers are long-lived: never freeze a
                # traceparent/tracestate into them (stale span context).
                # Trace context travels only on per-request headers.
                headers=_strip_trace_context_headers(headers),
                trust_env=True,
            )
            self._register_session_finalizer(self._session)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.close(_reason="context-exit")

    async def _ensure_session(self) -> AioClientSession:
        """Ensure session exists with current auth headers."""

        self._raise_if_backend_egress_disabled("backend request")
        existing = self._session
        if existing is not None and not _session_is_closed(existing):
            return cast(AioClientSession, existing)

        async with self._session_lock:
            existing = self._session
            if existing is not None and not _session_is_closed(existing):
                return cast(AioClientSession, existing)

            if not AIOHTTP_AVAILABLE:
                raise CloudServiceError(
                    "Client session not initialized and aiohttp not available"
                )

            # B4 ROUND 4: Fail closed on authentication errors. Previously, this
            # block caught any exception from ``get_headers()`` and silently
            # rebuilt headers from the raw stored API key (or from the
            # private ``_get_api_key_headers()`` helper) -- defeating the
            # round-3 fail-closed change in ``AuthManager.get_auth_headers()``
            # because the backend-rejected key would still be shipped on the
            # wire as ``X-API-Key`` / ``Authorization``.
            #
            # Now: ``AuthenticationError`` (and its ``InvalidCredentialsError``
            # subclass) propagates so the caller sees the rejection. Other
            # unexpected errors are surfaced as ``CloudServiceError`` rather
            # than silently swallowed -- still fail-closed for auth, but
            # without losing diagnostic context.
            try:
                headers = await self.auth_manager.auth.get_headers()
            except AuthenticationError:
                # Do NOT fall back to raw-key headers. Re-raise so callers
                # see the auth failure instead of getting a session that
                # silently emits a rejected key.
                raise
            except Exception as exc:
                logger.warning("Could not get auth headers: %s", exc)
                raise CloudServiceError(
                    f"Failed to build authenticated session: {exc}"
                ) from exc
            headers.setdefault("Content-Type", _JSON_CONTENT_TYPE)
            headers.setdefault("User-Agent", _SDK_USER_AGENT)

            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.timeout),
                # Same rule as __aenter__: no stale trace context in
                # long-lived session-default headers.
                headers=_strip_trace_context_headers(headers),
                trust_env=True,
            )
            self._register_session_finalizer(self._session)

            return cast(AioClientSession, self._session)

    def _raise_if_backend_egress_disabled(self, operation: str) -> None:
        """Fail closed before any backend HTTP request."""

        if getattr(self, "_url_invalid", False) is True:
            raise CloudEgressBlockedError(operation)
        raise_if_cloud_egress_disabled(operation, no_egress=self.no_egress)

    async def _reset_http_session(self, reason: str | None = None) -> None:
        """Close and discard the shared aiohttp session."""

        if not self._session:
            return

        session = self._session
        self._session = None
        self._detach_session_finalizer()
        try:
            if _session_is_closed(session):
                return
            await session.close()
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
                "Error closing backend HTTP session%s: %s",
                f" ({reason})" if reason else "",
                exc,
            )

    async def close(self, *, _reason: str = "shutdown") -> None:
        """Close any active HTTP session to avoid resource leaks."""

        await self._reset_http_session(_reason)

    def _register_session_finalizer(self, session: AioClientSession) -> None:
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

    # === Delegated Operations ===
    # The following methods delegate to the refactored sub-modules

    # Privacy Operations
    async def create_privacy_optimization_session(
        self,
        function_name: str,
        configuration_space: dict[str, Any],
        objectives: list[str],
        dataset_metadata: dict[str, Any],
        max_trials: int = 50,
        user_id: str | None = None,
    ) -> tuple[str, str, str]:
        """Create privacy-first optimization session.
        Delegates to privacy_operations module."""
        return cast(
            tuple[str, str, str],
            await self._privacy_ops.create_privacy_optimization_session(
                function_name,
                configuration_space,
                objectives,
                dataset_metadata,
                max_trials,
                user_id,
            ),
        )

    async def _deprecated_create_privacy_optimization_session(
        self,
        function_name: str,
        configuration_space: dict[str, Any],
        objectives: list[str],
        dataset_metadata: dict[str, Any],
        max_trials: int = 50,
        user_id: str | None = None,
    ) -> tuple[str, str, str]:
        """Compatibility shim for legacy tests expecting the deprecated method name."""

        logger.warning(
            "_deprecated_create_privacy_optimization_session is deprecated; use create_privacy_optimization_session"
        )
        return await self.create_privacy_optimization_session(
            function_name,
            configuration_space,
            objectives,
            dataset_metadata,
            max_trials,
            user_id,
        )

    async def get_next_privacy_trial(
        self,
        session_id: str,
        previous_results: list[TrialResultSubmission] | None = None,
    ) -> TrialSuggestion | None:
        """Get next trial suggestion for privacy-first optimization.
        Delegates to privacy_operations module."""
        return cast(
            TrialSuggestion | None,
            await self._privacy_ops.get_next_privacy_trial(
                session_id, previous_results
            ),
        )

    async def submit_privacy_trial_results(
        self,
        session_id: str,
        trial_id: str,
        config: dict[str, Any],
        metrics: dict[str, float],
        duration: float,
        error_message: str | None = None,
    ) -> bool:
        """Submit trial results for privacy-first optimization.
        Delegates to privacy_operations module."""
        return cast(
            bool,
            await self._privacy_ops.submit_privacy_trial_results(
                session_id, trial_id, config, metrics, duration, error_message
            ),
        )

    # Reserved Cloud Operations
    async def start_agent_optimization(
        self,
        agent_spec: AgentSpecification,
        dataset: Dataset,
        configuration_space: dict[str, Any],
        objectives: list[str],
        max_trials: int = 50,
        user_id: str | None = None,
    ) -> AgentOptimizationResponse:
        """Start reserved cloud agent optimization.
        Delegates to cloud_operations module."""
        return cast(
            AgentOptimizationResponse,
            await self._cloud_ops.start_agent_optimization(
                agent_spec,
                dataset,
                configuration_space,
                objectives,
                max_trials,
                user_id,
            ),
        )

    async def execute_agent(
        self,
        agent_spec: AgentSpecification,
        input_data: dict[str, Any],
        config_overrides: dict[str, Any] | None = None,
    ) -> AgentExecutionResponse:
        """Execute agent through the reserved cloud path.
        Delegates to cloud_operations module."""
        return cast(
            AgentExecutionResponse,
            await self._cloud_ops.execute_agent(
                agent_spec, input_data, config_overrides
            ),
        )

    # Session Operations
    def create_session(
        self,
        function_name: str,
        search_space: dict[str, Any],
        optimization_goal: str = "maximize",
        metadata: dict[str, Any] | None = None,
        objectives: list[Any] | None = None,
        default_config: dict[str, Any] | None = None,
        promotion_policy: dict[str, Any] | None = None,
        tvl_governance: dict[str, Any] | None = None,
        warm_start_from: str | None = None,
        artifact_fingerprints: dict[str, str | None] | None = None,
        fingerprint_meta: dict[str, Any] | None = None,
        evaluator_id: str | None = None,
        evaluator_definition_id: str | None = None,
        smart_pruning: dict[str, Any] | None = None,
        cost_limit: float | None = None,
        optimization_strategy: dict[str, Any] | None = None,
    ) -> SessionCreationResult:
        """Synchronous wrapper for creating a session.
        Delegates to session_operations module. Phase 8: objectives are
        METRIC names for the typed contract; promotion_policy/tvl_governance
        are the content-free governance wire (RFC 0001 P8)."""
        return self._session_ops.create_session(
            function_name,
            search_space,
            optimization_goal,
            metadata,
            objectives=objectives,
            default_config=default_config,
            promotion_policy=promotion_policy,
            tvl_governance=tvl_governance,
            warm_start_from=warm_start_from,
            artifact_fingerprints=artifact_fingerprints,
            fingerprint_meta=fingerprint_meta,
            evaluator_id=evaluator_id,
            evaluator_definition_id=evaluator_definition_id,
            smart_pruning=smart_pruning,
            cost_limit=cost_limit,
            optimization_strategy=optimization_strategy,
        )

    async def create_hybrid_session(
        self,
        problem_statement: str,
        search_space: dict[str, Any],
        optimization_config: dict[str, Any],
        metadata: dict[str, Any] | None = None,
    ) -> tuple[str, str, str]:
        """Create a hybrid execution session.
        Delegates to session_operations module."""
        return cast(
            tuple[str, str, str],
            await self._session_ops.create_hybrid_session(
                problem_statement, search_space, optimization_config, metadata
            ),
        )

    async def get_hybrid_session_status(self, session_id: str) -> dict[str, Any]:
        """Get status of a hybrid session.
        Delegates to session_operations module."""
        return cast(
            dict[str, Any],
            await self._session_ops.get_hybrid_session_status(session_id),
        )

    async def finalize_hybrid_session(self, session_id: str) -> dict[str, Any]:
        """Finalize hybrid session and get results.
        Delegates to session_operations module."""
        return cast(
            dict[str, Any],
            await self._session_ops.finalize_hybrid_session(session_id),
        )

    async def finalize_session(
        self,
        session_id: str,
        include_full_history: bool = False,
        certified_selection: dict[str, Any] | None = None,
        session_aggregation: dict[str, Any] | None = None,
    ) -> OptimizationFinalizationResponse:
        """Finalize optimization session and get results.
        Delegates to session_operations module. Phase 8: an optional
        client-attested certified-selection report (content-free).
        Traigent#1720/#1724 (g2:agg-summary): an optional content-free
        session-level rollup, threaded the same way."""
        return await self._session_ops.finalize_session(
            session_id,
            include_full_history,
            certified_selection=certified_selection,
            session_aggregation=session_aggregation,
        )

    async def delete_session(self, session_id: str, cascade: bool = True) -> bool:
        """Delete optimization session and optionally related data.
        Delegates to session_operations module."""
        return cast(bool, await self._session_ops.delete_session(session_id, cascade))

    def finalize_session_sync(
        self,
        session_id: str,
        include_full_history: bool = False,
        certified_selection: dict[str, Any] | None = None,
        session_aggregation: dict[str, Any] | None = None,
    ) -> OptimizationFinalizationResponse | None:
        """Synchronous wrapper for finalize_session.
        Delegates to session_operations module."""
        return self._session_ops.finalize_session_sync(
            session_id,
            include_full_history,
            certified_selection=certified_selection,
            session_aggregation=session_aggregation,
        )

    def delete_session_sync(self, session_id: str, cascade: bool = True) -> bool:
        """Synchronous wrapper for delete_session.
        Delegates to session_operations module."""
        return cast(bool, self._session_ops.delete_session_sync(session_id, cascade))

    def get_active_sessions(self) -> dict[str, OptimizationSession]:
        """Get all active optimization sessions.
        Delegates to session_operations module."""
        return cast(
            dict[str, OptimizationSession], self._session_ops.get_active_sessions()
        )

    def get_session_mapping(self, session_id: str) -> SessionExperimentMapping | None:
        """Get session to experiment mapping.
        Delegates to session_operations module."""
        return self._session_ops.get_session_mapping(session_id)

    def _get_sync_auth_headers(self, target: str = "backend") -> dict[str, str]:
        """Return auth headers for sync request paths.

        Uses the async auth manager to preserve the canonical header-generation
        path, including JWT refresh behavior when available.

        B4 ROUND 5: Fail closed on auth header generation. Previously this
        method caught every exception from ``get_headers()`` (including
        ``AuthenticationError`` raised by the round-3 fail-closed check in
        ``AuthManager.get_auth_headers()``) and returned bare default
        headers (``Content-Type`` + ``User-Agent``). The caller --
        ``upload_example_features()`` -- then issued an UNAUTHENTICATED
        ``requests.post`` to the backend, defeating round-3/round-4 closure
        of the auth chain.

        After round 5:

        * ``AuthenticationError`` propagates so the caller surfaces the
          backend rejection instead of silently shipping an unauthenticated
          request.
        * Any other unexpected failure is re-raised as ``CloudServiceError``
          so callers cannot mistake "header generation blew up" for
          "everything is fine, here are default headers".
        * The bare ``default_headers`` short-circuit only fires when there
          is *legitimately* no auth manager available (e.g. Edge Analytics
          mode); in that case the caller decides whether to proceed.
        """
        default_headers = {
            "Content-Type": _JSON_CONTENT_TYPE,
            "User-Agent": _SDK_USER_AGENT,
        }
        # B4 ROUND 6: Tighten the default-headers short-circuit. The bare
        # defaults are ONLY a legitimate response when the caller is
        # operating without any auth manager (e.g. anonymous Edge mode).
        # If we have an auth manager but its ``.auth`` is missing or
        # lacks ``get_headers``, that is an internal inconsistency -- not
        # a license to ship an unauthenticated request. Surface it as a
        # ``CloudServiceError`` so the caller fails closed instead.
        if getattr(self, "auth_manager", None) is None:
            # Anonymous Edge mode bypasses AuthManager._add_common_headers, so
            # inject the active trace context here (no-op when unavailable).
            _inject_trace_context(default_headers)
            return default_headers
        auth = getattr(self.auth_manager, "auth", None)
        if auth is None or not hasattr(auth, "get_headers"):
            raise CloudServiceError(
                "auth manager is in an invalid state; cannot construct auth headers"
            )

        async def _get_headers_async() -> dict[str, str]:
            return cast(dict[str, str], await auth.get_headers(target=target))

        def _run_in_new_loop() -> dict[str, str]:
            new_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(new_loop)
            try:
                return new_loop.run_until_complete(_get_headers_async())
            finally:
                new_loop.close()

        try:
            try:
                asyncio.get_running_loop()
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(_run_in_new_loop)
                    headers = future.result(timeout=min(self.timeout, 30.0))
            except RuntimeError:
                headers = asyncio.run(_get_headers_async())
        except AuthenticationError:
            # Re-raise: a backend-rejected key must NEVER be downgraded to a
            # silent unauthenticated POST. Callers (e.g.
            # ``upload_example_features``) catch this and abort before any
            # network I/O happens.
            raise
        except Exception as exc:
            # Unexpected header-generation failures: surface as
            # ``CloudServiceError`` rather than swallow into default headers.
            logger.warning(
                "Sync auth header generation failed; refusing to send "
                "unauthenticated request: %s",
                exc,
            )
            raise CloudServiceError(
                f"Failed to build sync auth headers: {exc}"
            ) from exc

        merged_headers = dict(headers or {})
        merged_headers.setdefault("Content-Type", _JSON_CONTENT_TYPE)
        merged_headers.setdefault("User-Agent", _SDK_USER_AGENT)
        project_id = read_optional_project_env()
        if project_id:
            merged_headers.setdefault("X-Project-Id", project_id)
        return merged_headers

    def _build_feature_upload_url(self, experiment_run_id: str) -> str | None:
        """Return a sanitized feature-upload URL for the configured backend API."""
        parsed = urlparse(self.api_base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            logger.debug(
                "Skipping example feature upload because api_base_url is invalid: %s",
                self.api_base_url,
            )
            return None

        if (
            parsed.username
            or parsed.password
            or parsed.params
            or parsed.query
            or parsed.fragment
        ):
            logger.debug(
                "Skipping example feature upload because api_base_url contains unsupported URL components"
            )
            return None

        encoded_run_id = quote(experiment_run_id, safe="")
        upload_path = (
            f"{parsed.path.rstrip('/')}/analytics/example-scoring/"
            f"{encoded_run_id}/features"
        )
        return cast(
            str,
            urlunparse(
                parsed._replace(path=upload_path, params="", query="", fragment="")
            ),
        )

    def _build_best_config_url(self, config_id: str | None = None) -> str | None:
        """Return a sanitized best-config URL for the configured backend API."""
        parsed = urlparse(self.api_base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            return None
        if (
            parsed.username
            or parsed.password
            or parsed.params
            or parsed.query
            or parsed.fragment
        ):
            return None

        best_config_path = f"{parsed.path.rstrip('/')}/best-configs"
        if config_id:
            best_config_path = f"{best_config_path}/{quote(config_id, safe='')}"
        return cast(
            str,
            urlunparse(
                parsed._replace(path=best_config_path, params="", query="", fragment="")
            ),
        )

    @staticmethod
    def _extract_backend_data(response: Any) -> dict[str, Any]:
        try:
            payload = response.json()
        except Exception as exc:  # noqa: BLE001
            raise CloudServiceError("Backend returned a non-JSON response") from exc
        if not isinstance(payload, dict):
            raise CloudServiceError("Backend returned an invalid JSON response")
        data = payload.get("data", payload)
        if not isinstance(data, dict):
            raise CloudServiceError("Backend response data was not an object")
        return data

    def publish_best_config_sync(
        self,
        spec: dict[str, Any],
        *,
        environment: str | None = None,
        if_match: str | None = None,
    ) -> dict[str, Any]:
        """Publish a canonical best-config spec to the backend."""
        self._raise_if_backend_egress_disabled("publish best config")
        try:
            import requests
        except ImportError as exc:  # pragma: no cover - requests is required
            raise CloudServiceError("requests is unavailable") from exc

        url = self._build_best_config_url()
        if not url:
            raise CloudServiceError("Backend best-config URL is invalid")

        headers = self._get_sync_auth_headers(target="backend")
        if if_match:
            headers["If-Match"] = if_match
        payload: dict[str, Any] = {"spec": spec}
        if environment:
            payload["environment"] = environment

        def _post_best_config() -> Any:
            try:
                response = requests.post(  # nosec B113 - timeout is provided
                    url,
                    json=payload,
                    headers=headers,
                    timeout=min(self.timeout, 30.0),
                )
            except (
                requests.exceptions.ConnectionError,
                requests.exceptions.Timeout,
                ConnectionError,
                TimeoutError,
            ) as exc:
                raise _SyncBackendTransientError(
                    f"Best-config publish transport failed: {exc}"
                ) from exc
            _raise_for_transient_backend_response(response, "Best-config publish")
            return response

        response = _run_sync_backend_request_with_retry(
            "Best-config publish",
            _post_best_config,
        )

        if response.status_code >= 400:
            raise CloudServiceError(
                f"Best-config publish rejected with HTTP {response.status_code}: "
                f"{response.text[:200]}"
            )
        return self._extract_backend_data(response)

    def fetch_best_config_sync(
        self,
        config_id: str,
        *,
        environment: str | None = None,
        function_ref: str | None = None,
        etag: str | None = None,
    ) -> dict[str, Any] | None:
        """Fetch the current backend best config, returning None on 304/404."""
        self._raise_if_backend_egress_disabled("fetch best config")
        try:
            import requests
        except ImportError as exc:  # pragma: no cover - requests is required
            raise CloudServiceError("requests is unavailable") from exc

        url = self._build_best_config_url(config_id)
        if not url:
            raise CloudServiceError("Backend best-config URL is invalid")

        headers = self._get_sync_auth_headers(target="backend")
        if etag:
            headers["If-None-Match"] = etag
        params = {
            key: value
            for key, value in {
                "environment": environment,
                "function_ref": function_ref,
            }.items()
            if value
        }

        def _get_best_config() -> Any:
            try:
                response = requests.get(  # nosec B113 - timeout is provided
                    url,
                    params=params or None,
                    headers=headers,
                    timeout=min(self.timeout, 30.0),
                )
            except (
                requests.exceptions.ConnectionError,
                requests.exceptions.Timeout,
                ConnectionError,
                TimeoutError,
            ) as exc:
                raise _SyncBackendTransientError(
                    f"Best-config fetch transport failed: {exc}"
                ) from exc
            _raise_for_transient_backend_response(response, "Best-config fetch")
            return response

        response = _run_sync_backend_request_with_retry(
            "Best-config fetch",
            _get_best_config,
        )

        if response.status_code in {304, 404}:
            return None
        if response.status_code >= 400:
            raise CloudServiceError(
                f"Best-config fetch rejected with HTTP {response.status_code}: "
                f"{response.text[:200]}"
            )
        return self._extract_backend_data(response)

    def upload_example_features(
        self,
        experiment_run_id: str,
        feature_kind: str,
        features: list[dict[str, Any]] | dict[str, Any],
    ) -> bool:
        """Upload per-run example features for backend-side analytics."""
        self._raise_if_backend_egress_disabled("upload example features")
        if not experiment_run_id or not feature_kind or not features:
            return False

        if feature_kind not in _ALLOWED_EXAMPLE_FEATURE_KINDS:
            logger.debug(
                "Skipping example feature upload for unsupported feature kind: %s",
                feature_kind,
            )
            return False

        try:
            import requests
        except ImportError:  # pragma: no cover - requests is a required dependency
            logger.debug(
                "Skipping example feature upload because requests is unavailable"
            )
            return False

        # B4 ROUND 5: Fail closed. If auth header generation raises
        # (``AuthenticationError`` from a backend-rejected key, or
        # ``CloudServiceError`` from an unexpected failure), do NOT issue an
        # unauthenticated POST. Short-circuit and return False -- the
        # request must never go out without auth.
        try:
            headers = self._get_sync_auth_headers(target="backend")
        except AuthenticationError as exc:
            logger.debug(
                "Skipping example feature upload for run %s: auth rejected (%s)",
                experiment_run_id,
                exc,
            )
            return False
        except CloudServiceError as exc:
            logger.debug(
                "Skipping example feature upload for run %s: header build failed (%s)",
                experiment_run_id,
                exc,
            )
            return False
        url = self._build_feature_upload_url(experiment_run_id)
        if not url:
            return False
        payload = {"feature_kind": feature_kind, "features": features}

        def _post_example_features() -> Any:
            try:
                response = requests.post(  # nosec B113 - timeout is provided
                    url,
                    json=payload,
                    headers=headers,
                    timeout=min(self.timeout, 30.0),
                )
            except (
                requests.exceptions.ConnectionError,
                requests.exceptions.Timeout,
                ConnectionError,
                TimeoutError,
            ) as exc:
                raise _SyncBackendTransientError(
                    f"Example feature upload transport failed: {exc}"
                ) from exc
            _raise_for_transient_backend_response(response, "Example feature upload")
            return response

        try:
            response = _run_sync_backend_request_with_retry(
                "Example feature upload",
                _post_example_features,
            )
        except Exception as exc:
            logger.debug(
                "Example feature upload failed for run %s: %s",
                experiment_run_id,
                exc,
            )
            return False

        if response.status_code >= 400:
            logger.debug(
                "Example feature upload rejected for run %s: HTTP %s %s",
                experiment_run_id,
                response.status_code,
                response.text[:200],
            )
            return False

        return True

    # Trial Operations
    async def register_trial_start(
        self,
        session_id: str,
        trial_id: str,
        config: dict[str, Any],
    ) -> bool | None:
        """Register a trial start with the backend.
        Delegates to trial_operations module.

        Returns:
            True if registration succeeded, False if it failed,
            None if the operation was skipped (e.g. offline mode).
        """
        return cast(
            bool | None,
            await self._trial_ops.register_trial_start(session_id, trial_id, config),
        )

    def register_trial_start_sync(
        self,
        session_id: str,
        trial_id: str,
        config: dict[str, Any],
    ) -> bool | None:
        """Synchronous wrapper for register_trial_start.
        Delegates to trial_operations module.

        Returns:
            True if registration succeeded, False if it failed,
            None if the operation was skipped (e.g. offline mode).
        """
        return cast(
            bool | None,
            self._trial_ops.register_trial_start_sync(session_id, trial_id, config),
        )

    async def request_trial_slot(self, session_id: str) -> TrialSlotResult:
        """Acquire a backend-minted trial id (configuration_run) for a session.
        Delegates to trial_operations module.

        Returns:
            A neutral slot result: acquired backend trial id, explicit
            optimization-complete signal, or unavailable/error.
        """
        return await self._trial_ops.request_trial_slot(session_id)

    def _generate_trial_id(
        self,
        session_id: str,
        config: dict[str, Any],
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Generate deterministic trial IDs compatible with legacy behaviour."""

        if metadata and isinstance(metadata.get("trial_id"), str):
            return cast(str, metadata["trial_id"])

        from traigent.utils.hashing import generate_trial_hash

        return cast(str, generate_trial_hash(session_id, config, ""))

    async def _submit_trial_result_via_session(
        self,
        session_id: str,
        trial_id: str,
        config: dict[str, Any],
        metrics: dict[str, float],
        status: str,
        error_message: str | None = None,
        execution_mode: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> bool | None | TrialSubmissionResult:
        """Submit trial results via the Traigent session endpoint.
        Delegates to trial_operations module.

        Returns:
            True if submission succeeded, False if it failed,
            TrialSubmissionResult for permanent backend rejections, or None if
            the operation was skipped (e.g. offline mode).
        """
        if metadata is None:
            return cast(
                bool | None | TrialSubmissionResult,
                await self._trial_ops.submit_trial_result_via_session(
                    session_id,
                    trial_id,
                    config,
                    metrics,
                    status,
                    error_message,
                    execution_mode,
                ),
            )
        return cast(
            bool | None | TrialSubmissionResult,
            await self._trial_ops.submit_trial_result_via_session(
                session_id,
                trial_id,
                config,
                metrics,
                status,
                error_message,
                execution_mode,
                metadata=metadata,
            ),
        )

    async def _submit_summary_stats(
        self,
        session_id: str,
        trial_id: str,
        config: dict[str, Any],
        summary_stats: dict[str, Any],
        status: str = "completed",
    ) -> bool | None:
        """Submit summary statistics for privacy-preserving mode.
        Delegates to trial_operations module.

        Returns:
            True if submission succeeded, False if it failed,
            None if the operation was skipped (e.g. offline mode).
        """
        return cast(
            bool | None,
            await self._trial_ops.submit_summary_stats(
                session_id, trial_id, config, summary_stats, status
            ),
        )

    async def update_trial_weighted_scores(
        self,
        trial_id: str,
        weighted_score: float,
        normalization_info: dict[str, Any] | None = None,
        objective_weights: dict[str, float] | None = None,
    ) -> bool | None:
        """Update configuration run with weighted multi-objective scores.
        Delegates to trial_operations module.

        Returns:
            True if update succeeded, False if it failed,
            None if the operation was skipped (e.g. offline mode).
        """
        return cast(
            bool | None,
            await self._trial_ops.update_trial_weighted_scores(
                trial_id, weighted_score, normalization_info, objective_weights
            ),
        )

    def submit_result(
        self,
        session_id: str,
        config: dict[str, Any],
        score: float | None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Compatibility shim for legacy synchronous result submission."""

        logger.debug("submit_result called for session=%s score=%s", session_id, score)

        if self.enable_fallback and self.local_storage:
            try:
                self.local_storage.add_trial_result(
                    session_id=session_id,
                    config=config,
                    score=score,
                    metadata=metadata or {},
                )
            except Exception as exc:
                # #1279: do not silently drop the local record — on connected
                # runs this is the only durable trace if the remote submit fails.
                logger.warning(
                    "Local storage trial result persistence failed for session %s: %s",
                    session_id,
                    exc,
                )

        session = self._active_sessions.get(session_id)
        if session:
            session.completed_trials += 1
            session.updated_at = datetime.now(UTC)

    # API Operations
    def _map_to_backend_status(
        self, status: str, *, endpoint: str = "experiment_run"
    ) -> str:
        """Map Traigent status values to backend-expected values.
        Delegates to api_operations module."""
        return cast(str, self._api_ops.map_to_backend_status(status, endpoint=endpoint))

    def _normalize_execution_mode(self, execution_mode: str | None) -> str:
        """Translate SDK execution modes to backend wire values.

        Delegates to the canonical resolution in ``traigent.config.types``
        (issue #1393 dedup): removed and fail-closed legacy selectors
        (``edge_analytics``, ``privacy``, ``cloud``) raise instead of being
        silently remapped, and unknown selectors raise instead of silently
        defaulting.
        """
        # Local import: avoid a module-level cloud -> config-resolution
        # dependency for a single compatibility shim.
        from traigent.config.types import validate_execution_mode

        return validate_execution_mode(execution_mode).value

    def _sanitize_error_message(self, error_message: str | None) -> str | None:
        """Sanitize error message for transmission.
        Delegates to api_operations module."""
        return cast(str | None, self._api_ops.sanitize_error_message(error_message))

    async def _create_traigent_session_via_api(
        self, session_request: SessionCreationRequest
    ) -> TraigentSessionApiResult:
        """Create a Traigent optimization session using the new session endpoints.

        Returns the typed session-CREATE result (a 3-tuple of session /
        experiment / run ids, plus owning-context and warm_start_transfer
        attributes — issue #1683 Bug B: the warm-start decision arrives at
        CREATE time and must not be narrowed away here).
        Delegates to api_operations module."""
        return cast(
            TraigentSessionApiResult,
            await self._api_ops.create_traigent_session_via_api(session_request),
        )

    async def _update_config_run_status(
        self,
        config_run_id: str,
        status: str,
        error_message: str | None = None,
    ) -> bool:
        """Update configuration run status in the backend.
        Delegates to api_operations module."""
        return cast(
            bool,
            await self._api_ops.update_config_run_status(
                config_run_id, status, error_message=error_message
            ),
        )

    async def _report_intermediate_progress(
        self, payload: dict[str, Any]
    ) -> dict[str, Any]:
        """Report intermediate trial progress for smart pruning."""
        return cast(
            dict[str, Any],
            await self._api_ops.report_intermediate_progress(payload),
        )

    async def _update_config_run_measures(
        self,
        config_run_id: str,
        metrics: dict[str, float],
        execution_time: float | None = None,
    ) -> bool:
        """Update configuration run measures in the backend.
        Delegates to api_operations module."""
        return cast(
            bool,
            await self._api_ops.update_config_run_measures(
                config_run_id, metrics, execution_time
            ),
        )

    async def _update_experiment_run_status_on_completion(
        self, experiment_run_id: str, status: str
    ) -> None:
        """Update experiment run status when session is finalized.
        Delegates to api_operations module."""
        await self._api_ops.update_experiment_run_status_on_completion(
            experiment_run_id, status
        )

    # Cloud API methods (remote execution not implemented; fail closed in api_operations)
    async def _create_cloud_session(
        self, request: SessionCreationRequest
    ) -> SessionCreationResponse:
        """Create cloud session for optimization.
        Delegates to api_operations module."""
        return await self._api_ops.create_cloud_session(request)

    async def _get_cloud_trial_suggestion(
        self, request: NextTrialRequest
    ) -> NextTrialResponse:
        """Get next trial suggestion from cloud optimizer.
        Delegates to api_operations module."""
        return await self._api_ops.get_cloud_trial_suggestion(request)

    async def _submit_cloud_trial_results(
        self, submission: TrialResultSubmission
    ) -> None:
        """Reserved cloud result path; fails closed in api_operations.
        Delegates to api_operations module."""
        await self._api_ops.submit_cloud_trial_results(submission)

    async def _submit_agent_optimization(
        self, request: AgentOptimizationRequest
    ) -> AgentOptimizationResponse:
        """Reserved cloud agent optimization path; fails closed in api_operations.
        Delegates to api_operations module."""
        return await self._api_ops.submit_agent_optimization(request)

    async def _execute_cloud_agent(
        self, request: AgentExecutionRequest
    ) -> AgentExecutionResponse:
        """Reserved cloud agent execution path; fails closed in api_operations.
        Delegates to api_operations module."""
        return await self._api_ops.execute_cloud_agent(request)


_backend_client: BackendIntegratedClient | None = None


def get_backend_client(**kwargs) -> BackendIntegratedClient:
    """Return a shared ``BackendIntegratedClient`` instance.

    The first invocation creates a client; subsequent calls reuse it unless
    tests reset the module-level `_backend_client`.
    """

    global _backend_client
    if _backend_client is None:
        _backend_client = BackendIntegratedClient(**kwargs)
    return _backend_client
