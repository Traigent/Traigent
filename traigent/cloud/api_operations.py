"""Backend API operations for Traigent Cloud Client.

This module contains methods for interacting with the backend API endpoints,
including session creation, status updates, and configuration run management.
"""

# Traceability: CONC-Layer-Infra CONC-Quality-Reliability FUNC-CLOUD-HYBRID FUNC-AGENTS REQ-CLOUD-009 REQ-AGNT-013

import time
from typing import TYPE_CHECKING, Any, NoReturn, cast
from urllib.parse import urlparse

from traigent.cloud.auth import AuthenticationError
from traigent.cloud.client import (
    CloudRemoteExecutionUnavailableError,
    CloudServiceError,
    SessionContractError,
    raise_if_cloud_egress_disabled,
)
from traigent.cloud.governance import promotion_policy_to_wire, tvl_governance_to_wire
from traigent.cloud.models import (
    AgentExecutionRequest,
    AgentExecutionResponse,
    AgentOptimizationRequest,
    AgentOptimizationResponse,
    NextTrialRequest,
    NextTrialResponse,
    SessionCreationRequest,
    SessionCreationResponse,
    TrialResultSubmission,
)
from traigent.config.backend_config import BackendConfig
from traigent.core.session_types import (
    SessionCreationFailureDetail,
    SessionCreationHTTPError,
)
from traigent.utils.env_config import is_backend_offline
from traigent.utils.exceptions import MetricExtractionError
from traigent.utils.logging import get_logger
from traigent.utils.validation import validate_numeric_metric

# Optional aiohttp dependency handling
try:
    import aiohttp

    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False

    class _AiohttpPlaceholder:
        """Minimal placeholder to satisfy type references when aiohttp missing."""

        class ClientConnectorError(RuntimeError):
            pass

        class ClientError(RuntimeError):
            pass

        _AIOHTTP_MISSING_MSG = "aiohttp is not installed"

        class ClientTimeout:  # pragma: no cover - placeholder
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                raise RuntimeError(_AiohttpPlaceholder._AIOHTTP_MISSING_MSG)

        class TCPConnector:  # pragma: no cover - placeholder
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                raise RuntimeError(_AiohttpPlaceholder._AIOHTTP_MISSING_MSG)

        class ClientSession:  # pragma: no cover - placeholder
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                raise RuntimeError(_AiohttpPlaceholder._AIOHTTP_MISSING_MSG)

    aiohttp = _AiohttpPlaceholder()

# Content-Type header for JSON requests
_JSON_CONTENT_TYPE = "application/json"

if TYPE_CHECKING:
    from traigent.cloud.backend_client import BackendIntegratedClient

logger = get_logger(__name__)


class TraigentSessionApiResult(tuple):
    """Three-ID session creation result with optional owning context attrs."""

    project_id: str | None
    tenant_id: str | None

    def __new__(
        cls,
        session_id: str,
        experiment_id: str,
        experiment_run_id: str,
        *,
        project_id: str | None = None,
        tenant_id: str | None = None,
    ):
        obj = super().__new__(cls, (session_id, experiment_id, experiment_run_id))
        obj.project_id = project_id
        obj.tenant_id = tenant_id
        return obj


# ---------------------------------------------------------------------------
# Canonical run / configuration-run wire status vocabulary (single source of
# truth — issue #1302).
#
# The backend's ``ExperimentRunStatus`` / ``ConfigurationRunStatus`` enums use
# the run-lifecycle vocabulary (RUNNING / PENDING / NOT_STARTED / ...), NOT the
# session-lifecycle vocabulary (ACTIVE / CREATED / PRUNED). Earlier code mapped
# in-flight SDK states onto ACTIVE / CREATED, which the backend 400-rejects, so
# an SDK-driven run could never persist its in-flight state server-side.
#
# Keep the mapping, the validation gate, and the SDK enums all derived from the
# constants below so they cannot drift apart again.
# ---------------------------------------------------------------------------

# Backend ExperimentRunStatus members (run-lifecycle). PRUNED / PENDING / PAUSED
# specifics differ between experiment-runs and config-runs; see below.
EXPERIMENT_RUN_WIRE_STATUSES: frozenset[str] = frozenset(
    {
        "NOT_STARTED",
        "PENDING",
        "PAUSED",
        "RUNNING",
        "FAILED",
        "COMPLETED",
        "CANCELLED",
        "UNKNOWN",
        "PARTIALLY_DELETED",
    }
)

# Backend ConfigurationRunStatus members (run-lifecycle). Note: config-runs
# accept PRUNED (early-stopping success) but lack PENDING / PAUSED /
# PARTIALLY_DELETED that experiment-runs carry.
CONFIGURATION_RUN_WIRE_STATUSES: frozenset[str] = frozenset(
    {
        "NOT_STARTED",
        "RUNNING",
        "COMPLETED",
        "FAILED",
        "UNKNOWN",
        "CANCELLED",
        "PRUNED",
    }
)

# Neutral fallback emitted/validated when an input status is not recognized.
# Both backend enums carry UNKNOWN, so it is always a safe wire value and never
# fakes success on a failed/running run (issue #1302, AC3).
UNKNOWN_WIRE_STATUS = "UNKNOWN"

# Canonical SDK-lowercase -> run-lifecycle wire mapping. Keys are the SDK's
# lowercase status tokens (and a couple of synonyms); values are the backend
# run-lifecycle members. ``pruned`` is intentionally absent here and handled
# per-endpoint by ``map_to_backend_status`` because experiment-runs do not
# accept PRUNED while config-runs do.
_RUN_STATUS_WIRE_MAP: dict[str, str] = {
    "not_started": "NOT_STARTED",
    "pending": "PENDING",
    "paused": "PAUSED",
    "running": "RUNNING",
    "in_progress": "RUNNING",
    "completed": "COMPLETED",
    "failed": "FAILED",
    "cancelled": "CANCELLED",
    "canceled": "CANCELLED",
    "unknown": "UNKNOWN",
    "partially_deleted": "PARTIALLY_DELETED",
}


def map_status_to_wire(status: str, *, endpoint: str = "experiment_run") -> str:
    """Pure mapper: SDK status token -> backend run-lifecycle wire status.

    Single source of truth for the run / configuration-run status vocabulary
    (issue #1302). The class method ``ApiOperations.map_to_backend_status`` and
    any other caller that needs the wire vocab delegate here so the mapping and
    the validation gate can never drift apart.

    See ``ApiOperations.map_to_backend_status`` for the per-endpoint ``PRUNED``
    handling and fallback semantics.
    """
    token = (status or "").strip().lower()
    valid = (
        CONFIGURATION_RUN_WIRE_STATUSES
        if endpoint == "config_run"
        else EXPERIMENT_RUN_WIRE_STATUSES
    )

    # PRUNED is endpoint-specific: config-runs keep it; the experiment-run enum
    # has no PRUNED member. A pruned run is NOT a successful completion, so fold
    # it to the neutral UNKNOWN rather than claiming COMPLETED — emitting
    # COMPLETED here would be fake completion (no-fake-completion rule).
    if token == "pruned":
        mapped = "PRUNED" if endpoint == "config_run" else UNKNOWN_WIRE_STATUS
    else:
        mapped = _RUN_STATUS_WIRE_MAP.get(token, "")
        if not mapped:
            # Allow already-canonical wire values (any case) to pass through.
            upper = (status or "").strip().upper()
            mapped = upper if upper in valid else ""

    if mapped not in valid:
        logger.warning(
            "Unrecognized %s status %r (resolved to %r); "
            "sending neutral %r so the backend is not given a wrong status",
            endpoint,
            status,
            mapped or None,
            UNKNOWN_WIRE_STATUS,
        )
        mapped = UNKNOWN_WIRE_STATUS

    return mapped


def _typed_configuration_space(space: Any) -> Any:
    """Normalize shorthand configuration space to the typed wire contract.

    Normalization rules applied in order:
    - list/tuple → {"type": "categorical", "choices": [...]}
    - dict with "type" already set → pass through unchanged
    - dict with "low"/"high" but no "type" → infer "int" (both int) or "float"
    - scalar (bool/int/float/str) → {"type": "categorical", "choices": [value]}
    - anything else → pass through for the backend to reject with a clear error
    """
    if not isinstance(space, dict):
        return space
    normalized: dict[str, Any] = {}
    for name, entry in space.items():
        if isinstance(entry, (list, tuple)):
            normalized[name] = {"type": "categorical", "choices": list(entry)}
        elif isinstance(entry, dict):
            if "type" not in entry and ("low" in entry or "high" in entry):
                low, high = entry.get("low"), entry.get("high")
                inferred = (
                    "int"
                    if isinstance(low, int)
                    and not isinstance(low, bool)
                    and isinstance(high, int)
                    and not isinstance(high, bool)
                    else "float"
                )
                normalized[name] = {**entry, "type": inferred}
            else:
                normalized[name] = entry
        elif isinstance(entry, (bool, int, float, str)):
            # scalar/fixed value -> single-choice categorical (must be wrapped,
            # not passed raw to the backend, or cloud session-create 400s)
            normalized[name] = {"type": "categorical", "choices": [entry]}
        else:
            normalized[name] = entry
    return normalized


class ApiOperations:
    """Handles backend API operations."""

    def __init__(self, client: "BackendIntegratedClient"):
        """Initialize API operations handler.

        Args:
            client: Parent BackendIntegratedClient instance
        """
        self.client = client

    def _raise_if_backend_egress_disabled(self, operation: str) -> None:
        """Fail closed before any backend HTTP request."""

        raise_if_cloud_egress_disabled(
            operation,
            no_egress=getattr(self.client, "no_egress", False),
        )

    def validate_and_sanitize_url(self, url: str) -> str:
        """Validate and sanitize URL to prevent injection attacks.

        Args:
            url: URL to validate

        Returns:
            Sanitized URL

        Raises:
            ValueError: If URL is invalid or uses unsafe scheme
        """
        if not url:
            raise ValueError("URL cannot be empty") from None

        # Strip whitespace to prevent CVE-2023-24329 bypass on older Python
        url = url.strip()

        # Parse URL
        parsed = urlparse(url)

        # Validate scheme
        allowed_schemes = {"http", "https"}
        if parsed.scheme not in allowed_schemes:
            raise ValueError(
                f"URL scheme must be one of {allowed_schemes}, got {parsed.scheme}"
            )

        # Validate host
        if not parsed.netloc:
            raise ValueError("URL must have a valid host")

        # Check for localhost/private IPs only in development
        if parsed.hostname:
            hostname = parsed.hostname.lower()
            # Allow localhost for development but provide helpful context
            if hostname in ["localhost", "127.0.0.1", "::1"]:
                logger.info(f"🔧 Using local backend URL: {url}")
                logger.info("💡 Traigent is configured for local development mode")
            # Block other private IP ranges
            elif (
                hostname.startswith("192.168.")
                or hostname.startswith("10.")
                or hostname.startswith("172.")
            ):
                logger.warning(f"Using private IP address: {hostname}")

        # Reconstruct clean URL without fragments or unnecessary components
        clean_url = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
        if parsed.query:
            clean_url += f"?{parsed.query}"

        return clean_url.rstrip("/")

    def map_to_backend_status(
        self, status: str, *, endpoint: str = "experiment_run"
    ) -> str:
        """Map Traigent SDK status values to the backend run-lifecycle wire vocab.

        The SDK uses lowercase tokens (``pending``, ``running``, ``completed``,
        ``not_started`` …). The backend's experiment-run and configuration-run
        status endpoints expect the run-lifecycle enum (``RUNNING``, ``PENDING``,
        ``NOT_STARTED``, ``COMPLETED``, ``FAILED``, ``CANCELLED`` …), NOT the
        session-lifecycle vocab (``ACTIVE`` / ``CREATED`` / ``PRUNED``). Mapping
        in-flight states onto ``ACTIVE`` / ``CREATED`` is exactly the bug from
        issue #1302: the backend 400-rejects those on the run/config PUT path,
        so in-flight states could never persist.

        ``PRUNED`` is handled per-endpoint:

        * ``endpoint="config_run"`` keeps ``pruned`` -> ``PRUNED`` (config-runs
          accept early-stopping as a distinct success-ish outcome).
        * ``endpoint="experiment_run"`` (the default) maps ``pruned`` ->
          ``UNKNOWN`` because the backend ``ExperimentRunStatus`` enum has no
          ``PRUNED`` member. A pruned run is NOT a successful completion, so it
          folds to the neutral ``UNKNOWN`` rather than claiming ``COMPLETED``
          (claiming completion would be fake completion); ``UNKNOWN`` is a valid
          member, so this never produces a 400.

        Unrecognized inputs map to ``UNKNOWN`` (a member of both backend enums),
        never to ``FAILED`` — we must not assert failure on a status we simply
        did not understand.

        Args:
            status: SDK status token (case-insensitive).
            endpoint: ``"experiment_run"`` (default) or ``"config_run"`` —
                selects the per-endpoint ``PRUNED`` handling and the validation
                vocabulary.

        Returns:
            A backend run-lifecycle status member valid for the chosen endpoint.
        """
        return map_status_to_wire(status, endpoint=endpoint)

    def sanitize_error_message(self, error_message: str | None) -> str | None:
        """Sanitize error message for transmission.

        Args:
            error_message: Raw error message

        Returns:
            Sanitized error message or None
        """
        if not error_message:
            return None

        # Remove sensitive information patterns
        import re

        # Remove potential secrets/tokens
        sanitized = re.sub(
            r"(api[_-]?key|token|secret|password)[\s=:]+[\w-]+",
            "[REDACTED]",
            error_message,
            flags=re.IGNORECASE,
        )

        # Remove file paths that might contain user info
        sanitized = re.sub(r"/home/[\w/.-]+", "[PATH]", sanitized)
        sanitized = re.sub(r"/Users/[\w/.-]+", "[PATH]", sanitized)

        # Truncate if too long
        max_length = 1000
        if len(sanitized) > max_length:
            sanitized = sanitized[:max_length] + "... [truncated]"

        return sanitized

    async def create_traigent_session_via_api(
        self, session_request: SessionCreationRequest
    ) -> tuple[str, str, str]:
        """Create a Traigent optimization session using the new session endpoints.

        Returns:
            Tuple of (session_id, experiment_id, experiment_run_id)
        """
        # Skip cloud session creation if backend is offline
        # Note: is_backend_offline() returns True if TRAIGENT_OFFLINE_MODE=true
        if is_backend_offline():
            logger.debug("Backend offline: using local session IDs")
            return (
                f"mock_session_{int(time.time())}",
                f"mock_exp_{int(time.time())}",
                f"mock_run_{int(time.time())}",
            )
        raise_if_cloud_egress_disabled(
            "create session",
            no_egress=getattr(self.client, "no_egress", False),
        )

        if not AIOHTTP_AVAILABLE:
            raise CloudServiceError(
                "aiohttp is required for backend session creation. Install "
                "traigent[hybrid] or set TRAIGENT_OFFLINE_MODE=true for "
                "explicit offline local-only mode."
            )

        try:
            max_trials_value = self._resolve_max_trials(session_request)
            session_payload = self._build_session_payload(
                session_request, max_trials_value
            )
            connector = self._build_connector()
            headers = await self.client.auth_manager.augment_headers(
                {"Content-Type": _JSON_CONTENT_TYPE}
            )
            try:
                return await self._post_session_creation(
                    session_payload, headers, connector
                )
            except CloudServiceError:
                # auto-contract fallback: a failed TYPED create may retry the
                # legacy shape ONCE — and only for NON-governed sessions
                # (governed sessions must fail loudly; the legacy contract
                # cannot carry strict mode and silently degrading would
                # launder it — RFC 0001 P7).
                if self._session_contract() != "auto" or self._is_governed_request(
                    session_request
                ):
                    raise
                logger.warning(
                    "Typed session create failed; retrying once with the "
                    "legacy contract (non-governed session, "
                    "TRAIGENT_SESSION_CONTRACT=auto)"
                )
                legacy_payload = self._build_legacy_session_payload(
                    session_request, max_trials_value
                )
                return await self._post_session_creation(
                    legacy_payload, headers, connector
                )
        except SessionContractError:
            # Review round 3: contract failures must reach the caller AS
            # THEMSELVES — the generic wrap below would demote them to a
            # plain CloudServiceError and the local-fallback layer would
            # absorb them.
            raise
        except AuthenticationError:
            # #1278: a 401/403 on session creation must reach the caller AS
            # ITSELF. _handle_session_error raises AuthenticationError with the
            # structured 403 detail attached; the generic wrap below would
            # demote it to a plain CloudServiceError (dropping the detail), so
            # session_operations would classify it SESSION_FAILED →
            # BACKEND_UNREACHABLE ("check your network/URL") instead of an
            # auth/permission error (MISSING_PERMISSION / INVALID_OR_REVOKED_KEY).
            raise
        except aiohttp.ClientConnectorError as e:
            self._handle_connector_error(e)
        except aiohttp.ClientError as e:
            self._handle_client_error(e)
        except Exception as e:
            self._handle_generic_session_exception(e)
        raise CloudServiceError("Unexpected session creation failure")

    def _resolve_max_trials(self, session_request: SessionCreationRequest) -> int:
        """Resolve max trials with logging when falling back to default."""

        value = session_request.max_trials
        if value is None:
            logger.debug("max_trials is None in session_request, using default 10")
            return 10
        return value

    @staticmethod
    def _session_contract() -> str:
        """Resolve the session-create contract (TRAIGENT_SESSION_CONTRACT).

        auto (default): typed create; for NON-governed sessions a failed
        typed create may fall back to the legacy shape once.
        typed: typed create only — failures raise.
        legacy: legacy shape — REFUSED for governed/strict sessions
        (falling back would silently launder strict mode; RFC 0001 P7).
        """
        import os

        contract = os.getenv("TRAIGENT_SESSION_CONTRACT", "auto").strip().lower()
        if contract not in {"auto", "typed", "legacy"}:
            raise SessionContractError(
                "TRAIGENT_SESSION_CONTRACT must be one of auto|typed|legacy; "
                f"got {contract!r}"
            )
        return contract

    @staticmethod
    def _is_governed_request(session_request: SessionCreationRequest) -> bool:
        """A session is governed when it declares a promotion policy or any
        TVL governance — governed sessions must NEVER take the legacy path
        (the legacy contract cannot carry them; dropping them silently is
        the Phase 7 laundering bug all over again)."""
        return bool(session_request.promotion_policy or session_request.tvl_governance)

    def _build_session_payload(
        self, session_request: SessionCreationRequest, max_trials: int
    ) -> dict[str, Any]:
        """Build the payload sent to the cloud session creation endpoint.

        Contract-gated (Phase 8): the TYPED shape is the default — it is the
        only shape the backend's governed path accepts (promotion_policy,
        tvl_governance, experiment records for FE hydration). The legacy
        shape survives behind TRAIGENT_SESSION_CONTRACT=legacy for
        non-governed compatibility only.
        """
        contract = self._session_contract()
        if contract == "legacy":
            if self._is_governed_request(session_request):
                raise SessionContractError(
                    "strict/governed sessions require the typed session "
                    "contract; TRAIGENT_SESSION_CONTRACT=legacy cannot carry "
                    "promotion_policy/tvl_governance (refusing to launder "
                    "strict mode)"
                )
            return self._build_legacy_session_payload(session_request, max_trials)
        return self._build_typed_session_payload(session_request, max_trials)

    def _build_typed_session_payload(
        self, session_request: SessionCreationRequest, max_trials: int
    ) -> dict[str, Any]:
        """The typed interactive-session contract (TraigentSchema sdk_tuning):
        function_name + configuration_space + objectives at top level select
        the backend's typed path, which preserves governance and creates the
        experiment records the FE hydrates."""
        metadata = dict(session_request.metadata or {})
        evaluation_set = metadata.get("evaluation_set", "default")
        dataset_metadata = dict(session_request.dataset_metadata or {})
        # The typed path requires a positive dataset size.
        size = dataset_metadata.get("size")
        if not isinstance(size, int) or size <= 0:
            dataset_metadata["size"] = 1

        payload: dict[str, Any] = {
            "function_name": session_request.function_name,
            "configuration_space": _typed_configuration_space(
                session_request.configuration_space
            ),
            "objectives": list(session_request.objectives or []),
            "dataset_metadata": dataset_metadata,
            "max_trials": max_trials,
            "metadata": {
                "function_name": session_request.function_name,
                "evaluation_set": evaluation_set,
                **metadata,
            },
        }
        # CHOKE POINT (review round 2): the allowlist serializer runs on the
        # actual request body, not only on the orchestrator path — a direct
        # SessionCreationRequest caller must not be able to serialize
        # unknown/value-shaped policy data (RFC 0001 P8 content freedom).
        wire_policy = promotion_policy_to_wire(session_request.promotion_policy)
        if wire_policy:
            payload["promotion_policy"] = wire_policy
        wire_governance = tvl_governance_to_wire(session_request.tvl_governance)
        if wire_governance:
            payload["tvl_governance"] = wire_governance
        if session_request.smart_pruning is not None:
            payload["smart_pruning"] = dict(session_request.smart_pruning)
        return payload

    def _build_legacy_session_payload(
        self, session_request: SessionCreationRequest, max_trials: int
    ) -> dict[str, Any]:
        """The pre-Phase-8 legacy shape (problem_statement/search_space) —
        non-governed compatibility only."""
        metadata = session_request.metadata or {}
        evaluation_set = metadata.get("evaluation_set", "default")

        return {
            "problem_statement": session_request.function_name,
            "dataset": {
                "examples": [],  # Privacy mode - no actual data sent
                "metadata": session_request.dataset_metadata,
            },
            "search_space": session_request.configuration_space,
            "optimization_config": {
                "algorithm": "grid",
                "max_trials": max_trials,
                "optimization_goal": (
                    session_request.objectives[0]
                    if session_request.objectives
                    else "maximize"
                ),
            },
            "metadata": {
                "function_name": session_request.function_name,
                "evaluation_set": evaluation_set,
                **metadata,
            },
        }

    def _build_connector(self) -> Any | None:
        """Create an aiohttp connector when custom transport settings are required."""

        return None

    async def _post_session_creation(
        self,
        payload: dict[str, Any],
        headers: dict[str, str],
        connector: Any | None,
    ) -> tuple[str, str, str]:
        """Send the session creation request and handle responses."""

        self._raise_if_backend_egress_disabled("create session")
        async with cast(Any, aiohttp).ClientSession(connector=connector) as session:
            api_base = (
                self.client.backend_config.api_base_url
                or BackendConfig.get_backend_api_url()
            )
            url = f"{api_base}/sessions"
            timeout = cast(Any, aiohttp).ClientTimeout(total=30)

            async with session.post(
                url,
                json=payload,
                headers=headers,
                timeout=timeout,
            ) as response:
                if response.status == 201:
                    return await self._parse_session_response(response)

                error_msg = await response.text()
                self._handle_session_error(response.status, error_msg)

        raise CloudServiceError("Unexpected session creation failure")

    async def _parse_session_response(self, response: Any) -> TraigentSessionApiResult:
        """Parse the JSON success payload returned by the session endpoint."""

        result = await response.json()
        metadata = result.get("metadata", {})
        if not isinstance(metadata, dict):
            metadata = {}
        session_id = result.get("session_id")
        experiment_id = metadata.get("experiment_id", session_id)
        experiment_run_id = metadata.get("experiment_run_id", session_id)
        project_id = self._optional_context_id(
            result.get("project_id") or metadata.get("project_id")
        )
        tenant_id = self._optional_context_id(
            result.get("tenant_id") or metadata.get("tenant_id")
        )

        logger.info(
            f"✅ Created Traigent session: {session_id} "
            f"(exp: {experiment_id}, run: {experiment_run_id})"
        )
        return TraigentSessionApiResult(
            session_id,
            experiment_id,
            experiment_run_id,
            project_id=project_id,
            tenant_id=tenant_id,
        )

    @staticmethod
    def _optional_context_id(value: Any) -> str | None:
        if value is None:
            return None
        normalized = str(value).strip()
        return normalized or None

    def _handle_session_error(self, status_code: int, error_msg: str) -> None:
        """Raise structured exceptions for non-success HTTP responses.

        All logging is DEBUG — user-facing warnings are emitted by
        ``BackendSessionManager.handle_session_creation_result()``.
        """
        detail = SessionCreationFailureDetail.from_http_response(status_code, error_msg)
        structured_cause = SessionCreationHTTPError(detail)
        if status_code in (401, 403):
            logger.debug("Backend auth failed: %s", status_code)
            auth_exc = AuthenticationError(
                f"Authentication failed ({status_code}): {detail.one_line_summary()}"
            )
            cast(Any, auth_exc).session_creation_failure = detail
            raise auth_exc from structured_cause

        if status_code in (500, 502, 503, 504):
            logger.debug("Backend HTTP error: %s", status_code)
            service_exc = CloudServiceError(f"Backend HTTP {status_code}")
            cast(Any, service_exc).session_creation_failure = detail
            raise service_exc from structured_cause

        logger.debug("Backend session error: %s - %s", status_code, error_msg[:200])
        service_exc = CloudServiceError(
            f"Session creation failed: HTTP {status_code}: {detail.one_line_summary()}"
        )
        cast(Any, service_exc).session_creation_failure = detail
        raise service_exc from structured_cause

    def _handle_connector_error(self, error: aiohttp.ClientConnectorError) -> None:
        """Handle aiohttp connector errors — DEBUG only."""
        logger.debug("Backend connection failed: %s", error)
        raise CloudServiceError("Backend unavailable (connection failed)") from error

    def _handle_client_error(self, error: aiohttp.ClientError) -> None:
        """Handle generic aiohttp client errors — DEBUG only."""
        logger.debug("Network error connecting to backend: %s", error)
        raise CloudServiceError(f"Network error: {error}") from error

    def _handle_generic_session_exception(self, error: Exception) -> None:
        """Handle unexpected exceptions during session creation — DEBUG only."""
        logger.debug("Unexpected error creating session: %s", error)
        raise CloudServiceError(f"Session creation failed: {error}") from error

    async def update_config_run_status(self, config_run_id: str, status: str) -> bool:
        """Update configuration run status in the backend.

        Args:
            config_run_id: Configuration run ID (same as trial_id)
            status: Status to set. Accepts either an SDK token (``running``,
                ``completed`` …) or an already-mapped wire value; it is always
                normalized to the configuration-run wire vocab before sending so
                the backend never receives a session-lifecycle value
                (issue #1302).

        Returns:
            True if successful, False otherwise
        """
        self._raise_if_backend_egress_disabled("update configuration run status")
        if not AIOHTTP_AVAILABLE:
            return False

        try:
            # Normalize to the configuration-run wire vocab (preserves PRUNED).
            backend_status = self.map_to_backend_status(status, endpoint="config_run")

            connector = self._build_connector()

            # Prepare headers with API key
            headers = await self.client.auth_manager.augment_headers(
                {"Content-Type": _JSON_CONTENT_TYPE}
            )

            async with cast(Any, aiohttp).ClientSession(connector=connector) as session:
                api_base = (
                    self.client.backend_config.api_base_url
                    or BackendConfig.get_backend_api_url()
                )
                url = f"{api_base}/configuration-runs/{config_run_id}/status"
                status_data = {"status": backend_status}

                async with session.put(
                    url,
                    json=status_data,
                    headers=headers,
                    timeout=cast(Any, aiohttp).ClientTimeout(total=10),
                ) as response:
                    if response.status in [200, 204]:
                        logger.debug(
                            f"Updated configuration run {config_run_id} "
                            f"status to {backend_status}"
                        )
                        return True
                    else:
                        error_msg = await response.text()
                        logger.debug(
                            f"Failed to update config run status: {response.status} - {error_msg[:100]}"
                        )
                        return False

        except Exception as e:
            logger.debug(f"Error updating config run status: {e}")
            return False

    async def update_config_run_measures(
        self,
        config_run_id: str,
        metrics: dict[str, float],
        execution_time: float | None = None,
    ) -> bool:
        """Update configuration run measures in the backend.

        Args:
            config_run_id: Configuration run ID (same as trial_id)
            metrics: Metrics to convert to measures
            execution_time: Optional execution time in seconds

        Returns:
            True if successful, False otherwise
        """
        self._raise_if_backend_egress_disabled("update configuration run measures")
        if not AIOHTTP_AVAILABLE or not metrics:
            return False

        try:
            # Convert metrics to backend measures format per Traigent schema
            # Map Traigent metrics to standard measure IDs
            mapped_metrics = {}

            # Standard Traigent measure mappings with strict validation
            if "accuracy" in metrics and metrics["accuracy"] is not None:
                try:
                    mapped_metrics["accuracy"] = validate_numeric_metric(
                        metrics["accuracy"],
                        field_name="accuracy",
                        trial_id=config_run_id,
                    )
                except MetricExtractionError as e:
                    logger.error(
                        f"Invalid 'accuracy' metric for config run {config_run_id}: {e.message}",
                        extra={
                            "hint": "Ensure your evaluation function returns numeric values for 'accuracy'. "
                            "Check that the value is not None, NaN, or Inf.",
                            "config_run_id": config_run_id,
                            "field": "accuracy",
                            "value": metrics["accuracy"],
                        },
                    )
                    return False

            if "score" in metrics and metrics["score"] is not None:
                try:
                    mapped_metrics["score"] = validate_numeric_metric(
                        metrics["score"],
                        field_name="score",
                        trial_id=config_run_id,
                    )
                except MetricExtractionError as e:
                    logger.error(
                        f"Invalid 'score' metric for config run {config_run_id}: {e.message}",
                        extra={
                            "hint": "Ensure your evaluation function returns numeric values for 'score'. "
                            "Check that the value is not None, NaN, or Inf.",
                            "config_run_id": config_run_id,
                            "field": "score",
                            "value": metrics["score"],
                        },
                    )
                    return False
            elif (
                "accuracy" in metrics and metrics["accuracy"] is not None
            ):  # Use accuracy as score if no explicit score
                mapped_metrics["score"] = mapped_metrics["accuracy"]

            # Include other standard measures if present
            for key in [
                "faithfulness",
                "relevance",
                "latency",
                "cost",
                "context_precision",
                "context_recall",
            ]:
                if key in metrics and metrics[key] is not None:
                    try:
                        mapped_metrics[key] = validate_numeric_metric(
                            metrics[key],
                            field_name=key,
                            trial_id=config_run_id,
                        )
                    except MetricExtractionError as e:
                        logger.error(
                            f"Invalid '{key}' metric for config run {config_run_id}: {e.message}",
                            extra={
                                "hint": f"Ensure your evaluation function returns numeric values for '{key}'. "
                                "Check that the value is not None, NaN, or Inf.",
                                "config_run_id": config_run_id,
                                "field": key,
                                "value": metrics[key],
                            },
                        )
                        return False

            # Include any other metrics as-is, ensuring they are numeric
            for key, value in metrics.items():
                if key not in mapped_metrics and value is not None:
                    try:
                        mapped_metrics[key] = validate_numeric_metric(
                            value,
                            field_name=key,
                            trial_id=config_run_id,
                        )
                    except MetricExtractionError as e:
                        logger.error(
                            f"Invalid '{key}' metric for config run {config_run_id}: {e.message}",
                            extra={
                                "hint": f"Ensure your evaluation function returns numeric values for '{key}'. "
                                "Check that the value is not None, NaN, or Inf. "
                                "Custom metrics must be numeric types (int, float).",
                                "config_run_id": config_run_id,
                                "field": key,
                                "value": value,
                            },
                        )
                        return False

            # Build measures data in schema-compliant array format
            # Per configuration_run_schema.json, measures must be array of MeasureResult objects
            # Type: dict[str, list[dict[str, float]] | None]
            measures_data: dict[str, list[dict[str, float]] | None]
            if mapped_metrics:
                measure_result = dict(mapped_metrics)
                measures_data = {"measures": [measure_result]}
            else:
                # Omit measures if empty (per schema - use null or omit when no results)
                measures_data = {"measures": None}

            # execution_time is not transmitted via the measures endpoint.
            # It is already included in the trial result payload submitted by
            # submit_trial_result_via_session (result_data["execution_time"]).
            if execution_time is not None:
                logger.debug(
                    "execution_time=%s recorded in trial result payload; "
                    "not duplicated in measures update.",
                    execution_time,
                )

            connector = self._build_connector()

            # Prepare headers with API key
            headers = await self.client.auth_manager.augment_headers(
                {"Content-Type": _JSON_CONTENT_TYPE}
            )

            async with cast(Any, aiohttp).ClientSession(connector=connector) as session:
                api_base = (
                    self.client.backend_config.api_base_url
                    or BackendConfig.get_backend_api_url()
                )
                url = f"{api_base}/configuration-runs/{config_run_id}/measures"

                async with session.put(
                    url,
                    json=measures_data,
                    headers=headers,
                    timeout=cast(Any, aiohttp).ClientTimeout(total=10),
                ) as response:
                    if response.status == 200:
                        logger.debug(
                            f"Updated configuration run {config_run_id} with {len(mapped_metrics)} measures"
                        )
                        return True
                    else:
                        error_msg = await response.text()
                        logger.debug(
                            f"Failed to update config run measures: {response.status} - {error_msg[:100]}"
                        )
                        return False

        except Exception as e:
            logger.debug(f"Error updating config run measures: {e}")
            return False

    async def update_experiment_run_status_on_completion(
        self, experiment_run_id: str, status: str
    ) -> None:
        """Update experiment run status when session is finalized.

        Args:
            experiment_run_id: Experiment run ID
            status: Status to set (COMPLETED, FAILED, etc.)
        """
        self._raise_if_backend_egress_disabled("update experiment run status")
        if not AIOHTTP_AVAILABLE:
            logger.debug("aiohttp not available, skipping experiment run status update")
            return

        try:
            # Map SDK status to the experiment-run wire vocab. The mapper is the
            # single source of truth and already guarantees a member of
            # EXPERIMENT_RUN_WIRE_STATUSES (UNKNOWN for anything it cannot
            # resolve). The gate below is a defense-in-depth assertion against
            # the same canonical set so the two can never drift (issue #1302).
            backend_status = self.map_to_backend_status(
                status, endpoint="experiment_run"
            )

            if backend_status not in EXPERIMENT_RUN_WIRE_STATUSES:
                logger.warning(
                    "Status %r mapped to non-canonical %r; sending neutral %r",
                    status,
                    backend_status,
                    UNKNOWN_WIRE_STATUS,
                )
                backend_status = UNKNOWN_WIRE_STATUS

            # Prepare headers with API key
            headers = await self.client.auth_manager.augment_headers(
                {"Content-Type": _JSON_CONTENT_TYPE}
            )

            connector = self._build_connector()

            async with cast(Any, aiohttp).ClientSession(connector=connector) as session:
                api_base = (
                    self.client.backend_config.api_base_url
                    or BackendConfig.get_backend_api_url()
                )
                url = f"{api_base}/experiment-runs/runs/{experiment_run_id}"

                async with session.put(
                    url,
                    json={"status": backend_status},
                    headers=headers,
                    timeout=cast(Any, aiohttp).ClientTimeout(total=10),
                ) as response:
                    if response.status in [200, 204]:
                        logger.info(
                            f"✅ Updated experiment run {experiment_run_id} to status {backend_status}"
                        )
                    else:
                        error_text = await response.text()
                        logger.warning(
                            f"Failed to update experiment run status (HTTP {response.status}): {error_text}"
                        )
        except Exception as e:
            logger.warning(f"Error updating experiment run status: {e}")

    # Cloud API Methods

    def _raise_cloud_remote_unavailable(self, operation: str) -> NoReturn:
        """Raise the standard not-implemented error for remote cloud execution."""
        raise CloudRemoteExecutionUnavailableError(operation)

    async def create_cloud_session(
        self, request: SessionCreationRequest
    ) -> SessionCreationResponse:
        """Reserved cloud session path; fails closed until remote execution exists."""
        _ = request
        self._raise_cloud_remote_unavailable("create_cloud_session")

    async def get_cloud_trial_suggestion(
        self, request: NextTrialRequest
    ) -> NextTrialResponse:
        """Reserved cloud suggestion path; fails closed until remote execution exists."""
        _ = request
        self._raise_cloud_remote_unavailable("get_cloud_trial_suggestion")

    async def submit_cloud_trial_results(
        self, submission: TrialResultSubmission
    ) -> None:
        """Reserved cloud result path; fails closed until remote execution exists."""
        _ = submission
        self._raise_cloud_remote_unavailable("submit_cloud_trial_results")

    async def submit_agent_optimization(
        self, request: AgentOptimizationRequest
    ) -> AgentOptimizationResponse:
        """Reserved cloud agent optimization path; fails closed until implemented."""
        _ = request
        self._raise_cloud_remote_unavailable("submit_agent_optimization")

    async def execute_cloud_agent(
        self, request: AgentExecutionRequest
    ) -> AgentExecutionResponse:
        """Reserved cloud agent execution path; fails closed until implemented."""
        _ = request
        self._raise_cloud_remote_unavailable("execute_cloud_agent")
