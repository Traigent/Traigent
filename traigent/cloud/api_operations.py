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
    CloudEgressBlockedError,
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
from traigent.cloud.smart_pruning import (
    normalize_intermediate_report_payload,
    normalize_smart_pruning_options,
)
from traigent.config.backend_config import BackendConfig
from traigent.config.types import _warn_deprecated_once
from traigent.core.session_types import (
    SessionCreationFailureDetail,
    SessionCreationHTTPError,
)
from traigent.utils.artifact_fingerprints import (
    artifact_fingerprints_to_wire,
    fingerprint_meta_to_wire,
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

_LEGACY_SESSION_CONTRACT_DEPRECATION = (
    "Legacy session contract payloads are deprecated. Use the typed session "
    "contract by leaving TRAIGENT_SESSION_CONTRACT unset or setting "
    "TRAIGENT_SESSION_CONTRACT=typed; use offline=True with algorithm='grid' "
    "or algorithm='random' for local-only optimization, and prefer local over "
    "edge_analytics where a compatibility wire value is still required. The "
    "legacy session contract will be removed in a future major release."
)


class TraigentSessionApiResult(tuple):
    """Three-ID session creation result with optional owning context attrs.

    ``warm_start_transfer`` carries the backend's opaque, IP-safe warm-start
    decision block (transfer_mode / refused_reason / search_space_overlap /
    n_seed_configs_applied / ...) when the session-CREATE response includes
    one. The backend communicates the warm-start decision at CREATE time and
    does not resend it at finalize, so dropping it here made the SDK report
    cold-start defaults for valid warm starts (issue #1683, Bug B).
    """

    project_id: str | None
    tenant_id: str | None
    warm_start_transfer: dict[str, Any] | None

    def __new__(
        cls,
        session_id: str,
        experiment_id: str,
        experiment_run_id: str,
        *,
        project_id: str | None = None,
        tenant_id: str | None = None,
        warm_start_transfer: dict[str, Any] | None = None,
    ):
        obj = super().__new__(cls, (session_id, experiment_id, experiment_run_id))
        obj.project_id = project_id
        obj.tenant_id = tenant_id
        obj.warm_start_transfer = warm_start_transfer
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


def _choice_sequence_contains_bool(values: Any) -> bool:
    """Return True when a categorical choice container contains a bool."""

    if not isinstance(values, (list, tuple)):
        return False
    return any(isinstance(value, bool) for value in values)


def _configuration_space_entry_uses_bool(entry: Any) -> bool:
    """Detect bool-valued knobs without changing their wire representation."""

    # bool is a subclass of int, so detect it before any numeric handling.
    if isinstance(entry, bool):
        return True
    if isinstance(entry, (list, tuple)):
        return _choice_sequence_contains_bool(entry)
    if isinstance(entry, dict):
        return _choice_sequence_contains_bool(
            entry.get("choices")
        ) or _choice_sequence_contains_bool(entry.get("values"))
    return False


def _warn_boolean_config_values(space: Any) -> None:
    """Warn when configuration_space contains bools the cloud API rejects."""

    if not isinstance(space, dict):
        return

    offending_parameters = [
        str(name)
        for name, entry in space.items()
        if _configuration_space_entry_uses_bool(entry)
    ]
    if not offending_parameters:
        return

    logger.warning(
        "configuration_space parameter(s) %s use boolean values, which the "
        "cloud session API does not accept and will reject with a generic "
        "HTTP 400. Encode boolean knobs as strings (e.g. ['with','without']) "
        "or integers (0/1) and map back at the call site. See issue #1488.",
        offending_parameters,
    )


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
    _warn_boolean_config_values(space)
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

        if getattr(self.client, "_url_invalid", False) is True:
            raise CloudEgressBlockedError(operation)
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
    ) -> TraigentSessionApiResult:
        """Create a Traigent optimization session using the new session endpoints.

        Returns:
            TraigentSessionApiResult — unpacks as a tuple of
            (session_id, experiment_id, experiment_run_id) and carries
            owning-context / warm_start_transfer attributes (issue #1683).
        """
        # Skip cloud session creation if backend is offline
        # Note: is_backend_offline() returns True if TRAIGENT_OFFLINE_MODE=true
        if is_backend_offline():
            logger.debug("Backend offline: using local session IDs")
            return TraigentSessionApiResult(
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
            except CloudServiceError as typed_exc:
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
                _warn_deprecated_once(
                    "session_contract:auto_legacy_retry",
                    "Auto retry from typed session contract to legacy session "
                    f"contract is deprecated. {_LEGACY_SESSION_CONTRACT_DEPRECATION}",
                    stacklevel=3,
                )
                legacy_payload = self._build_legacy_session_payload(
                    session_request,
                    max_trials_value,
                    warn_boolean_config_values=False,
                )
                try:
                    return await self._post_session_creation(
                        legacy_payload, headers, connector
                    )
                except CloudServiceError as legacy_exc:
                    typed_detail = getattr(typed_exc, "session_creation_failure", None)
                    legacy_detail = getattr(
                        legacy_exc, "session_creation_failure", None
                    )
                    typed_status = getattr(typed_detail, "status_code", None)
                    legacy_status = getattr(legacy_detail, "status_code", None)
                    if typed_status == 400 and legacy_status == 400:
                        cast(Any, legacy_exc).typed_legacy_session_create_400 = True
                    raise
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
        except CloudServiceError as e:
            if getattr(e, "typed_legacy_session_create_400", False):
                raise
            self._handle_generic_session_exception(e)
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
        return cast(int, value)

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
            _warn_deprecated_once(
                "session_contract:TRAIGENT_SESSION_CONTRACT=legacy",
                "TRAIGENT_SESSION_CONTRACT=legacy is deprecated. "
                f"{_LEGACY_SESSION_CONTRACT_DEPRECATION}",
                stacklevel=3,
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
        if getattr(session_request, "budget", None):
            payload["budget"] = session_request.budget
        if getattr(session_request, "default_config", None):
            payload["default_config"] = session_request.default_config
        if getattr(session_request, "warm_start_from", None):
            payload["warm_start_from"] = session_request.warm_start_from
        smart_pruning = getattr(session_request, "smart_pruning", None)
        wire_smart_pruning = normalize_smart_pruning_options(smart_pruning)
        if wire_smart_pruning:
            payload["smart_pruning"] = wire_smart_pruning
        self._attach_artifact_fingerprint_payload(payload, session_request)
        return payload

    def _build_legacy_session_payload(
        self,
        session_request: SessionCreationRequest,
        max_trials: int,
        *,
        warn_boolean_config_values: bool = True,
    ) -> dict[str, Any]:
        """The pre-Phase-8 legacy shape (problem_statement/search_space) —
        non-governed compatibility only."""
        metadata = session_request.metadata or {}
        evaluation_set = metadata.get("evaluation_set", "default")
        if warn_boolean_config_values:
            _warn_boolean_config_values(session_request.configuration_space)

        payload: dict[str, Any] = {
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
        # Warm-start: never silently drop a user-set prior experiment id, even
        # on the legacy contract (empty string treated as absent). The legacy
        # BE may not consume it yet; the SDK must still forward it.
        if getattr(session_request, "warm_start_from", None):
            payload["warm_start_from"] = session_request.warm_start_from
        if getattr(session_request, "default_config", None):
            payload["default_config"] = session_request.default_config
        smart_pruning = getattr(session_request, "smart_pruning", None)
        wire_smart_pruning = normalize_smart_pruning_options(smart_pruning)
        if wire_smart_pruning:
            payload["smart_pruning"] = wire_smart_pruning
        self._attach_artifact_fingerprint_payload(payload, session_request)
        return payload

    @staticmethod
    def _attach_artifact_fingerprint_payload(
        payload: dict[str, Any],
        session_request: SessionCreationRequest,
    ) -> None:
        artifact_fingerprints = artifact_fingerprints_to_wire(
            getattr(session_request, "artifact_fingerprints", None)
        )
        if artifact_fingerprints is not None:
            payload["artifact_fingerprints"] = artifact_fingerprints

        fingerprint_meta = fingerprint_meta_to_wire(
            getattr(session_request, "fingerprint_meta", None)
        )
        if fingerprint_meta is not None:
            payload["fingerprint_meta"] = fingerprint_meta

    def _build_connector(self) -> Any | None:
        """Create an aiohttp connector when custom transport settings are required."""

        return None

    async def _post_session_creation(
        self,
        payload: dict[str, Any],
        headers: dict[str, str],
        connector: Any | None,
    ) -> TraigentSessionApiResult:
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
        # The backend communicates the warm-start decision in the CREATE
        # response (top-level or under metadata) and does not resend it at
        # finalize. Retain the block verbatim (issue #1683, Bug B).
        warm_start_transfer = result.get("warm_start_transfer")
        if not isinstance(warm_start_transfer, dict):
            warm_start_transfer = metadata.get("warm_start_transfer")
        if not isinstance(warm_start_transfer, dict):
            warm_start_transfer = None

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
            warm_start_transfer=warm_start_transfer,
        )

    @staticmethod
    def _optional_context_id(value: Any) -> str | None:
        if value is None:
            return None
        normalized = str(value).strip()
        return normalized or None

    async def report_intermediate_progress(
        self, payload: dict[str, Any]
    ) -> dict[str, Any]:
        """Report per-trial intermediate progress for cloud smart pruning."""
        wire_payload = normalize_intermediate_report_payload(payload)
        session_id = str(wire_payload.get("session_id", "")).strip()
        trial_id = str(wire_payload.get("trial_id", "")).strip()
        if not session_id or not trial_id:
            return {"prune": False, "prune_reason": None}
        if is_backend_offline():
            logger.debug(
                "Offline mode: skipping intermediate report for session %s trial %s",
                session_id,
                trial_id,
            )
            return {"prune": False, "prune_reason": None}

        self._raise_if_backend_egress_disabled("intermediate report")
        if not AIOHTTP_AVAILABLE:
            logger.debug("aiohttp not available, skipping intermediate report")
            return {"prune": False, "prune_reason": None}

        try:
            headers = await self.client.auth_manager.augment_headers(
                {"Content-Type": _JSON_CONTENT_TYPE}
            )
            async with cast(Any, aiohttp).ClientSession(
                connector=self._build_connector()
            ) as session:
                api_base = (
                    self.client.backend_config.api_base_url
                    or BackendConfig.get_backend_api_url()
                )
                url = f"{api_base}/sessions/{session_id}/intermediate-report"
                timeout = cast(Any, aiohttp).ClientTimeout(total=10)
                async with session.post(
                    url,
                    json=wire_payload,
                    headers=headers,
                    timeout=timeout,
                ) as response:
                    if response.status == 200:
                        decision = await response.json()
                        if isinstance(decision, dict):
                            return {
                                "prune": bool(decision.get("prune", False)),
                                "prune_reason": decision.get("prune_reason"),
                            }
                        return {"prune": False, "prune_reason": None}

                    error_msg = await response.text()
                    logger.debug(
                        "Intermediate report rejected for session %s trial %s: "
                        "HTTP %s %s",
                        session_id,
                        trial_id,
                        response.status,
                        error_msg[:300],
                    )
                    return {"prune": False, "prune_reason": None}
        except aiohttp.ClientError as exc:
            logger.debug(
                "Network error sending intermediate report for session %s trial %s: %s",
                session_id,
                trial_id,
                exc,
            )
            return {"prune": False, "prune_reason": None}

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

    async def update_config_run_status(
        self,
        config_run_id: str,
        status: str,
        error_message: str | None = None,
    ) -> bool:
        """Update configuration run status in the backend.

        Args:
            config_run_id: Configuration run ID (same as trial_id)
            status: Status to set. Accepts either an SDK token (``running``,
                ``completed`` …) or an already-mapped wire value; it is always
                normalized to the configuration-run wire vocab before sending so
                the backend never receives a session-lifecycle value
                (issue #1302).
            error_message: Optional failure reason to persist alongside a
                terminal status (e.g. ``failed``/``pruned``). Before this,
                this status-only PUT never sent a failure reason at all, so
                run failures reported through it were lost client-side
                before the wire (Traigent#1885, companion to
                TraigentBackend#2002). Sent as ``error_message`` — the
                canonical key the backend reads; the backend also accepts a
                legacy ``error`` alias for old senders, but new SDK code
                must always use the canonical key. Sanitized/length-capped
                the same way as the session-results path
                (``sanitize_error_message``), so this is safe to call with
                a raw, unsanitized string.

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
                status_data: dict[str, Any] = {"status": backend_status}
                sanitized_error_message = self.sanitize_error_message(error_message)
                if sanitized_error_message:
                    status_data["error_message"] = sanitized_error_message

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
