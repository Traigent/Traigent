"""Backend session lifecycle management."""

# Traceability: CONC-Layer-Core CONC-Quality-Reliability CONC-Quality-Security FUNC-CLOUD-HYBRID FUNC-ORCH-LIFECYCLE REQ-CLOUD-009 REQ-ORCH-003 SYNC-CloudHybrid

from __future__ import annotations

import asyncio
import inspect
import re
import threading
import time
from collections.abc import Callable, Coroutine, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

from traigent._version import get_version
from traigent.api.types import (
    AgentConfiguration,
    OptimizationResult,
    OptimizationStatus,
    TrialResult,
    TrialStatus,
)
from traigent.config.types import TraigentConfig

if TYPE_CHECKING:
    from traigent.cloud.backend_client import BackendIntegratedClient

from traigent.config.backend_config import get_no_credentials_hint
from traigent.core.execution_policy_runtime import (
    RESULT_SOURCES,
    SOURCE_CLOUD_BRAIN,
    SOURCE_EXPLICIT_LOCAL,
    SOURCE_LOCAL_FALLBACK,
    SOURCE_OFFLINE,
    backend_egress_disabled,
    exception_status,
    fallback_reason_from_session_result,
    mark_local_fallback,
    policy_allows_cloud_fallback,
    policy_from_config,
    policy_is_cloud_brain,
    policy_requires_cloud,
    session_failure_is_connectivity,
    session_failure_is_session_create_400,
)
from traigent.core.metadata_helpers import build_backend_metadata
from traigent.core.objectives import ObjectiveSchema
from traigent.core.session_context import SessionContext
from traigent.core.session_types import (
    SessionCreationFailureClassification,
    SessionCreationFailureDetail,
    SessionCreationFailureReason,
    SessionCreationResult,
)
from traigent.evaluators.base import Dataset
from traigent.metrics.content_features import SimhashFeatureExtractor
from traigent.optimizers.base import BaseOptimizer
from traigent.utils.trial_costs import (
    TRIAL_COST_FIELDS,
    extract_trial_cost_metric,
    is_finite_numeric_cost,
)
from traigent.utils.env_config import is_untracked_fallback_allowed
from traigent.utils.exceptions import ConfigurationError
from traigent.utils.function_identity import FunctionDescriptor
from traigent.utils.logging import get_logger

logger = get_logger(__name__)

_WARNING_RULE = "=" * 72
_FINALIZE_MAX_ATTEMPTS = 3
_FINALIZE_BACKOFF_SECONDS = (1.0, 2.0, 4.0)
_FINALIZE_RETRYABLE_STATUSES = frozenset({403, 429, 500, 502, 503, 504})
_FINALIZE_CONNECTION_ERROR_NAMES = (
    "ClientConnectorError",
    "ClientConnectionError",
    "ConnectionError",
    "ConnectError",
    "ReadTimeout",
    "Timeout",
)


class _BackendFinalizeNotAcknowledgedError(RuntimeError):
    """Backend finalize returned without confirming remote finalization."""


def _status_from_exception(exc: BaseException) -> int | None:
    status = exception_status(exc)
    if status is not None:
        return status
    match = re.search(
        r"\b(?:http|status(?:_code)?|code)[=:\s-]*(\d{3})\b",
        str(exc),
        flags=re.IGNORECASE,
    )
    if match:
        return int(match.group(1))
    return None


def _is_transient_finalize_error(exc: BaseException) -> bool:
    if isinstance(exc, _BackendFinalizeNotAcknowledgedError):
        return True

    status = _status_from_exception(exc)
    if status is not None:
        return status in _FINALIZE_RETRYABLE_STATUSES

    if isinstance(exc, (ConnectionError, TimeoutError, OSError)):
        return True

    exc_name = type(exc).__name__
    if any(name in exc_name for name in _FINALIZE_CONNECTION_ERROR_NAMES):
        return True

    text = str(exc).lower()
    return any(
        pattern in text
        for pattern in (
            "connection refused",
            "connection reset",
            "network error",
            "timed out",
            "timeout",
            "temporary failure",
            "temporarily unavailable",
            "too many requests",
            "rate limit",
            "service unavailable",
        )
    )


def _finalize_acknowledged(result: Any) -> bool:
    if result is None:
        return False

    metadata = None
    if isinstance(result, dict):
        metadata = result.get("metadata")
    else:
        metadata = getattr(result, "metadata", None)

    if isinstance(metadata, dict) and metadata.get("finalized_via_api") is False:
        return False

    return True


def _contains_any(value: str, needles: tuple[str, ...]) -> bool:
    return any(needle in value for needle in needles)


def _classify_session_creation_failure(
    reason: SessionCreationFailureReason,
    detail: SessionCreationFailureDetail | None,
    *,
    failure_detail: str | None = None,
) -> SessionCreationFailureClassification:
    text = " ".join(
        part
        for part in (
            failure_detail,
            detail.error_code if detail else None,
            detail.message if detail else None,
            detail.raw_body if detail else None,
            detail.backend_url if detail else None,
        )
        if part
    ).lower()

    if _contains_any(
        text,
        (
            "edge_blocked",
            "edge blocked",
            "cloudflare",
            "cf-ray",
            "cf-mitigated",
            "error code: 1010",
            "browser_signature_banned",
            "waf",
        ),
    ):
        return SessionCreationFailureClassification.EDGE_BLOCKED

    key_validation_text = _contains_any(
        text,
        (
            "api key validation",
            "/keys/validate",
            "keys/validate",
        ),
    )
    if key_validation_text and _contains_any(
        text,
        (
            "timed out",
            "timeout",
            "transport error",
            "network",
            "connection",
            "rate limited",
            "temporarily unavailable",
            "service unavailable",
        ),
    ):
        return SessionCreationFailureClassification.NETWORK

    if reason == SessionCreationFailureReason.SESSION_FAILED:
        if detail and detail.status_code and 400 <= detail.status_code < 500:
            return SessionCreationFailureClassification.UNKNOWN
        return SessionCreationFailureClassification.BACKEND_UNREACHABLE

    if reason == SessionCreationFailureReason.NO_API_KEY:
        return SessionCreationFailureClassification.KEY_NOT_FOUND

    if detail and detail.missing_permissions:
        return SessionCreationFailureClassification.MISSING_PERMISSION

    if _contains_any(text, ("key_not_found", "api_key_not_found", "key not found")):
        return SessionCreationFailureClassification.KEY_NOT_FOUND
    if _contains_any(text, ("expired", "expiration", "key_expired")):
        return SessionCreationFailureClassification.EXPIRED_KEY
    if _contains_any(
        text,
        (
            "insufficient",
            "missing required scope",
            "missing scope",
            "permission denied",
            "forbidden",
            "scope",
        ),
    ):
        return SessionCreationFailureClassification.INSUFFICIENT_SCOPE
    if _contains_any(
        text,
        (
            "invalid",
            "revoked",
            "rejected",
            "unauthorized",
            "authentication",
        ),
    ):
        return SessionCreationFailureClassification.INVALID_OR_REVOKED_KEY
    if detail and detail.status_code == 401:
        return SessionCreationFailureClassification.INVALID_OR_REVOKED_KEY
    if detail and detail.status_code == 403:
        return SessionCreationFailureClassification.INSUFFICIENT_SCOPE

    return SessionCreationFailureClassification.UNKNOWN


def _format_raw_reason(result: SessionCreationResult) -> str | None:
    detail = result.failure_response
    raw_reason = result.failure_detail
    if detail and detail.raw_body:
        raw_reason = detail.raw_body
    if not raw_reason:
        return None
    return cast(str, raw_reason.replace("\n", "\\n")[:500])


def _backend_disabled_label(reason: SessionCreationFailureReason | None) -> str:
    """Return a log-safe, non-secret label for disabled backend tracking."""

    if reason == SessionCreationFailureReason.AUTH:
        return "authentication-failed"
    if reason == SessionCreationFailureReason.NO_API_KEY:
        return "credentials-unavailable"
    if reason == SessionCreationFailureReason.SESSION_FAILED:
        return "session-create-failed"
    return "unknown"


@dataclass(frozen=True)
class BackendTrialSlotAcquisition:
    """Core-local interpretation of a backend trial-slot request."""

    trial_id: str | None = None
    optimization_complete: bool = False
    reason: str | None = None


@dataclass(frozen=True)
class BackendTrialSubmissionOutcome:
    """Terminal submission outcome consumed by the orchestrator."""

    optimization_complete: bool = False
    reason: str | None = None

    @classmethod
    def complete(cls, reason: str | None = None) -> BackendTrialSubmissionOutcome:
        return cls(optimization_complete=True, reason=reason)


def _format_untracked_warning_block(
    result: SessionCreationResult,
    classification: SessionCreationFailureClassification,
    *,
    aborting: bool,
) -> str:
    detail = result.failure_response
    lines = [
        _WARNING_RULE,
        "TRAIGENT BACKEND TRACKING UNAVAILABLE",
        "This run will NOT be tracked in the portal.",
        f"Classification: {classification.value}",
    ]

    if detail and detail.status_code is not None:
        lines.append(f"HTTP status: {detail.status_code}")
    if detail and detail.backend_url:
        lines.append(f"Backend URL: {detail.backend_url}")
    if detail and detail.error_code:
        lines.append(f"Backend error_code: {detail.error_code}")
    if detail and detail.message:
        lines.append(f"Backend message: {detail.message}")
    if detail and detail.missing_permissions:
        permissions = ", ".join(f"`{p}`" for p in detail.missing_permissions)
        lines.append(f"Missing permissions: {permissions}")

    raw_reason = _format_raw_reason(result)
    if raw_reason:
        lines.append(f"Raw reason: {raw_reason}")

    if classification == SessionCreationFailureClassification.MISSING_PERMISSION:
        lines.append(
            "Remedy: grant the missing permission(s) to the API key, or if the "
            "key should already have them, verify the backend deployment and "
            "permission middleware that returned this structured response."
        )
    elif classification == SessionCreationFailureClassification.INSUFFICIENT_SCOPE:
        lines.append(
            "Remedy: grant the required API key scope or permission in the "
            "Traigent portal, then retry with that updated key."
        )
    elif classification == SessionCreationFailureClassification.EXPIRED_KEY:
        lines.append(
            "Remedy: create a new Traigent API key and set TRAIGENT_API_KEY to "
            "that active key."
        )
    elif classification == SessionCreationFailureClassification.EDGE_BLOCKED:
        lines.append(
            "Remedy: the validation request appears blocked by an edge/WAF "
            "layer. Allowlist the Traigent SDK User-Agent or retry from an "
            "unblocked network."
        )
    elif classification == SessionCreationFailureClassification.NETWORK:
        lines.append(
            "Remedy: check TRAIGENT_BACKEND_URL and network connectivity; retry "
            "with TRAIGENT_LOG_LEVEL=DEBUG for request diagnostics."
        )
    elif classification == SessionCreationFailureClassification.INVALID_OR_REVOKED_KEY:
        raw = (result.failure_detail or "").lower()
        if "invalid api key format" in raw or "invalid_api_key_format" in raw:
            lines.append(
                "Remedy: your API key does not match the expected format. "
                "Portal keys start with 'uk_' (46 chars) or 'tg_' (64 chars). "
                "Copy the full key from Settings → API Keys and set "
                "TRAIGENT_API_KEY to that exact value."
            )
        else:
            lines.append(
                "Remedy: re-authenticate with `traigent auth login` or provide a "
                "valid, non-revoked API key."
            )
    elif classification == SessionCreationFailureClassification.KEY_NOT_FOUND:
        lines.append(
            "Remedy: configure a valid Traigent API key or run `traigent auth login`."
        )
    elif classification == SessionCreationFailureClassification.BACKEND_UNREACHABLE:
        lines.append(
            "Remedy: check TRAIGENT_BACKEND_URL/network connectivity or retry later."
        )
    else:
        lines.append(
            "Remedy: inspect the raw backend response above; retry with "
            "TRAIGENT_LOG_LEVEL=DEBUG for more diagnostics."
        )

    if aborting:
        lines.append(
            "Aborting before trials or LLM calls. To knowingly proceed local-only, "
            "set TRAIGENT_ALLOW_UNTRACKED=1."
        )
    else:
        lines.append("Continuing local-only; results will be saved on this machine.")

    lines.append(_WARNING_RULE)
    return "\n".join(lines)


class BackendSessionManager:
    """Manages backend session lifecycle and trial synchronization.

    Owns all backend-related operations including:
    - Session creation and finalization
    - Trial submission with metadata
    - Weighted score updates for multi-objective optimization
    - Session aggregation and summary generation
    """

    def __init__(
        self,
        backend_client: BackendIntegratedClient | None,
        traigent_config: TraigentConfig,
        objectives: list[str],
        objective_schema: ObjectiveSchema | None,
        optimizer: BaseOptimizer,
        optimization_id: str,
        optimization_status: OptimizationStatus,
        strategy_preset_metadata: dict[str, Any] | None = None,
        smart_pruning: dict[str, Any] | None = None,
    ) -> None:
        """Initialize backend session manager.

        Args:
            backend_client: Backend API client (None disables backend sync)
            traigent_config: Global configuration
            objectives: List of optimization objectives
            objective_schema: Multi-objective schema with weights
            optimizer: Optimizer instance (for objectives and config_space)
            optimization_id: Unique optimization run identifier
            optimization_status: Current optimization status
        """
        self._backend_client: BackendIntegratedClient | None = backend_client
        self._traigent_config = traigent_config
        self._objectives = objectives
        self._objective_schema = objective_schema
        self._optimizer = optimizer
        self._optimization_id = optimization_id
        self._optimization_status = optimization_status
        self._strategy_preset_metadata = (
            dict(strategy_preset_metadata) if strategy_preset_metadata else None
        )
        self._smart_pruning = dict(smart_pruning) if smart_pruning else None

        # Run-scoped circuit breaker — once disabled, all backend writes skip
        self._no_egress = backend_egress_disabled(traigent_config)
        self._backend_tracking_enabled: bool = not self._no_egress
        if self._backend_client is not None and self._no_egress:
            self._backend_client.no_egress = True
        self._backend_disabled_reason: SessionCreationFailureReason | None = None
        self._fallback_reason: str | None = getattr(
            traigent_config, "fallback_reason", None
        )
        self._session_owning_context: dict[str, dict[str, str]] = {}
        self._session_cost_budget_armed: set[str] = set()
        self._cost_budget_zero_backfill_logged = False

        # Tracks (session_id, trial_id) pairs that have already been registered
        # with the backend, so retries of submit_trial don't re-register and
        # trip "already started" errors on the backend side.
        self._started_trials: set[tuple[str, str]] = set()

        # Tracks (session_id, backend_trial_id) pairs the backend MINTED and
        # ACCEPTED a result for. The certified-selection report binds the
        # winner to its backend trial id, so the report builder consults this
        # set: an incumbent whose id is absent here is unbindable and the
        # report is withheld (Rule 1/2: never attest a winner the backend
        # never acknowledged).
        self._acknowledged_trials: set[tuple[str, str]] = set()

        # Issue #1265: a backend that becomes unreachable *mid-run* (transient
        # outage / maintenance window) is distinct from the explicit
        # TRAIGENT_OFFLINE_MODE air-gap. When it happens in a backend-tracking
        # mode we degrade to local-only: trials are still optimized and
        # persisted locally, but the run is no longer cloud-tracked. We record
        # that here so the run's result can be marked source="local_fallback" and a
        # single prominent warning is emitted (instead of one per trial).
        self._runtime_degraded: bool = False
        self._degraded_warning_emitted: bool = False
        self._backend_rejection_reason: str | None = None

    @property
    def backend_tracking_enabled(self) -> bool:
        """Whether remote backend tracking is active for this run."""
        return self._backend_tracking_enabled

    @property
    def backend_degraded(self) -> bool:
        """Whether backend tracking degraded to local-only mid-run."""
        return self._runtime_degraded

    @property
    def backend_rejection_reason(self) -> str | None:
        """Return the permanent backend rejection reason, if one was recorded."""

        return self._backend_rejection_reason

    @property
    def fallback_reason(self) -> str | None:
        """Return the local-fallback reason when one was recorded."""

        return self._fallback_reason or getattr(
            self._traigent_config, "fallback_reason", None
        )

    def _egress_disabled(self) -> bool:
        """Return true when this manager must not touch backend egress paths."""

        if bool(getattr(self, "_no_egress", False)):
            return True
        config = getattr(self, "_traigent_config", None)
        return bool(config is not None and backend_egress_disabled(config))

    @staticmethod
    def _cost_budget_is_armed(cost_limit: float | None) -> bool:
        if cost_limit is None or isinstance(cost_limit, bool):
            return False
        try:
            return float(cost_limit) > 0
        except (TypeError, ValueError):
            return False

    def _ensure_budget_cost_metric(
        self,
        *,
        session_id: str,
        trial_result: TrialResult,
        metrics_payload: dict[str, Any],
        metadata: Mapping[str, Any],
    ) -> None:
        if is_finite_numeric_cost(metrics_payload.get("cost")):
            return

        attr_costs = {
            field: getattr(trial_result, field, None) for field in TRIAL_COST_FIELDS
        }
        cost = extract_trial_cost_metric(
            metrics_payload,
            getattr(trial_result, "metrics", None),
            getattr(trial_result, "metadata", None),
            metadata,
            attr_costs,
        )
        if cost is not None:
            metrics_payload["cost"] = cost
            return

        if session_id not in self._session_cost_budget_armed:
            return

        metrics_payload["cost"] = 0.0
        if not self._cost_budget_zero_backfill_logged:
            logger.debug("no cost telemetry; backfilling 0.0 for budget accounting")
            self._cost_budget_zero_backfill_logged = True

    def _flag_backend_degraded(
        self, context: str, *, rejection_reason: str | None = None
    ) -> None:
        """Mark the run as degraded to local-only and warn once (issue #1265).

        Called when a backend-tracking interaction fails at runtime. The first
        call emits a single, prominent warning; subsequent calls are quiet so a
        long run doesn't spam one warning per trial.
        """
        self._runtime_degraded = True
        self._fallback_reason = context
        mark_local_fallback(self._traigent_config, context)
        if rejection_reason:
            self._backend_rejection_reason = rejection_reason
            self._traigent_config.persistence_reason = "rejected"
            self._traigent_config.persistence_rejection_reason = rejection_reason
        if self._degraded_warning_emitted:
            return
        self._degraded_warning_emitted = True
        if rejection_reason:
            logger.warning(
                "⚠️  Traigent backend tracking was REJECTED during %s — "
                "continuing in LOCAL-ONLY mode. The backend was reachable and "
                "rejected the submitted config: %s. Trials are still optimized "
                "and saved to local storage, but this run is NOT tracked on the "
                "cloud backend; its result is marked source='local_fallback' "
                "and persistence_reason='rejected'.",
                context,
                rejection_reason,
            )
            return
        logger.warning(
            "⚠️  Traigent backend tracking became unavailable during %s — "
            "continuing in LOCAL-ONLY mode. Trials are still optimized and "
            "saved to local storage, but this run is NOT tracked on the cloud "
            "backend; its result is marked source='local_fallback'. Locally-stored "
            "results sync to the cloud on the next successful run.",
            context,
        )

    def result_source(self, trial_count: int) -> str:
        """Return the provenance of this run's result.

        Values are one of cloud_brain, local_fallback, explicit_local, or
        offline.
        """
        configured_source = getattr(self._traigent_config, "result_source", None)
        policy = policy_from_config(self._traigent_config)
        if configured_source == SOURCE_OFFLINE or (
            policy is not None and policy.offline
        ):
            return cast(str, SOURCE_OFFLINE)
        if configured_source in RESULT_SOURCES:
            if configured_source == SOURCE_CLOUD_BRAIN and (
                self._runtime_degraded
                or (trial_count > 0 and not self._acknowledged_trials)
            ):
                return cast(str, SOURCE_LOCAL_FALLBACK)
            return str(configured_source)
        if self._traigent_config.is_local_mode():
            return cast(str, SOURCE_EXPLICIT_LOCAL)
        if self._egress_disabled():
            return cast(str, SOURCE_OFFLINE)
        if not self._backend_tracking_enabled or self._runtime_degraded:
            return cast(str, SOURCE_LOCAL_FALLBACK)
        if trial_count > 0 and not self._acknowledged_trials:
            return cast(str, SOURCE_LOCAL_FALLBACK)
        return cast(str, SOURCE_CLOUD_BRAIN)

    def disable_backend_tracking(self, reason: SessionCreationFailureReason) -> None:
        """Disable backend tracking for this run. Idempotent — keeps first reason."""
        if not self._backend_tracking_enabled:
            return
        self._backend_tracking_enabled = False
        self._backend_disabled_reason = reason

    def handle_session_creation_result(
        self, result: SessionCreationResult, *, governed_session: bool = False
    ) -> str:
        """Consume a SessionCreationResult: flip breaker, emit warning, return session_id.

        Single place that interprets the structured result — ensures warning text
        and breaker state cannot diverge across call sites.
        """
        if not result.backend_connected:
            was_enabled = self._backend_tracking_enabled
            if result.failure_reason is None:
                raise ValueError(
                    "SessionCreationResult.failure_reason must be set "
                    "when backend_connected=False"
                )
            reason = result.failure_reason
            self.disable_backend_tracking(reason)

            if not was_enabled:
                # Even when tracking was already disabled, an AUTH-reason failure
                # must surface at WARNING so the user sees the remedy text
                # (missing_permissions, re-auth hint, etc.) — issue #1373 Case 2.
                # For non-auth reasons (NO_API_KEY, SESSION_FAILED) the normal
                # message was already emitted on the first disable; stay silent.
                if reason == SessionCreationFailureReason.AUTH:
                    classification = _classify_session_creation_failure(
                        reason,
                        result.failure_response,
                        failure_detail=result.failure_detail,
                    )
                    logger.warning(
                        "%s",
                        _format_untracked_warning_block(
                            result, classification, aborting=False
                        ),
                    )
                return str(result.session_id)

            policy = policy_from_config(self._traigent_config)
            fallback_reason = fallback_reason_from_session_result(result)
            if policy_requires_cloud(policy):
                raise ConfigurationError(
                    "Cloud execution is required, but backend session creation "
                    f"failed: {fallback_reason}"
                )
            if governed_session:
                raise ConfigurationError(
                    "Governed backend session creation failed; local fallback is "
                    f"not allowed: {fallback_reason}"
                )
            if policy_is_cloud_brain(policy):
                cloud_fallback_allowed = policy_allows_cloud_fallback(policy)
                if cloud_fallback_allowed and (
                    session_failure_is_connectivity(result)
                    or session_failure_is_session_create_400(result)
                ):
                    self._fallback_reason = fallback_reason
                    mark_local_fallback(self._traigent_config, fallback_reason)
                    logger.warning(
                        "traigent.cloud_brain_fallback source=%s "
                        "fallback_reason=%s stage=session-create",
                        SOURCE_LOCAL_FALLBACK,
                        fallback_reason,
                    )
                    return str(result.session_id)
                raise ConfigurationError(
                    "Cloud brain session creation failed without an allowed "
                    f"connectivity fallback: {fallback_reason}"
                )

            classification = _classify_session_creation_failure(
                reason,
                result.failure_response,
                failure_detail=result.failure_detail,
            )
            if reason == SessionCreationFailureReason.AUTH:
                allow_untracked = is_untracked_fallback_allowed()
                warning = _format_untracked_warning_block(
                    result,
                    classification,
                    aborting=not allow_untracked,
                )
                if not allow_untracked:
                    raise ConfigurationError(warning)
                logger.warning("%s", warning)
            elif reason == SessionCreationFailureReason.NO_API_KEY:
                logger.info(
                    "No API key found — results saved locally only. %s",
                    get_no_credentials_hint(),
                )
            else:
                logger.warning(
                    "%s",
                    _format_untracked_warning_block(
                        result,
                        classification,
                        aborting=False,
                    ),
                )
        else:
            logger.info("Created backend session: %s", result.session_id)

        return str(result.session_id)

    @staticmethod
    def normalize_session_creation_result(
        result: SessionCreationResult | str,
    ) -> SessionCreationResult:
        """Accept legacy string session IDs from tests and older stubs."""
        if isinstance(result, SessionCreationResult):
            return result
        return SessionCreationResult.connected(session_id=result)

    @staticmethod
    def create_backend_client(
        traigent_config: TraigentConfig,
    ) -> BackendIntegratedClient | None:
        """Initialize backend client if backend integration features are available.

        Returns None when backend integration modules are not installed
        (graceful degradation). Cloud remote execution remains unavailable
        and is validated elsewhere before use.

        Args:
            traigent_config: Global configuration for execution mode and storage

        Returns:
            BackendIntegratedClient if available, None otherwise
        """
        if backend_egress_disabled(traigent_config):
            logger.debug("Execution policy forbids backend client initialization")
            return None

        # Try to import backend integration module - may not be available in minimal installs.
        try:
            from traigent.cloud.backend_client import (
                BackendClientConfig,
                BackendIntegratedClient,
            )
            from traigent.config.backend_config import BackendConfig
        except ModuleNotFoundError as err:
            # Backend integration module not installed - check if this was the module itself.
            missing_module = getattr(err, "name", "") or ""
            if missing_module.startswith("traigent.cloud"):
                if traigent_config.execution_mode == "cloud":
                    # User explicitly requested reserved cloud mode but plugin not installed.
                    from traigent.utils.exceptions import FeatureNotAvailableError

                    raise FeatureNotAvailableError(
                        "Cloud remote execution is not available yet; use hybrid for portal-tracked optimization",
                        plugin_name="traigent-cloud",
                        install_hint="Use execution_mode='hybrid' for portal-tracked optimization",
                    ) from err
                # For local or other modes, gracefully degrade to local-only.
                logger.info(
                    f"Backend integration module not available for {traigent_config.execution_mode} mode. "
                    "Continuing with local storage only."
                )
                return None
            # Re-raise if it's a different missing module (broken install)
            raise

        backend_url = BackendConfig.get_backend_url()
        api_key = BackendConfig.get_api_key()

        if traigent_config.is_local_mode() or BackendConfig.is_local_backend():
            logger.info(
                f"Configuring for {traigent_config.execution_mode} mode "
                f"with backend at {backend_url} (fallback enabled)"
            )
        else:
            logger.info(
                f"Configuring for {traigent_config.execution_mode} mode "
                f"with backend at {backend_url}"
            )

        backend_config = BackendClientConfig(
            backend_base_url=backend_url,
            enable_session_sync=True,
        )
        local_storage_path = traigent_config.get_local_storage_path()
        no_egress = backend_egress_disabled(traigent_config)

        try:
            client = BackendIntegratedClient(
                api_key=api_key,
                backend_config=backend_config,
                enable_fallback=True,
                local_storage_path=local_storage_path,
                no_egress=no_egress,
            )
            logger.info(
                f"Backend client initialized for {traigent_config.execution_mode} mode - "
                f"session endpoints at {backend_config.backend_base_url}"
            )
            return client
        except Exception as exc:
            logger.warning(
                "Backend initialization warning. Continuing with local storage only. "
                "Results will not appear in backend UI.",
                exc_info=exc,
            )
            return BackendIntegratedClient(
                api_key=None,
                backend_config=backend_config,
                enable_fallback=True,
                local_storage_path=local_storage_path,
                no_egress=no_egress,
            )

    def _should_suppress_backend_warnings(self) -> bool:
        """Check if backend-related warnings should be suppressed.

        Returns True if:
        - Backend tracking is disabled (circuit breaker tripped), OR
        - Offline mode is enabled, OR
        - No API key is configured
        """
        if not self._backend_tracking_enabled:
            return True

        if self._egress_disabled():
            return True

        if self._backend_client:
            auth_manager = getattr(self._backend_client, "auth", None)
            has_api_key = bool(
                auth_manager
                and hasattr(auth_manager, "has_api_key")
                and auth_manager.has_api_key()
            )
            if not has_api_key:
                return True

        return False

    def _upload_dataset_features(
        self,
        *,
        session_id: str,
        dataset: Dataset,
        experiment_run_id: str | None,
    ) -> None:
        """Upload deterministic example features for backend-side content analytics."""
        if not bool(self._traigent_config.get("enable_dataset_feature_upload", False)):
            return

        if self._egress_disabled() or not self._backend_client or not experiment_run_id:
            return

        uploader = getattr(self._backend_client, "upload_example_features", None)
        if not callable(uploader):
            return

        try:
            feature_rows = SimhashFeatureExtractor().extract_dataset_features(dataset)
        except Exception as exc:
            logger.debug(
                "Failed to extract content features for session %s: %s",
                session_id,
                exc,
            )
            return

        if not feature_rows:
            return

        try:
            uploaded = uploader(experiment_run_id, "simhash_v1", feature_rows)
        except Exception as exc:
            logger.debug(
                "Failed to upload content features for session %s: %s",
                session_id,
                exc,
            )
            return

        if uploaded:
            logger.debug(
                "Uploaded %s example features for session %s",
                len(feature_rows),
                session_id,
            )

    def create_session(
        self,
        func: Callable[..., Any],
        dataset: Dataset,
        function_descriptor: FunctionDescriptor,
        max_trials: int | None,
        start_time: float,
        max_total_examples: int | None = None,
        agent_configuration: AgentConfiguration | None = None,
        objectives: list[Any] | None = None,
        default_config: dict[str, Any] | None = None,
        promotion_policy: dict[str, Any] | None = None,
        tvl_governance: dict[str, Any] | None = None,
        experiment_display_name: str | None = None,
        warm_start_from: str | None = None,
        smart_pruning: dict[str, Any] | None = None,
        artifact_fingerprints: dict[str, str | None] | None = None,
        fingerprint_meta: dict[str, Any] | None = None,
        cost_limit: float | None = None,
        optimization_strategy: dict[str, Any] | None = None,
    ) -> SessionContext:
        """Create backend session and return context.

        Args:
            func: Function being optimized
            dataset: Evaluation dataset
            function_descriptor: Descriptor for the function being optimized
            max_trials: Maximum number of trials
            start_time: Optimization start timestamp
            max_total_examples: Maximum total examples across all trials
            agent_configuration: Multi-agent configuration for parameter grouping
            default_config: Optional user-declared baseline configuration
                (``@optimize(default_config=...)``), projected onto the
                session-create wire so backend warm-start seed projection
                sees it. Omitted from the request entirely when None/empty.

        Returns:
            SessionContext with session_id (or None if backend disabled)
        """
        session_id = None
        function_identifier = function_descriptor.identifier
        # experiment_display_name (from @traigent.optimize(experiment_name=...)) overrides
        # the descriptor's __qualname__-derived display_name in portal/storage.
        function_display_name = (
            experiment_display_name or function_descriptor.display_name
        )
        function_slug = function_descriptor.slug

        if self._egress_disabled():
            return SessionContext(
                session_id=None,
                dataset_name=(getattr(dataset, "name", None) or "default_evaluation"),
                function_name=function_identifier,
                optimization_id=self._optimization_id,
                start_time=start_time,
            )

        if self._backend_client:
            evaluation_set_name = getattr(dataset, "name", None) or "default_evaluation"
            effective_smart_pruning = (
                dict(smart_pruning)
                if smart_pruning is not None
                else (
                    dict(self._smart_pruning)
                    if self._smart_pruning is not None
                    else None
                )
            )

            max_trials_value = max_trials if max_trials is not None else 10
            max_samples_value = (
                max_total_examples if max_total_examples is not None else None
            )
            logger.info(
                "Creating backend session with max_trials=%s for %s (remote_name=%s)",
                max_trials_value,
                function_identifier,
                function_slug,
            )

            # Build metadata including agent configuration if present
            session_metadata: dict[str, Any] = {
                "optimization_id": self._optimization_id,
                "max_trials": max_trials_value,
                "max_total_examples": max_samples_value,
                "dataset_size": len(dataset),
                "function_name": function_identifier,
                "function_display_name": function_display_name,
                "function_module": function_descriptor.module,
                "function_relative_path": function_descriptor.relative_path,
                "function_slug": function_slug,
                "evaluation_set": evaluation_set_name,
            }
            if agent_configuration is not None:
                session_metadata["agent_configuration"] = agent_configuration.to_dict()
            if self._strategy_preset_metadata is not None:
                session_metadata["strategy_preset"] = dict(
                    self._strategy_preset_metadata
                )
            if warm_start_from:
                session_metadata["warm_start_from"] = warm_start_from

            # Build the portal display name.
            # When the user supplied an explicit experiment_name, honour it.
            # When no name was given, weave in objectives and knob names so the
            # experiment is self-describing in the portal's Recent Experiments
            # list (issue #1422): e.g. "opt:accuracy,cost · model,temperature".
            slug_hash = function_slug.rsplit("_", 1)[-1] if function_slug else ""
            if experiment_display_name is None:
                # Collect up to 3 objective names and up to 4 knob names to
                # keep the label concise.
                obj_names = (self._objectives or [])[:3]
                knob_names = list(
                    (getattr(self._optimizer, "config_space", {}) or {}).keys()
                )[:4]
                label_parts: list[str] = []
                if obj_names:
                    label_parts.append("opt:" + ",".join(obj_names))
                if knob_names:
                    label_parts.append(",".join(knob_names))
                portal_name = (
                    " · ".join(label_parts)
                    if label_parts
                    else (function_display_name or function_identifier)
                )
            else:
                portal_name = function_display_name or function_identifier
            if slug_hash:
                portal_name = f"{portal_name} ({slug_hash})"

            # Tell create_session whether this run actually INTENDS cloud egress
            # (managed / auto-cloud) vs. a LOCAL-routed run (grid/random/offline/
            # runtime-resolved-to-local) that only wants OPTIONAL backend
            # tracking. A configured-but-invalid/rejected key must fail closed
            # for the former but must NOT hard-fail the latter (#1421). Mirrors
            # the existing ``no_egress`` flag set on the same client.
            _policy = policy_from_config(self._traigent_config)
            self._backend_client.cloud_egress_intent = bool(
                policy_requires_cloud(_policy) or policy_is_cloud_brain(_policy)
            )

            raw_result = self._backend_client.create_session(
                function_name=portal_name,
                search_space=getattr(self._optimizer, "config_space", {}),
                optimization_goal="maximize",
                metadata=session_metadata,
                objectives=objectives,
                default_config=default_config,
                promotion_policy=promotion_policy,
                tvl_governance=tvl_governance,
                warm_start_from=warm_start_from,
                smart_pruning=effective_smart_pruning,
                artifact_fingerprints=artifact_fingerprints,
                fingerprint_meta=fingerprint_meta,
                cost_limit=cost_limit,
                optimization_strategy=optimization_strategy,
            )
            result = self.normalize_session_creation_result(raw_result)
            session_id = self.handle_session_creation_result(
                result,
                governed_session=bool(promotion_policy or tvl_governance),
            )
            if session_id is not None:
                if self._cost_budget_is_armed(cost_limit):
                    self._session_cost_budget_armed.add(session_id)
                else:
                    self._session_cost_budget_armed.discard(session_id)
            if result.backend_connected:
                owning_context = {
                    key: normalized
                    for key, value in (
                        ("project_id", result.project_id),
                        ("tenant_id", result.tenant_id),
                    )
                    if value is not None and (normalized := str(value).strip())
                }
                if owning_context:
                    self._session_owning_context[session_id] = owning_context

            # On success, upload dataset features via the session mapping
            if result.backend_connected:
                session_mapping = None
                get_mapping = getattr(self._backend_client, "get_session_mapping", None)
                if callable(get_mapping):
                    try:
                        session_mapping = get_mapping(session_id)
                    except Exception as exc:
                        logger.debug(
                            "Unable to resolve session mapping for %s: %s",
                            session_id,
                            exc,
                        )

                if session_mapping is not None:
                    self._upload_dataset_features(
                        session_id=session_id,
                        dataset=dataset,
                        experiment_run_id=getattr(
                            session_mapping, "experiment_run_id", None
                        ),
                    )

        dataset_name = getattr(dataset, "name", None) or "default_evaluation"

        return SessionContext(
            session_id=session_id,
            dataset_name=dataset_name,
            function_name=function_identifier,
            optimization_id=self._optimization_id,
            start_time=start_time,
        )

    def update_backend_client(
        self, backend_client: BackendIntegratedClient | None
    ) -> None:
        """Swap the backend client while preserving existing session state."""

        self._backend_client = backend_client

    def should_report_intermediate_progress(self, session_id: str | None) -> bool:
        """Return whether smart-pruning intermediate reports are allowed."""
        if (
            not self._smart_pruning
            or self._egress_disabled()
            or not self._backend_client
            or not session_id
            or not self._backend_tracking_enabled
        ):
            return False

        policy = policy_from_config(self._traigent_config)
        return bool(policy_requires_cloud(policy) or policy_is_cloud_brain(policy))

    @staticmethod
    def _resolve_maybe_awaitable(value: Any) -> Any:
        """Resolve an awaitable from a synchronous progress callback."""
        if not inspect.isawaitable(value):
            return value

        coroutine = cast(Coroutine[Any, Any, Any], value)
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(coroutine)

        result_box: dict[str, Any] = {}

        def _runner() -> None:
            try:
                result_box["result"] = asyncio.run(coroutine)
            except BaseException as exc:  # pragma: no cover - defensive bridge
                result_box["error"] = exc

        thread = threading.Thread(
            target=_runner,
            name="traigent-smart-pruning-report",
            daemon=True,
        )
        thread.start()
        thread.join()
        error = result_box.get("error")
        if isinstance(error, BaseException):
            raise error
        return result_box.get("result")

    def report_intermediate_progress(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Send a smart-pruning intermediate report and return the prune decision."""
        session_id = payload.get("session_id")
        if not self.should_report_intermediate_progress(
            session_id if isinstance(session_id, str) else None
        ):
            return {"prune": False, "prune_reason": None}

        reporter = getattr(self._backend_client, "_report_intermediate_progress", None)
        if not callable(reporter):
            return {"prune": False, "prune_reason": None}

        try:
            decision = self._resolve_maybe_awaitable(reporter(payload))
        except Exception as exc:
            logger.debug(
                "Skipping smart-pruning decision for session %s trial %s: %s",
                payload.get("session_id"),
                payload.get("trial_id"),
                exc,
            )
            return {"prune": False, "prune_reason": None}

        if not isinstance(decision, dict):
            return {"prune": False, "prune_reason": None}

        prune_reason = decision.get("prune_reason")
        return {
            "prune": bool(decision.get("prune", False)),
            "prune_reason": (
                prune_reason if prune_reason is None else str(prune_reason)
            ),
        }

    async def submit_trial(
        self,
        trial_result: TrialResult,
        session_id: str | None,
        dataset_name: str = "dataset",
        content_scores: dict[str, dict[int, float]] | None = None,
    ) -> bool | BackendTrialSubmissionOutcome:
        """Submit trial to backend.

        Args:
            trial_result: Completed trial result
            session_id: Backend session identifier
            dataset_name: Name of the dataset (for stable example ID generation)
            content_scores: Deprecated and ignored. Content analytics now upload
                per-run features once at session creation.

        Returns:
            True if submission succeeded, False if skipped, or a terminal
            optimization-complete outcome when the backend gracefully closed the
            cloud run.
        """
        if self._egress_disabled() or not self._backend_client or not session_id:
            return False

        _ = content_scores

        primary_objective = (
            self._optimizer.objectives[0] if self._optimizer.objectives else "score"
        )

        score = trial_result.get_metric(primary_objective)
        trial_metadata = build_backend_metadata(
            trial_result,
            primary_objective,
            self._traigent_config,
            dataset_name,
            session_id=session_id,
        )
        if self._strategy_preset_metadata is not None:
            trial_metadata["strategy_preset"] = dict(self._strategy_preset_metadata)

        self._persist_trial_locally(
            session_id=session_id,
            trial_result=trial_result,
            score=score,
            metadata=trial_metadata,
        )

        if not self._backend_tracking_enabled:
            backend_disabled_label = _backend_disabled_label(
                self._backend_disabled_reason
            )
            logger.debug(
                "Skipping trial submission (backend disabled: %s)",
                backend_disabled_label,
            )
            return False

        submission_outcome = await self._log_trial_to_backend(
            session_id=session_id,
            trial_result=trial_result,
            score=score,
            metadata=trial_metadata,
        )
        if submission_outcome is not None and submission_outcome.optimization_complete:
            return submission_outcome

        return True

    def _persist_trial_locally(
        self,
        *,
        session_id: str,
        trial_result: TrialResult,
        score: float | None,
        metadata: dict[str, Any],
    ) -> None:
        """Record local analytics before any remote backend breaker can short-circuit."""

        if self._egress_disabled() or not self._backend_client or not session_id:
            return

        sanitized_score = float(score) if score is not None else None
        metadata_payload = dict(metadata)
        if score is None:
            metadata_payload["primary_objective_missing"] = True

        try:
            self._backend_client.submit_result(
                session_id=session_id,
                config=trial_result.config,
                score=sanitized_score,
                metadata=metadata_payload,
            )
        except Exception as exc:
            # #1279: a failed local write means the trial's only record is the
            # breaker-gated remote submit — if that also fails the completed
            # (paid-for) trial is lost. This must never be silent.
            logger.warning(
                "Local trial logging failed for session %s trial %s — trial has "
                "no durable local record if the remote submit also fails: %s",
                session_id,
                trial_result.trial_id,
                exc,
            )

    def is_trial_backend_acknowledged(self, session_id: str, trial_id: str) -> bool:
        """Whether the backend minted+accepted a result for ``trial_id``.

        The certified-selection report builder consults this to fail closed:
        an incumbent whose backend slot was never acknowledged cannot be bound
        to a server record, so no report is sent for it.
        """
        return (session_id, trial_id) in self._acknowledged_trials

    @staticmethod
    def _coerce_backend_trial_slot(slot: Any) -> BackendTrialSlotAcquisition:
        """Normalize legacy string/None and structured slot results."""

        if isinstance(slot, str) and slot:
            return BackendTrialSlotAcquisition(trial_id=slot)
        if getattr(slot, "optimization_complete", False) is True:
            reason = getattr(slot, "reason", None)
            return BackendTrialSlotAcquisition(
                optimization_complete=True,
                reason=str(reason) if reason else None,
            )
        trial_id = getattr(slot, "trial_id", None)
        if isinstance(trial_id, str) and trial_id:
            return BackendTrialSlotAcquisition(trial_id=trial_id)
        return BackendTrialSlotAcquisition()

    async def _acquire_backend_trial_id(
        self, session_id: str, trial_result: TrialResult
    ) -> BackendTrialSlotAcquisition:
        """Acquire (once per started trial) a backend-minted trial id.

        Reuses a previously acquired backend id for the same client trial so
        retries of submit_trial don't burn extra slots. Returns a structured
        outcome so graceful backend completion is not collapsed into the genuine
        no-slot/error path.
        """
        client = self._backend_client
        if self._egress_disabled() or client is None:
            return BackendTrialSlotAcquisition()

        client_trial_id = str(trial_result.trial_id)
        if (trial_result.metadata or {}).get("backend_trial_id_acquired"):
            self._started_trials.add((session_id, client_trial_id))
            return BackendTrialSlotAcquisition(trial_id=client_trial_id)
        # If this trial already mapped to an acknowledged backend id, reuse it.
        for sid, bid in self._started_trials:
            if sid == session_id and bid == client_trial_id:
                # client_trial_id already IS a backend id from a prior pass.
                return BackendTrialSlotAcquisition(trial_id=client_trial_id)

        requester = getattr(client, "request_trial_slot", None)
        if not callable(requester):
            return BackendTrialSlotAcquisition()
        try:
            slot = requester(session_id)
            slot_result = await slot if inspect.isawaitable(slot) else slot
        except Exception as exc:
            logger.debug(
                "Backend trial-slot request failed for session %s trial %s: %s",
                session_id,
                client_trial_id,
                exc,
            )
            return BackendTrialSlotAcquisition()

        acquisition = self._coerce_backend_trial_slot(slot_result)
        if acquisition.trial_id:
            self._started_trials.add((session_id, acquisition.trial_id))
        return acquisition

    async def _log_trial_to_backend(
        self,
        session_id: str,
        trial_result: TrialResult,
        score: float | None,
        metadata: dict[str, Any],
    ) -> BackendTrialSubmissionOutcome | None:
        """Submit trial metrics to backend when possible."""

        if self._egress_disabled() or not self._backend_client or not session_id:
            return None

        if not self._backend_tracking_enabled:
            return None

        sanitized_score = float(score) if score is not None else None
        metadata_payload = dict(metadata)
        if score is None:
            metadata_payload["primary_objective_missing"] = True

        # Skip remote submission when no API key is configured (offline/local only).
        auth_manager = getattr(self._backend_client, "auth_manager", None)
        has_api_key = bool(
            auth_manager
            and hasattr(auth_manager, "has_api_key")
            and auth_manager.has_api_key()
        )
        if not has_api_key:
            logger.debug(
                "Skipping backend trial submission for session %s trial %s (no API key)",
                session_id,
                trial_result.trial_id,
            )
            return None

        logger.debug(
            "Submitting trial %s for session %s",
            trial_result.trial_id,
            session_id,
        )

        # If the backend session was never registered (e.g., API fallback), avoid
        # posting to the SaaS endpoint to prevent 400 errors for unknown sessions.
        session_mapping = None
        get_mapping = getattr(self._backend_client, "get_session_mapping", None)
        if callable(get_mapping):
            try:
                session_mapping = get_mapping(session_id)
            except Exception as exc:  # pragma: no cover - defensive logging only
                logger.debug(
                    "Failed to retrieve backend session mapping for %s: %s",
                    session_id,
                    exc,
                )

        if session_mapping is None:
            logger.info(
                "No backend session mapping for %s; assuming offline fallback. "
                "Skipping remote submission for trial %s.",
                session_id,
                trial_result.trial_id,
            )
            return None

        # Build payload for backend session endpoint.
        metrics_payload: dict[str, Any] = dict(trial_result.metrics or {})

        # Surface summary stats / measures captured in metadata so the backend can persist them.
        if "summary_stats" in metadata and "summary_stats" not in metrics_payload:
            metrics_payload["summary_stats"] = metadata["summary_stats"]
            logger.debug(
                "Added summary_stats to metrics payload for trial %s",
                trial_result.trial_id,
            )
        if "measures" in metadata and "measures" not in metrics_payload:
            metrics_payload["measures"] = metadata["measures"]
            logger.info(
                "📊 Added %d measures to metrics payload for trial %s (status=%s)",
                len(metadata["measures"]),
                trial_result.trial_id,
                trial_result.status.value,
            )
        elif trial_result.status == TrialStatus.PRUNED:
            logger.warning(
                "⚠️ No measures found for PRUNED trial %s - metadata keys: %s",
                trial_result.trial_id,
                list(metadata.keys()),
            )

        if "score" not in metrics_payload and score is not None:
            metrics_payload["score"] = sanitized_score
        self._ensure_budget_cost_metric(
            session_id=session_id,
            trial_result=trial_result,
            metrics_payload=metrics_payload,
            metadata=metadata,
        )

        # Map SDK TrialStatus to the configuration-run wire vocab via the single
        # canonical mapper (issue #1302). This is a configuration-run submission,
        # so PRUNED (early-stopping success) is preserved rather than coerced.
        from traigent.cloud.api_operations import map_status_to_wire

        status = map_status_to_wire(trial_result.status.value, endpoint="config_run")

        if bool(getattr(self._traigent_config, "privacy_enabled", False)):
            logger.warning(
                "Skipping default backend trial submission for privacy-enabled "
                "session %s trial %s; use the privacy summary-stats submission "
                "path for remote tracking.",
                session_id,
                trial_result.trial_id,
            )
            return None

        try:
            # The backend binds a result to a session by the configuration_run
            # id IT minted (via next-trial). A client-hashed id 404s with
            # "Trial ... not found in session". So acquire a backend slot and
            # rebind this trial to it BEFORE submitting. The optimizer still
            # owns the config; the slot is purely the backend's trial handle.
            acquisition = await self._acquire_backend_trial_id(session_id, trial_result)
            if acquisition.optimization_complete:
                logger.info(
                    "Backend reported optimization complete for session %s while "
                    "submitting client trial %s (reason=%s)",
                    session_id,
                    trial_result.trial_id,
                    acquisition.reason,
                )
                return BackendTrialSubmissionOutcome.complete(acquisition.reason)
            backend_trial_id = acquisition.trial_id
            if backend_trial_id is None:
                # Fail closed: no acknowledged backend slot ⇒ NO submission and
                # NO acknowledgment. The certified-selection report will then
                # withhold for this incumbent (never attest an unbound winner).
                # A missing slot in a tracking-enabled run is the first symptom
                # of a backend outage — degrade to local-only (issue #1265).
                self._flag_backend_degraded("trial submission")
                logger.warning(
                    "No backend trial slot acquired for session %s "
                    "(client trial %s); skipping submission (fail closed)",
                    session_id,
                    trial_result.trial_id,
                )
                return None
            # Rebind in place: _best_trial_cached holds this same object, so the
            # incumbent's trial_id becomes the backend id the report needs.
            trial_result.trial_id = backend_trial_id

            # P8 content-freedom (live-E2E finding): the resolved trial
            # config carries injected CALIBRATED values; the wire submission
            # carries ONLY the tuned search-space projection — a fitted
            # calibrated value never rides to the backend (and the backend's
            # create-time key-subset validation would reject it anyway,
            # silently voiding the trial record).
            tuned_keys = set(getattr(self._optimizer, "config_space", {}) or {})
            wire_config = {
                k: v
                for k, v in (trial_result.config or {}).items()
                if not tuned_keys or k in tuned_keys
            }
            submitted_result = self._backend_client._submit_trial_result_via_session(
                session_id=session_id,
                trial_id=backend_trial_id,
                config=wire_config,
                metrics=metrics_payload,
                status=status,
                error_message=trial_result.error_message,
                execution_mode=cast(str | None, self._traigent_config.execution_mode),
                metadata=metadata_payload,
            )
            submitted = (
                await submitted_result
                if inspect.isawaitable(submitted_result)
                else submitted_result
            )
            if submitted is None:
                # None signals a transient, non-fatal skip (e.g. 400 session-not-found
                # from per-worker session storage — BE #1194).  The trial-operations
                # layer already logged an INFO message; don't flag backend degraded
                # and don't re-log here so the user isn't flooded with warnings for
                # a known transient condition.  Recover via `traigent sync`.
                pass
            elif not submitted:
                # False covers a real backend rejection (non-2xx that is not a
                # transient not-found) or a network failure; degrade to local-only
                # (#1265) so subsequent trials don't keep hitting a broken endpoint.
                if bool(getattr(submitted, "permanent_rejection", False)):
                    rejection_reason = (
                        getattr(submitted, "reason", None)
                        or "backend returned a permanent 4xx rejection"
                    )
                    self._flag_backend_degraded(
                        "trial submission", rejection_reason=rejection_reason
                    )
                    logger.warning(
                        "Backend session endpoint rejected trial %s for session %s: %s",
                        backend_trial_id,
                        session_id,
                        rejection_reason,
                    )
                else:
                    self._flag_backend_degraded("trial submission")
                    logger.warning(
                        "Backend session endpoint did not accept trial %s for session %s",
                        backend_trial_id,
                        session_id,
                    )
            else:
                # Record the acknowledged backend slot so the certified-selection
                # report can verify this incumbent is bindable.
                self._acknowledged_trials.add((session_id, backend_trial_id))
                logger.info(
                    "Submitted trial %s for session %s (status=%s, metrics_keys=%s)",
                    backend_trial_id,
                    session_id,
                    status,
                    sorted(metrics_payload.keys()),
                )
        except Exception as exc:
            # A raised error here is a backend interaction failure; the run is
            # no longer cloud-tracked for this trial, so degrade to local-only
            # (issue #1265) rather than retrying the backend every trial.
            self._flag_backend_degraded("trial submission")
            logger.warning(
                "Failed to submit trial %s for session %s to backend: %s",
                trial_result.trial_id,
                session_id,
                exc,
            )

    async def update_weighted_scores(
        self,
        result: OptimizationResult,
        session_id: str | None,
    ) -> int:
        """Update backend with weighted scores for multi-objective runs.

        Args:
            result: Optimization result with all trials
            session_id: Backend session identifier

        Returns:
            Number of trials successfully updated
        """
        if (
            self._egress_disabled()
            or not self._backend_client
            or session_id is None
            or len(self._objectives) <= 1
        ):
            return 0

        if not self._backend_tracking_enabled:
            backend_disabled_label = _backend_disabled_label(
                self._backend_disabled_reason
            )
            logger.debug(
                "Skipping weighted score updates (backend disabled: %s)",
                backend_disabled_label,
            )
            return 0

        backend_client = self._backend_client  # Capture for use in nested function

        try:
            logger.info("Calculating weighted scores for multi-objective optimization")

            if self._objective_schema:
                objective_weights = dict(self._objective_schema.weights_normalized)
            else:
                objective_weights = {
                    obj: 1.0 / len(self._objectives) for obj in self._objectives
                }

            weighted_results = result.calculate_weighted_scores(
                objective_weights=objective_weights
            )

            weighted_scores_data = list(weighted_results.get("weighted_scores", []))
            successful_trials_iter = iter(
                getattr(result, "successful_trials", result.trials)
            )

            weighted_updates: list[tuple[TrialResult, float]] = []

            for weighted_entry in weighted_scores_data:
                weighted_trial = None
                weighted_score = None

                if isinstance(weighted_entry, tuple):
                    candidate_trial = weighted_entry[0]
                    if hasattr(candidate_trial, "trial_id"):
                        weighted_trial = candidate_trial
                        if len(weighted_entry) > 1:
                            weighted_score = weighted_entry[1]
                    if weighted_score is None and len(weighted_entry) > 0:
                        weighted_score = weighted_entry[-1]
                else:
                    weighted_score = weighted_entry

                if weighted_trial is None:
                    try:
                        weighted_trial = next(successful_trials_iter)
                    except StopIteration:
                        logger.warning(
                            "No matching successful trial available for weighted score entry %s; skipping submission",
                            weighted_entry,
                        )
                        continue

                if weighted_score is None:
                    logger.debug(
                        "Weighted score entry %s lacked explicit score; defaulting to 0.0",
                        weighted_entry,
                    )
                    weighted_score = 0.0

                try:
                    weighted_score_value = float(weighted_score)
                except (TypeError, ValueError):
                    logger.warning(
                        "Unable to coerce weighted score %s for trial %s; skipping submission",
                        weighted_score,
                        getattr(weighted_trial, "trial_id", "unknown"),
                    )
                    continue

                weighted_updates.append((weighted_trial, weighted_score_value))

            attempted_updates = len(weighted_updates)

            if attempted_updates == 0:
                logger.info("No weighted score updates to submit")
                if hasattr(result, "metadata"):
                    if not result.metadata:
                        result.metadata = {}
                    result.metadata["weighted_results"] = {
                        "best_weighted_config": weighted_results.get(
                            "best_weighted_config"
                        ),
                        "best_weighted_score": weighted_results.get(
                            "best_weighted_score"
                        ),
                        "objective_weights": objective_weights,
                        "trials_attempted": 0,
                        "trials_updated": 0,
                        "trials_failed": 0,
                        "failed_trial_ids": [],
                    }
                return 0

            concurrency_limit = getattr(
                backend_client, "weighted_update_concurrency", 8
            )
            try:
                concurrency_limit_int = int(concurrency_limit)
            except (TypeError, ValueError):
                concurrency_limit_int = 8
            if concurrency_limit_int <= 0:
                concurrency_limit_int = 1

            semaphore = asyncio.Semaphore(concurrency_limit_int)
            normalization_info = weighted_results.get("normalization_ranges")

            async def submit_weighted_score(
                trial: TrialResult, score_value: float
            ) -> bool:
                async with semaphore:
                    trial_identifier = getattr(trial, "trial_id", "unknown")
                    try:
                        success = await backend_client.update_trial_weighted_scores(
                            trial_id=trial_identifier,
                            weighted_score=score_value,
                            normalization_info=normalization_info,
                            objective_weights=objective_weights,
                        )
                    except Exception as exc:  # pragma: no cover - defensive
                        if self._should_suppress_backend_warnings():
                            logger.debug(
                                "Exception while updating weighted score for trial %s: %s (backend unavailable)",
                                trial_identifier,
                                exc,
                            )
                        else:
                            logger.warning(
                                "Exception while updating weighted score for trial %s: %s",
                                trial_identifier,
                                exc,
                            )
                        return False

                    if success:
                        logger.debug(
                            "Updated trial %s with weighted score %.4f",
                            trial_identifier,
                            score_value,
                        )
                        return True

                    # Suppress warnings in mock mode or when no API key configured
                    # This prevents noisy output for users evaluating SDK without backend
                    if self._should_suppress_backend_warnings():
                        logger.debug(
                            "Failed to update weighted score for trial %s (backend unavailable or mock mode)",
                            trial_identifier,
                        )
                    else:
                        logger.warning(
                            "Failed to update weighted score for trial %s",
                            trial_identifier,
                        )
                    return False

            update_results = await asyncio.gather(
                *(
                    submit_weighted_score(trial, score)
                    for trial, score in weighted_updates
                )
            )
            update_count = sum(1 for result_flag in update_results if result_flag)
            # Traigent#1724: a partial failure previously vanished — only
            # trials_updated was ever persisted, so e.g. 3/10 successes and
            # 7/10 silent drops looked identical to "nothing attempted".
            # Persist attempted count and the specific failed trial ids too.
            failed_trial_ids = [
                getattr(trial, "trial_id", "unknown")
                for (trial, _score), result_flag in zip(
                    weighted_updates, update_results, strict=True
                )
                if not result_flag
            ]

            logger.info(
                "Updated %s/%s trials with weighted scores (concurrency=%s)",
                update_count,
                attempted_updates,
                concurrency_limit_int,
            )

            if hasattr(result, "metadata"):
                if not result.metadata:
                    result.metadata = {}
                result.metadata["weighted_results"] = {
                    "best_weighted_config": weighted_results.get(
                        "best_weighted_config"
                    ),
                    "best_weighted_score": weighted_results.get("best_weighted_score"),
                    "objective_weights": objective_weights,
                    "trials_attempted": attempted_updates,
                    "trials_updated": update_count,
                    "trials_failed": len(failed_trial_ids),
                    "failed_trial_ids": failed_trial_ids,
                }

            return update_count

        except Exception as exc:  # pragma: no cover - defensive logging
            logger.error("Error updating weighted scores: %s", exc)
            return 0

    def submit_session_aggregation(
        self, result: OptimizationResult, session_id: str | None
    ) -> bool | None:
        """Submit aggregated summary for non-edge modes.

        Args:
            result: Final optimization result
            session_id: Backend session identifier

        Returns:
            ``True`` if the session-level rollup was actually transmitted to
            the backend; ``False`` if a rollup payload was built but only
            reached the local-only ``BackendIntegratedClient.submit_result``
            shim (Traigent#1724 — this never leaves the process today, so
            callers must not treat it as persisted); ``None`` if aggregation
            was skipped entirely (offline/local mode, no session, or no
            session summary to submit).
        """
        if (
            self._egress_disabled()
            or not self._backend_client
            or session_id is None
            or self._traigent_config.is_local_mode()
            or not self._backend_tracking_enabled
        ):
            return None

        if (
            not self._backend_client
            or session_id is None
            or self._traigent_config.is_local_mode()
            or not self._backend_tracking_enabled
        ):
            return None

        session_summary = (
            result.metadata.get("session_summary")
            if hasattr(result, "metadata") and result.metadata
            else None
        )

        if not session_summary:
            return None

        aggregated_trial_id = f"{session_id}_AGG_SUMMARY"

        samples_per_config = session_summary.get("samples_per_config", {})
        total_examples = sum(samples_per_config.values()) if samples_per_config else 0

        summary_stats_with_aggregation = {
            "metrics": session_summary.get("metrics", {}),
            "execution_time": result.duration,
            "total_examples": total_examples,
            "metadata": {
                "aggregation_level": "session",
                "aggregation_summary": session_summary,
                "trial_id": aggregated_trial_id,
                "sdk_version": get_version(),
            },
        }

        # Include statistical significance badges if computed
        stat_sig = (
            result.metadata.get("statistical_significance") if result.metadata else None
        )
        if stat_sig:
            agg_meta = summary_stats_with_aggregation["metadata"]
            agg_meta["statistical_significance"] = stat_sig

        try:
            successful_trials = len([t for t in result.trials if t.is_successful])
            overlay_metrics: dict[str, float] = {
                "run_trials_completed": len(result.trials),
                "run_successful_trials": successful_trials,
                "run_success_rate": result.success_rate,
            }
            if isinstance(result.metrics, dict):
                for key, value in result.metrics.items():
                    if isinstance(value, (int, float)):
                        overlay_metrics[f"run_{key}"] = value
        except (KeyError, TypeError, ValueError) as exc:
            logger.debug("Failed building overlay run metrics: %s", exc)
            overlay_metrics = {}

        submission_metadata: dict[str, Any] = {
            "summary_stats": summary_stats_with_aggregation,
            "trial_id": aggregated_trial_id,
            **overlay_metrics,
        }
        if self._strategy_preset_metadata is not None:
            submission_metadata["strategy_preset"] = dict(
                self._strategy_preset_metadata
            )

        # NOTE (Traigent#1724): BackendIntegratedClient.submit_result is a
        # local-only compatibility shim (writes local_storage + bumps an
        # in-memory counter) — it never makes a network call. This rollup is
        # therefore built but NOT transmitted to the backend today. Return
        # False so the caller (orchestrator._finalize_optimization) does not
        # report persistence_status="succeeded" for a submission that never
        # left the process. Folding this rollup into the existing
        # POST /sessions/{id}/finalize payload (session_operations.py
        # finalize_body) is the real fix and is tracked separately; it needs
        # backend-side schema support this SDK change cannot verify alone.
        self._backend_client.submit_result(
            session_id=session_id,
            config=cast(dict[str, Any], result.best_config),
            score=result.best_score,
            metadata=submission_metadata,
        )
        logger.debug(
            "Submitted session aggregation with overlay metrics: %s",
            list(overlay_metrics.keys()),
        )
        logger.info(
            "Session aggregation rollup for session_id=%s was built locally but "
            "not transmitted to the backend (local-only submission path); "
            "trial-level results and session finalize are unaffected.",
            session_id,
        )
        return False

    def finalize_session(
        self,
        session_id: str | None,
        optimization_status: OptimizationStatus,
        certified_selection: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        """Finalize backend session and return summary.

        Args:
            session_id: Backend session identifier
            optimization_status: Final optimization status

        Returns:
            Session summary metadata (or None if backend disabled)
        """
        if (
            self._egress_disabled()
            or not self._backend_client
            or session_id is None
            or not self._backend_tracking_enabled
        ):
            return None

        if (
            not self._backend_client
            or session_id is None
            or not self._backend_tracking_enabled
        ):
            return None

        final_status = (
            "completed"
            if optimization_status == OptimizationStatus.COMPLETED
            else "failed"
        )

        # The certified report only accompanies a COMPLETED finalize — a
        # failed run must never carry a certified winner.
        report = certified_selection if final_status == "completed" else None

        last_error: BaseException | None = None
        for attempt in range(1, _FINALIZE_MAX_ATTEMPTS + 1):
            try:
                if hasattr(self._backend_client, "finalize_session_sync"):
                    result: dict[str, Any] | None = (
                        self._backend_client.finalize_session_sync(  # type: ignore[assignment]
                            session_id,
                            final_status == "completed",
                            certified_selection=report,
                        )
                    )
                else:
                    result = self._backend_client.finalize_session(  # type: ignore[assignment]
                        session_id,
                        final_status == "completed",
                        certified_selection=report,
                    )

                if _finalize_acknowledged(result):
                    return result
                raise _BackendFinalizeNotAcknowledgedError(
                    "backend did not acknowledge session finalization"
                )
            except Exception as exc:
                last_error = exc
                if (
                    attempt >= _FINALIZE_MAX_ATTEMPTS
                    or not _is_transient_finalize_error(exc)
                ):
                    raise

                delay = _FINALIZE_BACKOFF_SECONDS[
                    min(attempt - 1, len(_FINALIZE_BACKOFF_SECONDS) - 1)
                ]
                logger.warning(
                    "Backend session finalize failed transiently for session %s "
                    "(attempt %s/%s); retrying in %.1fs: %s",
                    session_id,
                    attempt,
                    _FINALIZE_MAX_ATTEMPTS,
                    delay,
                    exc,
                )
                time.sleep(delay)

        if last_error is not None:
            raise last_error
        raise _BackendFinalizeNotAcknowledgedError(
            "backend did not acknowledge session finalization"
        )

    def attach_session_metadata(
        self,
        result: OptimizationResult,
        session_id: str | None,
        session_summary: dict[str, Any] | None,
    ) -> None:
        """Attach session identifiers and summary to result metadata.

        Args:
            result: Optimization result to update
            session_id: Backend session identifier
            session_summary: Session summary from backend
        """
        if (
            session_id is None
            or self._egress_disabled()
            or not hasattr(result, "metadata")
        ):
            return

        if not result.metadata:
            result.metadata = {}
        update_payload: dict[str, Any] = {"local_session_id": session_id}
        if session_summary is not None:
            update_payload["local_session_summary"] = session_summary
            # session_summary may be a plain dict OR an OptimizationFinalizationResponse
            # dataclass (the real cloud finalize path), which exposes ``.metadata`` as an
            # attribute and has no ``.get()``. Calling ``.get()`` on the dataclass raised
            # AttributeError, which the persistence try/except swallowed as
            # "persistence failed" on every cloud run. Guard the type before extracting.
            if isinstance(session_summary, dict):
                summary_metadata = session_summary.get("metadata")
            else:
                summary_metadata = getattr(session_summary, "metadata", None)
            if isinstance(summary_metadata, dict):
                warm_start_transfer = summary_metadata.get("warm_start_transfer")
                if isinstance(warm_start_transfer, dict):
                    update_payload["warm_start_transfer"] = dict(warm_start_transfer)
        if owning_context := self._session_owning_context.get(session_id):
            update_payload.update(owning_context)

        # Add experiment_id from session mapping if available
        if self._backend_client is not None:
            try:
                mapping = self._backend_client.get_session_mapping(session_id)
                if mapping is not None:
                    update_payload["experiment_id"] = mapping.experiment_id
                    update_payload["experiment_run_id"] = mapping.experiment_run_id
            except Exception:
                pass  # Silently ignore if mapping not available

        result.metadata.update(update_payload)
        self._warn_on_refused_warm_start(result.metadata)

    @staticmethod
    def _warn_on_refused_warm_start(result_metadata: dict[str, Any]) -> None:
        """Warn loudly when an explicitly requested warm start applied 0 seeds.

        #1683: a user who set ``warm_start_from`` must not silently get a cold
        start. Emits aggregate info only (prior experiment id, refused_reason,
        seed count) — never seed contents.
        """
        warm_start_from = result_metadata.get("warm_start_from")
        transfer = result_metadata.get("warm_start_transfer")
        if not warm_start_from or not isinstance(transfer, dict):
            return
        refused_reason = transfer.get("refused_reason")
        n_applied = transfer.get("n_seed_configs_applied")
        zero_applied = (
            isinstance(n_applied, int)
            and not isinstance(n_applied, bool)
            and n_applied == 0
        )
        if refused_reason or zero_applied:
            logger.warning(
                "warm_start_from=%r was requested but the backend applied no "
                "seed configs (refused_reason=%r, n_seed_configs_applied=%r). "
                "The run effectively started cold.",
                warm_start_from,
                refused_reason,
                n_applied,
            )
