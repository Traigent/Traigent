"""Runtime helpers for execution-policy enforcement."""

from __future__ import annotations

import os
import re
from collections.abc import Iterable

from traigent.config.types import (
    ExecutionIntent,
    ResolvedExecutionPolicy,
    TraigentConfig,
    normalize_algorithm_name,
)
from traigent.core.session_types import (
    SessionCreationFailureReason,
    SessionCreationResult,
)
from traigent.utils.env_config import is_backend_offline
from traigent.utils.exceptions import OptimizationError

SOURCE_CLOUD_BRAIN = "cloud_brain"
SOURCE_LOCAL_FALLBACK = "local_fallback"
SOURCE_EXPLICIT_LOCAL = "explicit_local"
SOURCE_OFFLINE = "offline"

RESULT_SOURCES = frozenset(
    {
        SOURCE_CLOUD_BRAIN,
        SOURCE_LOCAL_FALLBACK,
        SOURCE_EXPLICIT_LOCAL,
        SOURCE_OFFLINE,
    }
)

_BACKEND_OPTIMIZATION_STRATEGIES: dict[str, dict[str, str]] = {
    "bayesian": {"algorithm": "optuna", "sampler": "tpe"},
    "tpe": {"algorithm": "optuna", "sampler": "tpe"},
    "optuna": {"algorithm": "optuna", "sampler": "tpe"},
    "optuna_tpe": {"algorithm": "optuna", "sampler": "tpe"},
    "optuna_random": {"algorithm": "optuna", "sampler": "random"},
}

CONNECTIVITY_STATUSES = frozenset({500, 502, 503, 504})
HARD_FAILURE_STATUSES = frozenset({400, 401, 402, 403, 404, 405, 409, 415, 422, 429})

CONNECTIVITY_PATTERNS = (
    "backend unavailable",
    "service unavailable",
    "health unavailable",
    "health check failed",
    "connection failed",
    "connect failed",
    "connection reset",
    "connection refused",
    "network error",
    "dns",
    "name resolution",
    "timed out",
    "timeout",
    "temporary failure",
    "502",
    "503",
    "504",
)

HARD_FAILURE_PATTERNS = (
    "401",
    "402",
    "403",
    "409",
    "422",
    "429",
    "unauthorized",
    "forbidden",
    "authentication",
    "invalid api key",
    "revoked",
    "expired",
    "billing",
    "payment required",
    "quota",
    "rate limit",
    "tenant disabled",
    "tenant-disabled",
    "policy denial",
    "content denial",
    "content policy",
    "schema mismatch",
    "invalid request",
    "validation",
)


class CloudBrainUnavailableError(OptimizationError):
    """Raised when the cloud optimization brain is unavailable.

    ``failure_reason`` carries the typed classification when the raiser has
    one, so a handler that re-stamps the run as locally-degraded keeps the
    classification instead of re-deriving it from ``reason`` — a string that
    embeds backend-controlled text (issue #2024).
    """

    def __init__(
        self,
        stage: str,
        reason: str,
        *,
        original: BaseException | None = None,
        failure_reason: SessionCreationFailureReason | None = None,
    ) -> None:
        super().__init__(f"Cloud brain unavailable during {stage}: {reason}")
        self.stage = stage
        self.reason = reason
        self.original = original
        self.failure_reason = failure_reason


def backend_optimization_strategy_for_algorithm(
    algorithm: str | None,
) -> dict[str, str] | None:
    """Return the typed backend strategy for SDK-dispatched smart algorithms."""

    strategy = _BACKEND_OPTIMIZATION_STRATEGIES.get(normalize_algorithm_name(algorithm))
    return dict(strategy) if strategy is not None else None


def backend_supported_smart_algorithms() -> tuple[str, ...]:
    """Return smart algorithm names the SDK can dispatch to the backend today."""

    return tuple(sorted(_BACKEND_OPTIMIZATION_STRATEGIES))


def unsupported_backend_smart_algorithm_message(algorithm: str | None) -> str:
    """Explain why a known smart algorithm is rejected before session create."""

    requested = normalize_algorithm_name(algorithm)
    supported = ", ".join(f"'{name}'" for name in backend_supported_smart_algorithms())
    return (
        f"Smart optimization ('{requested}') is not available as a first-party "
        "Traigent backend strategy yet. The SDK did not create a backend "
        f"session. Use one of the backend-guided smart algorithms ({supported}), "
        "or use algorithm='grid' / algorithm='random' for local optimization."
    )


def env_requires_cloud() -> bool:
    """Whether the environment requires cloud and forbids local fallback."""

    return str(os.environ.get("TRAIGENT_REQUIRE_CLOUD", "")).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def policy_from_config(config: TraigentConfig | None) -> ResolvedExecutionPolicy | None:
    """Read the resolved execution policy from a config object."""

    if config is None:
        return None
    policy = getattr(config, "execution_policy", None)
    return policy if isinstance(policy, ResolvedExecutionPolicy) else None


def policy_requires_cloud(policy: ResolvedExecutionPolicy | None) -> bool:
    """Return true when execution must not fall back to local."""

    return bool(
        env_requires_cloud()
        or (
            policy is not None
            and (
                policy.require_cloud or policy.intent is ExecutionIntent.CLOUD_REQUIRED
            )
        )
    )


def policy_allows_cloud_fallback(policy: ResolvedExecutionPolicy | None) -> bool:
    """Return true when cloud-brain connectivity failures may degrade locally."""

    return bool(
        policy is not None
        and policy.intent is ExecutionIntent.CLOUD_BRAIN
        and not policy_requires_cloud(policy)
    )


def policy_is_cloud_brain(policy: ResolvedExecutionPolicy | None) -> bool:
    """Return true for automatic cloud-brain runs."""

    return bool(policy is not None and policy.intent is ExecutionIntent.CLOUD_BRAIN)


def policy_is_cloud_required(policy: ResolvedExecutionPolicy | None) -> bool:
    """Return true for explicit smart algorithms that require cloud."""

    return bool(policy is not None and policy.intent is ExecutionIntent.CLOUD_REQUIRED)


def is_offline_requested(policy: ResolvedExecutionPolicy | None = None) -> bool:
    """Return true when any policy/env source requires no backend egress."""

    return bool(is_backend_offline() or (policy is not None and policy.offline))


def backend_egress_disabled(config: TraigentConfig | None) -> bool:
    """Return true when backend client/session/trial/trace creation is forbidden."""

    if config is None:
        return is_backend_offline()
    policy = policy_from_config(config)
    if is_offline_requested(policy):
        return True
    if bool(getattr(config, "no_egress", False)):
        return True
    return False


def initial_result_source(
    policy: ResolvedExecutionPolicy | None,
    *,
    external_evaluator: bool = False,
) -> str:
    """Resolve the initial provenance source for a run."""

    if is_offline_requested(policy):
        return SOURCE_OFFLINE
    if external_evaluator:
        return SOURCE_EXPLICIT_LOCAL
    if policy is None:
        return SOURCE_EXPLICIT_LOCAL
    if policy.intent is ExecutionIntent.LOCAL_ONLY:
        return SOURCE_EXPLICIT_LOCAL
    return SOURCE_CLOUD_BRAIN


def fallback_reason_from_session_result(result: SessionCreationResult) -> str:
    """Build a non-secret fallback reason from a failed session result."""

    parts: list[str] = []
    if result.failure_reason is not None:
        parts.append(result.failure_reason.value)
    if result.failure_response is not None:
        parts.append(result.failure_response.one_line_summary())
    elif result.failure_detail:
        parts.append(str(result.failure_detail))
    return ": ".join(parts) or "backend unavailable"


def failure_reason_from_code(code: str | None) -> SessionCreationFailureReason | None:
    """Rehydrate the typed failure reason stamped on a result's metadata.

    The only accepted input is a value this SDK itself wrote (see
    ``mark_local_fallback``), so this is an enum round-trip across the
    result-metadata boundary, not a classification of backend prose. An
    unrecognised or absent code yields ``None``, which makes reporting fall
    back to the reason-agnostic notice — a statement that is true for every
    local fallback — rather than guessing a remedy.
    """

    if not code:
        return None
    try:
        return SessionCreationFailureReason(code)
    except ValueError:
        return None


def local_fallback_notice(
    reason: str | None,
    failure_reason: SessionCreationFailureReason | None,
) -> str:
    """User-facing, one-paragraph explanation of a local-fallback run.

    Issue #2024: the machine-readable ``traigent.cloud_brain_fallback`` line
    tells a maintainer what happened but tells a user nothing, so an ``auto``
    run that degraded to a local search reads exactly like managed
    optimization. Single formatter so the session-create warning and the
    reporting-time banner cannot drift apart.

    ``failure_reason`` is the TYPED classification recorded when the run
    degraded, and it alone selects the remedy. It is deliberately never
    re-derived from ``reason``: that string embeds
    ``SessionCreationFailureDetail.one_line_summary()``, i.e.
    backend-controlled ``error_code``/``message``/``raw_body`` text, so a 500
    body mentioning ``no_api_key`` would otherwise tell a user with a valid
    key to set the key they already set. Only the no-key branch mentions
    ``TRAIGENT_API_KEY``; the other branch already knows the key is not the
    problem.
    """

    reason_text = (reason or "backend unavailable").strip()
    if failure_reason is SessionCreationFailureReason.NO_API_KEY:
        lead = (
            "No Traigent API key was found, so this ran a LOCAL search on your "
            "machine, not Traigent's managed optimization."
        )
        remedy = (
            "Set TRAIGENT_API_KEY to run managed optimization, or "
            "TRAIGENT_REQUIRE_CLOUD=1 to fail instead of falling back."
        )
    else:
        lead = (
            "Traigent's managed optimization was unavailable, so this ran a "
            "LOCAL search on your machine."
        )
        remedy = "Set TRAIGENT_REQUIRE_CLOUD=1 to fail instead of falling back."
    return (
        f"{lead} The reported best configuration was chosen by that local "
        f"search (reason: {reason_text}). {remedy}"
    )


def _combined_session_text(result: SessionCreationResult) -> str:
    detail = result.failure_response
    return " ".join(
        str(part)
        for part in (
            result.failure_reason.value if result.failure_reason else None,
            result.failure_detail,
            detail.status_code if detail else None,
            detail.error_code if detail else None,
            detail.message if detail else None,
            detail.raw_body if detail else None,
        )
        if part is not None
    ).lower()


def _contains_any(text: str, needles: Iterable[str]) -> bool:
    return any(needle in text for needle in needles)


def session_failure_is_connectivity(result: SessionCreationResult) -> bool:
    """Classify whether session-create failure may locally fall back."""

    if result.backend_connected:
        return False
    if result.failure_reason is SessionCreationFailureReason.NO_API_KEY:
        return True
    detail = result.failure_response
    status = detail.status_code if detail else None
    if status in HARD_FAILURE_STATUSES:
        return False
    text = _combined_session_text(result)
    if _contains_any(text, HARD_FAILURE_PATTERNS):
        return False
    if status in CONNECTIVITY_STATUSES:
        return True
    return bool(_contains_any(text, CONNECTIVITY_PATTERNS))


def session_failure_is_session_create_400(result: SessionCreationResult) -> bool:
    """Return true for the auto-contract typed+legacy HTTP 400 failure."""

    if not result.typed_legacy_session_create_400:
        return False
    detail = result.failure_response
    if detail and detail.status_code == 400:
        return True
    text = _combined_session_text(result)
    return "http 400" in text or "status 400" in text


def exception_status(exc: BaseException) -> int | None:
    """Extract an HTTP status code from common cloud/backend exception shapes."""

    for name in ("status", "status_code", "http_status"):
        value = getattr(exc, name, None)
        if isinstance(value, int):
            return value
    # httpx.HTTPStatusError and similar carry the code on .response.status_code
    response = getattr(exc, "response", None)
    response_status = getattr(response, "status_code", None)
    if isinstance(response_status, int):
        return response_status
    match = re.search(r"\bHTTP\s+(\d{3})\b", str(exc), flags=re.IGNORECASE)
    if match:
        return int(match.group(1))
    return None


def exception_is_connectivity(exc: BaseException) -> bool:
    """Return true only for connectivity/service-unavailable failures."""

    current: BaseException | None = exc
    visited: set[int] = set()
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        status = exception_status(current)
        text = str(current).lower()
        # A permanent client error (4xx) is NOT a connectivity failure and must
        # not trigger a silent local fallback -- even when wrapped in a
        # CloudBrainUnavailableError (e.g. a trial-result submission rejected
        # with HTTP 400 for an unknown objective metric). Surface it loudly.
        if status in HARD_FAILURE_STATUSES or _contains_any(
            text, HARD_FAILURE_PATTERNS
        ):
            return False
        if isinstance(current, CloudBrainUnavailableError):
            return True
        if status in CONNECTIVITY_STATUSES or _contains_any(
            text, CONNECTIVITY_PATTERNS
        ):
            return True
        current = current.__cause__ or current.__context__

    return isinstance(exc, (ConnectionError, TimeoutError, OSError))


def mark_local_fallback(
    config: TraigentConfig,
    reason: str,
    *,
    failure_reason: SessionCreationFailureReason | None = None,
    no_egress: bool = True,
) -> None:
    """Stamp a config as degraded to local fallback.

    ``failure_reason`` is the typed classification, when the caller has one.
    It is recorded alongside the human ``reason`` so reporting can pick the
    right remedy without re-parsing prose the backend controls. The two are
    always written together: a caller with no typed classification clears the
    code rather than leaving the previous fallback's code next to a new
    reason.
    """

    config.result_source = SOURCE_LOCAL_FALLBACK
    config.fallback_reason = reason
    config.fallback_reason_code = (
        failure_reason.value if failure_reason is not None else None
    )
    config.no_egress = no_egress
