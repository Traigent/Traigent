"""Structured types for backend session creation outcomes.

Leaf module with no cloud-package imports so core orchestration code can
depend on it even when the optional cloud package is unavailable.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

# Closed vocabulary of quota/limit error codes safe to render. Anything else
# (including backend-invented strings) is shown as "unknown" so a quota
# diagnostic can never carry backend or customer text (#2550).
QUOTA_ERROR_CODES = frozenset(
    {
        "quota_exceeded",
        "quota_limit_exceeded",
        "plan_limit_exceeded",
        "usage_limit_exceeded",
    }
)


# Closed vocabulary of backend refusals that are definitive answers from a
# reachable backend (e.g. HTTP 503 for a session shape the backend will not
# create). Never connectivity, never retried on the legacy contract, and the
# code (only) may be rendered (SDK-52).
CLOSED_REFUSAL_CODES = frozenset(
    {
        "typed_storage_mapping_unavailable",
        "director_tokens_required",
        "director_opaque_ids_required",
        "content_identity_keys_unavailable",
        # The workspace is limited to optimization (see
        # traigent.utils.optimization_only_egress): a closed 403, never retried or
        # degraded to a different request shape or a local fallback.
        "optimization_only",
    }
)


# Closed vocabulary of generic backend error codes that may appear in an SDK
# error message. Anything else is rendered as "unknown" so backend-invented
# text (which can echo customer content) never reaches an exception or a log
# (#2552).
SAFE_HTTP_ERROR_CODES = (
    QUOTA_ERROR_CODES
    | CLOSED_REFUSAL_CODES
    | frozenset(
        {
            "bad_request",
            "conflict",
            "forbidden",
            "insufficient_permissions",
            "internal_error",
            "invalid_request",
            "not_found",
            "rate_limited",
            "service_unavailable",
            "session_not_found",
            "unauthorized",
            "validation_error",
        }
    )
)


def safe_http_error_summary(status_code: int | None, raw_body: str | None) -> str:
    """Content-free summary of a backend HTTP error: status and closed code only.

    Never includes the backend message or body text. The error code is shown
    only when it is in ``SAFE_HTTP_ERROR_CODES``; otherwise ``unknown``.
    """

    code = "unknown"
    if raw_body:
        try:
            parsed = json.loads(raw_body)
        except (TypeError, ValueError):
            parsed = None
        if isinstance(parsed, dict):
            raw = parsed.get("error_code") or parsed.get("code")
            if isinstance(raw, str) and raw.strip().lower() in SAFE_HTTP_ERROR_CODES:
                code = raw.strip().lower()
    parts = []
    if status_code is not None:
        parts.append(f"HTTP {status_code}")
    parts.append(code)
    return " | ".join(parts)


class SessionCreationFailureReason(Enum):
    """Why backend session creation failed."""

    AUTH = "auth"  # 401/403 — non-retryable
    NO_API_KEY = "no_api_key"  # known before any HTTP call  # pragma: allowlist secret
    SESSION_FAILED = "session_failed"  # transient: 5xx, connection, timeout


class SessionCreationFailureClassification(Enum):
    """User-facing classification for a failed backend session creation."""

    MISSING_PERMISSION = "missing-permission"
    INSUFFICIENT_SCOPE = "insufficient-scope"
    INVALID_OR_REVOKED_KEY = "invalid-or-revoked-key"
    EXPIRED_KEY = "expired-key"
    EDGE_BLOCKED = "edge-blocked"
    NETWORK = "network"
    KEY_NOT_FOUND = "key-not-found"
    BACKEND_UNREACHABLE = "backend-unreachable"
    QUOTA_EXCEEDED = "quota-exceeded"
    BACKEND_REFUSED = "backend-refused"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class SessionCreationFailureDetail:
    """Structured backend response details for failed session creation."""

    status_code: int | None = None
    error_code: str | None = field(default=None, repr=False)
    message: str | None = field(default=None, repr=False)
    missing_permissions: tuple[str, ...] = field(default=(), repr=False)
    raw_body: str | None = field(default=None, repr=False)
    backend_url: str | None = None

    def __repr__(self) -> str:
        # Content-free: backend message, raw body, free-form code and permission
        # strings can echo customer text (#2552).
        return (
            f"SessionCreationFailureDetail(status_code={self.status_code!r}, "
            f"error_code={self.safe_error_code!r})"
        )

    __str__ = __repr__

    @classmethod
    def from_http_response(
        cls,
        status_code: int,
        raw_body: str | None,
        *,
        backend_url: str | None = None,
    ) -> SessionCreationFailureDetail:
        """Parse the backend's structured session-creation error response."""

        payload: dict[str, Any] = {}
        if raw_body:
            try:
                parsed = json.loads(raw_body)
            except (TypeError, ValueError):
                parsed = None
            if isinstance(parsed, dict):
                payload = parsed

        details = payload.get("details")
        if not isinstance(details, dict):
            details = {}

        raw_missing = details.get("missing_permissions")
        if isinstance(raw_missing, (list, tuple, set)):
            missing_permissions = tuple(str(item) for item in raw_missing)
        else:
            missing_permissions = ()

        error_code = payload.get("error_code") or payload.get("code")
        if not error_code:
            # Director/content-identity refusals nest the closed code as
            # ``{"error": {"code": ...}}`` (structured field only, never prose).
            nested = payload.get("error")
            if isinstance(nested, dict):
                error_code = nested.get("code")
        message = payload.get("message") or payload.get("error")

        return cls(
            status_code=status_code,
            error_code=str(error_code) if error_code else None,
            message=str(message) if message else None,
            missing_permissions=missing_permissions,
            raw_body=raw_body,
            backend_url=backend_url,
        )

    @property
    def is_quota_refusal(self) -> bool:
        """True when the backend refused because a plan quota/limit was hit.

        A quota refusal is a definitive, reachable-backend answer (typically
        HTTP 429 or 402 with the closed ``quota_exceeded`` error code), never a
        connectivity failure. Classification uses only the status and the
        backend's closed ``error_code`` (message text is consulted only when the
        backend sent no code).
        """

        code = (self.error_code or "").strip().lower()
        if code:
            # Every declared code is recognized (#2550). An undeclared code that
            # merely contains "quota" counts as a refusal only on HTTP 402/429
            # (rendered as "unknown" so its backend text is never echoed); on any
            # other status it follows the normal connectivity path.
            return code in QUOTA_ERROR_CODES or (
                self.status_code in (402, 429) and "quota" in code
            )
        return (
            self.status_code in (402, 429) and "quota" in (self.message or "").lower()
        )

    @property
    def is_closed_refusal(self) -> bool:
        """True when the backend refused with a closed refusal code (SDK-52).

        Read only from the structured ``error_code`` field; message text is
        never consulted. A closed refusal is a reachable-backend answer, never a
        connectivity failure, so it is not retried on the legacy contract.
        """

        code = (self.error_code or "").strip().lower()
        return code in CLOSED_REFUSAL_CODES

    @property
    def safe_quota_error_code(self) -> str:
        """The error code if it is in the closed quota vocabulary, else 'unknown'."""

        code = (self.error_code or "").strip().lower()
        return code if code in QUOTA_ERROR_CODES else "unknown"

    def quota_summary(self) -> str:
        """Content-free quota-refusal summary: status and closed error code only."""

        parts = []
        if self.status_code is not None:
            parts.append(f"HTTP {self.status_code}")
        parts.append(self.safe_quota_error_code)
        return " | ".join(parts)

    @property
    def safe_error_code(self) -> str:
        """The error code if it is in the closed vocabulary, else 'unknown'."""

        code = (self.error_code or "").strip().lower()
        return code if code in SAFE_HTTP_ERROR_CODES else "unknown"

    def one_line_summary(self) -> str:
        """Return a content-free summary: HTTP status and closed error code only.

        Never includes the backend message, raw body or any other backend text
        (it can echo customer content, #2552). ``message``/``raw_body`` stay on
        the detail for classification only.
        """

        parts: list[str] = []
        if self.status_code is not None:
            parts.append(f"HTTP {self.status_code}")
        parts.append(self.safe_error_code)
        return " | ".join(parts)


class SessionCreationHTTPError(Exception):
    """HTTP error raised while creating a backend tracking session."""

    def __init__(self, detail: SessionCreationFailureDetail) -> None:
        super().__init__(detail.one_line_summary())
        self.detail = detail


@dataclass
class SessionCreationResult:
    """Structured outcome of a session creation attempt."""

    session_id: str
    backend_connected: bool
    failure_reason: SessionCreationFailureReason | None = None
    failure_detail: str | None = None
    failure_response: SessionCreationFailureDetail | None = None
    project_id: str | None = None
    tenant_id: str | None = None
    typed_legacy_session_create_400: bool = False
    # The agent the backend bound this session to (API-key callers only; None
    # for other principals and for backends that do not disclose it).
    agent_id: str | None = None
    # Content-less (``content_less_v1``) sessions: the opaque ``agent_ref`` the
    # session-create 201 advertised. ``content_less`` is True when the backend
    # negotiated the capability; such a session never carries ``agent_id`` and its
    # promotions are protected by the ref-addressed head precondition.
    agent_ref: str | None = None
    content_less: bool = False
    # Execution-path transparency (so a caller never mistakes a local run for a
    # managed/cloud one). ``execution_path`` is "connected" for a real backend
    # session and "local_fallback" for any local result. ``backend_fallback`` is
    # True whenever the run is local. ``degraded`` is True ONLY when the user
    # intended a managed/cloud run but the backend was unreachable and we fell
    # back to local — it stays False for intentional local execution (offline
    # mode / no API key), which is a deliberate configuration, not a downgrade.
    execution_path: str | None = None
    backend_fallback: bool = False
    degraded: bool = False

    def __str__(self) -> str:
        """Return the session ID string for backwards compatibility.

        v0.10.x returned a plain str from create_session(); code that does
        ``session_id = create_session()`` and passes the result as a string
        argument will continue to work without changes.
        """
        return self.session_id

    def __post_init__(self) -> None:
        if not self.backend_connected and self.failure_reason is None:
            raise ValueError("failure_reason is required when backend_connected=False")
        if self.backend_connected and self.failure_reason is not None:
            raise ValueError("failure_reason must be None when backend_connected=True")

    @classmethod
    def connected(
        cls,
        session_id: str,
        *,
        project_id: str | None = None,
        tenant_id: str | None = None,
        agent_id: str | None = None,
        agent_ref: str | None = None,
        content_less: bool = False,
    ) -> SessionCreationResult:
        """Factory for a successful backend session."""
        return cls(
            session_id=session_id,
            backend_connected=True,
            project_id=project_id,
            tenant_id=tenant_id,
            agent_id=agent_id,
            agent_ref=agent_ref,
            content_less=content_less,
            execution_path="connected",
            backend_fallback=False,
            degraded=False,
        )

    @classmethod
    def fallback(
        cls,
        session_id: str,
        reason: SessionCreationFailureReason,
        detail: str | None = None,
        failure_response: SessionCreationFailureDetail | None = None,
        typed_legacy_session_create_400: bool = False,
        degraded: bool = False,
    ) -> SessionCreationResult:
        """Factory for a local-fallback session after backend failure.

        ``degraded`` must be set True by callers where the user intended a
        managed/cloud run but the backend was unreachable (an unintended
        downgrade); it stays False for intentional local execution (offline
        mode / no API key configured).
        """
        return cls(
            session_id=session_id,
            backend_connected=False,
            failure_reason=reason,
            failure_detail=detail,
            failure_response=failure_response,
            typed_legacy_session_create_400=typed_legacy_session_create_400,
            execution_path="local_fallback",
            backend_fallback=True,
            degraded=degraded,
        )
