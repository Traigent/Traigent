"""Client for retrieving client-safe next-step recommendations.

This module provides an async-first client for the backend next-steps endpoint:
``GET /api/v1/analytics/experiments/{experiment_run_id}/next-steps``.

The endpoint is available only on backend versions that include the next-steps
feature. Older backends should fail truthfully, especially with a 404 response.

Returned recommendations are category-level, templated advice for what to try
next. They expose safe labels, rationale text, action templates, and coarse
evidence levels only. Proprietary tuning signals, signal values, formulas, and
rankings are not returned to clients, following the platform's IP discipline.

Usage:
    >>> from traigent.analytics import NextStepsClient
    >>>
    >>> client = NextStepsClient(backend_url="https://portal.traigent.ai")
    >>> payload = await client.get_next_steps("run_123")
    >>> print(payload["caveat"])
"""

from __future__ import annotations

import os
import re
from datetime import datetime
from typing import Any, Literal, cast

from traigent.cloud.auth import _build_api_key_auth_headers
from traigent.cloud.url_security import validate_cloud_base_url
from traigent.config.backend_config import DEFAULT_LOCAL_URL
from traigent.utils.logging import get_logger

logger = get_logger(__name__)

try:
    import httpx

    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False
    httpx = None  # type: ignore


_REQUIRED_RESPONSE_KEYS = {
    "schema_version",
    "experiment_run_id",
    "caveat",
    "summary",
    "next_steps",
}  # matches the contract's required set (next_steps_schema.json)
_ALLOWED_RESPONSE_KEYS = _REQUIRED_RESPONSE_KEYS | {
    "posture",
    "guidance_meta",
    "decision",
}
_GUIDANCE_META_FIELDS = {
    "requested_variant",
    "served_variant",
    "engine",
    "policy_table_sha",
    "smartopt_version",
    "fallback_reason",
    "decision_id",
    "evidence_snapshot_hash",
}

_GUIDANCE_VARIANT_HEADER = "X-Traigent-Guidance-Variant"
_GUIDANCE_VARIANTS = frozenset({"rules", "policy"})
_RECEIPT_STATUSES = frozenset({"started", "completed", "failed", "skipped"})
_HOLDOUT_STATUSES = frozenset({"passed", "failed", "no_decision"})
_SAFETY_GATE_STATUSES = frozenset({"passed", "failed", "not_run"})
_ATTEMPT_ID_RE = re.compile(r"^[a-zA-Z0-9_.:-]+$")
_DECISION_ID_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_.:-]*$")
_RUN_ID_RE = re.compile(r"^[a-zA-Z0-9_-]+$")
_AUTHORITATIVE_CATEGORIES = frozenset(
    {
        "score_evaluation_set",
        "curate_evaluation_set",
        "audit_evaluator_quality",
        "improve_evaluator",
        "adjust_configuration_space",
        "run_optimization",
        "validate_holdout",
        "compare_with_baseline",
        "add_safety_gate",
        "promote_winner",
        "wait",
    }
)

GuidanceVariant = Literal["rules", "policy"]
ReceiptStatus = Literal["started", "completed", "failed", "skipped"]
HoldoutStatus = Literal["passed", "failed", "no_decision"]
SafetyGateStatus = Literal["passed", "failed", "not_run"]


class GuidanceVariantConflictError(RuntimeError):
    """The experiment run is durably pinned to the opposite treatment arm."""


def _normalize_guidance_variant(value: str | None) -> GuidanceVariant | None:
    """Normalize and validate an optional rules-vs-policy treatment."""
    if value is None:
        return None
    variant = value.strip().lower()
    if not variant:
        return None
    if variant not in _GUIDANCE_VARIANTS:
        raise ValueError(f"guidance_variant must be one of rules|policy; got {value!r}")
    return cast(GuidanceVariant, variant)


def _resolve_guidance_variant(
    explicit: str | None,
) -> GuidanceVariant | None:
    """Resolve explicit treatment first, then the environment fallback."""
    if explicit is not None:
        if not explicit.strip():
            raise ValueError(
                "guidance_variant must be one of rules|policy; got a blank value"
            )
        return _normalize_guidance_variant(explicit)
    return _normalize_guidance_variant(os.environ.get("TRAIGENT_GUIDANCE_VARIANT"))


def _validate_run_id(value: Any, field: str = "experiment_run_id") -> str:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > 100
        or not _RUN_ID_RE.fullmatch(value)
    ):
        raise ValueError(
            f"{field} must be 1-100 letters, digits, underscores, or hyphens"
        )
    return value


def _validate_guidance_response(
    payload: dict[str, Any],
    *,
    requested_variant: GuidanceVariant,
    strict_experiment: bool,
) -> None:
    """Fail closed when the server did not serve the requested treatment."""
    guidance_meta = payload.get("guidance_meta")
    if not isinstance(guidance_meta, dict):
        raise ValueError(
            "Guidance integrity error: response omitted guidance_meta for "
            f"requested variant {requested_variant!r}."
        )
    unexpected_root = set(payload) - _ALLOWED_RESPONSE_KEYS
    if unexpected_root:
        raise ValueError(
            "Guidance integrity error: unexpected top-level response field(s): "
            f"{', '.join(sorted(unexpected_root))}."
        )
    unexpected_meta = set(guidance_meta) - _GUIDANCE_META_FIELDS
    if unexpected_meta:
        raise ValueError(
            "Guidance integrity error: unexpected guidance_meta field(s): "
            f"{', '.join(sorted(unexpected_meta))}."
        )
    missing_meta = {"requested_variant", "served_variant", "engine"} - set(
        guidance_meta
    )
    if missing_meta:
        raise ValueError(
            "Guidance integrity error: guidance_meta missing required field(s): "
            f"{', '.join(sorted(missing_meta))}."
        )
    _validate_guidance_public_shape(payload, guidance_meta)

    echoed_request = guidance_meta.get("requested_variant")
    if echoed_request is not None and echoed_request != requested_variant:
        raise ValueError(
            "Guidance integrity error: server echoed requested_variant "
            f"{echoed_request!r}, expected {requested_variant!r}."
        )

    served_variant = guidance_meta.get("served_variant")
    if served_variant != requested_variant:
        raise ValueError(
            "Guidance integrity error: served_variant "
            f"{served_variant!r} does not match requested variant "
            f"{requested_variant!r}."
        )

    engine = guidance_meta.get("engine")
    if engine not in {"rules", "policy", "none"}:
        raise ValueError(f"Guidance integrity error: unsupported engine {engine!r}.")
    if strict_experiment and echoed_request != requested_variant:
        raise ValueError(
            "Strict experiment mode requires guidance_meta.requested_variant "
            f"to equal {requested_variant!r}."
        )
    if strict_experiment and engine != requested_variant:
        raise ValueError(
            "Strict experiment mode requires the actual guidance engine to "
            f"match the requested treatment; got engine={engine!r}."
        )
    if strict_experiment and guidance_meta.get("fallback_reason") not in (None, ""):
        raise ValueError(
            "Strict experiment mode rejects guidance fallbacks: "
            f"{guidance_meta.get('fallback_reason')!r}."
        )

    decision = payload.get("decision")
    if not isinstance(decision, dict):
        if "decision" in payload:
            raise ValueError("Guidance integrity error: decision must be an object.")
        if strict_experiment:
            raise ValueError(
                "Strict experiment mode requires the authoritative decision object."
            )
        if payload.get("next_steps") != []:
            raise ValueError(
                "Guidance integrity error: a guidance response without a decision "
                "must not contain executable next_steps."
            )
        return
    if engine not in {"rules", "policy"}:
        raise ValueError(
            "Guidance integrity error: engine='none' cannot produce an authoritative decision."
        )
    expected_decision_fields = {
        "id",
        "category",
        "source_engine",
        "evidence_snapshot_hash",
        "rationale",
        "action",
        "evidence_level",
    }
    if set(decision) != expected_decision_fields:
        raise ValueError(
            "Guidance integrity error: decision fields do not match the public contract."
        )
    if decision.get("source_engine") != engine:
        raise ValueError(
            "Guidance integrity error: decision.source_engine does not match "
            f"guidance_meta.engine ({engine!r})."
        )
    decision_id = guidance_meta.get("decision_id")
    if (
        not isinstance(decision_id, str)
        or len(decision_id) > 128
        or not _DECISION_ID_RE.fullmatch(decision_id)
        or decision_id != decision.get("id")
    ):
        raise ValueError(
            "Guidance integrity error: guidance_meta.decision_id does not match "
            "decision.id."
        )
    evidence_hash = guidance_meta.get("evidence_snapshot_hash")
    if (
        not isinstance(evidence_hash, str)
        or not evidence_hash
        or len(evidence_hash) > 200
        or evidence_hash != decision.get("evidence_snapshot_hash")
    ):
        raise ValueError(
            "Guidance integrity error: guidance_meta.evidence_snapshot_hash does "
            "not match the authoritative decision."
        )
    category = decision.get("category")
    if category not in _AUTHORITATIVE_CATEGORIES:
        raise ValueError(
            f"Guidance integrity error: unsupported decision category {category!r}."
        )
    action = decision.get("action")
    if not isinstance(action, dict) or set(action) != {"kind", "command_template"}:
        raise ValueError("Guidance integrity error: decision.action must be an object.")
    if not isinstance(decision.get("rationale"), str):
        raise ValueError(
            "Guidance integrity error: decision.rationale must be a string."
        )
    if decision.get("evidence_level") not in {"low", "medium", "high"}:
        raise ValueError(
            "Guidance integrity error: unsupported decision evidence_level."
        )
    action_kind = action.get("kind")
    command_template = action.get("command_template")
    next_steps = payload.get("next_steps")
    if next_steps != []:
        raise ValueError(
            "Guidance integrity error: authoritative decisions require an empty "
            "next_steps compatibility list."
        )
    if category == "wait":
        if action_kind != "none" or command_template != "":
            raise ValueError(
                "Guidance integrity error: wait must be non-executable with an "
                "empty command template."
            )
        return
    if (
        action_kind not in {"cli", "sdk", "skill"}
        or not isinstance(command_template, str)
        or not command_template
    ):
        raise ValueError(
            "Guidance integrity error: executable decisions require a declared "
            "action and command template."
        )


def _validate_guidance_public_shape(
    payload: dict[str, Any], guidance_meta: dict[str, Any]
) -> None:
    """Validate every public field on feature-lane responses before echoing JSON."""
    if not isinstance(payload.get("schema_version"), str):
        raise ValueError("Guidance integrity error: schema_version must be a string.")
    if not isinstance(payload.get("caveat"), str):
        raise ValueError("Guidance integrity error: caveat must be a string.")

    summary = payload.get("summary")
    summary_fields = {"winner_config_ref", "confidence_label", "trade_off_note"}
    if not isinstance(summary, dict) or set(summary) - summary_fields:
        raise ValueError("Guidance integrity error: summary fields are invalid.")
    if summary.get("confidence_label") not in {"low", "medium", "high"}:
        raise ValueError(
            "Guidance integrity error: summary.confidence_label is invalid."
        )
    for field in ("winner_config_ref", "trade_off_note"):
        if field in summary and not isinstance(summary[field], str):
            raise ValueError(
                f"Guidance integrity error: summary.{field} must be a string."
            )

    if "posture" in payload:
        posture = payload.get("posture")
        if not isinstance(posture, dict) or set(posture) != {
            "summary_text",
            "generated_at",
        }:
            raise ValueError("Guidance integrity error: posture fields are invalid.")
        if not isinstance(posture.get("summary_text"), str):
            raise ValueError(
                "Guidance integrity error: posture.summary_text must be a string."
            )
        _parse_rfc3339(posture.get("generated_at"), "posture.generated_at")

    if guidance_meta.get("requested_variant") not in _GUIDANCE_VARIANTS:
        raise ValueError(
            "Guidance integrity error: guidance_meta.requested_variant is invalid."
        )
    if guidance_meta.get("served_variant") not in _GUIDANCE_VARIANTS:
        raise ValueError(
            "Guidance integrity error: guidance_meta.served_variant is invalid."
        )
    if guidance_meta.get("engine") not in {"rules", "policy", "none"}:
        raise ValueError("Guidance integrity error: guidance_meta.engine is invalid.")
    for field in ("policy_table_sha", "smartopt_version", "fallback_reason"):
        value = guidance_meta.get(field)
        if value is not None and not isinstance(value, str):
            raise ValueError(
                f"Guidance integrity error: guidance_meta.{field} is invalid."
            )
    decision_id = guidance_meta.get("decision_id")
    if decision_id is not None and (
        not isinstance(decision_id, str)
        or len(decision_id) > 128
        or not _DECISION_ID_RE.fullmatch(decision_id)
    ):
        raise ValueError(
            "Guidance integrity error: guidance_meta.decision_id is invalid."
        )
    evidence_hash = guidance_meta.get("evidence_snapshot_hash")
    if evidence_hash is not None and (
        not isinstance(evidence_hash, str) or not evidence_hash
    ):
        raise ValueError(
            "Guidance integrity error: guidance_meta.evidence_snapshot_hash is invalid."
        )


def _parse_rfc3339(value: Any, field: str) -> None:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be an RFC3339 date-time string")
    candidate = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(candidate)
    except ValueError as exc:
        raise ValueError(f"{field} must be an RFC3339 date-time string") from exc
    if parsed.tzinfo is None:
        raise ValueError(f"{field} must include a timezone offset")


def _validate_receipt_response(
    payload: dict[str, Any],
    *,
    decision_id: str,
    attempt_id: str,
    status: str,
    successor_run_id: str | None,
    outcomes: dict[str, str],
) -> None:
    required = {
        "receipt_id",
        "decision_id",
        "attempt_id",
        "status",
        "created_at",
        "updated_at",
    }
    missing = sorted(required - payload.keys())
    if missing:
        raise ValueError(
            "Malformed next-steps receipt response: missing required key(s): "
            f"{', '.join(missing)}."
        )
    unknown = set(payload) - required - {"successor_run_id", "outcomes"}
    if unknown:
        raise ValueError(
            "Malformed next-steps receipt response: unexpected key(s): "
            f"{', '.join(sorted(unknown))}."
        )
    receipt_id = payload.get("receipt_id")
    if not isinstance(receipt_id, str) or not receipt_id or len(receipt_id) > 200:
        raise ValueError("Malformed next-steps receipt response: invalid receipt_id.")
    if payload.get("decision_id") != decision_id:
        raise ValueError(
            "Malformed next-steps receipt response: decision_id does not match the request."
        )
    if payload.get("attempt_id") != attempt_id:
        raise ValueError(
            "Malformed next-steps receipt response: attempt_id does not match the request."
        )
    if payload.get("status") != status:
        raise ValueError(
            "Malformed next-steps receipt response: status does not match the request."
        )
    response_successor = payload.get("successor_run_id")
    if response_successor != successor_run_id:
        raise ValueError(
            "Malformed next-steps receipt response: successor_run_id does not match the request."
        )
    response_outcomes = payload.get("outcomes", {})
    if response_outcomes != outcomes:
        raise ValueError(
            "Malformed next-steps receipt response: outcomes do not match the request."
        )
    _parse_rfc3339(payload.get("created_at"), "created_at")
    _parse_rfc3339(payload.get("updated_at"), "updated_at")


class NextStepsClient:
    """Client for retrieving category-level next-step recommendations.

    Thread Safety: Safe for concurrent use (httpx.AsyncClient is thread-safe).
    """

    def __init__(
        self,
        backend_url: str = DEFAULT_LOCAL_URL,
        api_key: str | None = None,
        timeout: float = 30.0,
    ):
        """Initialize NextStepsClient.

        Args:
            backend_url: Backend API base URL. The backend must expose the
                next-steps endpoint.
            api_key: API key for authentication (None uses env var
                TRAIGENT_API_KEY)
            timeout: Default timeout for HTTP requests in seconds

        Raises:
            ImportError: If httpx is not installed
        """
        if not HTTPX_AVAILABLE:
            raise ImportError(
                "httpx is required for NextStepsClient. "
                "Install with: pip install traigent[analytics]"
            )

        backend_url = backend_url.rstrip("/")
        self.backend_url = validate_cloud_base_url(
            backend_url, purpose="analytics request"
        )
        self.timeout = timeout

        # Import here to avoid circular dependency
        from traigent.config.backend_config import get_no_credentials_hint
        from traigent.utils.env_config import get_api_key

        self.api_key = api_key or get_api_key("traigent")
        if not self.api_key:
            logger.warning(
                "No API key found for NextStepsClient. %s",
                get_no_credentials_hint(),
            )

        self._client: httpx.AsyncClient | None = None

    def _get_client(self) -> httpx.AsyncClient:
        """Get or create async HTTP client.

        Returns:
            Configured httpx.AsyncClient instance
        """
        from traigent.utils.env_config import raise_if_backend_offline

        raise_if_backend_offline("NextStepsClient request")
        if self._client is None:
            headers: dict[str, str] = {}
            if self.api_key:
                headers.update(_build_api_key_auth_headers(self.api_key))

            self._client = httpx.AsyncClient(
                base_url=self.backend_url,
                headers=headers,
                timeout=self.timeout,
            )

        return self._client

    async def close(self) -> None:
        """Close HTTP client and release resources."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def __aenter__(self) -> NextStepsClient:
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        await self.close()

    async def get_next_steps(
        self,
        experiment_run_id: str,
        *,
        guidance_variant: GuidanceVariant | str | None = None,
        strict_experiment: bool = False,
    ) -> dict[str, Any]:
        """Retrieve category-level next-step recommendations for an experiment.

        The backend must include the next-steps feature and expose:
        ``GET /api/v1/analytics/experiments/{experiment_run_id}/next-steps``.

        ``guidance_variant`` accepts only ``rules`` or ``policy``. When omitted,
        ``TRAIGENT_GUIDANCE_VARIANT`` remains a fallback. Invalid values fail
        locally before an HTTP request is made. Any requested variant requires
        the response to echo a matching served variant. ``strict_experiment``
        additionally rejects fallback engines and requires a joined authoritative
        decision, preventing treatment contamination in comparison harnesses.

        Args:
            experiment_run_id: Experiment run ID to retrieve recommendations for.
            guidance_variant: Explicit rules-vs-policy treatment. Explicit values
                take precedence over ``TRAIGENT_GUIDANCE_VARIANT``.
            strict_experiment: Fail closed unless the requested treatment was
                actually served by the matching engine with decision provenance.

        Returns:
            Dict with the backend next-steps contract payload. The response
            includes a ``caveat`` field that callers should display near the
            recommendations. If the backend includes the optional opaque
            ``posture`` block, it is returned unchanged. If the backend
            includes the optional ``guidance_meta`` block (served_variant,
            engine, policy_table_sha, smartopt_version, fallback_reason), it
            is returned unchanged.

        Raises:
            httpx.HTTPError: If request fails. A 404 response is raised as
                httpx.HTTPStatusError with a message noting that the backend may
                predate the next-steps feature.
            ValueError: If the backend returns a JSON object missing required
                next-steps contract keys.
        """
        requested_variant = _resolve_guidance_variant(guidance_variant)
        normalized_run_id = _validate_run_id(experiment_run_id)
        if strict_experiment and requested_variant is None:
            raise ValueError(
                "strict_experiment requires guidance_variant=rules|policy or "
                "TRAIGENT_GUIDANCE_VARIANT."
            )

        client = self._get_client()
        headers = (
            {_GUIDANCE_VARIANT_HEADER: requested_variant}
            if requested_variant is not None
            else {}
        )

        try:
            response = await client.get(
                f"/api/v1/analytics/experiments/{normalized_run_id}/next-steps",
                headers=headers or None,
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 404:
                raise httpx.HTTPStatusError(
                    "Next steps endpoint returned 404. The backend may predate "
                    "the next-steps feature; use a backend version that exposes "
                    "GET /api/v1/analytics/experiments/{experiment_run_id}/next-steps.",
                    request=exc.request,
                    response=exc.response,
                ) from exc
            if exc.response.status_code == 409 and requested_variant is not None:
                raise GuidanceVariantConflictError(
                    "Guidance variant assignment conflict: this experiment run is "
                    "already pinned to a different rules-vs-policy treatment. Use "
                    "the assigned variant or create a new experimental run."
                ) from exc
            raise

        payload = response.json()
        if not isinstance(payload, dict):
            raise ValueError(
                "Malformed next-steps response: expected a JSON object matching "
                "the next-steps contract."
            )

        missing = sorted(_REQUIRED_RESPONSE_KEYS - payload.keys())
        if missing:
            raise ValueError(
                "Malformed next-steps response: missing required key(s): "
                f"{', '.join(missing)}."
            )
        response_run_id = payload.get("experiment_run_id")
        try:
            validated_response_run_id = _validate_run_id(response_run_id)
        except ValueError as exc:
            raise ValueError(
                "Malformed next-steps response: invalid experiment_run_id."
            ) from exc
        if validated_response_run_id != normalized_run_id:
            raise ValueError(
                "Malformed next-steps response: experiment_run_id does not match the request."
            )

        typed_payload = cast(dict[str, Any], payload)
        if (
            requested_variant is not None
            or "guidance_meta" in payload
            or "decision" in payload
        ):
            _validate_guidance_response(
                typed_payload,
                requested_variant=requested_variant or "rules",
                strict_experiment=strict_experiment,
            )
        return typed_payload

    async def record_decision_receipt(
        self,
        experiment_run_id: str,
        *,
        decision_id: str,
        attempt_id: str,
        status: ReceiptStatus | str,
        successor_run_id: str | None = None,
        holdout_status: HoldoutStatus | str | None = None,
        safety_gate_status: SafetyGateStatus | str | None = None,
    ) -> dict[str, Any]:
        """Record execution and lifecycle evidence for an authoritative decision.

        The backend proves that ``decision_id`` belongs to this run and permits
        outcomes only for the matching operation category. ``attempt_id`` makes
        retries idempotent within the decision's project scope.
        """
        if not isinstance(status, str):
            raise ValueError("status must be one of started|completed|failed|skipped")
        normalized_status = status.strip().lower()
        if normalized_status not in _RECEIPT_STATUSES:
            raise ValueError(
                "status must be one of started|completed|failed|skipped; "
                f"got {status!r}"
            )
        if not isinstance(decision_id, str) or not decision_id.strip():
            raise ValueError("decision_id must be a non-blank string")
        normalized_decision_id = decision_id.strip()
        if len(normalized_decision_id) > 128 or not _DECISION_ID_RE.fullmatch(
            normalized_decision_id
        ):
            raise ValueError(
                "decision_id must be at most 128 characters using only letters, "
                "digits, underscore, dot, colon, or hyphen"
            )
        if not isinstance(attempt_id, str) or not attempt_id.strip():
            raise ValueError("attempt_id must be a non-blank string")
        normalized_attempt_id = attempt_id.strip()
        if len(normalized_attempt_id) > 128 or not _ATTEMPT_ID_RE.fullmatch(
            normalized_attempt_id
        ):
            raise ValueError(
                "attempt_id must be at most 128 characters using only letters, "
                "digits, underscore, dot, colon, or hyphen"
            )
        if successor_run_id is not None:
            successor_run_id = _validate_run_id(successor_run_id, "successor_run_id")
        normalized_run_id = _validate_run_id(experiment_run_id)

        outcomes: dict[str, str] = {}
        for key, value, allowed in (
            ("holdout_status", holdout_status, _HOLDOUT_STATUSES),
            ("safety_gate_status", safety_gate_status, _SAFETY_GATE_STATUSES),
        ):
            if value is None:
                continue
            if not isinstance(value, str):
                raise ValueError(f"unsupported {key} {value!r}")
            normalized = value.strip().lower()
            if normalized not in allowed:
                raise ValueError(
                    f"unsupported {key} {value!r}; expected one of "
                    f"{'|'.join(sorted(allowed))}"
                )
            outcomes[key] = normalized
        if outcomes and normalized_status != "completed":
            raise ValueError("holdout/safety-gate outcomes require status='completed'")

        request_payload: dict[str, Any] = {
            "status": normalized_status,
            "attempt_id": normalized_attempt_id,
        }
        if successor_run_id is not None:
            request_payload["successor_run_id"] = successor_run_id
        if outcomes:
            request_payload["outcomes"] = outcomes

        response = await self._get_client().post(
            "/api/v1/analytics/experiments/"
            f"{normalized_run_id}/next-steps/{normalized_decision_id}/receipt",
            json=request_payload,
        )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            raise ValueError(
                "Malformed next-steps receipt response: expected a JSON object."
            )
        _validate_receipt_response(
            payload,
            decision_id=normalized_decision_id,
            attempt_id=normalized_attempt_id,
            status=normalized_status,
            successor_run_id=successor_run_id,
            outcomes=outcomes,
        )
        return cast(dict[str, Any], payload)
