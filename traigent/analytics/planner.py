"""Fail-closed client for the additive SmartOps Planner V2 protocol.

V2 deliberately lives beside :mod:`traigent.analytics.next_steps`.  The old
client remains available for servers and experiments pinned to the v1
contract; callers must opt in to the lifecycle, receipt, and execution-spec
semantics defined here.
"""

from __future__ import annotations

import hashlib
import json
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


PLANNER_SCHEMA_VERSION = "2.0.0"
_OPAQUE_ID_RE = re.compile(r"^[a-z][a-z0-9_]{2,31}_[A-Za-z0-9_-]{16,160}$")
_RUN_ID_RE = re.compile(r"^[A-Za-z0-9_-]+$")
_EVIDENCE_HASH_RE = re.compile(r"^ev_[A-Za-z0-9_-]{32,96}$")

_UTILITY_PROFILES = frozenset({"quality_first", "balanced", "cost_first"})
_TREATMENTS = frozenset({"rules_control", "policy_override"})
_DECISION_MODES = frozenset(
    {
        "rules_control",
        "policy_override",
        "rules_parity",
        "rules_fallback",
        "pending_wait",
        "safety_stop",
    }
)
_ADVANTAGE_LABELS = frozenset(
    {"model_certified_positive", "parity", "unavailable", "not_applicable"}
)
_SELECTOR_ENGINES = frozenset({"rules", "policy", "safety"})
_SOURCE_ENGINES = frozenset({"rules", "policy", "safety"})
_CONTEXT_STATUSES = frozenset({"complete", "incomplete"})
_EVIDENCE_LEVELS = frozenset({"low", "medium", "high"})
_CATEGORIES = frozenset(
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
        "stop",
    }
)
_RECEIPT_STATUSES = frozenset({"started", "submitted", "failed", "skipped"})
_VERIFICATION_STATUSES = frozenset({"pending", "verified", "rejected"})
_REOPEN_REASONS = frozenset({"new_artifact", "budget", "operator"})
_FALLBACK_REASONS = frozenset({"policy_unavailable", "calibration_unavailable"})
_RATIONALES_BY_MODE = {
    "rules_control": "rules control selected the next safe lifecycle action",
    "policy_override": "policy selected a model-certified improvement over rules",
    "rules_parity": "policy agreed with the safe rule action",
    "rules_fallback": "policy or calibration was unavailable; safe rules were used",
    "pending_wait": "an operation is already active; wait for its authoritative result",
    "safety_stop": "no positive affordable safe action remains",
}
_EXECUTION_VARIANTS = {
    "score_examples": frozenset({"score_probe_32", "score_full"}),
    "synth_harder_examples": frozenset(
        {"curate_invalid", "deduplicate", "expand_coverage", "synthesize_hard"}
    ),
    "audit_evaluator": frozenset({"audit_probe_32", "audit_full"}),
    "refine_metric": frozenset(
        {"fix_parser", "calibrate_threshold", "reduce_bias", "stabilize"}
    ),
    "run_optimization": frozenset(
        {"optimize_probe", "optimize_standard", "optimize_deep"}
    ),
    "adjust_configuration_space": frozenset(
        {"config_expand", "config_prune", "config_fix_inactive"}
    ),
    "compare_baseline": frozenset({"compare_probe_64", "compare_full"}),
    "run_holdout": frozenset({"holdout_full"}),
    "add_safety_gate": frozenset({"safety_gate"}),
    "promote_winner": frozenset({"promote"}),
    "wait": frozenset({"wait"}),
    "stop": frozenset({"stop"}),
}
_CATEGORY_TO_VARIANTS = {
    "score_evaluation_set": _EXECUTION_VARIANTS["score_examples"],
    "curate_evaluation_set": _EXECUTION_VARIANTS["synth_harder_examples"],
    "audit_evaluator_quality": _EXECUTION_VARIANTS["audit_evaluator"],
    "improve_evaluator": _EXECUTION_VARIANTS["refine_metric"],
    "adjust_configuration_space": _EXECUTION_VARIANTS["adjust_configuration_space"],
    "run_optimization": _EXECUTION_VARIANTS["run_optimization"],
    "validate_holdout": _EXECUTION_VARIANTS["run_holdout"],
    "compare_with_baseline": _EXECUTION_VARIANTS["compare_baseline"],
    "add_safety_gate": _EXECUTION_VARIANTS["add_safety_gate"],
    "promote_winner": _EXECUTION_VARIANTS["promote_winner"],
    "wait": _EXECUTION_VARIANTS["wait"],
    "stop": _EXECUTION_VARIANTS["stop"],
}
_EXECUTION_TOKEN_RE = re.compile(r"^[A-Za-z0-9._:/=-]{1,192}$")
_SIGNATURE_RE = re.compile(r"^sig_[A-Za-z0-9_-]{43,128}$")

UtilityProfile = Literal["quality_first", "balanced", "cost_first"]
PlannerTreatment = Literal["rules_control", "policy_override"]
PlannerReceiptStatus = Literal["started", "submitted", "failed", "skipped"]
ReopenReason = Literal["new_artifact", "budget", "operator"]


def _validate_run_id(value: Any, field: str = "run_id") -> str:
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


def _validate_opaque_id(value: Any, field: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > 192
        or not _OPAQUE_ID_RE.fullmatch(value)
    ):
        raise ValueError(f"{field} must be a valid opaque identifier")
    return value


def _normalize_enum(value: Any, field: str, allowed: frozenset[str]) -> str:
    if not isinstance(value, str) or value not in allowed:
        raise ValueError(
            f"{field} must be one of {'|'.join(sorted(allowed))}; got {value!r}"
        )
    return value


def _parse_rfc3339(value: Any, field: str) -> None:
    if not isinstance(value, str) or len(value) > 40 or not value.endswith("Z"):
        raise ValueError(f"{field} must be an RFC3339 UTC date-time ending in Z")
    candidate = value[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(candidate)
    except ValueError as exc:
        raise ValueError(f"{field} must be an RFC3339 UTC date-time ending in Z") from exc
    if parsed.utcoffset() is None or parsed.utcoffset().total_seconds() != 0:
        raise ValueError(f"{field} must be UTC")


def _validate_execution_spec(payload: Any, *, decision_id: str) -> dict[str, Any]:
    root = _require_exact_keys(
        payload,
        field="Planner V2 execution response",
        required={
            "schema_version",
            "decision_id",
            "argv",
            "execution_spec",
            "signature",
            "expires_at",
        },
    )
    if root["schema_version"] != PLANNER_SCHEMA_VERSION:
        raise ValueError("Planner V2 execution schema_version is unsupported")
    if root["decision_id"] != decision_id:
        raise ValueError("Planner V2 execution decision_id does not match request")
    if not isinstance(root["signature"], str) or not _SIGNATURE_RE.fullmatch(
        root["signature"]
    ):
        raise ValueError("Planner V2 execution signature is malformed")
    _parse_rfc3339(root["expires_at"], "expires_at")

    argv = root["argv"]
    if (
        not isinstance(argv, list)
        or not 3 <= len(argv) <= 12
        or argv[:3] != ["traigent", "guidance", "execute-resolved"]
        or any(
            not isinstance(token, str) or not _EXECUTION_TOKEN_RE.fullmatch(token)
            for token in argv
        )
    ):
        raise ValueError("Planner V2 execution argv is not allowlisted")

    spec = _require_exact_keys(
        root["execution_spec"],
        field="Planner V2 execution_spec",
        required={
            "operation_kind",
            "variant",
            "attempt_id",
            "receipt_url",
            "lease_expires_at",
        },
        optional={"sample_limit", "max_trials", "reserved_cost_usd"},
    )
    operation_kind = spec["operation_kind"]
    if operation_kind not in _EXECUTION_VARIANTS:
        raise ValueError(f"unsupported execution operation_kind {operation_kind!r}")
    if spec["variant"] not in _EXECUTION_VARIANTS[operation_kind]:
        raise ValueError(
            f"unsupported execution variant {spec['variant']!r} for {operation_kind!r}"
        )
    _validate_opaque_id(spec["attempt_id"], "execution_spec.attempt_id")
    receipt_url = spec["receipt_url"]
    expected_suffix = f"/decisions/{decision_id}/receipts"
    if (
        not isinstance(receipt_url, str)
        or len(receipt_url) > 512
        or not receipt_url.startswith("/api/v2/lifecycles/")
        or not receipt_url.endswith(expected_suffix)
        or ".." in receipt_url
        or "?" in receipt_url
        or "#" in receipt_url
    ):
        raise ValueError("execution_spec.receipt_url is not a scoped relative path")
    _parse_rfc3339(spec["lease_expires_at"], "execution_spec.lease_expires_at")
    for field in ("sample_limit", "max_trials"):
        value = spec.get(field)
        if value is not None and (
            not isinstance(value, int) or isinstance(value, bool) or value < 1
        ):
            raise ValueError(
                f"execution_spec.{field} must be a positive integer or null"
            )
    trial_caps = {
        "optimize_probe": 4,
        "optimize_standard": 12,
        "optimize_deep": 30,
    }
    max_trials = spec.get("max_trials")
    if spec["variant"] in trial_caps:
        if max_trials is None or max_trials > trial_caps[spec["variant"]]:
            raise ValueError("execution_spec.max_trials exceeds the variant cap")
    elif max_trials is not None:
        raise ValueError("execution_spec.max_trials is valid only for optimization")
    exact_probe_samples = {
        "score_probe_32": 32,
        "audit_probe_32": 32,
        "compare_probe_64": 64,
    }
    sample_limit = spec.get("sample_limit")
    if spec["variant"] in exact_probe_samples:
        if sample_limit != exact_probe_samples[spec["variant"]]:
            raise ValueError("execution_spec.sample_limit does not match probe variant")
    elif sample_limit is not None:
        raise ValueError("execution_spec.sample_limit is valid only for a probe")
    reserved = spec.get("reserved_cost_usd")
    if reserved is not None and (
        not isinstance(reserved, str)
        or not re.fullmatch(r"^(0|[1-9][0-9]{0,8})(\.[0-9]{1,6})?$", reserved)
    ):
        raise ValueError("execution_spec.reserved_cost_usd is malformed")
    return root


def _require_exact_keys(
    value: Any,
    *,
    field: str,
    required: set[str],
    optional: set[str] | None = None,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be an object")
    optional = optional or set()
    missing = required - set(value)
    unexpected = set(value) - required - optional
    if missing or unexpected:
        details: list[str] = []
        if missing:
            details.append(f"missing {', '.join(sorted(missing))}")
        if unexpected:
            details.append(f"unexpected {', '.join(sorted(unexpected))}")
        raise ValueError(f"{field} fields are invalid: {'; '.join(details)}")
    return cast(dict[str, Any], value)


def _validate_decision_response(
    payload: Any,
    *,
    requested_run_id: str,
    requested_treatment: PlannerTreatment,
    requested_profile: UtilityProfile,
    strict_experiment: bool,
) -> dict[str, Any]:
    root = _require_exact_keys(
        payload,
        field="Planner V2 response",
        required={"schema_version", "lifecycle_id", "run_id", "decision", "meta"},
    )
    if root["schema_version"] != PLANNER_SCHEMA_VERSION:
        raise ValueError(
            "Planner V2 response has unsupported schema_version "
            f"{root['schema_version']!r}"
        )
    if _validate_run_id(root["run_id"]) != requested_run_id:
        raise ValueError("Planner V2 response run_id does not match the request")
    _validate_opaque_id(root["lifecycle_id"], "lifecycle_id")

    meta = _require_exact_keys(
        root["meta"],
        field="Planner V2 meta",
        required={
            "requested_variant",
            "served_variant",
            "selector_engine",
            "context_status",
            "policy_version",
            "rule_version",
            "calibration_version",
            "shield_version",
            "fallback_reason",
        },
    )
    _normalize_enum(meta["requested_variant"], "meta.requested_variant", _TREATMENTS)
    _normalize_enum(meta["served_variant"], "meta.served_variant", _TREATMENTS)
    _normalize_enum(meta["selector_engine"], "meta.selector_engine", _SELECTOR_ENGINES)
    _normalize_enum(meta["context_status"], "meta.context_status", _CONTEXT_STATUSES)
    for field in ("rule_version", "shield_version"):
        if not isinstance(meta[field], str) or not meta[field]:
            raise ValueError(f"meta.{field} must be a non-empty string")
    for field in ("policy_version", "calibration_version"):
        if meta[field] is not None and (
            not isinstance(meta[field], str) or not meta[field]
        ):
            raise ValueError(f"meta.{field} must be a non-empty string or null")
    if meta["fallback_reason"] is not None:
        _normalize_enum(
            meta["fallback_reason"], "meta.fallback_reason", _FALLBACK_REASONS
        )
    if meta["requested_variant"] != requested_treatment:
        raise ValueError("Planner V2 server did not echo the requested treatment")
    if meta["served_variant"] != requested_treatment:
        raise ValueError("Planner V2 server served a different treatment")

    decision = _require_exact_keys(
        root["decision"],
        field="Planner V2 decision",
        required={
            "id",
            "mode",
            "category",
            "source_engine",
            "baseline_rule_category",
            "utility_profile",
            "certificate_ref",
            "advantage_label",
            "evidence_snapshot_hash",
            "rationale",
            "action",
            "evidence_level",
        },
    )
    _validate_opaque_id(decision["id"], "decision.id")
    mode = _normalize_enum(decision["mode"], "decision.mode", _DECISION_MODES)
    category = _normalize_enum(decision["category"], "decision.category", _CATEGORIES)
    _normalize_enum(
        decision["baseline_rule_category"],
        "decision.baseline_rule_category",
        _CATEGORIES,
    )
    _normalize_enum(
        decision["source_engine"], "decision.source_engine", _SOURCE_ENGINES
    )
    _normalize_enum(
        decision["utility_profile"], "decision.utility_profile", _UTILITY_PROFILES
    )
    if decision["utility_profile"] != requested_profile:
        raise ValueError(
            "Planner V2 decision utility_profile does not match the request"
        )
    advantage = _normalize_enum(
        decision["advantage_label"],
        "decision.advantage_label",
        _ADVANTAGE_LABELS,
    )
    _normalize_enum(
        decision["evidence_level"], "decision.evidence_level", _EVIDENCE_LEVELS
    )
    evidence_hash = decision["evidence_snapshot_hash"]
    if not isinstance(evidence_hash, str) or not _EVIDENCE_HASH_RE.fullmatch(
        evidence_hash
    ):
        raise ValueError("decision.evidence_snapshot_hash is malformed")
    if not isinstance(decision["rationale"], str) or not decision["rationale"]:
        raise ValueError("decision.rationale must be a non-empty string")
    if decision["rationale"] != _RATIONALES_BY_MODE[mode]:
        raise ValueError("decision.rationale is not the template for decision.mode")
    certificate_ref = decision["certificate_ref"]
    if certificate_ref is not None:
        _validate_opaque_id(certificate_ref, "decision.certificate_ref")

    action = _require_exact_keys(
        decision["action"],
        field="Planner V2 decision.action",
        required={"kind", "variant", "command_template"},
    )
    if action["variant"] not in _CATEGORY_TO_VARIANTS[category]:
        raise ValueError("decision.action.variant is invalid for decision.category")
    if category in {"wait", "stop"}:
        if action["kind"] != "none" or action["command_template"] != "":
            raise ValueError("WAIT and STOP decisions must be non-executable")
    else:
        expected_command = f"traigent guidance execute --decision {decision['id']}"
        if action["kind"] != "cli":
            raise ValueError("Executable decision.action.kind must be 'cli'")
        if action["command_template"] != expected_command:
            raise ValueError(
                "Executable Planner V2 decisions must use the static opaque "
                "guidance command"
            )

    if mode == "policy_override":
        if advantage != "model_certified_positive" or certificate_ref is None:
            raise ValueError(
                "policy_override requires a certified-positive opaque certificate"
            )
        if decision["source_engine"] != "policy":
            raise ValueError("policy_override requires source_engine='policy'")
        if decision["evidence_level"] != "high":
            raise ValueError("policy_override requires high evidence")
    elif advantage == "model_certified_positive":
        raise ValueError("Only policy_override may claim model_certified_positive")
    elif certificate_ref is not None:
        raise ValueError("Only policy_override may expose a certificate_ref")
    if mode == "pending_wait" and (
        category != "wait" or decision["baseline_rule_category"] != "wait"
    ):
        raise ValueError("pending_wait mode requires the wait category and baseline")
    if mode == "safety_stop" and (
        category != "stop" or decision["baseline_rule_category"] != "stop"
    ):
        raise ValueError("safety_stop mode requires the stop category and baseline")
    if mode in {"policy_override", "rules_parity", "rules_fallback"} and category in {
        "wait",
        "stop",
    }:
        raise ValueError(f"{mode} cannot carry a terminal action")
    if mode in {"rules_control", "rules_parity", "rules_fallback"} and (
        category != decision["baseline_rule_category"]
    ):
        raise ValueError(f"{mode} must return the frozen baseline category")
    if mode in {"policy_override", "rules_parity"} and (
        meta["policy_version"] is None or meta["calibration_version"] is None
    ):
        raise ValueError(f"{mode} requires policy and calibration version pins")
    if mode == "rules_fallback" and meta["fallback_reason"] is None:
        raise ValueError("rules_fallback requires an explicit fallback_reason")
    if mode != "rules_fallback" and meta["fallback_reason"] is not None:
        raise ValueError("fallback_reason is valid only for rules_fallback")

    expected_semantics = {
        "rules_control": ("rules", "not_applicable", "medium"),
        "rules_parity": ("rules", "parity", "medium"),
        "rules_fallback": ("rules", "unavailable", "low"),
        "pending_wait": ("rules", "not_applicable", "low"),
        "safety_stop": ("safety", "not_applicable", "high"),
    }
    if mode in expected_semantics:
        expected_source, expected_advantage, expected_evidence = expected_semantics[
            mode
        ]
        if (
            decision["source_engine"] != expected_source
            or advantage != expected_advantage
            or decision["evidence_level"] != expected_evidence
        ):
            raise ValueError(f"{mode} decision semantics are inconsistent")
    expected_selector = {
        "rules_control": "rules",
        "policy_override": "policy",
        "rules_parity": "policy",
        "rules_fallback": "rules",
        "pending_wait": "safety",
        "safety_stop": "safety",
    }[mode]
    if meta["selector_engine"] != expected_selector:
        raise ValueError(f"{mode} has an inconsistent selector_engine")

    if strict_experiment:
        allowed_modes = (
            {"rules_control", "pending_wait", "safety_stop"}
            if requested_treatment == "rules_control"
            else {"policy_override", "rules_parity", "pending_wait", "safety_stop"}
        )
        if mode not in allowed_modes:
            raise ValueError(
                "Strict experiment rejects treatment contamination or fallback"
            )
    return root


class PlannerV2Client:
    """Authenticated client for Planner V2 decisions and lifecycle receipts."""

    def __init__(
        self,
        backend_url: str = DEFAULT_LOCAL_URL,
        api_key: str | None = None,
        timeout: float = 30.0,
    ) -> None:
        if not HTTPX_AVAILABLE:
            raise ImportError(
                "httpx is required for PlannerV2Client. "
                "Install with: pip install traigent[analytics]"
            )
        self.backend_url = validate_cloud_base_url(
            backend_url.rstrip("/"), purpose="analytics request"
        )
        self.timeout = timeout
        from traigent.config.backend_config import get_no_credentials_hint
        from traigent.utils.env_config import get_api_key

        self.api_key = api_key or get_api_key("traigent")
        if not self.api_key:
            logger.warning(
                "No API key found for PlannerV2Client. %s", get_no_credentials_hint()
            )
        self._client: httpx.AsyncClient | None = None

    def _get_client(self) -> httpx.AsyncClient:
        from traigent.utils.env_config import raise_if_backend_offline

        raise_if_backend_offline("PlannerV2Client request")
        if self._client is None:
            headers: dict[str, str] = {}
            if self.api_key:
                headers.update(_build_api_key_auth_headers(self.api_key))
            self._client = httpx.AsyncClient(
                base_url=self.backend_url, headers=headers, timeout=self.timeout
            )
        return self._client

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def __aenter__(self) -> PlannerV2Client:
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.close()

    async def get_next_decision(
        self,
        run_id: str,
        *,
        utility_profile: UtilityProfile | str = "balanced",
        treatment: PlannerTreatment | str = "policy_override",
        strict_experiment: bool = False,
    ) -> dict[str, Any]:
        """Fetch one authoritative v2 decision for a run."""
        normalized_run_id = _validate_run_id(run_id)
        profile = cast(
            UtilityProfile,
            _normalize_enum(utility_profile, "utility_profile", _UTILITY_PROFILES),
        )
        normalized_treatment = cast(
            PlannerTreatment,
            _normalize_enum(treatment, "treatment", _TREATMENTS),
        )
        response = await self._get_client().post(
            f"/api/v2/experiment-runs/{normalized_run_id}/next-decision",
            json={
                "utility_profile": profile,
                "requested_variant": normalized_treatment,
            },
        )
        response.raise_for_status()
        return _validate_decision_response(
            response.json(),
            requested_run_id=normalized_run_id,
            requested_treatment=normalized_treatment,
            requested_profile=profile,
            strict_experiment=strict_experiment,
        )

    async def record_receipt(
        self,
        lifecycle_id: str,
        decision_id: str,
        *,
        status: PlannerReceiptStatus | str,
        attempt_id: str,
        successor_run_id: str | None = None,
        result_ref: str | None = None,
    ) -> dict[str, Any]:
        """Record execution without equating submission with verification."""
        lifecycle = _validate_opaque_id(lifecycle_id, "lifecycle_id")
        decision = _validate_opaque_id(decision_id, "decision_id")
        attempt = _validate_opaque_id(attempt_id, "attempt_id")
        normalized_status = _normalize_enum(status, "status", _RECEIPT_STATUSES)
        if normalized_status == "submitted" and result_ref is None:
            raise ValueError("status='submitted' requires result_ref")
        if normalized_status != "submitted" and result_ref is not None:
            raise ValueError("result_ref is valid only for status='submitted'")
        if normalized_status != "submitted" and successor_run_id is not None:
            raise ValueError("successor_run_id is valid only for status='submitted'")
        request: dict[str, Any] = {
            "status": normalized_status,
            "attempt_id": attempt,
        }
        if successor_run_id is not None:
            request["successor_run_id"] = _validate_run_id(
                successor_run_id, "successor_run_id"
            )
        if result_ref is not None:
            request["result_ref"] = _validate_opaque_id(result_ref, "result_ref")
        receipt_idempotency = (
            "receipt-"
            + hashlib.sha256(
                json.dumps(
                    {
                        "lifecycle": lifecycle,
                        "decision": decision,
                        "attempt": attempt,
                        **request,
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()[:32]
        )
        response = await self._get_client().post(
            f"/api/v2/lifecycles/{lifecycle}/decisions/{decision}/receipts",
            json=request,
            headers={"Idempotency-Key": receipt_idempotency},
        )
        response.raise_for_status()
        payload = _require_exact_keys(
            response.json(),
            field="Planner V2 receipt response",
            required={
                "schema_version",
                "receipt_id",
                "lifecycle_id",
                "decision_id",
                "attempt_id",
                "status",
                "verification_status",
                "idempotent_replay",
                "updated_at",
            },
            optional={"successor_run_id", "result_ref"},
        )
        if payload["schema_version"] != PLANNER_SCHEMA_VERSION:
            raise ValueError("Planner V2 receipt schema_version is unsupported")
        _validate_opaque_id(payload["receipt_id"], "receipt_id")
        for field, expected in (
            ("lifecycle_id", lifecycle),
            ("decision_id", decision),
            ("attempt_id", attempt),
            ("status", normalized_status),
        ):
            if payload[field] != expected:
                raise ValueError(f"Planner V2 receipt {field} does not match request")
        _normalize_enum(
            payload["verification_status"],
            "verification_status",
            _VERIFICATION_STATUSES,
        )
        if not isinstance(payload["idempotent_replay"], bool):
            raise ValueError("Planner V2 receipt idempotent_replay must be boolean")
        if payload.get("successor_run_id") != request.get("successor_run_id"):
            raise ValueError(
                "Planner V2 receipt successor_run_id does not match request"
            )
        if payload.get("result_ref") != request.get("result_ref"):
            raise ValueError("Planner V2 receipt result_ref does not match request")
        _parse_rfc3339(payload["updated_at"], "updated_at")
        return payload

    async def resolve_decision(self, decision_id: str) -> dict[str, Any]:
        """Resolve an opaque decision to a private, signed, shell-free spec."""
        decision = _validate_opaque_id(decision_id, "decision_id")
        response = await self._get_client().post(
            f"/api/v2/guidance/decisions/{decision}/resolve",
        )
        response.raise_for_status()
        return _validate_execution_spec(response.json(), decision_id=decision)

    async def reopen_lifecycle(
        self,
        lifecycle_id: str,
        *,
        reason: ReopenReason | str,
    ) -> dict[str, Any]:
        """Create a child lifecycle after an explicit reopen event."""
        lifecycle = _validate_opaque_id(lifecycle_id, "lifecycle_id")
        normalized_reason = _normalize_enum(reason, "reason", _REOPEN_REASONS)
        response = await self._get_client().post(
            f"/api/v2/lifecycles/{lifecycle}/reopen",
            json={"reason": normalized_reason},
            headers={
                "Idempotency-Key": "reopen-"
                + hashlib.sha256(
                    f"{lifecycle}:{normalized_reason}".encode()
                ).hexdigest()[:32]
            },
        )
        response.raise_for_status()
        payload = _require_exact_keys(
            response.json(),
            field="Planner V2 reopen response",
            required={
                "schema_version",
                "lifecycle_id",
                "predecessor_lifecycle_id",
                "reason",
                "requested_variant",
                "utility_profile",
                "status",
                "idempotent_replay",
                "created_at",
            },
        )
        if payload["schema_version"] != PLANNER_SCHEMA_VERSION:
            raise ValueError("Planner V2 reopen schema_version is unsupported")
        if payload["predecessor_lifecycle_id"] != lifecycle:
            raise ValueError("predecessor_lifecycle_id does not match the request")
        if payload["reason"] != normalized_reason or payload["status"] != "active":
            raise ValueError("reopen response does not describe an active child")
        _normalize_enum(payload["requested_variant"], "requested_variant", _TREATMENTS)
        _normalize_enum(
            payload["utility_profile"], "utility_profile", _UTILITY_PROFILES
        )
        if not isinstance(payload["idempotent_replay"], bool):
            raise ValueError("reopen idempotent_replay must be boolean")
        _validate_opaque_id(payload["lifecycle_id"], "lifecycle_id")
        _parse_rfc3339(payload["created_at"], "created_at")
        return payload
