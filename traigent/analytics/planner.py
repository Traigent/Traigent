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
from datetime import UTC, datetime
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
PLANNER_V2_ECONOMICS_HASH = (
    "df7d6d7c95e61716a55eec42f829c7ca2bb66c285ce4013e66f5dc1cac956d02"
)
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
    {
        "certified_session_utility_advantage_no_kpi_guarantee",
        "no_certified_override",
        "unavailable",
        "not_applicable",
    }
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
_FALLBACK_REASONS = frozenset(
    {
        "policy_unavailable",
        "calibration_unavailable",
        "artifact_invalid",
        "certificate_drift",
        "exact_support_mismatch",
        "override_denied",
    }
)
_RATIONALES_BY_MODE = {
    "rules_control": "rules control selected the next safe lifecycle action",
    "policy_override": (
        "certified session-utility advantage selected; no product KPI guarantee"
    ),
    "rules_parity": "no certified override applies; safe rule action retained",
    "rules_fallback": "policy artifact or certificate unavailable; safe rules were used",
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
_SIGNATURE_RE = re.compile(r"^sig_[A-Za-z0-9_-]{43,128}$")
_SHA256_RE = re.compile(r"^[a-f0-9]{64}$")
_RECEIPT_URL_RE = re.compile(
    r"^/api/v2/lifecycles/(?P<lifecycle>[a-z][a-z0-9_]{2,31}_"
    r"[A-Za-z0-9_-]{16,160})/decisions/(?P<decision>[a-z][a-z0-9_]{2,31}_"
    r"[A-Za-z0-9_-]{16,160})/receipts$"
)

_ACTION_SPECS: dict[
    str, tuple[str, str, int, int, float | None, int | None, int | None]
] = {
    "score_probe_32": ("score_examples", "dataset", 1, 1, None, None, 32),
    "score_full": ("score_examples", "dataset", 2, 2, None, None, None),
    "curate_invalid": ("synth_harder_examples", "dataset", 2, 2, None, None, None),
    "deduplicate": ("synth_harder_examples", "dataset", 1, 1, None, None, None),
    "expand_coverage": ("synth_harder_examples", "dataset", 2, 2, None, None, None),
    "synthesize_hard": ("synth_harder_examples", "dataset", 2, 2, None, None, None),
    "audit_probe_32": ("audit_evaluator", "evaluator", 1, 1, None, None, 32),
    "audit_full": ("audit_evaluator", "evaluator", 2, 2, None, None, None),
    "fix_parser": ("refine_metric", "evaluator", 1, 1, None, None, None),
    "calibrate_threshold": ("refine_metric", "evaluator", 1, 1, None, None, None),
    "reduce_bias": ("refine_metric", "evaluator", 2, 2, None, None, None),
    "stabilize": ("refine_metric", "evaluator", 2, 2, None, None, None),
    "optimize_probe": ("run_optimization", "agent", 2, 2, 0.15, 4, None),
    "optimize_standard": ("run_optimization", "agent", 3, 3, 0.4, 12, None),
    "optimize_deep": ("run_optimization", "agent", 5, 5, 0.75, 30, None),
    "config_expand": ("adjust_configuration_space", "agent", 2, 2, None, None, None),
    "config_prune": ("adjust_configuration_space", "agent", 1, 1, None, None, None),
    "config_fix_inactive": (
        "adjust_configuration_space",
        "agent",
        1,
        1,
        None,
        None,
        None,
    ),
    "compare_probe_64": ("compare_baseline", "agent", 1, 1, None, None, 64),
    "compare_full": ("compare_baseline", "agent", 2, 2, None, None, None),
    "holdout_full": ("run_holdout", "agent", 2, 2, None, None, None),
    "safety_gate": ("add_safety_gate", "agent", 0, 0, None, None, None),
    "promote": ("promote_winner", "agent", 0, 0, None, None, None),
    "wait": ("wait", "session", 0, 0, None, None, None),
    "stop": ("stop", "session", 0, 0, None, None, None),
}

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


def _parse_rfc3339(value: Any, field: str) -> datetime:
    if not isinstance(value, str) or len(value) > 40 or not value.endswith("Z"):
        raise ValueError(f"{field} must be an RFC3339 UTC date-time ending in Z")
    candidate = value[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(candidate)
    except ValueError as exc:
        raise ValueError(
            f"{field} must be an RFC3339 UTC date-time ending in Z"
        ) from exc
    offset = parsed.utcoffset()
    if offset is None or offset.total_seconds() != 0:
        raise ValueError(f"{field} must be UTC")
    return parsed


def _expected_action_signature(
    operation_kind: str,
    variant: str,
    target: str,
    cost_units: int,
    reservation_units: int,
    max_budget_fraction: float | None,
    max_trials: int | None,
    sample_limit: int | None,
) -> str:
    return (
        f"{operation_kind}:{variant}:{target}:{cost_units}:{reservation_units}:"
        f"{max_budget_fraction}:{max_trials}:{sample_limit}"
    )


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
    expires_at = _parse_rfc3339(root["expires_at"], "expires_at")

    argv = root["argv"]
    if (
        not isinstance(argv, list)
        or len(argv) != 5
        or argv[:4] != ["traigent", "guidance", "execute-resolved", "--attempt"]
    ):
        raise ValueError("Planner V2 execution argv is not allowlisted")
    _validate_opaque_id(argv[4], "argv attempt_id")

    spec = _require_exact_keys(
        root["execution_spec"],
        field="Planner V2 execution_spec",
        required={
            "operation_kind",
            "variant",
            "target",
            "cost_units",
            "reservation_units",
            "max_budget_fraction",
            "max_trials",
            "sample_limit",
            "action_signature",
            "economics_hash",
            "attempt_id",
            "receipt_url",
            "lease_expires_at",
        },
        optional={"reserved_cost_usd"},
    )
    operation_kind = spec["operation_kind"]
    if operation_kind not in _EXECUTION_VARIANTS:
        raise ValueError(f"unsupported execution operation_kind {operation_kind!r}")
    if spec["variant"] not in _EXECUTION_VARIANTS[operation_kind]:
        raise ValueError(
            f"unsupported execution variant {spec['variant']!r} for {operation_kind!r}"
        )
    _validate_opaque_id(spec["attempt_id"], "execution_spec.attempt_id")
    if argv[4] != spec["attempt_id"]:
        raise ValueError(
            "Planner V2 execution argv attempt does not match execution_spec"
        )
    receipt_url = spec["receipt_url"]
    receipt_match = (
        _RECEIPT_URL_RE.fullmatch(receipt_url)
        if isinstance(receipt_url, str) and len(receipt_url) <= 512
        else None
    )
    if receipt_match is None or receipt_match.group("decision") != decision_id:
        raise ValueError("execution_spec.receipt_url is not a scoped relative path")
    _validate_opaque_id(
        receipt_match.group("lifecycle"), "execution_spec receipt lifecycle"
    )
    lease_expires_at = _parse_rfc3339(
        spec["lease_expires_at"], "execution_spec.lease_expires_at"
    )
    if lease_expires_at != expires_at:
        raise ValueError("Planner V2 execution and lease expiries must match")
    if expires_at <= datetime.now(UTC):
        raise ValueError("Planner V2 execution lease is expired")

    for field in ("cost_units", "reservation_units"):
        if not isinstance(spec[field], int) or isinstance(spec[field], bool):
            raise ValueError(f"execution_spec.{field} must be an integer")
    for field in ("max_trials", "sample_limit"):
        if spec[field] is not None and (
            not isinstance(spec[field], int) or isinstance(spec[field], bool)
        ):
            raise ValueError(f"execution_spec.{field} must be an integer or null")
    if spec["max_budget_fraction"] is not None and (
        not isinstance(spec["max_budget_fraction"], (int, float))
        or isinstance(spec["max_budget_fraction"], bool)
    ):
        raise ValueError("execution_spec.max_budget_fraction must be numeric or null")

    expected = _ACTION_SPECS[spec["variant"]]
    actual = (
        spec["operation_kind"],
        spec["target"],
        spec["cost_units"],
        spec["reservation_units"],
        spec["max_budget_fraction"],
        spec["max_trials"],
        spec["sample_limit"],
    )
    if actual != expected:
        raise ValueError(
            "execution_spec action economics do not match the exact variant"
        )
    expected_signature = _expected_action_signature(
        spec["operation_kind"],
        spec["variant"],
        spec["target"],
        spec["cost_units"],
        spec["reservation_units"],
        spec["max_budget_fraction"],
        spec["max_trials"],
        spec["sample_limit"],
    )
    if spec["action_signature"] != expected_signature:
        raise ValueError(
            "execution_spec.action_signature does not match exact economics"
        )
    if (
        not isinstance(spec["economics_hash"], str)
        or not _SHA256_RE.fullmatch(spec["economics_hash"])
        or spec["economics_hash"] != PLANNER_V2_ECONOMICS_HASH
    ):
        raise ValueError("execution_spec.economics_hash is unsupported")
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
        if (
            advantage != "certified_session_utility_advantage_no_kpi_guarantee"
            or certificate_ref is None
        ):
            raise ValueError(
                "policy_override requires a certified-positive opaque certificate"
            )
        if decision["source_engine"] != "policy":
            raise ValueError("policy_override requires source_engine='policy'")
        if decision["evidence_level"] != "high":
            raise ValueError("policy_override requires high evidence")
    elif advantage == "certified_session_utility_advantage_no_kpi_guarantee":
        raise ValueError(
            "Only policy_override may claim certified session utility advantage"
        )
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
        "rules_parity": ("rules", "no_certified_override", "medium"),
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

    treatment_modes = (
        {"rules_control", "pending_wait", "safety_stop"}
        if requested_treatment == "rules_control"
        else {
            "policy_override",
            "rules_parity",
            "rules_fallback",
            "pending_wait",
            "safety_stop",
        }
    )
    if mode not in treatment_modes:
        raise ValueError("Planner V2 response contaminates the requested treatment")
    if strict_experiment and (
        mode == "rules_fallback" or meta["context_status"] == "incomplete"
    ):
        raise ValueError("Strict experiment rejects fallback or incomplete context")
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
        expected_verification = {
            "started": {"pending"},
            "submitted": {"pending", "verified", "rejected"},
            "failed": {"rejected"},
            "skipped": {"rejected"},
        }[normalized_status]
        if payload["verification_status"] not in expected_verification:
            raise ValueError(
                "Planner V2 receipt status and verification_status are inconsistent"
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
        """Resolve an opaque decision to a server-tagged, shell-free private spec.

        The integrity tag is verified by the service; this public client has no
        server signing key and therefore treats authenticated HTTPS plus the
        fail-closed structural, economics, scope, and expiry checks as its trust
        boundary.
        """
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
        expected_treatment: PlannerTreatment | str | None = None,
        expected_profile: UtilityProfile | str | None = None,
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
        inherited_treatment = _normalize_enum(
            payload["requested_variant"], "requested_variant", _TREATMENTS
        )
        inherited_profile = _normalize_enum(
            payload["utility_profile"], "utility_profile", _UTILITY_PROFILES
        )
        if not isinstance(payload["idempotent_replay"], bool):
            raise ValueError("reopen idempotent_replay must be boolean")
        child = _validate_opaque_id(payload["lifecycle_id"], "lifecycle_id")
        if child == lifecycle:
            raise ValueError("reopen response must create a distinct child lifecycle")
        if expected_treatment is not None:
            expected = _normalize_enum(
                expected_treatment, "expected_treatment", _TREATMENTS
            )
            if inherited_treatment != expected:
                raise ValueError("reopen response changed the inherited treatment")
        if expected_profile is not None:
            expected = _normalize_enum(
                expected_profile, "expected_profile", _UTILITY_PROFILES
            )
            if inherited_profile != expected:
                raise ValueError(
                    "reopen response changed the inherited utility profile"
                )
        _parse_rfc3339(payload["created_at"], "created_at")
        return payload
