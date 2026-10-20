"""Closed data model for runtime requirements and reactive policies (PY-1).

A :class:`RuntimeRequirement` is one latency requirement plus a ladder of
response rules.  Sugar kwargs and TVL fragments both build this object, and
:meth:`RuntimeRequirement.to_doc` renders the same TVL 1.3 runtime blocks for
either spelling.  Nothing here touches the network.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

import yaml

KIND_LATENCY = "latency_violation_rate"
TVL_VERSION = "1.3"
DEFAULT_ENVIRONMENT = "production"

#: Window strings accepted by the sugar (hours under the ``backend_v1`` profile).
WINDOWS: Mapping[str, int] = {"1h": 1, "3h": 3, "6h": 6, "12h": 12, "24h": 24}
WINDOW_HOURS_VALUES = tuple(WINDOWS.values())
CONFIDENCE_VALUES = (0.90, 0.95, 0.99)

#: Backend defaults (TraigentBackend ``src/models/production_requirement.py``).
PROFILE_DEFAULTS: Mapping[str, float | int] = {
    "window_hours": 6,
    "grace_minutes": 30,
    "min_samples": 200,
    "max_unresolved_fraction": 0.05,
    "closure_windows": 2,
    "closure_max_age_hours": 24,
    "false_alarm_budget": 0.05,
    "horizon_days": 30,
    "confidence": 0.95,
}
#: Evidence-profile fields sent on the wire, in a fixed order.
PROFILE_FIELDS = (
    "max_violation_rate",
    "confidence",
    "window_hours",
    "grace_minutes",
    "min_samples",
    "max_unresolved_fraction",
    "closure_windows",
    "closure_max_age_hours",
    "false_alarm_budget",
    "horizon_days",
)
_INT_RANGES: Mapping[str, tuple[int, int]] = {
    "latency_threshold_ms": (1, 600_000),
    "grace_minutes": (0, 360),
    "min_samples": (1, 10_000_000),
    "closure_windows": (1, 24),
    "closure_max_age_hours": (1, 720),
    "horizon_days": (1, 365),
}
ENVIRONMENT_MAX_LENGTH = 64

EVENTS = ("episode_opened", "episode_resolved", "episode_withdrawn")
ACTIONS = ("notify", "open_next_step", "optimize")
SEVERITIES = ("info", "warning", "critical")
ON_BREACH_VALUES = ("log", "alert", "fix")
_ON_BREACH_ALIASES = {"optimize": "fix"}
_DEPLOY_VALUES = ("fix_and_deploy", "optimize_and_deploy")

_NAME_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]{0,63}")

# Closed error codes (the only text an error carries besides a structural path).
E_INVALID = "invalid_requirement"
E_INCOMPLETE = "requirement_incomplete"
E_BAD_WINDOW = "invalid_window"
E_BAD_ON_BREACH = "invalid_on_breach"
E_BAD_APPROVAL = "invalid_approval"
E_FIX_CEILING = "fix_ceiling_required"
E_ACTION_UNSUPPORTED = "response_action_unsupported"
E_CAPABILITY = "runtime_unsupported_capability"
E_FRAGMENT = "tvl_fragment_invalid"
E_COMPILER = "tvl_compiler_unavailable"
E_EXISTS = "requirement_exists"
E_DISABLED = "production_requirements_disabled"
E_NOT_CURRENT = "receipt_not_current"
E_APPROVAL_REQUIRED = "approval_required"
E_APPROVAL_EXPIRED = "approval_expired"
E_OCCURRENCE_USED = "occurrence_already_used"
E_HOLD = "hold"
E_STATE = "local_state_invalid"
E_EGRESS = "egress_disabled"
E_NO_CREDENTIALS = "credentials_missing"
E_BACKEND = "backend_error"
E_NOT_APPLIED = "requirement_not_applied"


class RuntimeRequirementError(Exception):
    """A failure with a closed ``code`` and an optional structural ``path``.

    Messages never echo user values, request bodies or server bodies.
    """

    def __init__(self, code: str, path: str | None = None) -> None:
        self.code = code
        self.path = path
        super().__init__(code if path is None else f"{code} at {path}")


@dataclass(frozen=True)
class Issue:
    """One offline validation finding."""

    code: str
    path: str
    severity: str  # "error" | "warning"


@dataclass(frozen=True)
class Receipt:
    """What the backend returned when a requirement was registered."""

    head_id: str
    revision_id: str
    revision_number: int
    provenance: str
    status: str
    created: bool | None = None  # True 201, False 200/no-op

    def as_dict(self) -> dict[str, Any]:
        return {
            "head_id": self.head_id,
            "revision_id": self.revision_id,
            "revision_number": self.revision_number,
            "provenance": self.provenance,
            "status": self.status,
        }


@dataclass(frozen=True)
class ResponseRule:
    """One rung of the response ladder (``log``/``alert``/``fix`` compile to these)."""

    id: str
    event: str
    action: str
    severity: str | None = None
    approval: str | None = None

    def to_doc(self, monitor: str) -> dict[str, Any]:
        rule: dict[str, Any] = {
            "id": self.id,
            "trigger": {"monitor": monitor, "event": self.event},
            "do": self.action,
        }
        if self.action == "notify":
            rule["severity"] = self.severity
        if self.action == "optimize":
            rule["approval"] = self.approval or "human"
        return rule


@dataclass(frozen=True)
class RuntimeEvent:
    """A closed record handed to ``on_event`` and the ``traigent.runtime`` logger."""

    event: str
    breach_id: str
    requirement: str
    severity: str
    opened_at: str | None = None
    closed_at: str | None = None
    outcome: str | None = None
    withdrawn_reason: str | None = None


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def digest_of(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def normalise_environment(value: str) -> str:
    return value.strip().lower()


def _is_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def positive_finite(value: Any) -> bool:
    return _is_number(value) and value > 0


@dataclass(frozen=True)
class RuntimeRequirement:
    """One latency requirement and its response ladder."""

    name: str
    latency_threshold_ms: int
    environment: str
    max_violation_rate: float
    confidence: float = 0.95
    window_hours: int = 6
    grace_minutes: int = 30
    min_samples: int = 200
    max_unresolved_fraction: float = 0.05
    closure_windows: int = 2
    closure_max_age_hours: int = 24
    false_alarm_budget: float = 0.05
    horizon_days: int = 30
    rules: tuple[ResponseRule, ...] = ()
    response_name: str | None = None
    monitor_label: str | None = None
    on_breach: str = "alert"
    max_fix_cost_usd: float | None = None
    agent_id: str | None = None
    source: str = "kwargs"  # "kwargs" -> native create, "tvl" -> tvl-import
    fragment_path: str | None = None
    on_event: Callable[[RuntimeEvent], None] | None = field(
        default=None, compare=False, repr=False
    )

    # -- identity -----------------------------------------------------------
    @property
    def monitor_name(self) -> str:
        return self.monitor_label or f"{self.name}_monitor"

    @property
    def ladder_name(self) -> str:
        return self.response_name or f"{self.name}_ladder"

    def key(self) -> dict[str, Any]:
        return {
            "kind": KIND_LATENCY,
            "latency_threshold_ms": self.latency_threshold_ms,
            "environment": normalise_environment(self.environment),
        }

    def profile(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in PROFILE_FIELDS}

    def native_body(self, agent_id: str) -> dict[str, Any]:
        """The closed ``POST /production-requirements`` body (native create route)."""
        body: dict[str, Any] = {"agent_id": agent_id, **self.key(), **self.profile()}
        body["environment"] = self.environment.strip()
        return body

    def has_optimize_rule(self) -> bool:
        return any(rule.action == "optimize" for rule in self.rules)

    # -- TVL ------------------------------------------------------------------
    def to_doc(self) -> dict[str, Any]:
        """TVL 1.3 runtime blocks (requirement, monitor, responses) as a mapping."""
        doc: dict[str, Any] = {
            "tvl_version": TVL_VERSION,
            "requirements": [
                {
                    "name": self.name,
                    "kind": "rate_bound",
                    "unit": "trace",
                    "violation": {
                        "metric": "wall_clock_latency_ms",
                        "exceeds": self.latency_threshold_ms,
                    },
                    "max_rate": self.max_violation_rate,
                }
            ],
            "monitors": [
                {
                    "name": self.monitor_name,
                    "requirement": self.name,
                    "phase": "runtime",
                    "profile": "backend_v1",
                    "evidence": {
                        "source": "backend_trace_metadata",
                        "deployment_environment": self.environment.strip(),
                    },
                    "window": {
                        "tumbling_hours": self.window_hours,
                        "grace_minutes": self.grace_minutes,
                    },
                    "decision": {
                        "method": "clopper_pearson_exact_v1",
                        "confidence": self.confidence,
                        "false_alarm_budget": self.false_alarm_budget,
                        "min_samples": self.min_samples,
                        "max_unresolved_fraction": self.max_unresolved_fraction,
                        "horizon_days": self.horizon_days,
                    },
                    "episode": {
                        "closure_windows": self.closure_windows,
                        "closure_max_age_hours": self.closure_max_age_hours,
                    },
                }
            ],
        }
        if self.rules:
            doc["responses"] = [
                {
                    "name": self.ladder_name,
                    "rules": [rule.to_doc(self.monitor_name) for rule in self.rules],
                }
            ]
        return doc

    def to_tvl(self) -> str:
        return yaml.safe_dump(self.to_doc(), sort_keys=False, default_flow_style=False)

    def policy_digest(self) -> str:
        """Digest of the FULL policy (requirement, monitor and every response rule)."""
        return digest_of(self.to_doc())

    # -- validation -------------------------------------------------------------
    def validate(self) -> list[Issue]:
        issues: list[Issue] = []

        def bad(path: str) -> None:
            issues.append(Issue(E_INVALID, path, "error"))

        if not _NAME_RE.fullmatch(self.name):
            bad("name")
        for fname, (lo, hi) in _INT_RANGES.items():
            value = getattr(self, fname)
            if type(value) is not int or not lo <= value <= hi:
                bad(fname)
        rate = self.max_violation_rate
        if not _is_number(rate) or not 0 < rate < 1:
            bad("max_violation_rate")
        if self.confidence not in CONFIDENCE_VALUES:
            bad("confidence")
        if self.window_hours not in WINDOW_HOURS_VALUES:
            issues.append(Issue(E_BAD_WINDOW, "window", "error"))
        frac = self.max_unresolved_fraction
        if not _is_number(frac) or not 0 <= frac <= 0.5:
            bad("max_unresolved_fraction")
        alpha = self.false_alarm_budget
        if not _is_number(alpha) or not 0 < alpha <= 0.5:
            bad("false_alarm_budget")
        if not 1 <= len(self.environment.strip()) <= ENVIRONMENT_MAX_LENGTH:
            bad("environment")
        seen: set[str] = set()
        for index, rule in enumerate(self.rules):
            where = f"rules[{index}]"
            if rule.id in seen or not _NAME_RE.fullmatch(rule.id):
                bad(f"{where}.id")
            seen.add(rule.id)
            if rule.event not in EVENTS:
                bad(f"{where}.event")
            if rule.action not in ACTIONS:
                issues.append(Issue(E_ACTION_UNSUPPORTED, f"{where}.do", "error"))
            if (
                rule.action in ("optimize", "open_next_step")
                and rule.event != "episode_opened"
            ):
                issues.append(
                    Issue(E_ACTION_UNSUPPORTED, f"{where}.trigger.event", "error")
                )
            if rule.action == "notify" and rule.severity not in SEVERITIES:
                bad(f"{where}.severity")
            if rule.action == "optimize" and (rule.approval or "human") != "human":
                issues.append(Issue(E_ACTION_UNSUPPORTED, f"{where}.approval", "error"))
        if self.max_fix_cost_usd is not None and not positive_finite(
            self.max_fix_cost_usd
        ):
            bad("max_fix_cost_usd")
        return issues


def build_rules(on_breach: str, approval: str = "human") -> tuple[ResponseRule, ...]:
    """Map the owner's mechanisms onto response rules (design section 3)."""
    if on_breach == "log":
        return (
            ResponseRule("log_opened", "episode_opened", "notify", "info"),
            ResponseRule("log_resolved", "episode_resolved", "notify", "info"),
            ResponseRule("log_withdrawn", "episode_withdrawn", "notify", "info"),
        )
    alert = (
        ResponseRule("alert_opened", "episode_opened", "notify", "warning"),
        ResponseRule("alert_resolved", "episode_resolved", "notify", "info"),
        ResponseRule("alert_withdrawn", "episode_withdrawn", "notify", "info"),
    )
    if on_breach == "alert":
        return alert
    return alert + (
        ResponseRule("restore", "episode_opened", "open_next_step"),
        ResponseRule("fix", "episode_opened", "optimize", approval=approval),
    )


def normalise_on_breach(value: str) -> str:
    if value in _DEPLOY_VALUES:
        raise RuntimeRequirementError(E_ACTION_UNSUPPORTED, "on_breach")
    value = _ON_BREACH_ALIASES.get(value, value)
    if value not in ON_BREACH_VALUES:
        raise RuntimeRequirementError(E_BAD_ON_BREACH, "on_breach")
    return value
