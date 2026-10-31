"""Typed client for the TVL 1.4 reactive routes.

The wire contracts are the closed shapes in TraigentSchema
(``optimization/tvl_reactive_*.json``, ``execution/best_config_deploy_*.json``,
``docs/tvl-reactive-contracts.md``
there).  Two rules shape this module:

* **Requests carry only closed fields and hosted tokens.**  Every body is built by a
  function in this module from enums, bounded integers, digests, server refs and hosted
  tokens.  Customer names (monitor, response, rule, scope, environment) are translated
  to tokens through the persisted token map; when the map is unavailable the request is
  not sent (``local_state_invalid``), never a fallback to the name or a hash.  Nothing
  here accepts a prompt, an output, a question or any free text.  The one exception is
  :meth:`ReactiveClient.publish_deploy`, whose ``spec`` is the candidate configuration:
  customer content that this deploy target stores in Traigent by design.
* **Responses are parsed strictly.**  An unknown key, a wrong type, a value outside its
  enum or pattern, or a ref that does not belong to the request fails the whole response
  closed.  Errors are the closed ``Refusal`` envelope; anything else is
  ``backend_error``.  A failure that leaves the outcome of a write unknown (transport
  error, 5xx, an unparseable success) is flagged ``ambiguous`` so callers never read it
  as "did not happen".
"""

from __future__ import annotations

import re
import secrets
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from traigent.runtime import manifest
from traigent.runtime.client import RuntimeClient
from traigent.runtime.model import (
    E_BACKEND,
    E_STATE,
    RuntimeRequirementError,
)

SCHEMA_VERSION = 1
REACTIVE_PREFIX = "/api/v1/production-reactive/agents"
HEAD_PREFIX = "/api/v1/best-configs/agent-heads"
_BEST_CONFIG_ID = r"[A-Za-z0-9_-]{1,128}"
MAX_JS_INT = 9_007_199_254_740_991

# --- closed vocabularies (mirrors tvl_reactive_common_schema.json) -------------------------

REFUSAL_MESSAGES: dict[str, str] = {
    "invalid_request": "Request is invalid.",
    "unauthenticated": "Authentication is required.",
    "forbidden": "Not permitted.",
    "fact_source_not_permitted": "Fact source is not permitted.",
    "reference_not_found": "Reference not found.",
    "payload_too_large": "Request is too large.",
    "batch_out_of_order": "Batch is out of order.",
    "instant_not_increasing": "Instant does not increase.",
    "evaluation_inputs_frozen": "Evaluation inputs are frozen.",
    "instant_sealed": "Instant is sealed.",
    "fact_contradicts_record": "Fact contradicts the stored record.",
    "idempotency_conflict": "Key was used with a different request.",
    "grant_not_valid": "Grant is not valid.",
    "reservation_conflict": "Attempt is already reserved.",
    "reservation_generation_stale": "Reservation generation is stale.",
    "start_refused": "Start is not authorized.",
    "attempt_not_open": "Attempt is not open.",
    "head_generation_stale": "Agent head moved since approval.",
    "publication_cancelled": "Publication was cancelled.",
    "rate_limited": "Too many requests.",
    "internal_error": "Internal error.",
    "service_unavailable": "Service is temporarily unavailable.",
}
VERDICTS = ("MET", "BREACHED", "UNKNOWN")
UNKNOWN_REASONS = (
    "inconclusive",
    "insufficient",
    "unresolved",
    "unplaceable",
    "late_evaluation",
    "revision_gap",
)
CANDIDATE_VERDICTS = (*VERDICTS, *(f"UNKNOWN({r})" for r in UNKNOWN_REASONS))
LEGACY_DECISIONS = ("Promote", "Reject", "NoDecision")
EFFECTS = ("optimize", "deploy")
ATTEMPT_STATES = (
    "open_pending",
    "approved",
    "reserved",
    "started",
    "lapsed",
    "finished",
    "closed",
)
COMPLETIONS = ("normal", "error", "cancelled", "cost_limit")
FACT_KINDS = (
    "fired",
    "started",
    "finished",
    "approval",
    "authority",
    "deploy_started",
    "deploy_result",
)
NOTIFICATION_EVENTS = (
    "episode_opened",
    "episode_continued",
    "episode_resolved",
    "episode_withdrawn",
    "unknown_for",
    "stale",
    "live_again",
    "optimize_no_candidate",
    "optimize_failed",
    "deploy_refused",
    "deploy_failed",
)
PHASES = ("collect", "freeze", "authority", "seal")

HOSTED_TOKEN_RE = re.compile(r"t[A-Za-z0-9_-]{16}")
_DIGEST = r"[0-9a-f]{64}"
_AGENT_REF = r"agentref_[0-9a-f]{32}"
_ATTEMPT_REF = r"attemptref_[0-9a-f]{32}"
_GRANT_REF = r"grantref_[0-9a-f]{32}"
_RESERVATION_REF = r"reservationref_[0-9a-f]{32}"
_NOTIFICATION_REF = r"notificationref_[0-9a-f]{32}"
_POLICY_REF = r"policyrevisionref_[0-9a-f]{32}"
_KEY = r"[0-9a-f]{32}"
_DEPLOY_KEY = rf"deploy:{_ATTEMPT_REF}:{_DIGEST}:{_DIGEST}"
_USER_REF = r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"
MAX_GENERATION = 9007199254740991
MAX_HOURS = 1_000_000_000
MAX_OCCURRENCE = 1_000_000

Transport = Callable[
    [str, str, "Mapping[str, Any] | None", "Mapping[str, Any] | None"],
    "tuple[int, Any, Mapping[str, str]]",
]


class ReactiveError(RuntimeRequirementError):
    """A closed failure of a reactive route.

    ``code`` is a Refusal code, ``backend_error`` or an SDK code.  ``ambiguous`` is true
    when a write may or may not have been committed (transport failure, 5xx, a success
    body that did not parse): the caller must look up or finalize, never assume.
    ``retry_after`` is the parsed ``Retry-After`` (seconds) of a 429.
    """

    def __init__(
        self,
        code: str,
        path: str | None = None,
        *,
        status: int | None = None,
        ambiguous: bool = False,
        retry_after: int | None = None,
    ) -> None:
        super().__init__(code, path)
        self.status = status
        self.ambiguous = ambiguous
        self.retry_after = retry_after


# --- tiny strict validators -----------------------------------------------------------------


class _Bad(Exception):
    """A response value outside its contract (never carries the value)."""


Check = Callable[[Any], Any]


def _rx(pattern: str) -> Check:
    compiled = re.compile(pattern)

    def check(value: Any) -> Any:
        if not isinstance(value, str) or not compiled.fullmatch(value):
            raise _Bad
        return value

    return check


def _enum(*allowed: str) -> Check:
    def check(value: Any) -> Any:
        if not isinstance(value, str) or value not in allowed:
            raise _Bad
        return value

    return check


def _int(low: int, high: int) -> Check:
    def check(value: Any) -> Any:
        if isinstance(value, bool) or not isinstance(value, int):
            raise _Bad
        if not low <= value <= high:
            raise _Bad
        return value

    return check


def _bool(value: Any) -> Any:
    if not isinstance(value, bool):
        raise _Bad
    return value


def _const(expected: Any) -> Check:
    def check(value: Any) -> Any:
        if isinstance(value, bool) != isinstance(expected, bool) or value != expected:
            raise _Bad
        return value

    return check


def _ts(value: Any) -> Any:
    if not isinstance(value, str) or len(value) > 40:
        raise _Bad
    try:
        datetime.fromisoformat(value)
    except ValueError:
        raise _Bad from None
    return value


def _null(check: Check) -> Check:
    def inner(value: Any) -> Any:
        return None if value is None else check(value)

    return inner


def _arr(item: Check, max_items: int) -> Check:
    def check(value: Any) -> Any:
        if not isinstance(value, list) or len(value) > max_items:
            raise _Bad
        return [item(v) for v in value]

    return check


def _obj(
    required: Mapping[str, Check], optional: Mapping[str, Check] | None = None
) -> Check:
    optional = optional or {}

    def check(value: Any) -> Any:
        if not isinstance(value, dict):
            raise _Bad
        if not set(value) <= set(required) | set(optional):
            raise _Bad
        if not set(required) <= set(value):
            raise _Bad
        out: dict[str, Any] = {}
        for key, item in value.items():
            out[key] = (required.get(key) or optional[key])(item)
        return out

    return check


_VERSION = _const(SCHEMA_VERSION)
_TOKEN = _rx(HOSTED_TOKEN_RE.pattern)
_DIGEST_C = _rx(_DIGEST)
_GEN = _int(0, MAX_GENERATION)
_HOURS = _int(0, MAX_HOURS)
_OCC = _int(0, MAX_OCCURRENCE)
_ATTEMPT_C = _rx(_ATTEMPT_REF)
_GRANT_C = _rx(_GRANT_REF)
_RESERVATION_C = _rx(_RESERVATION_REF)
_POLICY_C = _rx(_POLICY_REF)
_CEILING = _obj(
    {
        "max_trials": _int(1, 10000),
        "max_cost_micro_usd": _int(0, 1_000_000_000_000),
    }
)
_DEPLOY_BINDING = _obj(
    {
        "candidate_fingerprint": _DIGEST_C,
        "gate_result_digest": _DIGEST_C,
        "expected_head_generation": _GEN,
    }
)
_CANDIDATE = _obj(
    {"evidence_valid": _bool},
    {
        "legacy_decision": _enum(*LEGACY_DECISIONS),
        "monitor_verdicts": _arr(_enum(*CANDIDATE_VERDICTS), 32),
        "fingerprint": _DIGEST_C,
        "gate_result_digest": _DIGEST_C,
    },
)

_GRANT = _obj(
    {
        "schema_version": _VERSION,
        "grant_ref": _GRANT_C,
        "attempt_ref": _ATTEMPT_C,
        "policy_revision_ref": _POLICY_C,
        "effect": _enum(*EFFECTS),
        "state": _enum("granted", "declined", "expired", "revoked"),
        "contract_digest": _DIGEST_C,
        "decided_by": _rx(_USER_REF),
        "decided_at": _ts,
    },
    {
        "ceiling": _CEILING,
        "deploy": _DEPLOY_BINDING,
        "head_moved_since_optimization_start": _bool,
        "expires_at": _ts,
    },
)
_ATTEMPT = _obj(
    {
        "schema_version": _VERSION,
        "attempt_ref": _ATTEMPT_C,
        "policy_revision_ref": _POLICY_C,
        "response": _TOKEN,
        "rule": _TOKEN,
        "scope": _TOKEN,
        "occurrence": _OCC,
        "effect": _enum(*EFFECTS),
        "state": _enum(*ATTEMPT_STATES),
        "optimize_outcome": _enum(
            "none", "optimize_no_candidate", "optimize_failed", "promoted"
        ),
        "deploy_outcome": _enum("none", "deployed", "deploy_refused", "deploy_failed"),
        "lapsed": _bool,
        "opened_at": _ts,
    },
    {"grant_ref": _GRANT_C},
)
_PAGE_AFTER = _null(_int(0, MAX_GENERATION))
_GUARD = _obj(
    {
        "kind": _enum("monitor_episode_open", "rule_attempt_open"),
        "list": _enum("only_while", "unless"),
        "value": _enum("TRUE", "FALSE", "UNDECIDED"),
    },
    {"monitor": _TOKEN, "rule": _TOKEN},
)
_NOTIFICATION = _obj(
    {
        "schema_version": _VERSION,
        "notification_ref": _rx(_NOTIFICATION_REF),
        "policy_revision_ref": _POLICY_C,
        "response": _TOKEN,
        "rule": _TOKEN,
        "event": _enum(*NOTIFICATION_EVENTS),
        "scope": _TOKEN,
        "occurrence_index": _OCC,
        "repeat_index": _OCC,
        "evaluation_instant": _HOURS,
        "decision": _enum("delivered", "suppressed"),
        "guards": _arr(_GUARD, 32),
        "evidence_source": _enum("backend_native", "recorded_by_customer", "mixed"),
        "producers_missing": _bool,
        "recorded_at": _ts,
    },
    {"attempt_ref": _ATTEMPT_C},
)
_DEPLOY_ONLY_APPROVAL = (
    "candidate_fingerprint",
    "gate_result_digest",
    "optimization_start_generation",
    "current_head_generation",
    "head_moved_since_optimization_start",
)
_APPROVAL_CONTEXT_BASE = _obj(
    {
        "schema_version": _VERSION,
        "attempt_ref": _ATTEMPT_C,
        "policy_revision_ref": _POLICY_C,
        "effect": _enum(*EFFECTS),
        "state": _enum(*ATTEMPT_STATES),
        "contract_digest": _DIGEST_C,
    },
    {
        "candidate_fingerprint": _DIGEST_C,
        "gate_result_digest": _DIGEST_C,
        "optimization_start_generation": _GEN,
        "current_head_generation": _GEN,
        "head_moved_since_optimization_start": _bool,
    },
)


def _parse_approval_context(value: Any) -> Any:
    """``ApprovalContext`` is ``oneOf`` by effect: deploy carries all the binding fields."""
    doc = _APPROVAL_CONTEXT_BASE(value)
    present = [k for k in _DEPLOY_ONLY_APPROVAL if k in doc]
    if doc["effect"] == "optimize" and present:
        raise _Bad
    if doc["effect"] == "deploy" and len(present) != len(_DEPLOY_ONLY_APPROVAL):
        raise _Bad
    return doc


_ACCEPTED = _int(0, 256)
_PARSERS: dict[str, Check] = {
    "grant": _GRANT,
    "attempt": _ATTEMPT,
    "attempt_page": _obj(
        {
            "schema_version": _VERSION,
            "items": _arr(_ATTEMPT, 100),
            "next_after": _PAGE_AFTER,
        }
    ),
    "notification": _NOTIFICATION,
    "notification_page": _obj(
        {
            "schema_version": _VERSION,
            "items": _arr(_NOTIFICATION, 100),
            "next_after": _PAGE_AFTER,
        }
    ),
    "stage_receipt": _obj(
        {
            "schema_version": _VERSION,
            "stage_key": _rx(_KEY),
            "phase": _enum(*PHASES),
            "accepted": _obj(
                {
                    "revisions": _ACCEPTED,
                    "windows": _ACCEPTED,
                    "episodes": _ACCEPTED,
                    "liveness": _ACCEPTED,
                    "optimizations": _ACCEPTED,
                }
            ),
            "replayed": _bool,
        }
    ),
    "complete_receipt": _obj(
        {
            "schema_version": _VERSION,
            "complete_key": _rx(_KEY),
            "phase": _enum(*PHASES),
            "producers_missing": _bool,
            "replayed": _bool,
        }
    ),
    "reservation": _obj(
        {
            "schema_version": _VERSION,
            "reservation_ref": _RESERVATION_C,
            "attempt_ref": _ATTEMPT_C,
            "grant_ref": _GRANT_C,
            "effect": _enum(*EFFECTS),
            "generation": _GEN,
            "state": _const("pending"),
            "expires_at": _ts,
            "replayed": _bool,
        }
    ),
    "start_receipt": _obj(
        {
            "schema_version": _VERSION,
            "attempt_ref": _ATTEMPT_C,
            "effect": _enum(*EFFECTS),
            "fact": _enum("started", "deploy_started"),
            "started_at": _ts,
            "lease_expires_at": _ts,
            "replayed": _bool,
        }
    ),
    "heartbeat_receipt": _obj(
        {
            "schema_version": _VERSION,
            "attempt_ref": _ATTEMPT_C,
            "lease_expires_at": _ts,
            "replayed": _bool,
        }
    ),
    "outcome_receipt": _obj(
        {
            "schema_version": _VERSION,
            "attempt_ref": _ATTEMPT_C,
            "recorded": _bool,
            "annotation": _enum("none", "late_result"),
            "replayed": _bool,
        }
    ),
    "cancel_receipt": _obj(
        {
            "schema_version": _VERSION,
            "attempt_ref": _ATTEMPT_C,
            "effect": _const("optimize"),
            "completion": _const("cancelled"),
            "replayed": _bool,
        }
    ),
}

_PUBLISH_RESULT_BASE = _obj(
    {
        "schema_version": _VERSION,
        "idempotency_key": _rx(_DEPLOY_KEY),
        "state": _enum("succeeded", "cancelled", "pending"),
        "head_generation": _null(_int(1, MAX_GENERATION)),
        "pinned_version": _null(_int(1, MAX_GENERATION)),
        "decided_at": _null(_ts),
    }
)


def _parse_publish_result(value: Any) -> Any:
    doc = _PUBLISH_RESULT_BASE(value)
    state = doc["state"]
    gens = (doc["head_generation"], doc["pinned_version"])
    if state == "succeeded":
        if None in gens or doc["decided_at"] is None:
            raise _Bad
    elif state == "cancelled":
        if any(g is not None for g in gens) or doc["decided_at"] is None:
            raise _Bad
    elif any(g is not None for g in gens) or doc["decided_at"] is not None:
        raise _Bad
    return doc


def _dict(value: Any) -> Any:
    if not isinstance(value, dict):
        raise _Bad
    return value


_PINNED = _obj(
    {
        "schema_version": _VERSION,
        "generation": _GEN,
        "version": _int(1, MAX_GENERATION),
        "spec": _dict,
    }
)


def _parse_pinned(value: Any) -> Any:
    return _PINNED(value)


def parse_refusal(parsed: Any) -> str | None:
    """The code of a well-formed closed Refusal envelope, else ``None``."""
    if not isinstance(parsed, dict) or set(parsed) != {
        "success",
        "error",
        "error_code",
        "message",
    }:
        return None
    code = parsed["error_code"]
    if (
        parsed["success"] is not False
        or not isinstance(code, str)
        or parsed["error"] != code
        or REFUSAL_MESSAGES.get(code) != parsed["message"]
    ):
        return None
    return code


# --- token map (fail closed) ----------------------------------------------------------------

NS_RESPONSE = "response"
NS_RULE = "rule"
NS_MONITOR = "monitor"
NS_ENVIRONMENT = "environment"
NS_SCOPE = "scope"


class TokenMap:
    """The persisted local-name to hosted-token map; every lookup fails closed.

    Names are translated here and nowhere else.  An unreadable, malformed or missing
    map, or a name with no token, is ``local_state_invalid``: there is no fallback to
    the name or to a hash of it (GNFR-PRIV-002/-003).  Only ``scope`` tokens are minted
    on demand (an episode scope is ingest-minted and announced by its first fact).
    """

    def __init__(self, state_dir: Path | None = None) -> None:
        self._state = state_dir or manifest.default_state_dir()

    def _entries(self) -> list[dict[str, str]]:
        manifest.check_state_dir(self._state)
        doc = manifest._read_json(self._state / "tvl-token-map.json", None)
        if (
            not isinstance(doc, dict)
            or doc.get("format") != manifest.TOKEN_MAP_FORMAT
            or not isinstance(doc.get("tokens"), list)
        ):
            raise RuntimeRequirementError(E_STATE, "tokens")
        out: list[dict[str, str]] = []
        for entry in doc["tokens"]:
            if (
                not isinstance(entry, dict)
                or not all(
                    isinstance(entry.get(k), str)
                    for k in ("namespace", "local", "hosted")
                )
                or not HOSTED_TOKEN_RE.fullmatch(entry["hosted"])
            ):
                raise RuntimeRequirementError(E_STATE, "tokens")
            out.append(entry)
        return out

    def hosted(self, namespace: str, local: str, *, mint: bool = False) -> str:
        if not isinstance(local, str) or not local:
            raise RuntimeRequirementError(E_STATE, "tokens")
        for entry in self._entries():
            if entry["namespace"] == namespace and entry["local"] == local:
                return entry["hosted"]
        if mint and namespace == NS_SCOPE:
            doc = manifest.ensure_tokens(self._state, [(namespace, local)])
            return str(manifest.token_for(doc, namespace, local))
        raise RuntimeRequirementError(E_STATE, "tokens")

    def local(self, namespace: str, hosted: str) -> str | None:
        for entry in self._entries():
            if entry["hosted"] == hosted and entry["namespace"] == namespace:
                return entry["local"]
        return None


# --- typed results --------------------------------------------------------------------------


@dataclass(frozen=True)
class ReservationHandle:
    """The holder's proof: the reservation ref and its server-issued generation."""

    ref: str
    generation: int
    attempt_ref: str
    effect: str
    expires_at: str


@dataclass(frozen=True)
class PublishResult:
    state: str
    idempotency_key: str
    head_generation: int | None
    pinned_version: int | None
    decided_at: str | None


@dataclass(frozen=True)
class PinnedConfig:
    generation: int
    version: int
    spec: dict[str, Any]


def new_key() -> str:
    """A fresh idempotency nonce: 32 lowercase hex characters."""
    return secrets.token_hex(16)


def deploy_key(
    attempt_ref: str, candidate_fingerprint: str, gate_result_digest: str
) -> str:
    """``deploy:{attempt_ref}:{fingerprint}:{gate digest}`` - one per deploy occurrence."""
    key = f"deploy:{attempt_ref}:{candidate_fingerprint}:{gate_result_digest}"
    if not re.fullmatch(_DEPLOY_KEY, key):
        raise RuntimeRequirementError("invalid_request", "deploy_key")
    return key


def _need(pattern: str, value: Any, field: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(pattern, value):
        raise RuntimeRequirementError("invalid_request", field)
    return value


def _need_int(value: Any, low: int, high: int, field: str) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not low <= value <= high
    ):
        raise RuntimeRequirementError("invalid_request", field)
    return value


def _need_enum(value: Any, allowed: Sequence[str], field: str) -> str:
    if not isinstance(value, str) or value not in allowed:
        raise RuntimeRequirementError("invalid_request", field)
    return value


def _need_digest(value: Any, field: str) -> str:
    return _need(_DIGEST, value, field)


def _check_candidate_join_inputs(out: dict[str, Any]) -> None:
    # Valid evidence carries the promotion join inputs; invalid evidence carries none.
    if out["evidence_valid"]:
        if "legacy_decision" not in out or "monitor_verdicts" not in out:
            raise RuntimeRequirementError("invalid_request", "candidate")
    elif set(out) != {"evidence_valid"}:
        raise RuntimeRequirementError("invalid_request", "candidate")


def build_candidate(candidate: Mapping[str, Any]) -> dict[str, Any]:
    """A closed candidate fact; any other key is refused (never dropped silently)."""
    allowed = {
        "evidence_valid",
        "legacy_decision",
        "monitor_verdicts",
        "fingerprint",
        "gate_result_digest",
    }
    if not isinstance(candidate, Mapping) or not set(candidate) <= allowed:
        raise RuntimeRequirementError("invalid_request", "candidate")
    if not isinstance(candidate.get("evidence_valid"), bool):
        raise RuntimeRequirementError("invalid_request", "candidate")
    out: dict[str, Any] = {"evidence_valid": candidate["evidence_valid"]}
    if "legacy_decision" in candidate:
        out["legacy_decision"] = _need_enum(
            candidate["legacy_decision"], LEGACY_DECISIONS, "candidate"
        )
    if "monitor_verdicts" in candidate:
        verdicts = candidate["monitor_verdicts"]
        if not isinstance(verdicts, (list, tuple)) or len(verdicts) > 32:
            raise RuntimeRequirementError("invalid_request", "candidate")
        out["monitor_verdicts"] = [
            _need_enum(v, CANDIDATE_VERDICTS, "candidate") for v in verdicts
        ]
    for name in ("fingerprint", "gate_result_digest"):
        if name in candidate:
            out[name] = _need_digest(candidate[name], "candidate")
    _check_candidate_join_inputs(out)
    return out


class ReactiveClient:
    """The reactive and deploy routes of one agent, request bodies closed, responses strict."""

    def __init__(
        self,
        agent_ref: str,
        tokens: TokenMap,
        *,
        project_id: str = "",
        base_url: str | None = None,
        api_key: str | None = None,
        transport: Transport | None = None,
    ) -> None:
        self.agent_ref = _need(_AGENT_REF, agent_ref, "agent_ref")
        self.tokens = tokens
        if transport is None:
            http = RuntimeClient(project_id, base_url=base_url, api_key=api_key)

            def _default(
                method: str,
                path: str,
                params: Mapping[str, Any] | None,
                body: Mapping[str, Any] | None,
            ) -> tuple[int, Any, Mapping[str, str]]:
                status, parsed, headers = http.send(
                    method,
                    f"{http.origin}{path}",
                    dict(body) if body is not None else None,
                    params=dict(params) if params else None,
                )
                return status, parsed, headers

            transport = _default
        self._transport: Transport = transport

    # -- plumbing ---------------------------------------------------------------------------
    def _path(self, tail: str) -> str:
        return f"{REACTIVE_PREFIX}/{self.agent_ref}{tail}"

    def _head_path(self, tail: str) -> str:
        return f"{HEAD_PREFIX}/{self.agent_ref}{tail}"

    def _call(
        self,
        method: str,
        path: str,
        parser: Check,
        *,
        ok: tuple[int, ...] = (200,),
        body: Mapping[str, Any] | None = None,
        params: Mapping[str, Any] | None = None,
    ) -> Any:
        try:
            status, parsed, headers = self._transport(method, path, params, body)
        except ReactiveError:
            raise
        except RuntimeRequirementError as exc:
            if exc.code == E_BACKEND:
                raise ReactiveError(E_BACKEND, "transport", ambiguous=True) from None
            raise
        except Exception:  # noqa: BLE001 - never surface transport text
            raise ReactiveError(E_BACKEND, "transport", ambiguous=True) from None
        if status in ok:
            try:
                return parser(parsed)
            except _Bad:
                raise ReactiveError(
                    E_BACKEND, "response", status=status, ambiguous=True
                ) from None
        code = parse_refusal(parsed)
        if code is None:
            raise ReactiveError(E_BACKEND, "response", status=status, ambiguous=True)
        raise ReactiveError(
            code,
            status=status,
            ambiguous=status >= 500,
            retry_after=_retry_after(headers) if status == 429 else None,
        )

    def _same_attempt(self, doc: Mapping[str, Any], attempt_ref: str) -> None:
        if doc.get("attempt_ref") != attempt_ref:
            raise ReactiveError(E_BACKEND, "response", ambiguous=True)

    # -- facts ------------------------------------------------------------------------------
    def _fact_revision(self, fact: Mapping[str, Any]) -> dict[str, Any]:
        kind = _need_enum(fact.get("kind"), ("response", "monitor"), "revisions")
        _closed(fact, {"kind", "target", "value"}, "revisions")
        return {
            "kind": kind,
            "target": self.tokens.hosted(kind, str(fact.get("target"))),
            "value": _need_digest(fact.get("value"), "revisions"),
        }

    def _fact_window(self, fact: Mapping[str, Any]) -> dict[str, Any]:
        _closed(
            fact,
            {"monitor", "start", "end", "verdict", "unknown_reason", "usable"},
            "windows",
        )
        verdict = _need_enum(fact.get("verdict"), VERDICTS, "windows")
        out: dict[str, Any] = {
            "monitor": self.tokens.hosted(NS_MONITOR, str(fact.get("monitor"))),
            "start": _need_int(fact.get("start"), 0, MAX_HOURS, "windows"),
            "end": _need_int(fact.get("end"), 0, MAX_HOURS, "windows"),
            "verdict": verdict,
        }
        if not isinstance(fact.get("usable"), bool):
            raise RuntimeRequirementError("invalid_request", "windows")
        out["usable"] = fact["usable"]
        if "unknown_reason" in fact:
            if verdict != "UNKNOWN":
                raise RuntimeRequirementError("invalid_request", "windows")
            out["unknown_reason"] = _need_enum(
                fact["unknown_reason"], UNKNOWN_REASONS, "windows"
            )
        return out

    def _fact_episode(self, fact: Mapping[str, Any]) -> dict[str, Any]:
        _closed(fact, {"monitor", "scope", "event", "reason"}, "episodes")
        event = _need_enum(
            fact.get("event"), ("opened", "continued", "closed"), "episodes"
        )
        out: dict[str, Any] = {
            "monitor": self.tokens.hosted(NS_MONITOR, str(fact.get("monitor"))),
            "scope": self.tokens.hosted(NS_SCOPE, str(fact.get("scope")), mint=True),
            "event": event,
        }
        if event == "closed":
            out["reason"] = _need_enum(
                fact.get("reason"), ("resolved", "withdrawn"), "episodes"
            )
        elif "reason" in fact:
            raise RuntimeRequirementError("invalid_request", "episodes")
        return out

    def _fact_liveness(self, fact: Mapping[str, Any]) -> dict[str, Any]:
        _closed(fact, {"monitor", "event", "scope"}, "liveness")
        return {
            "monitor": self.tokens.hosted(NS_MONITOR, str(fact.get("monitor"))),
            "event": _need_enum(fact.get("event"), ("stale", "live_again"), "liveness"),
            "scope": self.tokens.hosted(NS_SCOPE, str(fact.get("scope")), mint=True),
        }

    def _fact_optimization(self, fact: Mapping[str, Any]) -> dict[str, Any]:
        _closed(
            fact,
            {
                "response",
                "revision",
                "rule",
                "scope",
                "occurrence",
                "fact",
                "completion",
                "candidates",
                "effect",
                "state",
                "checks",
                "write",
                "retries_exhausted",
            },
            "optimizations",
        )
        kind = _need_enum(fact.get("fact"), FACT_KINDS, "optimizations")
        out: dict[str, Any] = {
            "response": self.tokens.hosted(NS_RESPONSE, str(fact.get("response"))),
            "revision": _need_digest(fact.get("revision"), "optimizations"),
            "rule": self.tokens.hosted(NS_RULE, str(fact.get("rule"))),
            "scope": self.tokens.hosted(NS_SCOPE, str(fact.get("scope")), mint=True),
            "occurrence": _need_int(
                fact.get("occurrence"), 0, MAX_OCCURRENCE, "optimizations"
            ),
            "fact": kind,
        }
        if "completion" in fact:
            out["completion"] = _need_enum(
                fact["completion"], COMPLETIONS, "optimizations"
            )
        if "candidates" in fact:
            cands = fact["candidates"]
            if not isinstance(cands, (list, tuple)) or len(cands) > 64:
                raise RuntimeRequirementError("invalid_request", "optimizations")
            out["candidates"] = [build_candidate(c) for c in cands]
        if "effect" in fact:
            out["effect"] = _need_enum(fact["effect"], EFFECTS, "optimizations")
        if "state" in fact:
            out["state"] = _need_enum(
                fact["state"],
                ("granted", "declined", "expired", "reachable", "unreachable"),
                "optimizations",
            )
        if "checks" in fact:
            out["checks"] = _need_enum(
                fact["checks"], ("passed", "refused", "unreachable"), "optimizations"
            )
        if "write" in fact:
            out["write"] = _need_enum(
                fact["write"], ("succeeded", "failed", "not_attempted"), "optimizations"
            )
        if "retries_exhausted" in fact:
            if not isinstance(fact["retries_exhausted"], bool):
                raise RuntimeRequirementError("invalid_request", "optimizations")
            out["retries_exhausted"] = fact["retries_exhausted"]
        if kind in ("approval", "authority") and "state" not in out:
            raise RuntimeRequirementError("invalid_request", "optimizations")
        return out

    def stage(
        self,
        instant: int,
        *,
        policy_revision_ref: str,
        environment: str,
        producer_kind: str,
        revisions: Sequence[Mapping[str, Any]] = (),
        windows: Sequence[Mapping[str, Any]] = (),
        episodes: Sequence[Mapping[str, Any]] = (),
        liveness: Sequence[Mapping[str, Any]] = (),
        optimizations: Sequence[Mapping[str, Any]] = (),
        key: str | None = None,
    ) -> dict[str, Any]:
        """Stage facts for one evaluation instant.  Names are local; tokens are applied here."""
        _need_int(instant, 0, MAX_HOURS, "instant")
        body: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "stage_key": _need(_KEY, key or new_key(), "stage_key"),
            "policy_revision_ref": _need(
                _POLICY_REF, policy_revision_ref, "policy_revision_ref"
            ),
            "environment": self.tokens.hosted(NS_ENVIRONMENT, environment),
            "producer_kind": _need_enum(
                producer_kind,
                ("customer_monitor", "optimization_reporter"),
                "producer_kind",
            ),
        }
        for name, facts, build, cap in (
            ("revisions", revisions, self._fact_revision, 32),
            ("windows", windows, self._fact_window, 256),
            ("episodes", episodes, self._fact_episode, 256),
            ("liveness", liveness, self._fact_liveness, 64),
            ("optimizations", optimizations, self._fact_optimization, 256),
        ):
            if len(facts) > cap:
                raise RuntimeRequirementError("invalid_request", name)
            if facts:
                body[name] = [build(f) for f in facts]
        receipt: dict[str, Any] = self._call(
            "POST",
            self._path(f"/evaluations/{instant}/stage"),
            _PARSERS["stage_receipt"],
            body=body,
        )
        if receipt["stage_key"] != body["stage_key"]:
            raise ReactiveError(E_BACKEND, "response", ambiguous=True)
        return receipt

    def complete(
        self,
        instant: int,
        *,
        policy_revision_ref: str,
        environment: str,
        producer_kind: str,
        disposition: str = "complete",
        monitor: str | None = None,
        completed_through_end: int | None = None,
        key: str | None = None,
    ) -> dict[str, Any]:
        """Customer monitors post a watermark; reporters complete or defer their slot."""
        _need_int(instant, 0, MAX_HOURS, "instant")
        kind = _need_enum(
            producer_kind,
            ("customer_monitor", "optimization_reporter"),
            "producer_kind",
        )
        disp = _need_enum(disposition, ("complete", "defer"), "disposition")
        if kind == "customer_monitor" and disp != "complete":
            raise RuntimeRequirementError("invalid_request", "disposition")
        body: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "complete_key": _need(_KEY, key or new_key(), "complete_key"),
            "policy_revision_ref": _need(
                _POLICY_REF, policy_revision_ref, "policy_revision_ref"
            ),
            "environment": self.tokens.hosted(NS_ENVIRONMENT, environment),
            "producer_kind": kind,
            "disposition": disp,
        }
        if monitor is not None:
            body["monitor"] = self.tokens.hosted(NS_MONITOR, monitor)
        if completed_through_end is not None:
            body["completed_through_end"] = _need_int(
                completed_through_end, 0, MAX_HOURS, "completed_through_end"
            )
        receipt: dict[str, Any] = self._call(
            "POST",
            self._path(f"/evaluations/{instant}/complete"),
            _PARSERS["complete_receipt"],
            body=body,
        )
        if receipt["complete_key"] != body["complete_key"]:
            raise ReactiveError(E_BACKEND, "response", ambiguous=True)
        return receipt

    # -- attempts / grants / notifications (read-only GETs; closed shapes) --------------------
    def list_attempts(
        self, *, limit: int = 50, after: int | None = None
    ) -> dict[str, Any]:
        params: dict[str, Any] = {"limit": _need_int(limit, 1, 100, "limit")}
        if after is not None:
            params["after"] = _need_int(after, 0, MAX_GENERATION, "after")
        page: dict[str, Any] = self._call(
            "GET", self._path("/attempts"), _PARSERS["attempt_page"], params=params
        )
        return page

    def get_attempt(self, attempt_ref: str) -> dict[str, Any]:
        _need(_ATTEMPT_REF, attempt_ref, "attempt_ref")
        doc: dict[str, Any] = self._call(
            "GET", self._path(f"/attempts/{attempt_ref}"), _PARSERS["attempt"]
        )
        self._same_attempt(doc, attempt_ref)
        return doc

    def get_grant(self, grant_ref: str) -> dict[str, Any]:
        _need(_GRANT_REF, grant_ref, "grant_ref")
        doc: dict[str, Any] = self._call(
            "GET", self._path(f"/grants/{grant_ref}"), _PARSERS["grant"]
        )
        if doc["grant_ref"] != grant_ref:
            raise ReactiveError(E_BACKEND, "response", ambiguous=True)
        return doc

    def list_notifications(
        self, *, limit: int = 50, after: int | None = None, decision: str | None = None
    ) -> dict[str, Any]:
        params: dict[str, Any] = {"limit": _need_int(limit, 1, 100, "limit")}
        if after is not None:
            params["after"] = _need_int(after, 0, MAX_GENERATION, "after")
        if decision is not None:
            params["decision"] = _need_enum(
                decision, ("delivered", "suppressed"), "decision"
            )
        page: dict[str, Any] = self._call(
            "GET",
            self._path("/notifications"),
            _PARSERS["notification_page"],
            params=params,
        )
        return page

    def get_notification(self, notification_ref: str) -> dict[str, Any]:
        _need(_NOTIFICATION_REF, notification_ref, "notification_ref")
        doc: dict[str, Any] = self._call(
            "GET",
            self._path(f"/notifications/{notification_ref}"),
            _PARSERS["notification"],
        )
        if doc["notification_ref"] != notification_ref:
            raise ReactiveError(E_BACKEND, "response", ambiguous=True)
        return doc

    def get_approval_context(self, attempt_ref: str) -> dict[str, Any]:
        """What a portal approver sees for one attempt (read-only).

        P-human only: an SDK API key is answered ``forbidden`` (a closed refusal).
        """
        _need(_ATTEMPT_REF, attempt_ref, "attempt_ref")
        doc: dict[str, Any] = self._call(
            "GET",
            self._path(f"/attempts/{attempt_ref}/approval-context"),
            _parse_approval_context,
        )
        self._same_attempt(doc, attempt_ref)
        return doc

    # -- authority: reserve / start / heartbeat / results / cancel --------------------------
    def reserve(
        self, attempt_ref: str, grant_ref: str, *, key: str | None = None
    ) -> ReservationHandle:
        _need(_ATTEMPT_REF, attempt_ref, "attempt_ref")
        body = {
            "schema_version": SCHEMA_VERSION,
            "reservation_key": _need(_KEY, key or new_key(), "reservation_key"),
            "grant_ref": _need(_GRANT_REF, grant_ref, "grant_ref"),
        }
        doc = self._call(
            "POST",
            self._path(f"/attempts/{attempt_ref}/reservations"),
            _PARSERS["reservation"],
            ok=(201,),
            body=body,
        )
        if doc["attempt_ref"] != attempt_ref or doc["grant_ref"] != grant_ref:
            raise ReactiveError(E_BACKEND, "response", ambiguous=True)
        return ReservationHandle(
            doc["reservation_ref"],
            doc["generation"],
            attempt_ref,
            doc["effect"],
            doc["expires_at"],
        )

    def _holder_body(
        self, reservation: ReservationHandle, key_name: str, key: str | None
    ) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            key_name: _need(_KEY, key or new_key(), key_name),
            "reservation_ref": _need(
                _RESERVATION_REF, reservation.ref, "reservation_ref"
            ),
            "reservation_generation": _need_int(
                reservation.generation, 0, MAX_GENERATION, "reservation_generation"
            ),
        }

    def start(
        self, reservation: ReservationHandle, *, key: str | None = None
    ) -> dict[str, Any]:
        """Start authorization.  The external run may launch ONLY after this returns."""
        body = self._holder_body(reservation, "start_key", key)
        doc: dict[str, Any] = self._call(
            "POST",
            self._path(f"/attempts/{reservation.attempt_ref}/start"),
            _PARSERS["start_receipt"],
            ok=(201,),
            body=body,
        )
        self._same_attempt(doc, reservation.attempt_ref)
        if doc["effect"] != reservation.effect:
            raise ReactiveError(E_BACKEND, "response", ambiguous=True)
        return doc

    def heartbeat(
        self, reservation: ReservationHandle, *, key: str | None = None
    ) -> dict[str, Any]:
        body = self._holder_body(reservation, "heartbeat_key", key)
        doc: dict[str, Any] = self._call(
            "POST",
            self._path(f"/attempts/{reservation.attempt_ref}/heartbeat"),
            _PARSERS["heartbeat_receipt"],
            body=body,
        )
        self._same_attempt(doc, reservation.attempt_ref)
        return doc

    def _report(
        self, reservation: ReservationHandle, key: str | None, extra: dict[str, Any]
    ) -> dict[str, Any]:
        body = self._holder_body(reservation, "result_key", key)
        body.update(extra)
        doc: dict[str, Any] = self._call(
            "POST",
            self._path(f"/attempts/{reservation.attempt_ref}/results"),
            _PARSERS["outcome_receipt"],
            body=body,
        )
        self._same_attempt(doc, reservation.attempt_ref)
        return doc

    def report_finished(
        self,
        reservation: ReservationHandle,
        *,
        completion: str,
        candidates: Sequence[Mapping[str, Any]] = (),
        key: str | None = None,
    ) -> dict[str, Any]:
        """The optimization result, with the reservation generation."""
        if len(candidates) > 64:
            raise RuntimeRequirementError("invalid_request", "candidates")
        return self._report(
            reservation,
            key,
            {
                "fact": "finished",
                "completion": _need_enum(completion, COMPLETIONS, "completion"),
                "candidates": [build_candidate(c) for c in candidates],
            },
        )

    def report_deploy_result(
        self,
        reservation: ReservationHandle,
        *,
        checks: str,
        write: str,
        retries_exhausted: bool,
        publish_result_key: str | None = None,
        key: str | None = None,
    ) -> dict[str, Any]:
        extra: dict[str, Any] = {
            "fact": "deploy_result",
            "checks": _need_enum(
                checks, ("passed", "refused", "unreachable"), "checks"
            ),
            "write": _need_enum(
                write, ("succeeded", "failed", "not_attempted"), "write"
            ),
            "retries_exhausted": retries_exhausted,
        }
        if not isinstance(retries_exhausted, bool):
            raise RuntimeRequirementError("invalid_request", "retries_exhausted")
        if publish_result_key is not None:
            extra["publish_result_key"] = _need(
                _DEPLOY_KEY, publish_result_key, "publish_result_key"
            )
        elif write in ("succeeded", "failed"):
            raise RuntimeRequirementError("invalid_request", "publish_result_key")
        return self._report(reservation, key, extra)

    def report_authority(
        self, reservation: ReservationHandle, *, state: str, key: str | None = None
    ) -> dict[str, Any]:
        return self._report(
            reservation,
            key,
            {
                "fact": "authority",
                "state": _need_enum(state, ("reachable", "unreachable"), "state"),
            },
        )

    def cancel(self, attempt_ref: str, *, key: str | None = None) -> dict[str, Any]:
        """P-human only; optimization attempts only (an SDK key is refused with ``forbidden``)."""
        _need(_ATTEMPT_REF, attempt_ref, "attempt_ref")
        body = {
            "schema_version": SCHEMA_VERSION,
            "cancel_key": _need(_KEY, key or new_key(), "cancel_key"),
            "effect": "optimize",
            "confirmation": "process_gone",
        }
        doc: dict[str, Any] = self._call(
            "POST",
            self._path(f"/attempts/{attempt_ref}/cancel"),
            _PARSERS["cancel_receipt"],
            body=body,
        )
        self._same_attempt(doc, attempt_ref)
        return doc

    # -- deploy -----------------------------------------------------------------------------
    def finalize_publish(self, key: str) -> PublishResult:
        """Head-lock finalization: the stored success or a committed cancellation tombstone."""
        _need(_DEPLOY_KEY, key, "key")
        doc = self._call(
            "POST",
            self._head_path(f"/publish-results/{key}/finalize"),
            _parse_publish_result,
            body={"schema_version": SCHEMA_VERSION},
        )
        return self._publish_result(doc, key)

    def publish_deploy(
        self, *, key: str, best_config_id: str, version: int
    ) -> PublishResult:
        """Publish the server-issued best-config version onto the grant's environment head.

        ``POST /best-configs/agent-heads/{agent_ref}/publish-results/{key}`` (P-ingest, the
        reservation holder, after a recorded start).  The body is the closed
        ``DeployPublishRequest``: ids and an integer only, never the configuration.
        """
        _need(_DEPLOY_KEY, key, "idempotency_key")
        body = {
            "schema_version": SCHEMA_VERSION,
            "best_config_id": _need(_BEST_CONFIG_ID, best_config_id, "best_config_id"),
            "version": _need_int(version, 1, MAX_JS_INT, "version"),
        }
        doc = self._call(
            "POST",
            self._head_path(f"/publish-results/{key}"),
            _parse_publish_result,
            body=body,
        )
        return self._publish_result(doc, key)

    def get_pinned_config(self) -> PinnedConfig:
        """The spec of EXACTLY the version the head pins, with its generation and version.

        ``GET /best-configs/agent-heads/{agent_ref}/config`` (ETag = generation).
        """
        doc = self._call("GET", self._head_path("/config"), _parse_pinned)
        return PinnedConfig(doc["generation"], doc["version"], doc["spec"])

    def _publish_result(self, doc: Mapping[str, Any], key: str) -> PublishResult:
        if doc["idempotency_key"] != key:
            raise ReactiveError(E_BACKEND, "response", ambiguous=True)
        return PublishResult(
            doc["state"],
            doc["idempotency_key"],
            doc["head_generation"],
            doc["pinned_version"],
            doc["decided_at"],
        )


def _closed(fact: Mapping[str, Any], allowed: set[str], field: str) -> None:
    if not isinstance(fact, Mapping) or not set(fact) <= allowed:
        raise RuntimeRequirementError("invalid_request", field)


def _retry_after(headers: Mapping[str, str]) -> int | None:
    raw = next((v for k, v in headers.items() if k.lower() == "retry-after"), None)
    if raw is None or not raw.isdigit():
        return None
    return min(int(raw), 3600)


__all__ = [
    "PinnedConfig",
    "PublishResult",
    "ReactiveClient",
    "ReactiveError",
    "ReservationHandle",
    "TokenMap",
    "build_candidate",
    "deploy_key",
    "new_key",
    "parse_refusal",
]
