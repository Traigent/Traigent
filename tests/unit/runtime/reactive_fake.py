"""A strict in-process fake of the TRAIG-101 reactive + deploy routes.

"Strict" means the fake is bound to the vendored TraigentSchema contracts
(``tests/fixtures/runtime/reactive_contracts``, byte-exact, sha-pinned):

* a request is matched against the documented routes (an undocumented method or path is
  a violation), its path/query parameters and JSON body are validated against the
  contract schemas, and
* every response the fake produces is validated against the schema documented for that
  status code.

Violations never raise inside the client's call stack (the client would swallow them as
transport noise): they are recorded in ``violations`` and answered with the closed 400
envelope, and a test asserts the list is empty.  State transitions follow the plan
(sections 5.3, 7, 8.7): atomic reservation, start authorized before launch, one start,
lease renewal, same-key replay before stale-generation, tombstoned publications.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

from jsonschema import Draft7Validator
from referencing import Registry, Resource
from referencing.jsonschema import DRAFT7

SCHEMAS = (
    Path(__file__).resolve().parents[2] / "fixtures" / "runtime" / "reactive_contracts"
)
SCHEMA_ROOT = SCHEMAS / "schemas"
BASE = "https://schemas.traigent.ai/"
ENDPOINT_FILES = (
    "optimization/tvl_reactive_endpoints.json",
    "execution/best_config_deploy_endpoints.json",
    "execution/best_config_agent_head_config_endpoints.json",
)


def _retrieve(uri: str) -> Resource[Any]:
    assert uri.startswith(BASE), uri
    doc = json.loads((SCHEMA_ROOT / uri[len(BASE) :]).read_text())
    return Resource.from_contents(doc, default_specification=DRAFT7)


REGISTRY: Registry[Any] = Registry(retrieve=_retrieve)  # type: ignore[call-arg]


def _validator(schema: dict[str, Any], base_uri: str) -> Draft7Validator:
    """Validate against ``schema`` whose relative ``$ref``s resolve from ``base_uri``."""
    resolved = copy.deepcopy(schema)

    def absolutise(node: Any) -> None:
        if isinstance(node, dict):
            ref = node.get("$ref")
            if isinstance(ref, str) and not ref.startswith("#"):
                node["$ref"] = urljoin(base_uri, ref)
            for value in node.values():
                absolutise(value)
        elif isinstance(node, list):
            for value in node:
                absolutise(value)

    absolutise(resolved)
    return Draft7Validator(resolved, registry=REGISTRY)


@dataclass
class Route:
    method: str
    regex: re.Pattern[str]
    operation: str
    op: dict[str, Any]
    base_uri: str
    param_names: list[str]


#: Operations the Backend serves (``production_reactive_routes.py`` incl. the read routes
#: ``feature/traig101-reactive-reads``).  Anything else the vendored endpoint files document
#: is a violation if called, so the SDK cannot rely on it.
SERVED = frozenset(
    {
        "stageTvlReactiveEvaluation",
        "completeTvlReactiveEvaluationProducer",
        "createTvlReactiveGrant",
        "reserveTvlReactiveAttempt",
        "startTvlReactiveAttempt",
        "reportTvlReactiveAttemptResult",
        "renewTvlReactiveAttemptLease",
        "cancelTvlReactiveAttempt",
        "publishTvlDeployResult",
        "finalizeTvlDeployPublishResult",
        "getAgentHeadPinnedConfig",
        # read-only GETs, closed shapes
        "listTvlReactiveAttempts",
        "getTvlReactiveAttempt",
        "getTvlReactiveGrant",
        "listTvlReactiveNotifications",
        "getTvlReactiveNotification",
        "getTvlReactiveApprovalContext",
    }
)
HOLDER_OPS = frozenset(
    {
        "startTvlReactiveAttempt",
        "renewTvlReactiveAttemptLease",
        "reportTvlReactiveAttemptResult",
    }
)


def _no_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    seen: dict[str, Any] = {}
    for key, value in pairs:
        if key in seen:
            raise ValueError("duplicate key")
        seen[key] = value
    return seen


def _load_routes() -> list[Route]:
    routes: list[Route] = []
    for name in ENDPOINT_FILES:
        doc = json.loads((SCHEMA_ROOT / name).read_text())
        for path, methods in doc["paths"].items():
            names = re.findall(r"\{([a-z_]+)\}", path)
            pattern = re.sub(r"\{[a-z_]+\}", r"([^/]+)", path)
            for method, op in methods.items():
                if op["operationId"] not in SERVED:
                    continue
                routes.append(
                    Route(
                        method.upper(),
                        re.compile(f"^{pattern}$"),
                        op["operationId"],
                        op,
                        BASE + name,
                        names,
                    )
                )
    return routes


ROUTES = _load_routes()

MESSAGES = {
    "invalid_request": "Request is invalid.",
    "forbidden": "Not permitted.",
    "reference_not_found": "Reference not found.",
    "grant_not_valid": "Grant is not valid.",
    "reservation_conflict": "Attempt is already reserved.",
    "reservation_generation_stale": "Reservation generation is stale.",
    "start_refused": "Start is not authorized.",
    "attempt_not_open": "Attempt is not open.",
    "head_generation_stale": "Agent head moved since approval.",
    "publication_cancelled": "Publication was cancelled.",
    "idempotency_conflict": "Key was used with a different request.",
    "fact_contradicts_record": "Fact contradicts the stored record.",
    "rate_limited": "Too many requests.",
    "service_unavailable": "Service is temporarily unavailable.",
    "internal_error": "Internal error.",
}


def refusal(code: str) -> dict[str, Any]:
    return {
        "success": False,
        "error": code,
        "error_code": code,
        "message": MESSAGES[code],
    }


class Clock:
    """One shared mutable clock (plan section 9: no client-controlled time)."""

    def __init__(self) -> None:
        self.now = datetime(2026, 10, 23, 12, 0, 0, tzinfo=UTC)

    def __call__(self) -> datetime:
        return self.now

    def advance(self, **kw: float) -> None:
        self.now += timedelta(**kw)


class DroppedResponse(ConnectionError):
    """The response was lost after the server committed."""


def canonical_digest(value: Any) -> str:
    """The Backend's independent re-derivation of TVL's canonical_json_digest."""
    text = json.dumps(value, sort_keys=True, indent=2, ensure_ascii=True) + "\n"
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _iso(moment: datetime) -> str:
    return moment.isoformat().replace("+00:00", "Z")


def make_ref(kind: str, n: int) -> str:
    return f"{kind}ref_{n:032x}"


@dataclass
class Head:
    generation: int = 3
    version: int = 7
    spec: dict[str, Any] = field(
        default_factory=lambda: {
            "schema_version": "traigent.best_config.v1",
            "config_id": "answer",
            "config": {"temperature": 0.1},
        }
    )


class StrictBackend:
    """Contract-bound state machine behind a ``Transport`` callable."""

    RESERVATION_TTL = timedelta(minutes=5)
    LEASE = timedelta(minutes=2)

    def __init__(self, agent_ref: str, clock: Clock | None = None) -> None:
        self.agent_ref = agent_ref
        self.principal = "ingest-key-1"  # the calling ingest key (P-ingest)
        self.principal_kind = "ingest"  # "human" for a portal session
        self.notifications: list[dict[str, Any]] = []
        self.approval_facts: dict[str, dict[str, Any]] = {}
        self.best_configs: dict[tuple[str, int], dict[str, Any]] = {}
        self.clock = clock or Clock()
        self.log: list[dict[str, Any]] = []
        self.violations: list[str] = []
        self.attempts: dict[str, dict[str, Any]] = {}
        self.grants: dict[str, dict[str, Any]] = {}
        self.reservations: dict[str, dict[str, Any]] = {}
        self.head = Head()
        self.publish_results: dict[str, dict[str, Any]] = {}
        self.publications: dict[str, dict[str, Any]] = {}
        self.starts_accepted: dict[str, int] = defaultdict(int)
        self.results: list[dict[str, Any]] = []
        self.staged: list[dict[str, Any]] = []
        self.faults: dict[str, list[str]] = defaultdict(list)
        self.finalize_hook: Callable[[], None] | None = None
        self._n = 0
        self._validators: dict[tuple[str, str], Draft7Validator] = {}

    # -- test setup helpers ---------------------------------------------------------------
    def _next(self) -> int:
        self._n += 1
        return self._n

    def add_attempt(
        self,
        *,
        effect: str = "optimize",
        response: str,
        rule: str,
        scope: str,
        occurrence: int = 0,
    ) -> str:
        ref = make_ref("attempt", self._next())
        self.attempts[ref] = {
            "schema_version": 1,
            "attempt_ref": ref,
            "policy_revision_ref": make_ref("policyrevision", 1),
            "response": response,
            "rule": rule,
            "scope": scope,
            "occurrence": occurrence,
            "effect": effect,
            "state": "open_pending",
            "optimize_outcome": "none",
            "deploy_outcome": "none",
            "lapsed": False,
            "opened_at": _iso(self.clock()),
        }
        return ref

    def grant(
        self,
        attempt_ref: str,
        *,
        hours: int = 4,
        max_trials: int = 20,
        max_cost_micro_usd: int = 5_000_000,
        deploy: dict[str, Any] | None = None,
    ) -> str:
        """A P-human grant (the SDK cannot create one: not part of the client)."""
        attempt = self.attempts[attempt_ref]
        ref = make_ref("grant", self._next())
        doc: dict[str, Any] = {
            "schema_version": 1,
            "grant_ref": ref,
            "attempt_ref": attempt_ref,
            "policy_revision_ref": attempt["policy_revision_ref"],
            "effect": attempt["effect"],
            "state": "granted",
            "contract_digest": "a" * 64,
            "decided_by": "00000000-0000-4000-8000-000000000001",
            "decided_at": _iso(self.clock()),
            "expires_at": _iso(self.clock() + timedelta(hours=hours)),
        }
        if attempt["effect"] == "optimize":
            doc["ceiling"] = {
                "max_trials": max_trials,
                "max_cost_micro_usd": max_cost_micro_usd,
            }
        else:
            assert deploy is not None
            doc["deploy"] = deploy
        self.grants[ref] = doc
        attempt["state"] = "approved"
        attempt["grant_ref"] = ref
        return ref

    def revoke(self, grant_ref: str) -> None:
        self.grants[grant_ref]["state"] = "revoked"

    def fail_next(self, operation: str, *modes: str) -> None:
        """Queue faults for an operation: ``transport``, ``503``, ``429``, ``drop_after_commit``."""
        self.faults[operation].extend(modes)

    def requests_of(self, operation: str) -> list[dict[str, Any]]:
        return [r for r in self.log if r["operation"] == operation]

    # -- transport ------------------------------------------------------------------------
    def transport(
        self,
        method: str,
        path: str,
        params: Any,
        body: Any,
    ) -> tuple[int, Any, dict[str, str]]:
        # The wire is JSON: round-trip so what the fake sees is what a server would.
        if isinstance(body, str):  # a pre-serialised body: duplicate keys are refused
            try:
                wire = json.loads(body, object_pairs_hook=_no_duplicate_keys)
            except ValueError:
                entry_: dict[str, Any] = {"method": method, "path": path, "body": None}
                entry_["operation"] = "duplicate_keys"
                self.log.append(entry_)
                return 400, refusal("invalid_request"), {}
        else:
            wire = json.loads(json.dumps(body)) if body is not None else None
        query = dict(params) if params else {}
        route = next(
            (r for r in ROUTES if r.method == method and r.regex.match(path)), None
        )
        entry: dict[str, Any] = {
            "method": method,
            "path": path,
            "params": query,
            "body": wire,
            "operation": route.operation if route else None,
        }
        self.log.append(entry)
        if route is None:
            self.violations.append(f"undocumented route {method} {path}")
            return self._answer(entry, None, 404, refusal("reference_not_found"))
        values = dict(
            zip(route.param_names, route.regex.match(path).groups(), strict=True)
        )  # type: ignore[union-attr]
        problems = self._check_request(route, values, query, wire)
        if problems:
            self.violations.extend(f"{route.operation}: {p}" for p in problems)
            return self._answer(entry, route, 400, refusal("invalid_request"))
        if values.get("agent_ref") != self.agent_ref:
            return self._answer(entry, route, 404, refusal("reference_not_found"))
        fault = (
            self.faults[route.operation].pop(0)
            if self.faults[route.operation]
            else None
        )
        if fault == "transport":
            entry["fault"] = fault
            raise ConnectionError("injected transport failure")
        if fault == "503":
            entry["fault"] = fault
            return self._answer(entry, route, 503, refusal("service_unavailable"))
        if fault == "429":
            entry["fault"] = fault
            return self._answer(
                entry, route, 429, refusal("rate_limited"), {"Retry-After": "2"}
            )
        if route.operation in HOLDER_OPS:
            held = self._attempt(values["attempt_ref"])
            res = self.reservations.get((held or {}).get("reservation_ref", ""))
            if res is not None and res["holder"] != self.principal:
                return self._answer(entry, route, 404, refusal("reference_not_found"))
        if (
            route.operation == "cancelTvlReactiveAttempt"
            and self.principal_kind != "human"
        ):
            return self._answer(entry, route, 403, refusal("forbidden"))
        handler = getattr(self, f"_op_{route.operation}")
        status, payload = handler(values, query, wire)
        entry["committed"] = status < 300
        if fault == "drop_after_commit":
            entry["fault"] = fault
            self._answer(entry, route, status, payload)  # validate, then lose it
            raise DroppedResponse("response lost after commit")
        return self._answer(entry, route, status, payload)

    def _answer(
        self,
        entry: dict[str, Any],
        route: Route | None,
        status: int,
        payload: Any,
        headers: dict[str, str] | None = None,
    ) -> tuple[int, Any, dict[str, str]]:
        entry["status"] = status
        entry["response"] = payload
        if route is not None:
            declared = route.op["responses"].get(str(status))
            if declared is None:
                self.violations.append(f"{route.operation}: undeclared status {status}")
            else:
                schema = declared["content"]["application/json"]["schema"]
                for error in _validator(schema, route.base_uri).iter_errors(payload):
                    self.violations.append(
                        f"{route.operation}: response {status}: {error.message[:120]}"
                    )
        return status, json.loads(json.dumps(payload)), headers or {}

    def _check_request(
        self, route: Route, values: dict[str, str], query: dict[str, Any], body: Any
    ) -> list[str]:
        problems: list[str] = []
        declared = {p["name"]: p for p in route.op.get("parameters", [])}
        for name, value in values.items():
            schema = declared[name]["schema"]
            raw: Any = (
                int(value)
                if schema.get("type") == "integer" and value.isdigit()
                else value
            )
            for error in _validator(schema, route.base_uri).iter_errors(raw):
                problems.append(f"path {name}: {error.message[:100]}")
        for name, value in query.items():
            if name not in declared or declared[name]["in"] != "query":
                problems.append(f"unknown query parameter {name}")
                continue
            for error in _validator(
                declared[name]["schema"], route.base_uri
            ).iter_errors(value):
                problems.append(f"query {name}: {error.message[:100]}")
        request_body = route.op.get("requestBody")
        if request_body is None:
            if body is not None:
                problems.append("unexpected body")
        else:
            schema = request_body["content"]["application/json"]["schema"]
            for error in _validator(schema, route.base_uri).iter_errors(body):
                problems.append(
                    f"body {list(error.absolute_path)}: {error.message[:100]}"
                )
        return problems

    # -- helpers --------------------------------------------------------------------------
    def _grant_valid(self, grant_ref: str | None) -> bool:
        grant = self.grants.get(grant_ref or "")
        if grant is None or grant["state"] != "granted":
            return False
        return _parse(grant["expires_at"]) > self.clock()

    def _lapse_check(self, attempt: dict[str, Any]) -> None:
        res = self.reservations.get(attempt.get("reservation_ref", ""))
        if (
            res
            and attempt["state"] == "reserved"
            and _parse(res["expires_at"]) <= self.clock()
        ):
            # Plan 7.1: a reservation that expires before start means nothing started
            # and simply frees; the attempt is approvable/reservable again.
            attempt["state"] = "approved"
            res["released"] = True
            return
        if (
            res
            and attempt["state"] == "started"
            and _parse(res["lease_expires_at"]) <= self.clock()
        ):
            attempt["state"] = "lapsed"
            attempt["lapsed"] = True

    def _attempt(self, ref: str) -> dict[str, Any] | None:
        attempt = self.attempts.get(ref)
        if attempt is not None:
            self._lapse_check(attempt)
        return attempt

    @staticmethod
    def _public(attempt: dict[str, Any]) -> dict[str, Any]:
        keys = {
            "schema_version",
            "attempt_ref",
            "policy_revision_ref",
            "response",
            "rule",
            "scope",
            "occurrence",
            "effect",
            "state",
            "optimize_outcome",
            "deploy_outcome",
            "lapsed",
            "opened_at",
            "grant_ref",
        }
        return {k: v for k, v in attempt.items() if k in keys}

    def _holder(self, attempt: dict[str, Any], body: dict[str, Any]) -> str | None:
        """None when the reservation matches, else the refusal code."""
        res = self.reservations.get(attempt.get("reservation_ref", ""))
        if (
            res is None
            or res["reservation_ref"] != body["reservation_ref"]
            or res["generation"] != body["reservation_generation"]
        ):
            return "reservation_generation_stale"
        return None

    # -- operations -----------------------------------------------------------------------
    def _op_listTvlReactiveAttempts(
        self, v: dict, q: dict, body: Any
    ) -> tuple[int, Any]:
        items = [
            self._public(a)
            for a in self.attempts.values()
            if self._attempt(a["attempt_ref"])
        ]
        items.reverse()
        limit = int(q.get("limit", 50))
        after = int(q.get("after", 0))
        page = items[after : after + limit]
        nxt = after + limit if after + limit < len(items) else None
        return 200, {"schema_version": 1, "items": page, "next_after": nxt}

    def _op_getTvlReactiveAttempt(self, v: dict, q: dict, body: Any) -> tuple[int, Any]:
        attempt = self._attempt(v["attempt_ref"])
        if attempt is None:
            return 404, refusal("reference_not_found")
        return 200, self._public(attempt)

    def _op_getTvlReactiveGrant(self, v: dict, q: dict, body: Any) -> tuple[int, Any]:
        grant = self.grants.get(v["grant_ref"])
        if grant is None:
            return 404, refusal("reference_not_found")
        out = dict(grant)
        if grant["state"] == "granted" and _parse(grant["expires_at"]) <= self.clock():
            out["state"] = "expired"
        return 200, out

    def _op_reserveTvlReactiveAttempt(
        self, v: dict, q: dict, body: Any
    ) -> tuple[int, Any]:
        attempt = self._attempt(v["attempt_ref"])
        if attempt is None:
            return 404, refusal("reference_not_found")
        existing = self.reservations.get(attempt.get("reservation_ref", ""))
        if (
            existing
            and not existing.get("released")
            and existing["reservation_key"] == body["reservation_key"]
        ):
            return 201, self._reservation_doc(existing, replayed=True)
        if attempt["state"] != "approved":
            return 409, refusal(
                "reservation_conflict"
                if attempt["state"] == "reserved"
                else "attempt_not_open"
            )
        if body["grant_ref"] != attempt.get("grant_ref") or not self._grant_valid(
            body["grant_ref"]
        ):
            return 409, refusal("grant_not_valid")
        ref = make_ref("reservation", self._next())
        res = {
            "reservation_ref": ref,
            "attempt_ref": attempt["attempt_ref"],
            "grant_ref": body["grant_ref"],
            "effect": attempt["effect"],
            "generation": len(
                [
                    r
                    for r in self.reservations.values()
                    if r["attempt_ref"] == attempt["attempt_ref"]
                ]
            )
            + 1,
            "expires_at": _iso(self.clock() + self.RESERVATION_TTL),
            "lease_expires_at": _iso(self.clock() + self.RESERVATION_TTL),
            "reservation_key": body["reservation_key"],
            "holder": self.principal,
        }
        self.reservations[ref] = res
        attempt["reservation_ref"] = ref
        attempt["state"] = "reserved"
        return 201, self._reservation_doc(res, replayed=False)

    @staticmethod
    def _reservation_doc(res: dict[str, Any], *, replayed: bool) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "reservation_ref": res["reservation_ref"],
            "attempt_ref": res["attempt_ref"],
            "grant_ref": res["grant_ref"],
            "effect": res["effect"],
            "generation": res["generation"],
            "state": "pending",
            "expires_at": res["expires_at"],
            "replayed": replayed,
        }

    def _op_startTvlReactiveAttempt(
        self, v: dict, q: dict, body: Any
    ) -> tuple[int, Any]:
        attempt = self._attempt(v["attempt_ref"])
        if attempt is None:
            return 404, refusal("reference_not_found")
        stale = self._holder(attempt, body)
        if stale:
            return 409, refusal(stale)
        res = self.reservations[attempt["reservation_ref"]]
        fact = "started" if attempt["effect"] == "optimize" else "deploy_started"
        if res.get("start_key") == body["start_key"]:
            return 201, self._start_doc(attempt, res, fact, replayed=True)
        if attempt["state"] != "reserved" or res.get("start_key"):
            return 409, refusal("start_refused")  # no second start on an occurrence
        if (
            not self._grant_valid(res["grant_ref"])
            or _parse(res["expires_at"]) <= self.clock()
        ):
            return 409, refusal("start_refused")
        res["start_key"] = body["start_key"]
        res["started_at"] = _iso(self.clock())
        res["lease_expires_at"] = _iso(self.clock() + self.LEASE)
        attempt["state"] = "started"
        self.starts_accepted[attempt["attempt_ref"]] += 1
        return 201, self._start_doc(attempt, res, fact, replayed=False)

    @staticmethod
    def _start_doc(
        attempt: dict, res: dict, fact: str, *, replayed: bool
    ) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "attempt_ref": attempt["attempt_ref"],
            "effect": attempt["effect"],
            "fact": fact,
            "started_at": res["started_at"],
            "lease_expires_at": res["lease_expires_at"],
            "replayed": replayed,
        }

    def _op_renewTvlReactiveAttemptLease(
        self, v: dict, q: dict, body: Any
    ) -> tuple[int, Any]:
        attempt = self._attempt(v["attempt_ref"])
        if attempt is None:
            return 404, refusal("reference_not_found")
        stale = self._holder(attempt, body)
        if stale:
            return 409, refusal(stale)
        if attempt["state"] not in ("reserved", "started"):
            return 409, refusal("attempt_not_open")
        if (
            not self._grant_valid(attempt["grant_ref"])
            and attempt["state"] == "reserved"
        ):
            return 409, refusal("grant_not_valid")
        res = self.reservations[attempt["reservation_ref"]]
        res["lease_expires_at"] = _iso(self.clock() + self.LEASE)
        if attempt["state"] == "reserved":
            res["expires_at"] = res["lease_expires_at"]
        return 200, {
            "schema_version": 1,
            "attempt_ref": attempt["attempt_ref"],
            "lease_expires_at": res["lease_expires_at"],
            "replayed": False,
        }

    def _op_reportTvlReactiveAttemptResult(
        self, v: dict, q: dict, body: Any
    ) -> tuple[int, Any]:
        attempt = self._attempt(v["attempt_ref"])
        if attempt is None:
            return 404, refusal("reference_not_found")
        stale = self._holder(attempt, body)
        if stale:
            return 409, refusal(stale)
        res = self.reservations[attempt["reservation_ref"]]
        if res.get("result_key") == body["result_key"]:
            return 200, self._outcome_doc(attempt, replayed=True)
        if attempt["state"] not in ("started", "lapsed"):
            return 409, refusal("attempt_not_open")
        expected = "finished" if attempt["effect"] == "optimize" else "deploy_result"
        if body["fact"] not in (expected, "authority"):
            return 409, refusal("fact_contradicts_record")
        if body["fact"] == "deploy_result":
            key = body.get("publish_result_key")
            stored = self.publish_results.get(key or "")
            if body["write"] == "succeeded" and (
                stored is None or stored["state"] != "succeeded"
            ):
                return 409, refusal("fact_contradicts_record")
            if body["write"] == "failed" and (
                stored is None or stored["state"] != "cancelled"
            ):
                return 409, refusal("fact_contradicts_record")
        res["result_key"] = body["result_key"]
        self.results.append({"attempt_ref": attempt["attempt_ref"], **body})
        if body["fact"] != "authority":
            attempt["state"] = "finished"
            if body["fact"] == "finished":
                promoted = any(
                    c.get("legacy_decision") == "Promote"
                    and all(v == "MET" for v in c.get("monitor_verdicts", []))
                    for c in body.get(
                        "candidates", []
                    )  # the gate join (Reject/NoDecision win)
                )
                attempt["optimize_outcome"] = (
                    "promoted"
                    if promoted
                    else "optimize_failed"
                    if body["completion"] in ("error", "cancelled")
                    else "optimize_no_candidate"
                )
            else:
                attempt["deploy_outcome"] = (
                    "deployed"
                    if body["write"] == "succeeded"
                    else "deploy_failed"
                    if body["write"] == "failed"
                    else "deploy_refused"
                )
        return 200, self._outcome_doc(attempt, replayed=False)

    @staticmethod
    def _outcome_doc(attempt: dict[str, Any], *, replayed: bool) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "attempt_ref": attempt["attempt_ref"],
            "recorded": True,
            "annotation": "none",
            "replayed": replayed,
        }

    def _op_stageTvlReactiveEvaluation(
        self, v: dict, q: dict, body: Any
    ) -> tuple[int, Any]:
        self.staged.append(body)
        accepted = {
            name: len(body.get(name, []))
            for name in (
                "revisions",
                "windows",
                "episodes",
                "liveness",
                "optimizations",
            )
        }
        return 200, {
            "schema_version": 1,
            "stage_key": body["stage_key"],
            "phase": "collect",
            "accepted": accepted,
            "replayed": False,
        }

    def _op_completeTvlReactiveEvaluationProducer(
        self, v: dict, q: dict, body: Any
    ) -> tuple[int, Any]:
        return 200, {
            "schema_version": 1,
            "complete_key": body["complete_key"],
            "phase": "collect",
            "producers_missing": False,
            "replayed": False,
        }

    def add_notification(
        self,
        *,
        event: str = "episode_opened",
        decision: str = "delivered",
        attempt_ref: str | None = None,
    ) -> str:
        """A recorded notification (the Backend writes these; the SDK only reads them)."""
        ref = make_ref("notification", self._next())
        doc: dict[str, Any] = {
            "schema_version": 1,
            "notification_ref": ref,
            "policy_revision_ref": make_ref("policyrevision", 1),
            "response": "t" + "r" * 16,
            "rule": "t" + "u" * 16,
            "event": event,
            "scope": "t" + "s" * 16,
            "occurrence_index": 1,
            "repeat_index": 0,
            "evaluation_instant": 3,
            "decision": decision,
            "guards": [],
            "evidence_source": "backend_native",
            "producers_missing": False,
            "recorded_at": _iso(self.clock()),
        }
        if attempt_ref is not None:
            doc["attempt_ref"] = attempt_ref
        self.notifications.append(doc)
        return ref

    def _op_listTvlReactiveNotifications(
        self, v: dict, q: dict, body: Any
    ) -> tuple[int, Any]:
        items = [
            copy.deepcopy(n)
            for n in reversed(self.notifications)
            if q.get("decision") in (None, n["decision"])
        ]
        limit = int(q.get("limit", 50))
        after = int(q.get("after", 0))
        page = items[after : after + limit]
        nxt = after + limit if after + limit < len(items) else None
        return 200, {"schema_version": 1, "items": page, "next_after": nxt}

    def _op_getTvlReactiveNotification(
        self, v: dict, q: dict, body: Any
    ) -> tuple[int, Any]:
        for n in self.notifications:
            if n["notification_ref"] == v["notification_ref"]:
                return 200, copy.deepcopy(n)
        return 404, refusal("reference_not_found")

    def _op_cancelTvlReactiveAttempt(
        self, v: dict, q: dict, body: Any
    ) -> tuple[int, Any]:
        return 403, refusal("forbidden")  # an SDK key is never a P-human

    def _op_createTvlReactiveGrant(
        self, v: dict, q: dict, body: Any
    ) -> tuple[int, Any]:
        return 403, refusal("forbidden")

    def _op_getTvlReactiveApprovalContext(
        self, v: dict, q: dict, body: Any
    ) -> tuple[int, Any]:
        if self.principal_kind != "human":
            return 403, refusal("forbidden")  # P-human only: an SDK key is refused
        attempt = self._attempt(v["attempt_ref"])
        if attempt is None:
            return 404, refusal("reference_not_found")
        doc: dict[str, Any] = {
            "schema_version": 1,
            "attempt_ref": attempt["attempt_ref"],
            "policy_revision_ref": attempt["policy_revision_ref"],
            "effect": attempt["effect"],
            "state": attempt["state"],
            "contract_digest": "a" * 64,
        }
        if attempt["effect"] == "deploy":
            facts = self.approval_facts.get(attempt["attempt_ref"])
            if facts is None:
                return 404, refusal("reference_not_found")
            doc.update(facts)
        return 200, doc

    # deploy -----------------------------------------------------------------------------
    def add_best_config(self, best_config_id: str, version: int, config: dict) -> None:
        """A server-issued best-config version the publish route can pin onto the head."""
        self.best_configs[(best_config_id, version)] = {
            "schema_version": "traigent.best_config.v1",
            "config_id": best_config_id,
            "config": copy.deepcopy(config),
        }

    def _deploy_holder(self, key: str) -> tuple[dict[str, Any] | None, str | None]:
        """The deploy attempt of ``key`` when the caller is its reservation holder."""
        attempt = self._attempt(key.split(":")[1])
        res = self.reservations.get((attempt or {}).get("reservation_ref", ""))
        if (
            attempt is None
            or attempt["effect"] != "deploy"
            or res is None
            or res["holder"] != self.principal
        ):
            return None, "reference_not_found"
        return attempt, None

    def _op_publishTvlDeployResult(
        self, v: dict, q: dict, body: Any
    ) -> tuple[int, Any]:
        key = v["key"]
        attempt, missing = self._deploy_holder(key)
        if missing or attempt is None:
            return 404, refusal("reference_not_found")
        # Same-key replay answers first.
        stored = self.publish_results.get(key)
        if stored is not None:
            if stored["state"] == "cancelled":
                return 409, refusal("publication_cancelled")
            return 200, self._publish_doc(stored)
        if self.starts_accepted[attempt["attempt_ref"]] < 1 or attempt["state"] not in (
            "started",
            "lapsed",
        ):
            return 409, refusal("attempt_not_open")  # start-before-publish
        grant = self.grants.get(attempt.get("grant_ref", ""))
        binding = grant["deploy"] if grant else None
        if binding is None or not self._grant_valid(attempt["grant_ref"]):
            return 409, refusal("grant_not_valid")
        expected = f"deploy:{attempt['attempt_ref']}:{binding['candidate_fingerprint']}:{binding['gate_result_digest']}"
        spec = self.best_configs.get((body["best_config_id"], body["version"]))
        if key != expected or spec is None:
            return 409, refusal("idempotency_conflict")
        if canonical_digest(spec["config"]) != binding["candidate_fingerprint"]:
            return 409, refusal("idempotency_conflict")
        if binding["expected_head_generation"] != self.head.generation:
            return 409, refusal("head_generation_stale")
        self.head.generation += 1
        self.head.version += 1
        self.head.spec = copy.deepcopy(spec)
        self.publications[key] = copy.deepcopy(spec)
        self.publish_results[key] = {
            "state": "succeeded",
            "idempotency_key": key,
            "head_generation": self.head.generation,
            "pinned_version": self.head.version,
            "decided_at": _iso(self.clock()),
        }
        return 200, self._publish_doc(self.publish_results[key])

    @staticmethod
    def _publish_doc(stored: dict[str, Any]) -> dict[str, Any]:
        return {"schema_version": 1, **stored}

    def _op_finalizeTvlDeployPublishResult(
        self, v: dict, q: dict, body: Any
    ) -> tuple[int, Any]:
        if self.finalize_hook:
            self.finalize_hook()
        _, missing = self._deploy_holder(v["key"])
        if missing:
            return 404, refusal("reference_not_found")
        stored = self.publish_results.get(v["key"])
        if stored is None:
            stored = {
                "state": "cancelled",
                "idempotency_key": v["key"],
                "head_generation": None,
                "pinned_version": None,
                "decided_at": _iso(self.clock()),
            }
            self.publish_results[v["key"]] = (
                stored  # the tombstone: later publish cannot commit
            )
        return 200, self._publish_doc(stored)

    def _op_getAgentHeadPinnedConfig(
        self, v: dict, q: dict, body: Any
    ) -> tuple[int, Any]:
        return 200, {
            "schema_version": 1,
            "generation": self.head.generation,
            "version": self.head.version,
            "spec": copy.deepcopy(self.head.spec),
        }


def _parse(text: str) -> datetime:
    return datetime.fromisoformat(text.replace("Z", "+00:00"))
