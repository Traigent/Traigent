"""A local loop controller: the closed Continuous Tuning loop on the customer side.

TVL defines the semantics (``tvl.runtime_conditions.conditions_trace``); this module
only *executes* them locally.  Everything the relation consumes is recorded as an
append-only fact in a durable store; ``Controller.step`` re-derives the full trace over
the whole history and acts ONLY on derived facts:

* ``attempt_opened``: run the optimization, but only after a recorded approval;
* ``acceptance ... written``: move the head (generation + 1), atomically.

Store layout (one directory, default ``.traigent/tuning/<agent>/``)::

    meta.json        agent, environment, origin, contract digest, initial head
    head.json        {fingerprint, config, generation}         (atomic temp+rename)
    heads.jsonl      every head ever written (rollback targets, config lookup)
    history.jsonl    recorded facts (checksummed JSONL): events, windows, approvals, finishes, acceptances
    grants.json      standing grants (closed records, with their consumed uses)
    controller.jsonl controller facts: refused optimizations and head writes
    approvals.json   optimization approvals keyed by occurrence ref
    consumed.json    acceptance ids already consumed
    attempts/        one marker per attempt: in-progress, then the persisted result

Nothing here decides a promotion: a candidate without gate evidence is recorded as
``NoDecision`` and never deploys.  No network is used.
"""

from __future__ import annotations

import asyncio
import contextlib
import copy
import fcntl
import hashlib
import json
import logging
import os
import re
import tempfile
import threading
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, cast

from traigent.tuning.errors import TuningError
from traigent.tuning.state import TuningState
from traigent.tuning.validator import conditions_trace

E_INVALID_RECORD = "E_INVALID_RECORD"
E_ATTEMPT_IN_PROGRESS = "E_ATTEMPT_IN_PROGRESS"
E_NO_OPTIMIZER = "E_NO_OPTIMIZER"
E_NO_DEPLOY_AWAITING = "E_NO_DEPLOY_AWAITING"
E_UNKNOWN_GENERATION = "E_UNKNOWN_GENERATION"
E_STORE_MISMATCH = "E_STORE_MISMATCH"
E_NOT_AN_OCCURRENCE = "E_NOT_AN_OCCURRENCE"
E_STORE_CORRUPT = "E_STORE_CORRUPT"
E_STORE_FORMAT = "E_STORE_FORMAT"
E_ACCEPTANCE_CONFLICT = "E_ACCEPTANCE_CONFLICT"
E_INVALID_GRANT = "E_INVALID_GRANT"
E_NO_CANDIDATE_RUN = "E_NO_CANDIDATE_RUN"

GATE_FORMAT = "traigent-gate-result/1"
ACCEPTANCE_FORMAT = "tvl-release-acceptance/1"
DEFAULT_STORE_ROOT = Path(".traigent") / "tuning"
_REF = re.compile(r"^[A-Za-z0-9_.:-]{1,128}$")
_LEGACY = {"Promote", "Reject", "NoDecision"}
_COMPLETIONS = {"normal", "error", "cancelled", "cost_limit"}
OptimizeFn = Callable[[dict[str, Any]], Any]
MAX_GRANT_DAYS = 30  # RFC 0003 section 7.3: a standing grant expires within 30 days
logger = logging.getLogger(__name__)


def default_store_dir(agent: str) -> Path:
    """``.traigent/tuning/<agent>/`` relative to the working directory."""
    if not _REF.match(agent):
        raise TuningError(E_INVALID_RECORD, "agent must be an opaque reference")
    return DEFAULT_STORE_ROOT / agent


def _digest(value: Any) -> str:
    from tvl.canonical import canonical_json_digest

    return str(canonical_json_digest(value))


def config_fingerprint(config: Mapping[str, Any]) -> str:
    """TVL canonical-JSON sha256 (hex) of a configuration."""
    return _digest(dict(config))


def _utc(now: datetime | None) -> datetime:
    value = now or datetime.now(UTC)
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.astimezone(UTC).replace(microsecond=0)


def _instant_text(moment: datetime) -> str:
    return moment.strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse_instant(text: str) -> datetime:
    return datetime.strptime(text, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC)


def scope_ref(occ: Mapping[str, Any]) -> str:
    """The relation's own ``scope_ref`` for an occurrence (never built here).

    Facts render an external scope as ``{issuer, event_id}``; TVL's reference
    ``scope_ref`` takes the raw ``(issuer, event_id)`` identity.
    """
    from tvl.runtime_conditions import scope_ref as tvl_scope_ref

    scope = occ["scope"]
    if not isinstance(scope, Mapping):
        return str(scope)  # already a reference (parsed from a deploy ref)
    raw = (scope["issuer"], scope["event_id"])
    return str(tvl_scope_ref(occ["scope_kind"], raw))


def occurrence_ref(occ: Mapping[str, Any]) -> str:
    """``response/rule/scope_kind/scope_ref/j``; refs never contain ``/`` (OpaqueRef)."""
    return "/".join(
        str(x)
        for x in (
            occ["response"],
            occ["rule"],
            occ["scope_kind"],
            scope_ref(occ),
            occ["j"],
        )
    )


def parse_occurrence_ref(ref: str) -> dict[str, Any]:
    parts = ref.split("/")
    if len(parts) != 5 or not parts[4].isdigit():
        raise TuningError(E_NOT_AN_OCCURRENCE, "expected response/rule/kind/scope/j")
    return {
        "response": parts[0],
        "rule": parts[1],
        "scope_kind": parts[2],
        "scope": parts[3],
        "j": int(parts[4]),
    }


# ---------------------------------------------------------------------------------
# atomic file helpers
# ---------------------------------------------------------------------------------
def _write_json(path: Path, value: Any) -> None:
    """Write JSON atomically: temp file in the same directory, fsync, rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(value, fh, sort_keys=True, indent=2)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise


def _read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return default


def _frame(value: Any) -> str:
    """One JSONL record: ``<sha256[:16] of the JSON>\t<canonical JSON>``."""
    body = json.dumps(value, sort_keys=True)
    return hashlib.sha256(body.encode("utf-8")).hexdigest()[:16] + "\t" + body


def _unframe(line: str) -> Any:
    """The value of a framed line; ``ValueError`` when torn or tampered."""
    check, sep, body = line.partition("\t")
    if not sep or hashlib.sha256(body.encode("utf-8")).hexdigest()[:16] != check:
        raise ValueError("checksum mismatch")
    return json.loads(body)


_FRAME_PREFIX = re.compile(rb"[0-9a-f]{16}\t")


def _is_legacy_line(piece: bytes) -> bool:
    """A pre-checksum record: a bare JSON object with no checksum frame."""
    if _FRAME_PREFIX.match(piece):
        return False
    try:
        return isinstance(json.loads(piece.decode("utf-8")), dict)
    except (ValueError, UnicodeDecodeError):
        return False


def _scan_lines(path: Path) -> tuple[list[Any], int, bool]:
    """``(values, valid_byte_length, torn)`` of a framed JSONL file.

    Only a final line WITHOUT its trailing newline can be a torn append (a crash
    mid-write): it is reported, not returned.  A newline-terminated record that
    fails its checksum was committed, so it is corruption and raises, as does a
    bad line anywhere else.  A file in the pre-checksum format (plain JSON lines)
    raises ``E_STORE_FORMAT`` untouched.
    """
    try:
        raw = path.read_bytes()
    except FileNotFoundError:
        return [], 0, False
    pieces = raw.split(b"\n")
    present = [p for p in pieces if p]
    if present and all(_is_legacy_line(p) for p in present):
        raise TuningError(
            E_STORE_FORMAT,
            f"the tuning store at {path.parent} was written in an older format "
            "without checksums and cannot be read; re-initialize the store "
            "(remove the directory and start again). Its files were not modified.",
        )
    values: list[Any] = []
    valid = 0
    for index, piece in enumerate(pieces):
        last = index == len(pieces) - 1
        if not piece:
            if last:
                break
            valid += 1  # blank line
            continue
        try:
            values.append(_unframe(piece.decode("utf-8")))
        except (ValueError, UnicodeDecodeError):
            if last:  # no trailing newline: the append never completed
                return values, valid, True
            raise TuningError(
                E_STORE_CORRUPT, f"{path.name} has a damaged record"
            ) from None
        valid += len(piece) + (0 if last else 1)
    return values, valid, False


def _repair_tail(path: Path) -> None:
    """Truncate a torn final line (logged) so the next append starts clean."""
    _, valid, torn = _scan_lines(path)
    if torn:
        logger.warning("truncating a torn final record of %s", path.name)
        with open(path, "r+b") as fh:
            fh.truncate(valid)
            fh.flush()
            os.fsync(fh.fileno())
    elif path.exists():
        with open(path, "rb") as fh:
            data = fh.read()
        if data and not data.endswith(b"\n"):
            with open(path, "ab") as out:
                out.write(b"\n")
                out.flush()
                os.fsync(out.fileno())


def _append_line(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _repair_tail(path)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(_frame(value) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def _read_lines(path: Path) -> list[dict[str, Any]]:
    values, _, torn = _scan_lines(path)
    if torn:
        logger.warning("ignoring a torn final record of %s", path.name)
    return cast(list[dict[str, Any]], values)


def read_head(store_dir: Path | str) -> dict[str, Any] | None:
    """The head ``{fingerprint, config, generation}`` of a store, or None."""
    head = _read_json(Path(store_dir) / "head.json")
    return head if isinstance(head, dict) else None


# ---------------------------------------------------------------------------------
# results and reports
# ---------------------------------------------------------------------------------
@dataclass
class StepReport:
    """What one ``step`` did, derived facts included."""

    facts: list[dict[str, Any]] = field(default_factory=list)
    started: list[str] = field(default_factory=list)  # attempts whose optimize ran
    approved: list[str] = field(default_factory=list)
    finished: list[str] = field(default_factory=list)
    awaiting_approval: list[str] = field(default_factory=list)  # occurrence refs
    refused: list[str] = field(default_factory=list)  # "<attempt>:<closed reason>"
    head_written: int | None = None  # new generation, when the head moved

    @property
    def changed(self) -> bool:
        return bool(self.started or self.approved or self.finished or self.head_written)


@dataclass(frozen=True)
class AcceptanceOutcome:
    """The relation's verdict on one submitted acceptance."""

    acceptance_id: str
    result: str  # written | reject_recorded | refused | invalid | consumed
    reason: str | None = None
    generation: int | None = None

    @property
    def ok(self) -> bool:
        return self.result in {"written", "reject_recorded"}


class Controller:
    """Executes the TVL relation locally over a durable store.

    ``optimize_fn(search_space)`` is the decorated function's ``.optimize()`` in real
    use (see :func:`traigent_optimize_fn`); it is injectable for tests.
    """

    def __init__(
        self,
        policy_state: TuningState,
        store_dir: Path | str | None = None,
        optimize_fn: OptimizeFn | None = None,
        *,
        environment: str = "production",
        initial_config: Mapping[str, Any] | None = None,
        authorized_approvers: set[str] | None = None,
        now: datetime | None = None,
    ) -> None:
        policy_state.require_validated()
        self.state = policy_state
        self.module: dict[str, Any] = policy_state.tvl
        agent = policy_state.agent
        if store_dir is None:
            if agent is None:
                raise TuningError(E_INVALID_RECORD, "store_dir or an agent is required")
            store_dir = default_store_dir(agent)
        self.dir = Path(store_dir)
        self.optimize_fn = optimize_fn
        self.authorized_approvers = authorized_approvers
        self._init_store(agent or self.dir.name, environment, initial_config, now)

    # -- store ----------------------------------------------------------------------
    def _p(self, name: str) -> Path:
        return self.dir / name

    @contextlib.contextmanager
    def _locked(self) -> Iterator[None]:
        self.dir.mkdir(parents=True, exist_ok=True)
        with open(self._p(".lock"), "a+") as fh:
            fcntl.flock(fh, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(fh, fcntl.LOCK_UN)

    def _init_store(
        self,
        agent: str,
        environment: str,
        initial_config: Mapping[str, Any] | None,
        now: datetime | None,
    ) -> None:
        digest = "sha256:" + _digest(self.module)
        with self._locked():
            meta = _read_json(self._p("meta.json"))
            if meta is None:
                # Init is a recoverable transaction: ``meta.json`` is written LAST,
                # so a crash earlier leaves the store uninitialized and the next
                # start clears the partial files and redoes the whole init.
                for stale in ("head.json", "heads.jsonl"):
                    with contextlib.suppress(FileNotFoundError):
                        self._p(stale).unlink()
                origin = _utc(now).replace(minute=0, second=0)
                config = dict(initial_config or {})
                fingerprint = config_fingerprint(config)
                head = {"fingerprint": fingerprint, "config": config, "generation": 0}
                _write_json(self._p("head.json"), head)
                _append_line(self._p("heads.jsonl"), head)
                meta = {
                    "agent": agent,
                    "environment": environment,
                    "origin": _instant_text(origin),
                    "contract_digest": digest,
                    "initial_head": {"fingerprint": fingerprint, "generation": 0},
                }
                _write_json(self._p("meta.json"), meta)
            elif meta["contract_digest"] != digest or meta["agent"] != agent:
                raise TuningError(E_STORE_MISMATCH, "store belongs to another policy")
            else:
                for name in ("history.jsonl", "heads.jsonl", "controller.jsonl"):
                    _repair_tail(self._p(name))
            self.meta: dict[str, Any] = meta
            self.origin = _parse_instant(meta["origin"])

    # -- history --------------------------------------------------------------------
    def _lines(self) -> list[dict[str, Any]]:
        return _read_lines(self._p("history.jsonl"))

    def _history(self, extra: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        steps: list[dict[str, Any]] = []
        for line in [*self._lines(), *(extra or [])]:
            step: dict[str, Any] = {"instant": line["instant"]}
            if line["category"] != "tick":
                step[line["category"]] = [line["record"]]
            steps.append(step)
        return {
            "agent": self.meta["agent"],
            "environment": self.meta["environment"],
            "origin": self.meta["origin"],
            "head": self.meta["initial_head"],
            "contract_digest": self.meta["contract_digest"],
            "steps": steps,
        }

    def _derive(self, extra: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        return conditions_trace(self.module, self._history(extra))

    def _hour(self, now: datetime, *, ceil: bool = False) -> int:
        seconds = (_utc(now) - self.origin).total_seconds()
        hours = -(-int(seconds) // 3600) if ceil else int(seconds // 3600)
        lines = self._lines()
        last = lines[-1]["instant"] if lines else 0
        return max(hours, last, 0)

    def _append(
        self, category: str, record: dict[str, Any] | None, instant: int
    ) -> None:
        """Append one recorded fact; a fact the relation rejects never lands."""
        line: dict[str, Any] = {"instant": instant, "category": category}
        if record is not None:
            line["record"] = record
        try:
            self._derive([line])
        except TuningError:
            raise
        except Exception as exc:  # noqa: BLE001 - TVL closed errors; never echo content
            raise TuningError(
                E_INVALID_RECORD, f"{category} rejected by the TVL relation"
            ) from exc
        _append_line(self._p("history.jsonl"), line)

    # -- recording ------------------------------------------------------------------
    def record_event(
        self, envelope: Mapping[str, Any], now: datetime | None = None
    ) -> int:
        """Append an external event envelope; returns its history instant."""
        self.state.require_validated()
        record = copy.deepcopy(dict(envelope))
        issued = record.get("issued_at")
        if isinstance(issued, str):  # convenience: an instant becomes an hour offset
            try:
                delta = _parse_instant(issued) - self.origin
            except ValueError:
                raise TuningError(
                    E_INVALID_RECORD, "issued_at is not an instant"
                ) from None
            record["issued_at"] = int(delta.total_seconds() // 3600)
        with self._locked():
            instant = self._hour(_utc(now))
            self._append("events", record, instant)
        return instant

    def record_window(
        self, report: Mapping[str, Any], now: datetime | None = None
    ) -> int:
        """Append a monitor window report (a verdict computed by the monitor)."""
        self.state.require_validated()
        record = copy.deepcopy(dict(report))
        with self._locked():
            instant = self._hour(_utc(now))
            end = record.get("end")
            if isinstance(end, int) and end > instant:
                instant = end  # a window is recorded no earlier than it ends
            self._append("windows", record, instant)
        return instant

    def approve_optimization(
        self,
        occurrence_ref_: str | Mapping[str, Any],
        approver: str,
        now: datetime | None = None,
    ) -> None:
        """Record a human approval for the attempt an occurrence will open."""
        self.state.require_validated()
        ref = (
            occurrence_ref_
            if isinstance(occurrence_ref_, str)
            else occurrence_ref(occurrence_ref_)
        )
        parse_occurrence_ref(ref)
        if not _REF.match(approver):
            raise TuningError(E_INVALID_RECORD, "approver must be an opaque reference")
        with self._locked():
            approvals = _read_json(self._p("approvals.json"), {})
            approvals.setdefault(
                ref, {"approver": approver, "at": _instant_text(_utc(now))}
            )
            _write_json(self._p("approvals.json"), approvals)

    def record_standing_grant(
        self,
        grant_id: str,
        rule_ref: str,
        max_occurrences: int,
        expires_at: datetime,
        issuer: str,
        now: datetime | None = None,
    ) -> dict[str, Any]:
        """Record a standing grant for an ``approval: standing`` optimize rule.

        RFC 0003 section 7.3: a closed, owner-issued record bound to this agent and
        contract, naming the rule (``response/rule``), a maximum occurrence count and
        an expiry (exclusive) at most 30 days after issue.  Each occurrence consumes
        one use; a retry of the same occurrence does not.
        """
        self.state.require_validated()
        issued = _utc(now)
        expires = _utc(expires_at)
        rule_ok = False
        parts = rule_ref.split("/") if isinstance(rule_ref, str) else []
        if len(parts) == 2:
            for response in self.module.get("responses", []):
                for rule in response["rules"]:
                    if (response["name"], rule["id"]) == tuple(parts):
                        rule_ok = (
                            rule.get("do") == "optimize"
                            and rule.get("approval") == "standing"
                        )
        if not (
            isinstance(grant_id, str)
            and _REF.match(grant_id)
            and isinstance(issuer, str)
            and _REF.match(issuer)
            and rule_ok
            and isinstance(max_occurrences, int)
            and not isinstance(max_occurrences, bool)
            and 1 <= max_occurrences <= 10_000
            and issued < expires <= issued + timedelta(days=MAX_GRANT_DAYS)
        ):
            raise TuningError(E_INVALID_GRANT, "standing grant is not a valid record")
        grant = {
            "grant_id": grant_id,
            "rule_ref": rule_ref,
            "agent_ref": self.meta["agent"],
            "contract_digest": self.meta["contract_digest"],
            "issuer": issuer,
            "issued_at": _instant_text(issued),
            "expires_at": _instant_text(expires),
            "max_occurrences": max_occurrences,
            "used": [],
        }
        with self._locked():
            grants = _read_json(self._p("grants.json"), {})
            if grant_id in grants:
                raise TuningError(E_INVALID_GRANT, "grant id already recorded")
            grants[grant_id] = grant
            _write_json(self._p("grants.json"), grants)
        return copy.deepcopy(grant)

    def _consume_grant(
        self, occ: Mapping[str, Any], ref: str, moment: datetime
    ) -> bool:
        """Consume one use of a matching standing grant (idempotent per occurrence)."""
        rule_ref = f"{occ['response']}/{occ['rule']}"
        grants = _read_json(self._p("grants.json"), {})
        for grant_id in sorted(grants):
            grant = grants[grant_id]
            if (
                grant["rule_ref"] != rule_ref
                or grant["agent_ref"] != self.meta["agent"]
                or grant["contract_digest"] != self.meta["contract_digest"]
            ):
                continue
            if moment >= _parse_instant(grant["expires_at"]):
                continue  # an expired grant authorizes nothing, used or not
            if ref in grant["used"]:
                return True  # a retry of one occurrence does not consume again
            if len(grant["used"]) >= int(grant["max_occurrences"]):
                continue
            grant["used"].append(ref)
            _write_json(self._p("grants.json"), grants)
            return True
        return False

    def controller_facts(self) -> list[dict[str, Any]]:
        """Closed facts the controller itself recorded (refusals); TVL never sees them."""
        return _read_lines(self._p("controller.jsonl"))

    def _record_controller_fact(self, fact: dict[str, Any]) -> None:
        if fact not in self.controller_facts():
            _append_line(self._p("controller.jsonl"), fact)

    # -- the step -------------------------------------------------------------------
    def step(
        self, now: datetime | None = None, *, abandon_interrupted: bool = False
    ) -> StepReport:
        """Re-derive over the whole history and act only on derived facts.

        Idempotent: with an unchanged history nothing is optimized or written twice.
        An attempt interrupted mid-optimization (marker without a result) is refused
        until ``abandon_interrupted`` records it as cancelled.
        """
        self.state.require_validated()
        moment = _utc(now)
        report = StepReport()
        with self._locked():
            if self._has_schedule():
                lines = self._lines()
                hour = self._hour(moment)
                if not lines or hour > lines[-1]["instant"]:
                    self._append("tick", None, hour)
            for _ in range(64):
                trace = self._derive()
                if not self._advance(trace, moment, report, abandon_interrupted):
                    break
            trace = self._derive()
            report.facts = trace["facts"]
            report.head_written = self._write_head(trace)
        return report

    def _has_schedule(self) -> bool:
        return any(
            "schedule" in rule["trigger"]
            for response in self.module.get("responses", [])
            for rule in response["rules"]
        )

    def _rule(self, occ: Mapping[str, Any]) -> dict[str, Any]:
        for response in self.module.get("responses", []):
            if response["name"] != occ["response"]:
                continue
            for rule in response["rules"]:
                if rule["id"] == occ["rule"]:
                    return cast(dict[str, Any], rule)
        raise TuningError(E_NOT_AN_OCCURRENCE, "unknown rule")

    @staticmethod
    def _open_attempts(facts: list[dict[str, Any]]) -> list[dict[str, Any]]:
        closed = {f["attempt"] for f in facts if f["fact"] == "attempt_closed"}
        finished = {f["attempt"] for f in facts if f["fact"] == "optimization_finished"}
        out = []
        for fact in facts:
            if fact["fact"] != "attempt_opened" or fact["attempt"] in closed:
                continue
            out.append({**fact, "finished": fact["attempt"] in finished})
        return out

    def _approval_recorded(self, attempt: str) -> bool:
        return any(
            ln["category"] == "optimizations"
            and ln["record"].get("attempt") == attempt
            and ln["record"].get("fact") == "approval"
            for ln in self._lines()
        )

    def _authorized_now(
        self, occ: Mapping[str, Any], ref: str, moment: datetime
    ) -> bool:
        """A recorded authorization for THIS occurrence, whatever the rule's mode.

        ``standing`` consumes a recorded standing grant; every other mode (``human``
        and any mode the controller does not know) needs ``approve_optimization``.
        """
        if self._rule(occ).get("approval", "human") == "standing":
            return self._consume_grant(occ, ref, moment)
        return ref in _read_json(self._p("approvals.json"), {})

    def _authorization_lapse(
        self, occ: Mapping[str, Any], ref: str, moment: datetime
    ) -> str | None:
        """Why the occurrence is no longer authorized at ``moment``; None when it is.

        Checked immediately before the optimizer runs, on every path (recovery
        included): a recorded approval never outlives its grant.
        """
        if self._rule(occ).get("approval", "human") == "standing":
            rule_ref = f"{occ['response']}/{occ['rule']}"
            grants = _read_json(self._p("grants.json"), {})
            for grant in grants.values():
                if (
                    grant["rule_ref"] == rule_ref
                    and grant["agent_ref"] == self.meta["agent"]
                    and grant["contract_digest"] == self.meta["contract_digest"]
                    and ref in grant["used"]
                    and moment < _parse_instant(grant["expires_at"])
                    and len(grant["used"]) <= int(grant["max_occurrences"])
                ):
                    return None
            return "grant_expired"
        if ref in _read_json(self._p("approvals.json"), {}):
            return None
        return "approval_missing"

    def _scope_requires_intervention(self, occ: Mapping[str, Any], hour: int) -> bool:
        """Ask the relation, run to ``hour``, whether the occurrence's scope still counts.

        An attempt the relation has closed (any reason, ``scope_expired`` included)
        is final: it never counts as requiring intervention.
        """
        from tvl.runtime_conditions import occurrence_status

        history = self._history([{"instant": hour, "category": "tick"}])
        status = occurrence_status(self.module, history, dict(occ))
        if status["phase"] == "closed":
            return False
        return bool(status["scope_active"])

    def _marker_path(self, attempt: str, ref: str) -> Path:
        key = hashlib.sha256(f"{attempt}|{ref}".encode()).hexdigest()[:24]
        return self._p("attempts") / f"{attempt}-{key}.json"

    def _advance(
        self,
        trace: dict[str, Any],
        moment: datetime,
        report: StepReport,
        abandon_interrupted: bool,
    ) -> bool:
        """Do at most one thing the derived facts call for; True when history grew."""
        facts = trace["facts"]
        domains = {f["attempt"]: f for f in facts if f["fact"] == "search_domain"}
        for opened in self._open_attempts(facts):
            if opened["finished"]:
                continue
            attempt, occ = opened["attempt"], opened["occurrence"]
            ref = occurrence_ref(occ)
            hour = self._hour(moment)
            marker_path = self._marker_path(attempt, ref)
            marker = _read_json(marker_path)
            # Immediately before any authorization is used or the optimizer is
            # invoked: the relation, over the history up to now, must still show
            # this occurrence's scope as requiring intervention.
            if marker is None and not self._scope_requires_intervention(occ, hour):
                self._record_controller_fact(
                    {
                        "fact": "optimize_refused",
                        "attempt": attempt,
                        "reason": "scope_expired",
                    }
                )
                report.refused.append(f"{attempt}:scope_expired")
                continue
            if not self._approval_recorded(attempt):
                if not self._authorized_now(occ, ref, moment):
                    if ref not in report.awaiting_approval:
                        report.awaiting_approval.append(ref)
                    continue
                self._append(
                    "optimizations",
                    {"attempt": attempt, "fact": "approval", "state": "granted"},
                    hour,
                )
                report.approved.append(attempt)
                return True
            if marker is not None and "finished_record" in marker:
                self._append("optimizations", marker["finished_record"], hour)
                report.finished.append(attempt)
                return True
            if marker is not None:
                if not abandon_interrupted:
                    raise TuningError(
                        E_ATTEMPT_IN_PROGRESS,
                        f"{attempt} was interrupted; it is not started twice",
                    )
                record = self._finished_record(attempt, "cancelled", [])
                _write_json(marker_path, {**marker, "finished_record": record})
                return True
            lapse = self._authorization_lapse(occ, ref, moment)
            if lapse is not None:
                self._record_controller_fact(
                    {"fact": "optimize_refused", "attempt": attempt, "reason": lapse}
                )
                report.refused.append(f"{attempt}:{lapse}")
                continue
            if self.optimize_fn is None:
                raise TuningError(E_NO_OPTIMIZER, f"{attempt} needs an optimize_fn")
            space = self._search_space(domains.get(attempt))
            _write_json(
                marker_path,
                {"attempt": attempt, "occurrence": ref, "state": "started"},
            )
            report.started.append(attempt)
            try:
                if getattr(self.optimize_fn, "accepts_candidate_dir", False):
                    isolated = cast(Callable[..., Any], self.optimize_fn)
                    result = isolated(
                        copy.deepcopy(space),
                        candidate_dir=self._p("candidates") / attempt,
                    )
                else:
                    result = self.optimize_fn(copy.deepcopy(space))
                completion, candidates, configs = self._map_result(attempt, result)
            except Exception:  # noqa: BLE001 - a failed run is recorded, content never echoed
                completion, candidates, configs = "error", [], {}
            record = self._finished_record(attempt, completion, candidates)
            # Persist the outcome BEFORE recording it: a crash in between must
            # never re-run a finished optimization.
            _write_json(
                marker_path,
                {
                    "attempt": attempt,
                    "occurrence": ref,
                    "state": "finished",
                    "finished_record": record,
                    "configs": configs,
                },
            )
            self._append("optimizations", record, hour)
            report.finished.append(attempt)
            return True
        return False

    @staticmethod
    def _finished_record(
        attempt: str, completion: str, candidates: list[dict[str, Any]]
    ) -> dict[str, Any]:
        return {
            "attempt": attempt,
            "fact": "finished",
            "completion": completion,
            "candidates": candidates,
        }

    def _search_space(self, domain_fact: Mapping[str, Any] | None) -> dict[str, Any]:
        space: dict[str, Any] = {
            t["name"]: copy.deepcopy(t["domain"]) for t in self.module.get("tvars", [])
        }
        if domain_fact is not None:
            space[domain_fact["tvar"]] = list(domain_fact["domain"])
        return space

    # -- mapping Traigent results to the relation's candidate shape ------------------
    def _promotion_monitors(self) -> list[str]:
        return sorted(
            m["name"]
            for m in self.module.get("monitors", [])
            if m.get("phase") == "promotion"
        )

    def _measure_candidate(self, attempt: str, config: Mapping[str, Any]) -> Any:
        """Promotion evidence for the frozen candidate, or None (-> ``NoDecision``).

        The agent function runs with the candidate config (and the current head, paired)
        on the customer's holdout; the customer's checkers label the outputs.  A missing
        holdout or checker, or anything not computable, records an
        ``E_NO_PROMOTION_EVIDENCE`` controller fact and the candidate stays undecided.
        """
        from traigent.tuning.evidence import (
            E_NO_PROMOTION_EVIDENCE,
            EvidenceUnavailable,
            build_promotion_evidence,
        )

        policy = self.state.policy
        head = read_head(self.dir) or {}
        try:
            return build_promotion_evidence(
                self.module,
                fn=getattr(self.state, "optimized", None),
                checkers=getattr(policy, "checkers", None),
                holdout=getattr(policy, "holdout", None),
                candidate_config=config,
                incumbent_config=head.get("config") or {},
            )
        except EvidenceUnavailable as exc:
            self._record_controller_fact(
                {
                    "fact": "no_promotion_evidence",
                    "code": E_NO_PROMOTION_EVIDENCE,
                    "attempt": attempt,
                    "reason": exc.reason,
                }
            )
            return None

    def _map_result(
        self, attempt: str, result: Any
    ) -> tuple[str, list[dict[str, Any]], dict[str, Any]]:
        """``(completion, candidates, configs)`` for the ``finished`` record.

        A result may carry ``candidates`` (dicts with ``config`` and, when the
        optimizer produced evidence, ``legacy_decision`` and ``monitor_verdicts``) or
        just a ``best_config`` (a Traigent ``OptimizationResult``).  Without gate
        evidence the candidate is ``NoDecision`` with every promotion monitor
        ``UNKNOWN(insufficient)``: absence of evidence never promotes.
        """
        from tvl.promotion_join import join_promotion_decisions

        def pick(name: str, default: Any = None) -> Any:
            if isinstance(result, Mapping):
                return result.get(name, default)
            return getattr(result, name, default)

        completion = pick("completion", "normal")
        if completion not in _COMPLETIONS:
            completion = "normal"
        raw = pick("candidates")
        measure = False  # a bare best config is measured on the customer's holdout
        if raw is None:
            best = pick("best_config")
            raw = [{"config": best}] if best else []
            measure = True
        monitors = self._promotion_monitors()
        candidates: list[dict[str, Any]] = []
        configs: dict[str, Any] = {}
        for item in raw:
            item = item if isinstance(item, Mapping) else {"config": item}
            config = item.get("config")
            try:
                if not isinstance(config, Mapping) or not config:
                    raise ValueError("no config")
                fingerprint = config_fingerprint(config)
                legacy = item.get("legacy_decision", "NoDecision")
                given = item.get("monitor_verdicts") or {}
                holdout_digest: str | None = None
                if measure:
                    evidence = self._measure_candidate(attempt, config)
                    if evidence is not None:
                        legacy = evidence.legacy_decision
                        given = evidence.monitor_verdicts
                        holdout_digest = evidence.holdout_digest
                if legacy not in _LEGACY:
                    legacy = "NoDecision"
                verdicts = {
                    n: str(given.get(n, "UNKNOWN(insufficient)")) for n in monitors
                }
                joined = join_promotion_decisions(
                    legacy, [verdicts[n] for n in monitors]
                )
            except Exception:  # noqa: BLE001 - unusable evidence is invalid, never guessed
                candidates.append({"evidence_valid": False})
                continue
            gate_input: dict[str, Any] = {
                "format": GATE_FORMAT,
                "contract_digest": self.meta["contract_digest"],
                "fingerprint": fingerprint,
                "legacy_decision": legacy,
                "monitors": verdicts,
                "decision": str(joined),
            }
            if holdout_digest is not None:
                gate_input["holdout_digest"] = holdout_digest
            gate_digest = "sha256:" + _digest(gate_input)
            configs[fingerprint] = dict(config)
            record: dict[str, Any] = {
                "evidence_valid": True,
                "fingerprint": fingerprint,
                "gate_digest": gate_digest,
                "legacy_decision": legacy,
                "monitor_verdicts": verdicts,
            }
            if holdout_digest is not None:
                record["holdout_digest"] = holdout_digest
            candidates.append(record)
        return str(completion), candidates, configs

    # -- acceptance -----------------------------------------------------------------
    def _generation_fingerprint(self, generation: int) -> str:
        for head in _read_lines(self._p("heads.jsonl")):
            if head["generation"] == generation:
                return str(head["fingerprint"])
        raise TuningError(
            E_UNKNOWN_GENERATION, "no head was written at that generation"
        )

    def build_acceptance(
        self,
        deploy_ref: str,
        approver: str,
        expected_head_generation: int,
        decision: str = "accept",
        now: datetime | None = None,
    ) -> dict[str, Any]:
        """A validated ``tvl-release-acceptance/1`` record for a deploy awaiting."""
        from tvl.release_acceptance import validate_acceptance

        self.state.require_validated()
        if decision not in ("accept", "reject"):
            raise TuningError(E_INVALID_RECORD, "decision must be accept or reject")
        occ = parse_occurrence_ref(deploy_ref)
        deploy = self._deploy_fact(occ)
        fingerprint = str(deploy["candidate"])
        gate_digest = self._gate_digest(fingerprint)
        moment = _utc(now)
        record: dict[str, Any] = {
            "format": ACCEPTANCE_FORMAT,
            "acceptance_id": "R-"
            + hashlib.sha256(
                f"{deploy_ref}|{fingerprint}|{expected_head_generation}|{decision}".encode()
            ).hexdigest()[:24],
            "agent_ref": self.meta["agent"],
            "environment_ref": self.meta["environment"],
            "occurrence": {
                "response_ref": occ["response"],
                "rule_ref": occ["rule"],
                "scope_kind": occ["scope_kind"],
                "scope_ref": occ["scope"],
                "occurrence": occ["j"],
            },
            "candidate_fingerprint": fingerprint,
            "gate_result_digest": gate_digest,
            "contract_digest": self.meta["contract_digest"],
            "expected_head_generation": int(expected_head_generation),
            "rollback_target": self._generation_fingerprint(expected_head_generation),
            "approver_ref": approver,
            "decision": decision,
            "decided_at": _instant_text(moment),
            "expires_at": _instant_text(moment + timedelta(days=7)),
        }
        holdout = self._holdout_digest(fingerprint)
        if holdout is not None:  # RFC 0012: bind the holdout the evidence used
            record["holdout_digest"] = holdout
        if occ["scope_kind"] == "episode":  # RFC 0012: repairs the originating episode
            record["repairs_episode"] = occ["scope"]
        if validate_acceptance(record):
            raise TuningError(E_INVALID_RECORD, "acceptance record is not valid")
        return record

    def _deploy_fact(self, occ: Mapping[str, Any]) -> dict[str, Any]:
        trace = self._derive()
        facts = trace["facts"]
        closed = {f["attempt"] for f in facts if f["fact"] == "attempt_closed"}
        for fact in facts:
            if (
                fact["fact"] == "deploy_awaiting"
                and fact["attempt"] not in closed
                and occurrence_ref(fact["occurrence"]) == occurrence_ref(occ)
            ):
                return cast(dict[str, Any], fact)
        raise TuningError(E_NO_DEPLOY_AWAITING, "no deploy awaits that occurrence")

    def _holdout_digest(self, fingerprint: str) -> str | None:
        for marker in self._markers():
            for cand in marker.get("finished_record", {}).get("candidates", []):
                if cand.get("fingerprint") == fingerprint:
                    value = cand.get("holdout_digest")
                    return str(value) if value else None
        return None

    def _gate_digest(self, fingerprint: str) -> str:
        for marker in self._markers():
            for cand in marker.get("finished_record", {}).get("candidates", []):
                if cand.get("fingerprint") == fingerprint:
                    return str(cand["gate_digest"])
        raise TuningError(E_NO_DEPLOY_AWAITING, "candidate evidence not found")

    def _markers(self) -> list[dict[str, Any]]:
        folder = self._p("attempts")
        if not folder.is_dir():
            return []
        return [
            m
            for m in (_read_json(p) for p in sorted(folder.glob("*.json")))
            if isinstance(m, dict)
        ]

    def submit_acceptance(
        self,
        record: Mapping[str, Any],
        now: datetime | None = None,
        *,
        approver_authorized: bool | None = None,
    ) -> AcceptanceOutcome:
        """Record an acceptance, step, and return the relation's verdict on it.

        The head moves only when the derived fact is ``acceptance ... written``.
        """
        self.state.require_validated()
        doc = copy.deepcopy(dict(record))
        approver = doc.get("approver_ref")
        if approver_authorized is None:
            approver_authorized = (
                self.authorized_approvers is None
                or approver in self.authorized_approvers
            )
        moment = _utc(now)
        ident = doc.get("acceptance_id")
        with self._locked():
            prior = [
                ln["record"]["record"]
                for ln in self._lines()
                if ln["category"] == "acceptances"
            ]
            if isinstance(ident, str) and any(
                isinstance(rec, dict)
                and rec.get("acceptance_id") == ident
                and rec != doc
                for rec in prior
            ):
                # One acceptance id, one content: a conflicting replay never lands.
                raise TuningError(
                    E_ACCEPTANCE_CONFLICT,
                    "that acceptance id was already submitted with other content",
                )
            position = len(prior)
            instant = self._hour(moment, ceil=True)
            self._append(
                "acceptances",
                {
                    "record": doc,
                    "authenticated": True,
                    "approver_authorized": approver_authorized,
                },
                instant,
            )
        self.step(moment)
        # The relation emits exactly one ``acceptance`` fact per acceptance record,
        # in order: the verdict on THIS record is the fact at its position.
        verdicts = [f for f in self._derive()["facts"] if f["fact"] == "acceptance"]
        last = verdicts[position]
        return AcceptanceOutcome(
            acceptance_id=str(ident),
            result=str(last["result"]),
            reason=last.get("reason"),
            generation=last.get("generation"),
        )

    def accept(
        self,
        deploy_ref: str,
        approver: str,
        expected_head_generation: int,
        decision: str = "accept",
        now: datetime | None = None,
    ) -> AcceptanceOutcome:
        """Accept (or reject) the deploy awaiting ``deploy_ref``."""
        record = self.build_acceptance(
            deploy_ref, approver, expected_head_generation, decision, now
        )
        return self.submit_acceptance(record, now)

    # -- head -----------------------------------------------------------------------
    def _write_head(self, trace: dict[str, Any]) -> int | None:
        """Write the head(s) the derived facts call for; only ever moves forward.

        Each derived write is bound to the exact acceptance record that produced it
        (the relation emits one ``acceptance`` fact per record, in history order),
        and its approver must be authorized NOW, at effect time (RFC 0012 section 2.3).
        """
        current = read_head(self.dir) or {"generation": 0}
        records = [
            ln["record"]["record"]
            for ln in self._lines()
            if ln["category"] == "acceptances"
        ]
        verdicts = [f for f in trace["facts"] if f["fact"] == "acceptance"]
        if len(verdicts) != len(records):
            raise TuningError(E_STORE_CORRUPT, "history and derived facts disagree")
        position = 0
        consumed: list[str] = []
        writes: list[tuple[int, str, str | None, str | None]] = []
        for fact in trace["facts"]:
            if fact["fact"] == "acceptance":
                rec = records[position]
                position += 1
                if fact["result"] in ("written", "reject_recorded"):
                    consumed.append(str(fact["acceptance"]))
                if fact["result"] == "written":
                    writes.append(
                        (
                            int(fact["generation"]),
                            str(rec["candidate_fingerprint"]),
                            str(fact["acceptance"]),
                            str(rec.get("approver_ref")),
                        )
                    )
            elif fact["fact"] == "rollback" and fact["result"] == "written":
                writes.append(
                    (int(fact["generation"]), str(fact["fingerprint"]), None, None)
                )
        _write_json(self._p("consumed.json"), sorted(set(consumed)))
        written: int | None = None
        for generation, fingerprint, acceptance, approver in writes:
            if generation <= int(current["generation"]):
                continue
            if acceptance is not None and not self._approver_authorized_now(approver):
                self._record_controller_fact(
                    {
                        "fact": "head_write_refused",
                        "acceptance": acceptance,
                        "generation": generation,
                        "reason": "approver_not_authorized",
                    }
                )
                break  # a later generation never skips an unwritten one
            config = self._config_for(fingerprint)
            head = {
                "fingerprint": fingerprint,
                "config": config,
                "generation": generation,
            }
            if not any(
                h["generation"] == generation
                for h in _read_lines(self._p("heads.jsonl"))
            ):
                _append_line(self._p("heads.jsonl"), head)
            _write_json(self._p("head.json"), head)
            current = head
            written = generation
        return written

    def _approver_authorized_now(self, approver: str | None) -> bool:
        return (
            self.authorized_approvers is None or approver in self.authorized_approvers
        )

    def _config_for(self, fingerprint: str) -> dict[str, Any]:
        for marker in self._markers():
            if fingerprint in marker.get("configs", {}):
                return dict(marker["configs"][fingerprint])
        for head in _read_lines(self._p("heads.jsonl")):
            if head["fingerprint"] == fingerprint:
                return dict(head["config"])
        raise TuningError(E_UNKNOWN_GENERATION, "configuration for the head not found")

    # -- status ---------------------------------------------------------------------
    def status(self) -> dict[str, Any]:
        """Head, open attempt and pending deploys (derived, read-only)."""
        self.state.require_validated()
        facts = self._derive()["facts"]
        closed = {f["attempt"] for f in facts if f["fact"] == "attempt_closed"}
        approvals = _read_json(self._p("approvals.json"), {})
        attempts = []
        for opened in self._open_attempts(facts):
            ref = occurrence_ref(opened["occurrence"])
            if opened["finished"]:
                state = "deploy_awaiting"
            elif self._approval_recorded(opened["attempt"]) or (
                self._rule(opened["occurrence"]).get("approval", "human") != "standing"
                and ref in approvals
            ):
                marker = _read_json(self._marker_path(opened["attempt"], ref))
                state = (
                    "interrupted"
                    if marker and "finished_record" not in marker
                    else "approved"
                )
            else:
                state = "awaiting_approval"
            attempts.append(
                {"attempt": opened["attempt"], "occurrence": ref, "state": state}
            )
        deploys = [
            {
                "attempt": f["attempt"],
                "occurrence": occurrence_ref(f["occurrence"]),
                "candidate": f["candidate"],
            }
            for f in facts
            if f["fact"] == "deploy_awaiting" and f["attempt"] not in closed
        ]
        head = read_head(self.dir) or {}
        return {
            "head": {
                "generation": head.get("generation"),
                "fingerprint": head.get("fingerprint"),
            },
            "open_attempts": attempts,
            "pending_deploys": deploys,
        }


def _candidate_copy(fn: Any) -> Any:
    """A throwaway copy of an optimized function with its OWN config state.

    ``optimize(apply=False)`` is withdrawn in the SDK, and ``optimize(apply=True)``
    applies the winner to the wrapper it ran on.  Running it on a copy that owns a
    fresh state manager returns the result while the live function keeps serving
    exactly what it served before.
    """
    init = getattr(fn, "_initialize_state_and_wrapper", None)
    if not callable(init) or not hasattr(fn, "_csm"):
        raise TuningError(
            E_NO_CANDIDATE_RUN, "the function cannot run an isolated optimization"
        )
    clone = copy.copy(fn)
    clone.__dict__["_exclusive_slot"] = None
    clone._initialize_state_and_wrapper()
    if clone._csm is fn._csm:
        raise TuningError(E_NO_CANDIDATE_RUN, "candidate run is not isolated")
    return clone


# The SDK resolves its artifact roots from the process environment; these are the
# knobs that route the optimization log (best-config artifacts included) and local
# storage.  They are swapped for the duration of one candidate run.
_ARTIFACT_ROOT_ENV = ("TRAIGENT_OPTIMIZATION_LOG_DIR", "TRAIGENT_RESULTS_FOLDER")
_ARTIFACT_ENV_LOCK = threading.Lock()


@contextlib.contextmanager
def _isolated_artifacts(root: Path) -> Iterator[None]:
    """Route every SDK artifact written inside the block under ``root``."""
    root.mkdir(parents=True, exist_ok=True)
    with _ARTIFACT_ENV_LOCK:
        saved = {name: os.environ.get(name) for name in _ARTIFACT_ROOT_ENV}
        for name in _ARTIFACT_ROOT_ENV:
            os.environ[name] = str(root)
        try:
            yield
        finally:
            for name, value in saved.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value


def traigent_optimize_fn(fn: Any, **kwargs: Any) -> OptimizeFn:
    """Adapt a ``@traigent.optimize`` function: run it on the search space.

    The optimization runs on an isolated copy (see :func:`_candidate_copy`), so the
    live function's configuration is never changed, and every artifact it writes
    (including best-config files) lands in ``candidate_dir`` -- inside the tuning
    store -- never where a live wrapper's best-config loaders look.  A deploy only
    follows a recorded human acceptance.
    """

    def run(search_space: dict[str, Any], *, candidate_dir: Path | None = None) -> Any:
        root = (
            Path(candidate_dir)
            if candidate_dir is not None
            else Path(tempfile.mkdtemp(prefix="traigent-candidate-"))
        )
        clone = _candidate_copy(fn)
        with _isolated_artifacts(root):
            return asyncio.run(
                clone.optimize(configuration_space=search_space, **kwargs)
            )

    run.accepts_candidate_dir = True  # type: ignore[attr-defined]
    return run


class HeadLoader:
    """Loads the head configuration from a store at call time (cached by generation)."""

    def __init__(self, store_dir: Path | str) -> None:
        self.dir = Path(store_dir)
        self._stamp: tuple[int, int] | None = None
        self._head: dict[str, Any] | None = None

    def head(self) -> dict[str, Any] | None:
        try:
            st = (self.dir / "head.json").stat()
        except FileNotFoundError:
            return None
        stamp = (st.st_mtime_ns, st.st_size)
        if stamp != self._stamp:
            self._head = read_head(self.dir)
            self._stamp = stamp
        return self._head

    def config(self) -> dict[str, Any] | None:
        head = self.head()
        return dict(head["config"]) if head else None

    def generation(self) -> int | None:
        head = self.head()
        return int(head["generation"]) if head else None


__all__ = [
    "AcceptanceOutcome",
    "Controller",
    "HeadLoader",
    "StepReport",
    "config_fingerprint",
    "default_store_dir",
    "occurrence_ref",
    "parse_occurrence_ref",
    "read_head",
    "traigent_optimize_fn",
]
