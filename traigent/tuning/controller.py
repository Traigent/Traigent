"""A local loop controller: the closed Continuous Tuning loop on the customer side.

TVL defines the semantics (``tvl.runtime_conditions.conditions_trace``); this module
only *executes* them locally.  Everything the relation consumes is recorded as an
append-only fact in a durable store; ``Controller.step`` re-derives the full trace over
the whole history and acts ONLY on derived facts:

* ``attempt_opened``: run the optimization, but only after a recorded approval
  (**Approve run**) by an approver the configured authority allows;
* ``acceptance ... written``: move the head (generation + 1).  The head move is NOT a
  reaction to a replayed verdict: ``submit_acceptance`` commits it under the store
  lock (see "Atomic deploy" below).

Store layout (one directory, default ``.traigent/tuning/<agent>/``)::

    meta.json        agent, environment, origin, contract digest, initial head
    head.json        {fingerprint, config, generation}         (atomic temp+rename)
    heads.jsonl      every head ever written (config lookup, binding of an acceptance)
    history.jsonl    recorded facts (checksummed JSONL): events, windows, approvals, finishes, acceptances
    grants.json      standing grants (closed records, with their consumed uses)
    controller.jsonl controller facts: refused optimizations and refused head writes
    commits.jsonl    the head-move journal: intent, then done or abandoned
    approvals.json   optimization approvals keyed by occurrence ref
    attempts/        one marker per attempt: in-progress, then the persisted result

All instants are whole seconds after ``meta.origin`` (UTC midnight of the store's
creation), exactly the reference relation's clock; nothing is rounded to an hour and
no recorded instant lies in the future.

**Atomic deploy.**  Under the store lock, ``submit_acceptance`` dry-runs the relation;
only when it derives ``written`` does it check, at that instant, expiry
(``decided_at <= t < expires_at``), the approver's authority, the physical head
generation and fingerprint, and the gate's measured incumbent; it then journals an
intent, installs the head, records the acceptance in the history, and marks the
intent done.  A refused write never reaches the history (so the relation and
``head.json`` cannot diverge) and is journalled.  A crash is repaired at the next
start: an installed head gets its acceptance recorded at the original instant; a head
that was never installed is abandoned (an expired acceptance is never installed late).

Nothing here decides a promotion: every candidate is measured here, on the customer's
holdout against the incumbent; supplied verdicts are ignored.  No network is used.
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
import secrets
import tempfile
import threading
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, cast

from traigent.tuning.authority import (
    LOCAL_TRUSTED,
    ApproverAuthority,
    EventVerifier,
)
from traigent.tuning.errors import (
    E_APPROVER_NOT_AUTHORIZED,
    E_NO_APPROVER_AUTHORITY,
    TuningError,
)
from traigent.tuning.state import TuningState
from traigent.tuning.validator import conditions_trace

E_INVALID_RECORD = "E_INVALID_RECORD"
E_ATTEMPT_IN_PROGRESS = "E_ATTEMPT_IN_PROGRESS"
E_NO_OPTIMIZER = "E_NO_OPTIMIZER"
E_NO_DEPLOY_AWAITING = "E_NO_DEPLOY_AWAITING"
E_UNKNOWN_GENERATION = "E_UNKNOWN_GENERATION"
E_STORE_MISMATCH = "E_STORE_MISMATCH"
E_STORE_ENVIRONMENT_MISMATCH = "E_STORE_ENVIRONMENT_MISMATCH"
E_NOT_AN_OCCURRENCE = "E_NOT_AN_OCCURRENCE"
E_STORE_CORRUPT = "E_STORE_CORRUPT"
E_STORE_FORMAT = "E_STORE_FORMAT"
E_ACCEPTANCE_CONFLICT = "E_ACCEPTANCE_CONFLICT"
E_INVALID_GRANT = "E_INVALID_GRANT"
E_NO_CANDIDATE_RUN = "E_NO_CANDIDATE_RUN"
E_HEAD_DIVERGED = "E_HEAD_DIVERGED"
E_LOCAL_AUTHORITY_IN_PRODUCTION = "E_LOCAL_AUTHORITY_IN_PRODUCTION"
E_STANDING_DEPLOY_UNSUPPORTED = "E_STANDING_DEPLOY_UNSUPPORTED"

GATE_FORMAT = "traigent-gate-result/1"
ACCEPTANCE_FORMAT = "tvl-release-acceptance/1"
DEFAULT_STORE_ROOT = Path(".traigent") / "tuning"
_REF = re.compile(r"^[A-Za-z0-9_.:-]{1,128}$")
_LEGACY = {"Promote", "Reject", "NoDecision"}
_COMPLETIONS = {"normal", "error", "cancelled", "cost_limit"}
OptimizeFn = Callable[[dict[str, Any]], Any]
MAX_GRANT_DAYS = 30  # RFC 0003 section 7.3: a standing grant expires within 30 days
logger = logging.getLogger(__name__)


DEFAULT_ENVIRONMENT = "production"


def default_store_dir(agent: str, environment: str | None = None) -> Path:
    """``.traigent/tuning/<environment>/<agent>/`` relative to the working directory.

    The environment is part of the path so a staging store and a production store
    of the same agent are never the same directory.
    """
    if not _REF.match(agent):
        raise TuningError(E_INVALID_RECORD, "agent must be an opaque reference")
    env = environment or DEFAULT_ENVIRONMENT
    if not _REF.match(env):
        raise TuningError(E_INVALID_RECORD, "environment must be an opaque reference")
    return DEFAULT_STORE_ROOT / env / agent


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


def _approve_context(ref: str) -> dict[str, Any]:
    """What the authenticator is told when a run approval is decided or re-checked."""
    return {"decision": "approve_run", "occurrence": ref}


def _grant_context(grant_id: Any, rule_ref: Any) -> dict[str, Any]:
    """What the authenticator is told when a standing grant is issued or used."""
    return {"decision": "standing_grant", "grant_id": grant_id, "rule_ref": rule_ref}


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
def _fsync_dir(directory: Path) -> None:
    """Make a create or rename in ``directory`` durable (its entry, not just the data)."""
    fd = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


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
        _fsync_dir(path.parent)
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
    created = not path.exists()
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(_frame(value) + "\n")
        fh.flush()
        os.fsync(fh.fileno())
    if created:
        _fsync_dir(path.parent)


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
    head_written: int | None = None  # generation completed by crash recovery

    @property
    def changed(self) -> bool:
        return bool(self.started or self.approved or self.finished or self.head_written)


@dataclass(frozen=True)
class AcceptanceOutcome:
    """What one submitted acceptance did: the committed result, never a replay.

    ``written`` means the head file now serves ``generation``.  A write the relation
    allowed but the physical head refused is ``refused`` with the physical reason
    (and nothing was recorded as consumed).
    """

    acceptance_id: str
    result: str  # written | reject_recorded | refused | invalid | consumed
    reason: str | None = None
    generation: int | None = None

    @property
    def ok(self) -> bool:
        return self.result in {"written", "reject_recorded"}


@dataclass(frozen=True)
class EventReceipt:
    """What recording an external event did: where it landed and whether it counted."""

    instant: int
    authenticated: bool


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
        authority: ApproverAuthority | None = None,
        authorized_approvers: set[str] | frozenset[str] | None = None,
        event_verifier: EventVerifier | None = None,
        now: datetime | None = None,
    ) -> None:
        policy_state.require_validated()
        self.state = policy_state
        self.module: dict[str, Any] = policy_state.tvl
        self._refuse_standing_deploy()
        agent = policy_state.agent
        if store_dir is None:
            if agent is None:
                raise TuningError(E_INVALID_RECORD, "store_dir or an agent is required")
            store_dir = default_store_dir(agent, environment)
        self.dir = Path(store_dir)
        self.optimize_fn = optimize_fn
        if authority is None and authorized_approvers is not None:
            authority = ApproverAuthority(frozenset(authorized_approvers))
        self.authority = authority
        self.event_verifier = event_verifier
        self._init_store(agent or self.dir.name, environment, initial_config, now)
        with self._locked():  # finish or abandon a head move a crash interrupted
            self._recover_commits()

    def _refuse_standing_deploy(self) -> None:
        """``approval: standing`` on a rule whose deploy is ``if_promoted`` is refused.

        A standing grant for a DEPLOY would need the acceptance-equivalent binding
        fields (RFC 0012 section 2.3, last sentence); neither the reference relation
        nor this controller implements them, and the deploy always waits for Accept
        release.  Refusing is the honest answer: no fake support.
        """
        for response in self.module.get("responses", []):
            for rule in response.get("rules", []):
                if rule.get("approval") == "standing" and (
                    rule.get("deploy", "never") == "if_promoted"
                ):
                    raise TuningError(
                        E_STANDING_DEPLOY_UNSUPPORTED,
                        "approval: standing is not supported on a rule whose deploy "
                        "is if_promoted; use human approval, or deploy: never",
                    )

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
                origin = _utc(now).replace(hour=0, minute=0, second=0)
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
            elif meta.get("environment") != environment:
                raise TuningError(
                    E_STORE_ENVIRONMENT_MISMATCH,
                    "store was created for another environment",
                )
            else:
                for name in (
                    "history.jsonl",
                    "heads.jsonl",
                    "controller.jsonl",
                    "commits.jsonl",
                ):
                    _repair_tail(self._p(name))
            self.meta: dict[str, Any] = meta
            self.origin = _parse_instant(meta["origin"])

    # -- history --------------------------------------------------------------------
    def _lines(self) -> list[dict[str, Any]]:
        return _read_lines(self._p("history.jsonl"))

    def _history(
        self,
        extra: list[dict[str, Any]] | None = None,
        lines: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        steps: list[dict[str, Any]] = []
        base = self._lines() if lines is None else lines
        for line in [*base, *(extra or [])]:
            # The external events of one whole-second instant are delivered as ONE
            # step: the relation orders a complete instant's events itself
            # (RFC 0011), so they are never split across steps.  Evaluation
            # categories (windows, optimizations, acceptances) keep one step per
            # line: each is a later observation of the instant.
            prev = steps[-1] if steps else None
            if (
                line["category"] == "events"
                and prev is not None
                and prev["instant"] == line["instant"]
                and set(prev) == {"instant", "events"}
            ):
                prev["events"].append(line["record"])
                continue
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

    def _instant(self, moment: datetime) -> int:
        """Whole seconds after the origin: the relation's clock, never rounded.

        Monotonic only because real time is: an instant earlier than the last
        recorded one (a clock set back) is held at the last, never moved forward.
        """
        seconds = int((_utc(moment) - self.origin).total_seconds())
        lines = self._lines()
        last = lines[-1]["instant"] if lines else 0
        return max(seconds, last, 0)

    def _append(
        self,
        category: str,
        record: dict[str, Any] | None,
        instant: int,
        meta: Mapping[str, Any] | None = None,
    ) -> None:
        """Append one recorded fact; a fact the relation rejects never lands.

        ``meta`` rides on the line (audit only; the relation never sees it).
        """
        line: dict[str, Any] = {"instant": instant, "category": category}
        if record is not None:
            line["record"] = record
        if meta:
            line["meta"] = dict(meta)
        try:
            self._derive([line])
        except TuningError:
            raise
        except Exception as exc:  # noqa: BLE001 - TVL closed errors; never echo content
            raise TuningError(
                E_INVALID_RECORD, f"{category} rejected by the TVL relation"
            ) from exc
        _append_line(self._p("history.jsonl"), line)

    # -- authority ------------------------------------------------------------------
    def require_authority(self) -> ApproverAuthority:
        """The configured approver authority, or a closed refusal.

        There is no default: without an explicit authority nobody can approve or
        accept.  A ``local-trusted`` authority is refused when the environment is
        production (production needs the platform to authenticate the person).
        """
        if self.authority is None:
            raise TuningError(
                E_NO_APPROVER_AUTHORITY,
                "no approver authority is configured; approving a run or accepting "
                "a release needs one",
            )
        if self.authority.mode == LOCAL_TRUSTED:
            from traigent.utils.env_config import is_production_strict_env

            if is_production_strict_env():
                raise TuningError(
                    E_LOCAL_AUTHORITY_IN_PRODUCTION,
                    "a local-trusted authority does not authenticate anyone; "
                    "production needs the platform's authenticator",
                )
        return self.authority

    def _on_list_now(
        self, approver: object, context: Mapping[str, Any] | None = None
    ) -> bool:
        """Is ``approver`` listed AND authenticated at this instant?

        The same authenticator that decides a release acceptance decides every
        other authority decision (a run approval, a standing grant): a platform
        authenticator that answers False refuses.  No authority means no.
        """
        authority = self.authority
        if authority is None or not authority.authorized(approver):
            return False
        return authority.authenticated(approver, context or {"decision": "authority"})

    # -- recording ------------------------------------------------------------------
    def record_event(
        self, envelope: Mapping[str, Any], now: datetime | None = None
    ) -> EventReceipt:
        """Append an external event envelope; authentication comes from the verifier.

        The envelope's own ``authenticated`` field is never read: it is replaced by
        the result of the configured :class:`EventVerifier` (an HMAC ``signature``
        from a trusted issuer).  An unverifiable event is still recorded, as the
        relation's ``rejected(unauthenticated)``, and fires nothing.
        """
        self.state.require_validated()
        record = copy.deepcopy(dict(envelope))
        authenticated = bool(
            self.event_verifier is not None and self.event_verifier.verify(record)
        )
        signature = record.pop("signature", None)
        record["authenticated"] = authenticated
        issued = record.get("issued_at")
        if isinstance(issued, str):  # convenience: an instant becomes whole seconds
            try:
                delta = _parse_instant(issued) - self.origin
            except ValueError:
                raise TuningError(
                    E_INVALID_RECORD, "issued_at is not an instant"
                ) from None
            record["issued_at"] = int(delta.total_seconds())
        meta: dict[str, Any] = {"verified_by": "hmac-sha256" if authenticated else None}
        if authenticated and isinstance(signature, str):
            meta["signature"] = signature
        with self._locked():
            self._recover_commits()  # history must hold a crashed commit first
            instant = self._instant(_utc(now))
            self._append("events", record, instant, meta)
        return EventReceipt(instant, authenticated)

    def record_window(
        self, report: Mapping[str, Any], now: datetime | None = None
    ) -> int:
        """Append a monitor window report (a verdict computed by the monitor).

        A window is recorded at the instant it is reported; a report for a window
        that has not ended yet is refused by the relation, never moved forward.
        """
        self.state.require_validated()
        record = copy.deepcopy(dict(report))
        with self._locked():
            self._recover_commits()  # history must hold a crashed commit first
            instant = self._instant(_utc(now))
            self._append("windows", record, instant)
        return instant

    def approve_optimization(
        self,
        occurrence_ref_: str | Mapping[str, Any],
        approver: str,
        now: datetime | None = None,
    ) -> None:
        """**Approve run**: record a human approval for the attempt an occurrence opens.

        The approver must be on the configured authority's list; it is checked again
        immediately before the optimizer runs.
        """
        self.state.require_validated()
        ref = (
            occurrence_ref_
            if isinstance(occurrence_ref_, str)
            else occurrence_ref(occurrence_ref_)
        )
        parse_occurrence_ref(ref)
        if not _REF.match(approver):
            raise TuningError(E_INVALID_RECORD, "approver must be an opaque reference")
        authority = self.require_authority()
        if not authority.authorized(approver):
            raise TuningError(
                E_APPROVER_NOT_AUTHORIZED, "that approver is not on the authority list"
            )
        if not authority.authenticated(approver, _approve_context(ref)):
            raise TuningError(
                E_APPROVER_NOT_AUTHORIZED, "that approver was not authenticated"
            )
        with self._locked():
            approvals = _read_json(self._p("approvals.json"), {})
            approvals.setdefault(
                ref,
                {
                    "approver": approver,
                    "at": _instant_text(_utc(now)),
                    "authentication": authority.mode,
                },
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

        RFC 0003 section 7.3: a closed record bound to this agent and contract,
        naming the rule (``response/rule``), a maximum occurrence count and an expiry
        (exclusive) at most 30 days after issue.  Each occurrence consumes one use; a
        retry of the same occurrence does not.  The grant covers the optimization
        ONLY (``covers: "optimize"``): its deploy still needs **Accept release** by a
        human (section 7.6 item 2).  The issuer must be on the authority list, now and
        at every use.
        """
        self.state.require_validated()
        self._refuse_standing_deploy()
        authority = self.require_authority()
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
        if not authority.authorized(issuer):
            raise TuningError(
                E_APPROVER_NOT_AUTHORIZED,
                "the grant issuer is not on the authority list",
            )
        if not authority.authenticated(issuer, _grant_context(grant_id, rule_ref)):
            raise TuningError(
                E_APPROVER_NOT_AUTHORIZED, "the grant issuer was not authenticated"
            )
        grant = {
            "grant_id": grant_id,
            "rule_ref": rule_ref,
            "agent_ref": self.meta["agent"],
            "contract_digest": self.meta["contract_digest"],
            "covers": "optimize",
            "issuer": issuer,
            "authentication": authority.mode,
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

    def _grant_usable(
        self, grant: Mapping[str, Any], rule_ref: str, moment: datetime
    ) -> bool:
        """A grant for this rule, agent and contract, optimize-only, unexpired, issued
        by someone still on the authority list."""
        return (
            grant["rule_ref"] == rule_ref
            and grant["agent_ref"] == self.meta["agent"]
            and grant["contract_digest"] == self.meta["contract_digest"]
            and grant.get("covers") == "optimize"
            and moment < _parse_instant(grant["expires_at"])
            and self._on_list_now(
                grant.get("issuer"),
                _grant_context(grant.get("grant_id"), grant.get("rule_ref")),
            )
        )

    def _consume_grant(
        self, occ: Mapping[str, Any], ref: str, moment: datetime
    ) -> bool:
        """Consume one use of a matching standing grant (idempotent per occurrence)."""
        rule_ref = f"{occ['response']}/{occ['rule']}"
        grants = _read_json(self._p("grants.json"), {})
        for grant_id in sorted(grants):
            grant = grants[grant_id]
            if not self._grant_usable(grant, rule_ref, moment):
                continue  # an expired or unauthorized grant authorizes nothing
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
        report = StepReport()
        with self._locked():
            moment = _utc(now)  # read after the lock: the time of the effect
            # an explicit ``now`` is a fixed clock (tests, replays); otherwise the
            # clock is re-read before every paid run and at its completion
            clock: Callable[[], datetime] = (
                (lambda: moment) if now is not None else (lambda: _utc(None))
            )
            report.head_written = self._recover_commits()
            if self._has_schedule():
                self._tick(self._instant(moment))
            for _ in range(64):
                trace = self._derive()
                if not self._advance(trace, clock, report, abandon_interrupted):
                    break
            trace = self._derive()
            report.facts = trace["facts"]
            self._check_head_consistent(trace)
        return report

    def _tick(self, instant: int) -> None:
        """Advance the relation's clock, only when time passing derives something.

        A tick line is appended when evaluating at ``instant`` yields a fact the
        history does not have yet (a schedule tick fired, or coalesced); a step that
        changes nothing leaves the history untouched.
        """
        before = len(self._derive()["facts"])
        tick = {"instant": instant, "category": "tick"}
        if len(self._derive([tick])["facts"]) > before:
            self._append("tick", None, instant)

    def _tvl_generation(self, trace: Mapping[str, Any]) -> int:
        generations = trace.get("head_generations") or []
        if generations:
            return int(generations[-1]["generation"])
        return int(self.meta["initial_head"]["generation"])

    def _check_head_consistent(self, trace: Mapping[str, Any]) -> None:
        """The relation's head and ``head.json`` must name the same generation."""
        head = read_head(self.dir) or {"generation": -1}
        derived = self._tvl_generation(trace)
        if int(head["generation"]) != derived:
            raise TuningError(
                E_HEAD_DIVERGED,
                f"the relation is at generation {derived} but head.json serves "
                f"{head['generation']}; the store is not usable until reconciled",
            )

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
        and any mode the controller does not know) needs ``approve_optimization``,
        by an approver the authority allows now.
        """
        if self._rule(occ).get("approval", "human") == "standing":
            return self._consume_grant(occ, ref, moment)
        approval = _read_json(self._p("approvals.json"), {}).get(ref)
        return approval is not None and self._on_list_now(
            approval.get("approver"), _approve_context(ref)
        )

    def _authorization_lapse(
        self, occ: Mapping[str, Any], ref: str, moment: datetime
    ) -> str | None:
        """Why the occurrence is no longer authorized at ``moment``; None when it is.

        Checked immediately before the optimizer runs, on every path (recovery
        included): a recorded approval never outlives its grant or its approver's
        authority.
        """
        if self._rule(occ).get("approval", "human") == "standing":
            rule_ref = f"{occ['response']}/{occ['rule']}"
            grants = _read_json(self._p("grants.json"), {})
            for grant in grants.values():
                if (
                    self._grant_usable(grant, rule_ref, moment)
                    and ref in grant["used"]
                    and len(grant["used"]) <= int(grant["max_occurrences"])
                ):
                    return None
            return "grant_expired"
        approval = _read_json(self._p("approvals.json"), {}).get(ref)
        if approval is None:
            return "approval_missing"
        if not self._on_list_now(approval.get("approver"), _approve_context(ref)):
            return "approver_not_authorized"
        return None

    def _scope_requires_intervention(
        self, occ: Mapping[str, Any], instant: int
    ) -> bool:
        """Ask the relation, run to ``instant`` (seconds), whether the occurrence's scope still counts.

        An attempt the relation has closed (any reason, ``scope_expired`` included)
        is final: it never counts as requiring intervention.
        """
        from tvl.runtime_conditions import occurrence_status

        history = self._history([{"instant": instant, "category": "tick"}])
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
        clock: Callable[[], datetime],
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
            moment = clock()  # fresh before each decision that can spend
            instant = self._instant(moment)  # whole seconds
            marker_path = self._marker_path(attempt, ref)
            marker = _read_json(marker_path)
            # Immediately before any authorization is used or the optimizer is
            # invoked: the relation, over the history up to now, must still show
            # this occurrence's scope as requiring intervention.
            if marker is None and not self._scope_requires_intervention(occ, instant):
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
                    instant,
                )
                report.approved.append(attempt)
                return True
            if marker is not None and "finished_record" in marker:
                self._append("optimizations", marker["finished_record"], instant)
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
            admitted = self._admitted_values(domains.get(attempt))
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
                        admitted_models=admitted,
                    )
                else:
                    result = self.optimize_fn(copy.deepcopy(space))
                completion, candidates, configs, gates = self._map_result(
                    attempt, result
                )
            except Exception:  # noqa: BLE001 - a failed run is recorded, content never echoed
                completion, candidates, configs, gates = "error", [], {}, {}
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
                    "gates": gates,
                },
            )
            self._append("optimizations", record, self._instant(clock()))
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
            t["name"]: _normalized_domain(t["domain"])
            for t in self.module.get("tvars", [])
        }
        if domain_fact is not None:
            space[domain_fact["tvar"]] = list(domain_fact["domain"])
        return space

    def _admitted_values(self, domain_fact: Mapping[str, Any] | None) -> list[str]:
        """Values the event admitted into the search domain (not in the declared one)."""
        if domain_fact is None:
            return []
        declared = [
            t["domain"]
            for t in self.module.get("tvars", [])
            if t["name"] == domain_fact["tvar"]
        ]
        return [
            v
            for v in domain_fact["domain"]
            if not any(_declared_contains(spec, v) for spec in declared)
        ]

    # -- mapping Traigent results to the relation's candidate shape ------------------
    def _promotion_monitors(self) -> list[str]:
        return sorted(
            m["name"]
            for m in self.module.get("monitors", [])
            if m.get("phase") == "promotion"
        )

    def _measure_candidate(
        self, attempt: str, config: Mapping[str, Any], incumbent: Mapping[str, Any]
    ) -> Any:
        """Promotion evidence for the frozen candidate, or None (-> ``NoDecision``).

        The agent function runs with the candidate config and the incumbent
        (``incumbent``: the head measured, paired, row by row) on the customer's
        holdout; the customer's checkers label the outputs.  A missing holdout or
        checker, or anything not computable, records an ``E_NO_PROMOTION_EVIDENCE``
        controller fact and the candidate stays undecided.
        """
        from traigent.tuning.evidence import (
            E_NO_PROMOTION_EVIDENCE,
            EvidenceUnavailable,
            build_promotion_evidence,
            eval_set_identity,
        )

        policy = self.state.policy
        optimized = getattr(self.state, "optimized", None)
        unpriced = _unpriced_models_of(config)
        try:
            source = getattr(optimized, "_dataset_from_source", None)
            eval_digests, _rows = eval_set_identity(
                getattr(optimized, "eval_dataset", None),
                source if callable(source) else None,
            )
            return build_promotion_evidence(
                self.module,
                fn=optimized,
                checkers=getattr(policy, "checkers", None),
                holdout=getattr(policy, "holdout", None),
                candidate_config=config,
                incumbent_config=incumbent.get("config") or {},
                checker_labels=getattr(policy, "checker_labels", None),
                eval_set_digests=eval_digests,
                unpriced_models=unpriced,
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

    def code_version(self) -> str:
        """The agent code version in effect now (``TuningError`` if undeterminable)."""
        from traigent.tuning.code_version import resolve_code_version

        return resolve_code_version(
            getattr(self.state.policy, "code_version", None),
            getattr(self.state, "source", None),
        )

    def _holdout_prior_uses(self, digest: str) -> int:
        """How many earlier gate results were measured on this very holdout."""
        return sum(
            1
            for marker in self._markers()
            for gate in (marker.get("gates") or {}).values()
            if gate.get("holdout_digest") == digest
        )

    def _map_result(
        self, attempt: str, result: Any
    ) -> tuple[str, list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
        """``(completion, candidates, configs, gates)`` for the ``finished`` record.

        A result carries ``candidates`` (dicts with a ``config``) or just a
        ``best_config`` (a Traigent ``OptimizationResult``).  EVERY candidate on EVERY
        path is measured here, on the customer's holdout, against the incumbent head;
        a ``legacy_decision`` or ``monitor_verdicts`` the optimizer supplies is
        ignored (recorded as a controller fact), because a verdict that was not
        measured against this holdout and this incumbent is not evidence.  Without
        measurable evidence the candidate is ``NoDecision`` with every promotion
        monitor ``UNKNOWN(insufficient)``: absence of evidence never promotes.
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
        if raw is None:
            best = pick("best_config")
            raw = [{"config": best}] if best else []
        if any(
            isinstance(i, Mapping)
            and ("legacy_decision" in i or "monitor_verdicts" in i)
            for i in raw
        ):
            self._record_controller_fact(
                {"fact": "external_verdicts_ignored", "attempt": attempt}
            )
        monitors = self._promotion_monitors()
        incumbent = read_head(self.dir) or {}
        try:
            code_version: str | None = self.code_version()
        except TuningError as exc:
            # No default: without a code version no candidate has promotion evidence.
            code_version = None
            self._record_controller_fact(
                {"fact": "no_code_version", "code": exc.code, "attempt": attempt}
            )
        candidates: list[dict[str, Any]] = []
        configs: dict[str, Any] = {}
        gates: dict[str, Any] = {}
        for item in raw:
            item = item if isinstance(item, Mapping) else {"config": item}
            config = item.get("config")
            try:
                if code_version is None:
                    raise ValueError("no code version")
                if not isinstance(config, Mapping) or not config:
                    raise ValueError("no config")
                config = copy.deepcopy(dict(config))  # frozen snapshot at selection
                fingerprint = config_fingerprint(config)
                legacy = "NoDecision"
                given: Mapping[str, str] = {}
                holdout_digest: str | None = None
                evidence = self._measure_candidate(attempt, config, incumbent)
                if evidence is not None:
                    legacy = evidence.legacy_decision
                    given = evidence.monitor_verdicts
                    holdout_digest = evidence.holdout_digest
                if config_fingerprint(config) != fingerprint:
                    self._record_controller_fact(
                        {
                            "fact": "candidate_config_changed",
                            "attempt": attempt,
                            "fingerprint": fingerprint,
                        }
                    )
                    raise ValueError("config changed during evaluation")
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
                "code_version": code_version,
                "fingerprint": fingerprint,
                "incumbent_fingerprint": incumbent.get("fingerprint"),
                "incumbent_generation": incumbent.get("generation"),
                "legacy_decision": legacy,
                "monitors": verdicts,
                "decision": str(joined),
            }
            if holdout_digest is not None:
                gate_input["holdout_digest"] = holdout_digest
            gate_digest = "sha256:" + _digest(gate_input)
            configs[fingerprint] = copy.deepcopy(config)
            gates[fingerprint] = {
                **gate_input,
                "gate_digest": gate_digest,
                **(
                    {
                        "holdout_rows": evidence.n,
                        "holdout_unresolved_rows": evidence.unresolved_rows,
                        "holdout_counts": evidence.counts,
                        "holdout_prior_uses": self._holdout_prior_uses(holdout_digest),
                        "errors": list(evidence.errors),
                    }
                    if evidence is not None and holdout_digest is not None
                    else {}
                ),
            }
            record: dict[str, Any] = {
                "evidence_valid": True,
                "fingerprint": fingerprint,
                "gate_digest": gate_digest,
                "code_version": code_version,
                "legacy_decision": legacy,
                "monitor_verdicts": verdicts,
            }
            if holdout_digest is not None:
                record["holdout_digest"] = holdout_digest
            candidates.append(record)
        return str(completion), candidates, configs, gates

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
        *,
        nonce: str | None = None,
    ) -> dict[str, Any]:
        """A validated ``tvl-release-acceptance/1`` record for a deploy awaiting.

        The ``acceptance_id`` covers the approver, the decision instant and a nonce,
        so a retry after an expiry or an unauthorized attempt is a new record, never
        a conflict with the earlier one.
        """
        from tvl.release_acceptance import validate_acceptance

        self.state.require_validated()
        if decision not in ("accept", "reject"):
            raise TuningError(E_INVALID_RECORD, "decision must be accept or reject")
        occ = parse_occurrence_ref(deploy_ref)
        deploy = self._deploy_fact(occ)
        fingerprint = str(deploy["candidate"])
        candidate = self._candidate_record(deploy)
        moment = _utc(now)
        decided_at = _instant_text(moment)
        nonce = nonce or secrets.token_hex(8)
        record: dict[str, Any] = {
            "format": ACCEPTANCE_FORMAT,
            "acceptance_id": "R-"
            + hashlib.sha256(
                "|".join(
                    (
                        deploy_ref,
                        fingerprint,
                        str(int(expected_head_generation)),
                        decision,
                        str(approver),
                        decided_at,
                        nonce,
                    )
                ).encode()
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
            "gate_result_digest": str(candidate["gate_digest"]),
            "contract_digest": self.meta["contract_digest"],
            "agent_code_version": str(candidate["code_version"]),
            "expected_head_generation": int(expected_head_generation),
            "rollback_target": self._generation_fingerprint(expected_head_generation),
            "approver_ref": approver,
            "decision": decision,
            "decided_at": decided_at,
            "expires_at": _instant_text(moment + timedelta(days=7)),
        }
        holdout = candidate.get("holdout_digest")
        if holdout:  # RFC 0012: bind the holdout the evidence used
            record["holdout_digest"] = str(holdout)
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

    def _attempt_marker(self, deploy: Mapping[str, Any]) -> dict[str, Any]:
        """The marker of THE deploy's own attempt (by attempt id, never a scan)."""
        ref = occurrence_ref(deploy["occurrence"])
        marker = _read_json(self._marker_path(str(deploy["attempt"]), ref))
        if not isinstance(marker, dict):
            raise TuningError(E_NO_DEPLOY_AWAITING, "candidate evidence not found")
        return marker

    def _candidate_record(self, deploy: Mapping[str, Any]) -> dict[str, Any]:
        marker = self._attempt_marker(deploy)
        for cand in marker.get("finished_record", {}).get("candidates", []):
            if cand.get("fingerprint") == deploy["candidate"]:
                return dict(cand)
        raise TuningError(E_NO_DEPLOY_AWAITING, "candidate evidence not found")

    def _gate_of(self, deploy: Mapping[str, Any]) -> dict[str, Any]:
        gate = (self._attempt_marker(deploy).get("gates") or {}).get(
            str(deploy["candidate"])
        )
        return dict(gate) if isinstance(gate, dict) else {}

    def _markers(self) -> list[dict[str, Any]]:
        folder = self._p("attempts")
        if not folder.is_dir():
            return []
        return [
            m
            for m in (_read_json(p) for p in sorted(folder.glob("*.json")))
            if isinstance(m, dict)
        ]

    def _acceptance_lines(self) -> list[dict[str, Any]]:
        return [ln for ln in self._lines() if ln["category"] == "acceptances"]

    def submit_acceptance(
        self, record: Mapping[str, Any], now: datetime | None = None
    ) -> AcceptanceOutcome:
        """**Accept release**: decide an acceptance and, if it deploys, commit it.

        Authentication and authority come from the configured authority, never from
        the caller.  Under the store lock the relation is dry-run first; only a
        derived ``written`` proceeds to the physical commit (see the module
        docstring), and the returned outcome is what was committed.  Only an exact
        redelivery of a recorded acceptance is deduplicated; the same id with other
        content is ``E_ACCEPTANCE_CONFLICT``.
        """
        self.state.require_validated()
        authority = self.require_authority()
        doc = copy.deepcopy(dict(record))
        approver = doc.get("approver_ref")
        ident = doc.get("acceptance_id")
        with self._locked():
            # The effect time is read after the store lock is held (and again at the
            # commit): an acceptance that expired while this call waited behind
            # another writer is judged at its real effect time.
            clock: Callable[[], datetime] = (
                (lambda: _utc(now)) if now is not None else (lambda: _utc(None))
            )
            self._recover_commits()
            prior = self._acceptance_lines()
            recorded = [ln["record"]["record"] for ln in prior]
            if isinstance(ident, str):
                for position, rec in enumerate(recorded):
                    if isinstance(rec, dict) and rec.get("acceptance_id") == ident:
                        if rec != doc:
                            # One acceptance id, one content.
                            raise TuningError(
                                E_ACCEPTANCE_CONFLICT,
                                "that acceptance id was already submitted with "
                                "other content",
                            )
                        return self._recorded_outcome(position, ident)  # redelivery
            if isinstance(ident, str):  # a journalled intent is a submission too
                for intent in _read_lines(self._p("commits.jsonl")):
                    if (
                        intent.get("fact") == "commit_intent"
                        and intent["acceptance"] == ident
                        and intent["line_record"]["record"] != doc
                    ):
                        raise TuningError(
                            E_ACCEPTANCE_CONFLICT,
                            "that acceptance id was already submitted with "
                            "other content",
                        )
            position = len(recorded)
            # Authentication can be slow (a platform call): do it BEFORE the time
            # and the serving code version are sampled, so neither is stale.
            authenticated = bool(authority.authenticated(approver, doc))
            authorized = bool(authority.authorized(approver))
            meta = {"authentication": authority.mode}

            def decide() -> tuple[dict[str, Any], int, dict[str, Any]]:
                """The relation's verdict at a freshly read time and code version."""
                # Undeterminable code version -> refuse, no default.
                serving = self.code_version()
                instant = self._instant(clock())
                line_record = {
                    "record": doc,
                    "authenticated": authenticated,
                    "approver_authorized": authorized,
                    "serving_code_version": serving,
                }
                line = {
                    "instant": instant,
                    "category": "acceptances",
                    "record": line_record,
                }
                try:
                    trace = self._derive([line])
                except TuningError:
                    raise
                except Exception as exc:  # noqa: BLE001 - TVL closed errors; never echo content
                    raise TuningError(
                        E_INVALID_RECORD, "acceptances rejected by the TVL relation"
                    ) from exc
                verdicts = [f for f in trace["facts"] if f["fact"] == "acceptance"]
                return line_record, instant, verdicts[position]

            line_record, instant, verdict = decide()
            if verdict["result"] != "written":
                self._append("acceptances", line_record, instant, meta)
                return AcceptanceOutcome(
                    acceptance_id=str(ident),
                    result=str(verdict["result"]),
                    reason=verdict.get("reason"),
                    generation=verdict.get("generation"),
                )
            return self._commit_acceptance(
                doc, line_record, meta, instant, str(ident), decide
            )

    def _recorded_outcome(self, position: int, ident: str) -> AcceptanceOutcome:
        verdicts = [f for f in self._derive()["facts"] if f["fact"] == "acceptance"]
        last = verdicts[position]
        return AcceptanceOutcome(
            acceptance_id=ident,
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
    def _physical_refusal(
        self,
        doc: Mapping[str, Any],
        instant: int,
        current: Mapping[str, Any],
        serving: str | None = None,
    ) -> str | None:
        """Why the physical head must not take this write at ``instant``; None if it may.

        Checked at the instant of the write (RFC 0012 section 2.3): the record is
        usable (``decided_at <= t < expires_at``), the approver is authorized now, the
        head is still at the expected generation and fingerprint, the gate measured
        THIS incumbent, and the candidate's configuration is the one fingerprinted.
        """
        at = self.origin + timedelta(seconds=instant)
        try:
            decided = _parse_instant(str(doc["decided_at"]))
            expires = _parse_instant(str(doc["expires_at"]))
        except (KeyError, ValueError):
            return "acceptance_not_usable"
        if not (decided <= at < expires):
            return "acceptance_not_usable"
        if not self._on_list_now(doc.get("approver_ref"), doc):
            return "approver_not_authorized"
        if int(current["generation"]) != doc["expected_head_generation"]:
            return "stale_generation"
        if current["fingerprint"] != doc["rollback_target"]:
            return "rollback_target_mismatch"
        if serving is None:
            try:
                serving = self.code_version()
            except TuningError:
                return "code_version_mismatch"
        if doc.get("agent_code_version") != serving:
            return "code_version_mismatch"
        try:
            deploy = self._deploy_fact(
                {
                    "response": doc["occurrence"]["response_ref"],
                    "rule": doc["occurrence"]["rule_ref"],
                    "scope_kind": doc["occurrence"]["scope_kind"],
                    "scope": doc["occurrence"]["scope_ref"],
                    "j": doc["occurrence"]["occurrence"],
                }
            )
            gate = self._gate_of(deploy)
        except TuningError:
            return "gate_evidence_missing"
        if (
            gate.get("incumbent_generation") != current["generation"]
            or gate.get("incumbent_fingerprint") != current["fingerprint"]
        ):
            return "gate_incumbent_stale"
        try:
            config = self._config_for(str(doc["candidate_fingerprint"]))
        except TuningError:
            return "candidate_config_missing"
        if config_fingerprint(config) != doc["candidate_fingerprint"]:
            return "config_fingerprint_mismatch"
        return None

    def _commit_acceptance(
        self,
        doc: Mapping[str, Any],
        line_record: Mapping[str, Any],
        meta: Mapping[str, Any],
        instant: int,
        ident: str,
        decide: Callable[[], tuple[dict[str, Any], int, dict[str, Any]]] | None = None,
    ) -> AcceptanceOutcome:
        """Move the head and consume the acceptance as ONE commit (store lock held).

        Order: refusal checks, journal intent, install the head, record the
        acceptance, journal done.  A refusal records nothing in the history (the
        relation stays at the serving generation) and is journalled; a crash is
        repaired by :meth:`_recover_commits`.
        """
        current = read_head(self.dir)
        if current is None:
            raise TuningError(E_STORE_CORRUPT, "head.json is missing")
        if decide is not None:
            # At the commit, under the lock: expiry, scope (the relation's verdict)
            # and the serving code version are re-decided at a fresh time.
            fresh_record, fresh_instant, fresh_verdict = decide()
            if fresh_verdict["result"] != "written":
                self._append("acceptances", fresh_record, fresh_instant, meta)
                return AcceptanceOutcome(
                    acceptance_id=ident,
                    result=str(fresh_verdict["result"]),
                    reason=fresh_verdict.get("reason"),
                    generation=fresh_verdict.get("generation"),
                )
            line_record, instant = fresh_record, fresh_instant
        generation = int(current["generation"]) + 1
        refusal = self._physical_refusal(
            doc, instant, current, str(line_record["serving_code_version"])
        )
        if refusal is not None:
            self._record_controller_fact(
                {
                    "fact": "head_write_refused",
                    "acceptance": ident,
                    "generation": generation,
                    "reason": refusal,
                }
            )
            return AcceptanceOutcome(ident, "refused", refusal)
        fingerprint = str(doc["candidate_fingerprint"])
        new_head = {
            "fingerprint": fingerprint,
            "config": self._config_for(fingerprint),
            "generation": generation,
            # the agent code this head was accepted for (RFC 0012 section 2.2)
            "code_version": str(doc["agent_code_version"]),
        }
        commit = secrets.token_hex(8)  # one id per attempt, never the acceptance id
        _append_line(
            self._p("commits.jsonl"),
            {
                "fact": "commit_intent",
                "commit": commit,
                "acceptance": ident,
                "instant": instant,
                "from_generation": int(current["generation"]),
                "to_generation": generation,
                "fingerprint": fingerprint,
                "line_record": dict(line_record),
                "meta": dict(meta),
            },
        )
        self._install_head(new_head)
        self._append("acceptances", dict(line_record), instant, meta)
        _append_line(
            self._p("commits.jsonl"),
            {
                "fact": "commit_done",
                "commit": commit,
                "acceptance": ident,
                "generation": generation,
            },
        )
        return AcceptanceOutcome(ident, "written", None, generation)

    def _install_head(self, head: Mapping[str, Any]) -> None:
        """Write ``head.json`` (atomic), then make sure ``heads.jsonl`` names it."""
        _write_json(self._p("head.json"), dict(head))
        known = {h["generation"] for h in _read_lines(self._p("heads.jsonl"))}
        if head["generation"] not in known:
            _append_line(self._p("heads.jsonl"), dict(head))

    def _record_recovered_acceptance(self, intent: Mapping[str, Any]) -> None:
        """Put a crashed commit's acceptance into history at its ORIGINAL instant.

        The head moved at that instant, so history records the consumption there
        (never at ``max(original, last)``: a later instant could be past the
        record's expiry and the relation would then reject what production already
        did).  Every history-appending entry point recovers first, so normally the
        line simply lands at the tail.  If history has meanwhile grown past that
        instant (a writer that did not recover), the line is spliced in at its
        instant: lines after it stay ordered after it, and the file is rewritten
        atomically under the store lock.  If the relation still rejects the
        result, nothing is written, the journal is NOT marked done, and the
        store is refused with ``E_HEAD_DIVERGED`` (loud, never silently split).
        """
        line: dict[str, Any] = {
            "instant": int(intent["instant"]),
            "category": "acceptances",
            "record": intent["line_record"],
        }
        if intent.get("meta"):
            line["meta"] = dict(intent["meta"])
        lines = self._lines()
        index = len(lines)
        while index > 0 and int(lines[index - 1]["instant"]) > line["instant"]:
            index -= 1
        merged = [*lines[:index], line, *lines[index:]]
        try:
            trace = conditions_trace(self.module, self._history(lines=merged))
            derived = self._tvl_generation(trace)
        except Exception:  # noqa: BLE001 - TVL closed errors; never echo content
            derived = -1
        if derived != int(intent["to_generation"]):
            raise TuningError(
                E_HEAD_DIVERGED,
                "head.json moved but the relation does not accept the journalled "
                "acceptance at its original instant; the commit stays open and the "
                "store is not usable until reconciled",
            )
        if index == len(lines):
            _append_line(self._p("history.jsonl"), line)
            return
        text = "".join(_frame(entry) + "\n" for entry in merged)
        fd, tmp = tempfile.mkstemp(dir=self.dir, prefix=".history.", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(text)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, self._p("history.jsonl"))
            _fsync_dir(self.dir)
        except BaseException:
            with contextlib.suppress(OSError):
                os.unlink(tmp)
            raise

    def _recover_commits(self) -> int | None:
        """Finish or abandon a head move a crash interrupted (store lock held).

        * the head was installed: record the acceptance at its ORIGINAL instant (the
          effect happened then), repair ``heads.jsonl``, mark done;
        * the head was not installed: abandon, never install late (the acceptance
          may have expired meanwhile); the deploy stays awaiting.

        Journal entries are keyed by the per-attempt ``commit`` id, never by the
        acceptance id alone: an identical resubmission after an abandoned attempt
        is a new attempt and is recovered on its own.  Returns the generation
        completed, if any.
        """

        def key(entry: Mapping[str, Any]) -> str:
            return str(entry.get("commit") or entry["acceptance"])  # legacy: id

        entries = _read_lines(self._p("commits.jsonl"))
        finished = {
            key(e)
            for e in entries
            if e.get("fact") in ("commit_done", "commit_abandoned")
        }
        completed: int | None = None
        for intent in entries:
            if intent.get("fact") != "commit_intent" or key(intent) in finished:
                continue
            ident = str(intent["acceptance"])
            commit = intent.get("commit")
            head = read_head(self.dir)
            if head is None:
                raise TuningError(E_STORE_CORRUPT, "head.json is missing")
            if (
                int(head["generation"]) == intent["to_generation"]
                and head["fingerprint"] == intent["fingerprint"]
            ):
                self._install_head(head)  # heads.jsonl may be missing its line
                recorded = any(
                    ln["record"]["record"].get("acceptance_id") == ident
                    for ln in self._acceptance_lines()
                )
                if not recorded:
                    self._record_recovered_acceptance(intent)
                _append_line(
                    self._p("commits.jsonl"),
                    {
                        "fact": "commit_done",
                        "commit": commit,
                        "acceptance": ident,
                        "generation": intent["to_generation"],
                        "recovered": True,
                    },
                )
                completed = int(intent["to_generation"])
            elif int(head["generation"]) == intent["from_generation"]:
                _append_line(
                    self._p("commits.jsonl"),
                    {"fact": "commit_abandoned", "commit": commit, "acceptance": ident},
                )
                self._record_controller_fact(
                    {
                        "fact": "head_write_refused",
                        "acceptance": ident,
                        "generation": intent["to_generation"],
                        "reason": "interrupted_before_install",
                    }
                )
            else:
                raise TuningError(
                    E_HEAD_DIVERGED,
                    "head.json matches neither side of a journalled move",
                )
        return completed

    def _config_for(self, fingerprint: str) -> dict[str, Any]:
        for marker in self._markers():
            if fingerprint in marker.get("configs", {}):
                return dict(marker["configs"][fingerprint])
        for head in _read_lines(self._p("heads.jsonl")):
            if head["fingerprint"] == fingerprint:
                return dict(head["config"])
        raise TuningError(E_UNKNOWN_GENERATION, "configuration for the head not found")

    # -- status ---------------------------------------------------------------------
    def _trigger_view(self, occ: Mapping[str, Any]) -> dict[str, Any]:
        """The rule's trigger as the TVL module declares it (a closed record)."""
        rule = self._rule(occ)
        return {
            "response": occ["response"],
            "rule": occ["rule"],
            "trigger": copy.deepcopy(rule["trigger"]),
            "approval": rule.get("approval", "human"),
        }

    def spend_limit(self) -> dict[str, Any] | None:
        """The limits the optimizer will run under, or None when none is configured."""
        limit = getattr(self.optimize_fn, "spend_limit", None)
        return dict(limit) if isinstance(limit, Mapping) and limit else None

    def _approval_text(self, occ: Mapping[str, Any], ref: str) -> str:
        """Who or what approved the run, as a person reads it.

        TVL's relation models a standing grant as platform state (the attempt is
        auto-approved); here the grant record is the platform state.  Both layers
        mean the same, and ``status`` words it the same way: "approved by standing
        grant <id>".
        """
        if self._rule(occ).get("approval", "human") == "standing":
            grants = _read_json(self._p("grants.json"), {})
            for grant_id in sorted(grants):
                if ref in grants[grant_id].get("used", []):
                    return f"approved by standing grant {grant_id}"
            return "approved by standing grant"
        approval = _read_json(self._p("approvals.json"), {}).get(ref) or {}
        return f"approved by {approval.get('approver', 'an approver')}"

    def status(self) -> dict[str, Any]:
        """Head, open attempts and pending deploys (derived, read-only).

        Each attempt awaiting approval carries ``approve_run`` (what the human
        decision **Approve run** spends and why); each pending deploy carries
        ``accept_release`` (what **Accept release** puts into production).
        """
        self.state.require_validated()
        facts = self._derive()["facts"]
        closed = {f["attempt"] for f in facts if f["fact"] == "attempt_closed"}
        approvals = _read_json(self._p("approvals.json"), {})
        domains = {f["attempt"]: f for f in facts if f["fact"] == "search_domain"}
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
            entry: dict[str, Any] = {
                "attempt": opened["attempt"],
                "occurrence": ref,
                "state": state,
            }
            if state in ("approved", "interrupted"):
                entry["approval"] = self._approval_text(opened["occurrence"], ref)
            if state == "awaiting_approval":
                domain = domains.get(opened["attempt"])
                entry["approve_run"] = {
                    **self._trigger_view(opened["occurrence"]),
                    "spend_limit": self.spend_limit(),
                    "admits": (
                        {"tvar": domain["tvar"], "domain": list(domain["domain"])}
                        if domain
                        else None
                    ),
                }
            attempts.append(entry)
        head = read_head(self.dir) or {}
        deploys = []
        for f in facts:
            if f["fact"] != "deploy_awaiting" or f["attempt"] in closed:
                continue
            gate = self._gate_of(f)
            deploys.append(
                {
                    "attempt": f["attempt"],
                    "occurrence": occurrence_ref(f["occurrence"]),
                    "candidate": f["candidate"],
                    "accept_release": {
                        "incumbent": {
                            "fingerprint": gate.get("incumbent_fingerprint"),
                            "generation": gate.get("incumbent_generation"),
                        },
                        "candidate": {"fingerprint": f["candidate"]},
                        "code_version": gate.get("code_version"),
                        "holdout": {
                            "digest": gate.get("holdout_digest"),
                            "rows": gate.get("holdout_rows"),
                            "unresolved_rows": gate.get("holdout_unresolved_rows"),
                            "counts": gate.get("holdout_counts"),
                            "prior_uses": gate.get("holdout_prior_uses"),
                        },
                        "gate": {
                            "legacy_decision": gate.get("legacy_decision"),
                            "monitors": gate.get("monitors"),
                            "decision": gate.get("decision"),
                            "errors": gate.get("errors", []),
                        },
                        "observed_generation": head.get("generation"),
                        "gate_incumbent_current": (
                            gate.get("incumbent_generation") == head.get("generation")
                            and gate.get("incumbent_fingerprint")
                            == head.get("fingerprint")
                        ),
                    },
                }
            )
        return {
            "head": {
                "generation": head.get("generation"),
                "fingerprint": head.get("fingerprint"),
            },
            "open_attempts": attempts,
            "pending_deploys": deploys,
        }


def _normalized_domain(spec: Any) -> Any:
    """A declared TVL domain as the optimizer's search space expects it.

    ``{set: [...]}`` becomes the list; any other form is passed through unchanged.
    """
    if isinstance(spec, Mapping) and isinstance(spec.get("set"), list):
        return copy.deepcopy(list(spec["set"]))
    return copy.deepcopy(spec)


def _declared_contains(spec: Any, value: Any) -> bool:
    """Is ``value`` inside the declared domain ``spec``?

    Fails closed: a domain shape that cannot be read counts as containing the value,
    so the value is NOT "newly admitted" and gets no unpriced-model exemption.
    """
    if isinstance(spec, (list, tuple)):
        return value in spec
    if isinstance(spec, Mapping):
        if isinstance(spec.get("set"), list):
            return value in spec["set"]
        bounds = spec.get("range")
        if isinstance(bounds, (list, tuple)) and len(bounds) == 2:
            try:
                return bool(float(bounds[0]) <= float(value) <= float(bounds[1]))
            except (TypeError, ValueError):
                return False
    return True


def _unpriced_models_of(config: Mapping[str, Any]) -> list[str]:
    """Models named in ``config`` that have no price (their cost is unknown)."""
    from traigent.core.optimized_function import _extract_config_model_ids
    from traigent.utils.cost_calculator import find_models_missing_price_coverage

    ids = _extract_config_model_ids(config)
    try:
        return find_models_missing_price_coverage(ids)
    except Exception:  # noqa: BLE001 - an unreadable price table: nothing is known priced
        return sorted(str(i) for i in ids)  # fail closed: cost unknown, never "priced"


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

    def run(
        search_space: dict[str, Any],
        *,
        candidate_dir: Path | None = None,
        admitted_models: Sequence[str] = (),
    ) -> Any:
        root = (
            Path(candidate_dir)
            if candidate_dir is not None
            else Path(tempfile.mkdtemp(prefix="traigent-candidate-"))
        )
        clone = _candidate_copy(fn)
        # a model the event admitted may be newer than the price table: it may run
        # with its cost unknown; every other unpriced model still fails the preflight
        clone._unpriced_models_allowed = frozenset(admitted_models)
        with _isolated_artifacts(root):
            return asyncio.run(
                clone.optimize(configuration_space=search_space, **kwargs)
            )

    run.accepts_candidate_dir = True  # type: ignore[attr-defined]
    limits = {
        key: kwargs.get(key, getattr(fn, key, None))
        for key in ("max_trials", "cost_limit", "timeout")
    }
    run.spend_limit = {k: v for k, v in limits.items() if v is not None}  # type: ignore[attr-defined]
    return run


class HeadLoader:
    """Loads the head configuration from a store at call time (cached by generation)."""

    def __init__(self, store_dir: Path | str, environment: str | None = None) -> None:
        self.dir = Path(store_dir)
        self.environment = environment
        self._stamp: tuple[int, int] | None = None
        self._head: dict[str, Any] | None = None

    def head(self) -> dict[str, Any] | None:
        try:
            st = (self.dir / "head.json").stat()
        except FileNotFoundError:
            return None
        stamp = (st.st_mtime_ns, st.st_size)
        if stamp != self._stamp:
            if self.environment is not None:
                meta = _read_json(self.dir / "meta.json") or {}
                if meta.get("environment") != self.environment:
                    raise TuningError(
                        E_STORE_ENVIRONMENT_MISMATCH,
                        "store was created for another environment",
                    )
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
    "EventReceipt",
    "HeadLoader",
    "StepReport",
    "config_fingerprint",
    "E_STORE_ENVIRONMENT_MISMATCH",
    "default_store_dir",
    "occurrence_ref",
    "parse_occurrence_ref",
    "read_head",
    "traigent_optimize_fn",
]
