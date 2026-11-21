"""A local loop controller: the spend half of the closed Continuous Tuning loop.

TVL defines the semantics (``tvl.runtime_conditions.conditions_trace``); this module
only *executes* them locally.  Everything the relation consumes is recorded as an
append-only fact in a durable store; ``Controller.step`` re-derives the full trace over
the whole history and acts ONLY on derived facts:

* ``attempt_opened``: run the optimization, but only after a recorded approval
  (**Approve run**) by an approver the configured authority allows.

This controller runs the spend step only.  It never installs a head: releasing a
candidate is planned, not built, and the release entry points
(``submit_acceptance``, ``accept``, ``build_acceptance``) refuse with
``E_UNSUPPORTED_IN_RELEASE`` before touching the store.  The head it reads
(``head.json``, written once when the store is created) is read-only here.

Store layout (one directory, default ``.traigent/tuning/<environment>/<agent>/``)::

    meta.json        agent, environment, origin, contract digest, initial head
    head.json        {fingerprint, config, generation}   (the initial head; read-only)
    history.jsonl    recorded facts (checksummed JSONL): windows, approvals, finishes
    controller.jsonl controller facts: refused optimizations
    approvals.json   optimization approvals keyed by occurrence ref
    attempts/        one marker per attempt: in-progress, then the persisted result

All instants are whole seconds after ``meta.origin`` (UTC midnight of the store's
creation), exactly the reference relation's clock; nothing is rounded to an hour and
no recorded instant lies in the future.

**One gated step for the spend.**  Spending (invoking the optimizer) goes through
``Controller._gated_effect``.  The operation builds its clock at entry (a fixed
instant is refused in production).  Under the store lock it prepares everything:
history replay, scope verdict and deadline, argument copies, the repaired journal tail
and the prepared append, and the last persisted instant.  Then, with no further replay
and no journal scan: the platform authenticator is called afresh, the clock is sampled
ONCE (a persisted instant ahead of it refuses), deadlines are checked against that
sample, and that sample is the instant recorded for the start fact just before the
optimizer runs.  Every failure denies; an approved attempt the final gate refuses is
recorded as cancelled, so it never holds a later schedule.

Nothing here decides a promotion: every candidate is measured here, on the customer's
holdout against the incumbent; supplied verdicts are ignored.  No network is used.
"""

from __future__ import annotations

import asyncio
import contextlib
import copy
import dataclasses
import hashlib
import json
import logging
import os
import re
import tempfile
import threading
import time
import types
import weakref
from collections.abc import Callable, Collection, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, TypeVar, cast

from traigent.tuning.authority import LOCAL_TRUSTED, ApproverAuthority
from traigent.tuning.errors import (
    E_APPROVER_NOT_AUTHORIZED,
    E_NO_APPROVER_AUTHORITY,
    E_UNSUPPORTED_IN_RELEASE,
    RELEASE_REFUSAL,
    TuningError,
)
from traigent.tuning.state import TuningState
from traigent.tuning.validator import conditions_trace

E_INVALID_RECORD = "E_INVALID_RECORD"
E_ATTEMPT_IN_PROGRESS = "E_ATTEMPT_IN_PROGRESS"
E_NO_OPTIMIZER = "E_NO_OPTIMIZER"
E_NO_DEPLOY_AWAITING = "E_NO_DEPLOY_AWAITING"
E_STORE_MISMATCH = "E_STORE_MISMATCH"
E_STORE_ENVIRONMENT_MISMATCH = "E_STORE_ENVIRONMENT_MISMATCH"
E_NOT_AN_OCCURRENCE = "E_NOT_AN_OCCURRENCE"
E_STORE_CORRUPT = "E_STORE_CORRUPT"
E_STORE_FORMAT = "E_STORE_FORMAT"
E_NO_CANDIDATE_RUN = "E_NO_CANDIDATE_RUN"
E_HEAD_DIVERGED = "E_HEAD_DIVERGED"
E_LOCAL_AUTHORITY_IN_PRODUCTION = "E_LOCAL_AUTHORITY_IN_PRODUCTION"
E_FIXED_CLOCK_IN_PRODUCTION = "E_FIXED_CLOCK_IN_PRODUCTION"
E_LOCK_UNSUPPORTED_PLATFORM = "E_LOCK_UNSUPPORTED_PLATFORM"
E_CONNECTED_TUNING_REFUSED = "E_CONNECTED_TUNING_REFUSED"

GATE_FORMAT = "traigent-gate-result/1"
DEFAULT_STORE_ROOT = Path(".traigent") / "tuning"
_REF = re.compile(r"[A-Za-z0-9_.:-]{1,128}")
# The rule keys this release executes; every other key (``when``, cooldowns,
# ``only_while`` / ``unless`` conditions, repeat limits, ...) is refused at construction.
_SUPPORTED_RULE_KEYS = frozenset(
    {"id", "trigger", "do", "approval", "deploy", "admit", "protects"}
)
# The actions this release runs: "optimize" invokes the optimizer for an approved
# attempt; "notify" only derives a TVL fact (the controller itself does nothing).
_SUPPORTED_ACTIONS = frozenset({"optimize", "notify"})
_LEGACY = {"Promote", "Reject", "NoDecision"}
_COMPLETIONS = {"normal", "error", "cancelled", "cost_limit"}
OptimizeFn = Callable[[dict[str, Any]], Any]
logger = logging.getLogger(__name__)


DEFAULT_ENVIRONMENT = "production"


def default_store_dir(agent: str, environment: str | None = None) -> Path:
    """``.traigent/tuning/<environment>/<agent>/`` relative to the working directory.

    The environment is part of the path so a staging store and a production store
    of the same agent are never the same directory.
    """
    if not _REF.fullmatch(agent) or agent in {".", ".."}:
        raise TuningError(E_INVALID_RECORD, "agent must be an opaque reference")
    env = environment or DEFAULT_ENVIRONMENT
    if not _REF.fullmatch(env) or env in {".", ".."}:
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


def _effect_clock(now: datetime | None) -> Callable[[], datetime]:
    """The clock an effect path (an optimizer run, a head write) reads.

    ``now=None`` is the live clock, read afresh every time.  A fixed ``now`` is a
    simulation or replay clock: refused when the process counts as production
    (``treat_as_production_policy``, which also refuses when the environment is not
    known), and otherwise a start instant that still advances with real time, so
    nothing sampled after a lock wait or a slow step is frozen.
    """
    if now is None:
        return lambda: _utc(None)
    from traigent.utils.env_config import treat_as_production_policy

    if treat_as_production_policy():
        raise TuningError(
            E_FIXED_CLOCK_IN_PRODUCTION,
            "a fixed instant is for simulation and replay; production reads the "
            "live clock",
        )
    base = _utc(now)
    started = time.monotonic()
    return lambda: _utc(base + timedelta(seconds=time.monotonic() - started))


def _approve_context(ref: str) -> dict[str, Any]:
    """What the authenticator is told when a run approval is decided or re-checked."""
    return {"decision": "approve_run", "occurrence": ref}


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
    if os.name == "nt":  # a directory cannot be opened for fsync on Windows
        return
    fd = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _mkdir_durable(directory: Path) -> None:
    """Create ``directory`` (and parents); fsync the parent of each directory created."""
    missing: list[Path] = []
    current = directory
    while not current.exists() and current != current.parent:
        missing.append(current)
        current = current.parent
    directory.mkdir(parents=True, exist_ok=True)
    for created in reversed(missing):
        _fsync_dir(created.parent)


def _write_json(path: Path, value: Any) -> None:
    """Write JSON atomically: temp file in the same directory, fsync, rename."""
    _mkdir_durable(path.parent)
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
    _prepare_append(path)(value)


def _prepare_append(path: Path) -> Callable[[Any], None]:
    """Repair the tail of ``path`` NOW and return the append that only writes.

    The journal scan (checksum of every record) happens here, under the store lock
    and before the final gate; the returned callable frames one record and appends it
    without reading the journal again.
    """
    _mkdir_durable(path.parent)
    _repair_tail(path)
    created = not path.exists()

    def append(value: Any) -> None:
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(_frame(value) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        if created:
            _fsync_dir(path.parent)

    return append


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
@dataclass(frozen=True)
class OptimizerRun:
    """What one optimizer invocation returns: its result AND the identity of the
    datasets THAT run actually loaded (digests, never content).

    An optimizer that cannot say what it loaded returns its bare result; the
    candidates of such a run cannot be promoted (``eval_set_identity_unknown``).
    A mapping result may carry the same information under ``loaded_dataset_digests``.
    """

    result: Any
    loaded_dataset_digests: frozenset[str] | None

    def __getattr__(self, name: str) -> Any:  # the wrapped result stays readable
        if name.startswith("__") or name in {"result", "loaded_dataset_digests"}:
            raise AttributeError(name)
        return getattr(self.result, name)


@dataclass
class StepReport:
    """What one ``step`` did, derived facts included."""

    facts: list[dict[str, Any]] = field(default_factory=list)
    started: list[str] = field(default_factory=list)  # attempts whose optimize ran
    approved: list[str] = field(default_factory=list)
    finished: list[str] = field(default_factory=list)
    awaiting_approval: list[str] = field(default_factory=list)  # occurrence refs
    refused: list[str] = field(default_factory=list)  # "<attempt>:<closed reason>"

    @property
    def changed(self) -> bool:
        return bool(self.started or self.approved or self.finished)


@dataclass(frozen=True)
class AcceptanceOutcome:
    """The result shape of a release acceptance.

    This release's local controller never produces one: every release entry point
    refuses with ``E_UNSUPPORTED_IN_RELEASE`` (releasing is planned, not built).
    """

    acceptance_id: str
    result: str  # written | reject_recorded | refused | invalid | consumed
    reason: str | None = None
    generation: int | None = None

    @property
    def ok(self) -> bool:
        return self.result in {"written", "reject_recorded"}


_T = TypeVar("_T")


class _GateRefused(Exception):
    """The final gate of an effect refused: nothing was spent or installed."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


class _GateFailed(Exception):
    """Something in the final gate raised (not a refusal): the effect is not run."""

    def __init__(self, original: Exception) -> None:
        super().__init__(str(original))
        self.original = original


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
        now: datetime | None = None,
    ) -> None:
        policy_state.require_validated()
        # A private snapshot, taken once after validation: everything executed (and
        # the store digest) comes from it, so a later change to the caller's
        # dictionary, or to the state this controller exposes, runs nothing new.
        self._module: dict[str, Any] = copy.deepcopy(policy_state.tvl)
        self.state = dataclasses.replace(
            policy_state, tvl=copy.deepcopy(policy_state.tvl)
        )
        self._refuse_unsupported()
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
        self._init_store(agent or self.dir.name, environment, initial_config, now)

    def _refuse_unsupported(self) -> None:
        """Refuse a policy that uses a feature this release does not run.

        TVL specifies standing approval, external-event triggers, conditions on a
        rule (``only_while``, ``unless``, cooldowns and repeat limits) and more; this
        release runs a schedule trigger that starts an optimization with human
        approval, and nothing else.  The rule keys and
        trigger shapes it executes are allow-listed: anything else is refused here
        rather than silently ignored.  The policy may still be compiled and checked;
        it is never run.
        """
        for response in self._module.get("responses", []):
            for rule in response.get("rules", []):
                if rule.get("approval") == "standing":
                    raise TuningError(
                        E_UNSUPPORTED_IN_RELEASE,
                        "standing approval is not supported in this release; "
                        "use human approval",
                    )
                trigger = rule.get("trigger")
                if isinstance(trigger, Mapping) and "external" in trigger:
                    raise TuningError(
                        E_UNSUPPORTED_IN_RELEASE,
                        "external-event triggers (ModelReleased) are not supported "
                        "in this release; use a schedule",
                    )
                unsupported = sorted(set(rule) - _SUPPORTED_RULE_KEYS)
                if unsupported:
                    raise TuningError(
                        E_UNSUPPORTED_IN_RELEASE,
                        "rule condition(s) not supported in this release: "
                        + ", ".join(str(k) for k in unsupported),
                    )
                if rule.get("do") not in _SUPPORTED_ACTIONS:
                    raise TuningError(
                        E_UNSUPPORTED_IN_RELEASE,
                        "only the optimize and notify actions are supported in this release",
                    )
                if not _supported_trigger(trigger):
                    raise TuningError(
                        E_UNSUPPORTED_IN_RELEASE,
                        "only a schedule trigger is supported in this release",
                    )

    # -- store ----------------------------------------------------------------------
    def _p(self, name: str) -> Path:
        return self.dir / name

    @contextlib.contextmanager
    def _locked(self) -> Iterator[None]:
        _mkdir_durable(self.dir)
        try:
            import fcntl  # POSIX only: imported when tuning is used, never at CLI start
        except ImportError:
            raise TuningError(
                E_LOCK_UNSUPPORTED_PLATFORM,
                "the tuning store lock needs a POSIX platform (fcntl is unavailable)",
            ) from None
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
        digest = "sha256:" + _digest(self._module)
        with self._locked():
            meta = _read_json(self._p("meta.json"))
            if meta is None:
                # Init is a recoverable transaction: ``meta.json`` is written LAST,
                # so a crash earlier leaves the store uninitialized and the next
                # start clears the partial files and redoes the whole init.
                with contextlib.suppress(FileNotFoundError):
                    self._p("head.json").unlink()
                origin = _utc(now).replace(hour=0, minute=0, second=0)
                config = dict(initial_config or {})
                fingerprint = config_fingerprint(config)
                head = {"fingerprint": fingerprint, "config": config, "generation": 0}
                _write_json(self._p("head.json"), head)
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
                for name in ("history.jsonl", "controller.jsonl"):
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
        return conditions_trace(self._module, self._history(extra))

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

    def _last_persisted_instant(self) -> int:
        """The last instant the history journal holds (0 when it is empty)."""
        lines = self._lines()
        return int(lines[-1]["instant"]) if lines else 0

    def _gated_effect(
        self,
        *,
        clock: Callable[[], datetime],
        last_persisted: int,
        authenticate: Callable[[], bool],
        lapse_at: Callable[[int], str | None],
        record: Callable[[int], None],
        effect: Callable[[Callable[[], int]], _T],
    ) -> tuple[_T | None, str | None, int | None]:
        """The ONE path the spend (invoking the optimizer) takes.

        The caller has already done ALL preparation under the store lock (history
        replay, scope verdict, deadline, argument copies, the repaired and prepared
        journal append) and captured ``last_persisted``.  What runs here, in this
        order, and nothing slow after it:

        1. the platform authenticator, called afresh (never a cached answer);
        2. the clock, sampled ONCE: a persisted instant ahead of it refuses;
        3. ``lapse_at(instant)``: deadlines and scope expiry against that sample;
        4. ``record(instant)``: the one effect instant goes into the journal (an
           append that scans nothing);
        5. the effect itself.

        ``effect`` receives ``final_check``, which runs steps 1-4 and returns the
        instant; the effect calls it immediately before it acts (an adapter that waits
        for a lock calls it after the wait).  Every failure denies.  Returns
        ``(value, refusal, instant)``.
        """
        gated: list[int] = []

        def final_check() -> int:
            if gated:
                raise TuningError(E_INVALID_RECORD, "the final check runs once")
            try:
                authenticated = bool(authenticate())
            except Exception:  # noqa: BLE001 - an authenticator that fails never authenticates
                authenticated = False
            if not authenticated:
                raise _GateRefused("approver_not_authorized")
            try:
                sample = _utc(clock())
                instant = int((sample - self.origin).total_seconds())
                if instant < last_persisted:
                    raise _GateRefused("persisted_time_ahead")
                reason = lapse_at(instant)
                if reason is not None:
                    raise _GateRefused(reason)
                record(instant)
            except _GateRefused:
                raise
            except Exception as exc:
                raise _GateFailed(exc) from exc
            gated.append(instant)
            return instant

        try:
            value = effect(final_check)
        except _GateRefused as refusal:
            return None, refusal.reason, None
        except _GateFailed as failed:
            raise failed.original from None
        return value, None, gated[0] if gated else None

    # -- authority ------------------------------------------------------------------
    def require_authority(self) -> ApproverAuthority:
        """The configured approver authority, or a closed refusal.

        There is no default: without an explicit authority nobody can approve a
        run.  A ``local-trusted`` authority is refused when the environment is
        production (production needs the platform to authenticate the person).
        """
        if self.authority is None:
            raise TuningError(
                E_NO_APPROVER_AUTHORITY,
                "no approver authority is configured; approving a run needs one",
            )
        if self.authority.mode == LOCAL_TRUSTED:
            from traigent.tuning.authority import local_trusted_refused

            if local_trusted_refused(self.meta.get("environment")):
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

        The same authenticator decides every authority decision (a run approval, and
        its re-check just before the optimizer runs): a platform
        authenticator that answers False refuses.  No authority means no.
        """
        authority = self.authority
        if authority is None or not authority.authorized(approver):
            return False
        return authority.authenticated(
            approver,
            context or {"decision": "authority"},
            self.meta.get("environment"),
        )

    # -- recording ------------------------------------------------------------------
    def record_event(self, *_args: Any, **_kwargs: Any) -> None:
        """External events are not supported in this release."""
        raise TuningError(
            E_UNSUPPORTED_IN_RELEASE,
            "recording external events is not supported in this release",
        )

    def record_window(
        self, report: Mapping[str, Any], now: datetime | None = None
    ) -> int:
        """Append a monitor window report (a verdict computed by the monitor).

        A window is recorded at the instant it is reported; a report for a window
        that has not ended yet is refused by the relation, never moved forward.
        """
        self.state.require_validated()
        clock = _effect_clock(now)  # operation entry: refused when fixed in production
        record = copy.deepcopy(dict(report))
        with self._locked():
            instant = self._instant(clock())
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
        if not _REF.fullmatch(approver):
            raise TuningError(E_INVALID_RECORD, "approver must be an opaque reference")
        authority = self.require_authority()
        if not authority.authorized(approver):
            raise TuningError(
                E_APPROVER_NOT_AUTHORIZED, "that approver is not on the authority list"
            )
        if not authority.authenticated(
            approver, _approve_context(ref), self.meta.get("environment")
        ):
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

    def record_standing_grant(self, *_args: Any, **_kwargs: Any) -> None:
        """Standing grants are not supported in this release."""
        raise TuningError(
            E_UNSUPPORTED_IN_RELEASE,
            "standing grants are not supported in this release",
        )

    def controller_facts(self) -> list[dict[str, Any]]:
        """Closed facts the controller itself recorded (refusals); TVL never sees them."""
        return _read_lines(self._p("controller.jsonl"))

    def _record_controller_fact(self, fact: dict[str, Any]) -> None:
        if fact not in self.controller_facts():
            _append_line(self._p("controller.jsonl"), fact)

    # -- the step -------------------------------------------------------------------
    def step(
        self,
        now: datetime | None = None,
        *,
        abandon_interrupted: bool = False,
    ) -> StepReport:
        """Re-derive over the whole history and act only on derived facts.

        Idempotent: with an unchanged history nothing is optimized or written twice.
        An attempt interrupted mid-optimization (marker without a result) is refused
        until ``abandon_interrupted`` records it as cancelled.
        """
        self.state.require_validated()
        report = StepReport()
        # Operation entry: a live clock is read afresh on every read; a fixed ``now``
        # (simulation only; production refuses it) starts advancing HERE, so a wait
        # for the store lock moves the first scheduling read.
        clock = _effect_clock(now)
        with self._locked():
            # Read-only divergence check BEFORE any journal write or spend.
            self._check_head_consistent(self._derive())
            moment = clock()
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
            for response in self._module.get("responses", [])
            for rule in response["rules"]
        )

    def _rule(self, occ: Mapping[str, Any]) -> dict[str, Any]:
        for response in self._module.get("responses", []):
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
            if fact["fact"] != "attempt_opened":
                continue
            # A closed attempt is kept (flagged): its optimization may still be
            # running or hold a saved finish marker that must be recorded.
            out.append(
                {
                    **fact,
                    "finished": fact["attempt"] in finished,
                    "closed": fact["attempt"] in closed,
                }
            )
        return out

    def _start_recorded(self, attempt: str) -> bool:
        return any(
            ln["category"] == "optimizations"
            and ln["record"].get("attempt") == attempt
            and ln["record"].get("fact") == "started"
            for ln in self._lines()
        )

    def _finish_recorded(self, attempt: str) -> bool:
        return any(
            ln["category"] == "optimizations"
            and ln["record"].get("attempt") == attempt
            and ln["record"].get("fact") == "finished"
            for ln in self._lines()
        )

    def _approval_recorded(self, attempt: str) -> bool:
        return any(
            ln["category"] == "optimizations"
            and ln["record"].get("attempt") == attempt
            and ln["record"].get("fact") == "approval"
            for ln in self._lines()
        )

    def _authorized_now(self, occ: Mapping[str, Any], ref: str) -> bool:
        """A recorded approval for THIS occurrence by an approver the authority allows
        now (every approval mode other than ``human`` is refused at construction)."""
        approval = _read_json(self._p("approvals.json"), {}).get(ref)
        return approval is not None and self._on_list_now(
            approval.get("approver"), _approve_context(ref)
        )

    def _scope_deadline(
        self, response: str, rule_id: str, scope_kind: str, scope: Any
    ) -> int | None:
        """The whole second (after the origin) at which a schedule scope ends.

        A schedule scope lasts until the next tick, so its end is known without
        replaying the history.  A monitor-breach scope ends only when its episode
        closes (a recorded window report, never the clock): no deadline.
        """
        if scope_kind != "schedule":
            return None
        from tvl.lints import parse_utc_instant

        try:
            rule = next(
                r
                for resp in self._module.get("responses", [])
                if resp["name"] == response
                for r in resp["rules"]
                if r["id"] == rule_id
            )
            spec = rule["trigger"]["schedule"]
            anchor = parse_utc_instant(spec["anchor"])
            tick = int(scope)
            every = int(spec["every_days"])
        except (StopIteration, KeyError, TypeError, ValueError):
            raise TuningError(E_NOT_AN_OCCURRENCE, "unknown schedule scope") from None
        if anchor is None:
            raise TuningError(E_NOT_AN_OCCURRENCE, "unknown schedule scope")
        return int((anchor - self.origin).total_seconds()) + (tick + 1) * every * 86400

    def _scope_requires_intervention(
        self, occ: Mapping[str, Any], instant: int
    ) -> bool:
        """Ask the relation, run to ``instant`` (seconds), whether the occurrence's scope still counts.

        An attempt the relation has closed (any reason, ``scope_expired`` included)
        is final: it never counts as requiring intervention.
        """
        from tvl.runtime_conditions import occurrence_status

        history = self._history([{"instant": instant, "category": "tick"}])
        status = occurrence_status(self._module, history, dict(occ))
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
            if (
                abandon_interrupted
                and marker is None
                and attempt not in report.approved
                and self._approval_recorded(attempt)
            ):
                # A granted human approval already counts as the start (TVL), but the
                # process stopped before the attempt marker existed: nothing else would
                # ever finish it and every later schedule would stay held.  Explicit
                # abandonment records it as cancelled.
                record = self._finished_record(attempt, "cancelled", [])
                _write_json(
                    marker_path,
                    {
                        "attempt": attempt,
                        "occurrence": ref,
                        "state": "finished",
                        "finished_record": record,
                    },
                )
                return True
            if opened["closed"] and marker is None:
                if self._approval_recorded(attempt):
                    # a recorded grant already counts as the start (TVL): record the
                    # cancellation so the attempt never holds a later schedule.
                    self._cancel_refused(
                        report, attempt, ref, marker_path, "attempt_closed"
                    )
                    return True
                continue  # closed and never started: nothing to finish, nothing blocks
            # Immediately before any authorization is used or the optimizer is
            # invoked: the relation, over the history up to now, must still show
            # this occurrence's scope as requiring intervention.
            if marker is None and not self._scope_requires_intervention(occ, instant):
                if self._approval_recorded(attempt):
                    # granted, then the scope lapsed before any start: a granted
                    # approval counts as the start, so cancel rather than hold.
                    self._cancel_refused(
                        report, attempt, ref, marker_path, "scope_expired"
                    )
                    return True
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
                if not self._authorized_now(occ, ref):
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
                if self._finish_recorded(attempt):
                    continue  # delivered once already (a closed attempt may ignore it)
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
            if self._spend(
                report, attempt, occ, ref, domains.get(attempt), marker_path, clock
            ):
                return True
            continue
        return False

    def _spend(
        self,
        report: StepReport,
        attempt: str,
        occ: Mapping[str, Any],
        ref: str,
        domain_fact: Mapping[str, Any] | None,
        marker_path: Path,
        clock: Callable[[], datetime],
    ) -> bool:
        """Invoke the optimizer for an approved attempt through the gated effect.

        True when history will grow: the attempt ran and its outcome was recorded, or
        the spend was refused and the attempt is cancelled (a controller fact says
        why).  False when the spend could not be judged (no approval on file).
        """
        if self.optimize_fn is None:
            raise TuningError(E_NO_OPTIMIZER, f"{attempt} needs an optimize_fn")
        # -- preparation: every read and write the spend needs, under the store
        # lock and BEFORE the final gate ------------------------------------
        approval = _read_json(self._p("approvals.json"), {}).get(ref)
        if approval is None:
            self._refuse_spend(report, attempt, "approval_missing")
            return False
        approver = approval.get("approver")
        space = self._search_space(domain_fact)
        admitted = self._admitted_values(domain_fact)
        in_scope = self._scope_requires_intervention(occ, self._instant(clock()))
        deadline = self._scope_deadline(
            occ["response"], occ["rule"], occ["scope_kind"], occ["scope"]
        )
        if not in_scope:
            self._cancel_refused(report, attempt, ref, marker_path, "scope_expired")
            return True
        call_space = copy.deepcopy(space)
        candidate_dir = self._p("candidates") / attempt
        start_recorded = self._start_recorded(attempt)
        last_persisted = self._last_persisted_instant()
        # The one journal scan the final gate would otherwise need: repair the tail
        # now, under the lock and before authentication; the gate only appends.
        append_start = _prepare_append(self._p("history.jsonl"))
        # Only the adapter this SDK built may be handed the final check.  Decided
        # before the marker: nothing after the marker may fail before the optimizer runs.
        optimize_fn = self.optimize_fn
        sdk_adapter = _is_sdk_adapter(optimize_fn)
        accepts_dir = bool(getattr(optimize_fn, "accepts_candidate_dir", False))
        # Journal first (the marker), then the start fact at the effect instant:
        # a crash between the two leaves a marker that abandon_interrupted finds.
        _write_json(
            marker_path,
            {"attempt": attempt, "occurrence": ref, "state": "started"},
        )

        def lapse_at(instant: int, deadline: int | None = deadline) -> str | None:
            if deadline is not None and instant >= deadline:
                return "scope_expired"
            return None

        def record_start(instant: int, attempt: str = attempt) -> None:
            if not start_recorded:
                append_start(
                    {
                        "instant": instant,
                        "category": "optimizations",
                        "record": {"attempt": attempt, "fact": "started"},
                    }
                )

        def run(
            final_check: Callable[[], int],
            attempt: str = attempt,
            call_space: dict[str, Any] = call_space,
        ) -> tuple[str, list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
            ran: list[int] = []

            def checked() -> int:
                ran.append(1)
                return final_check()

            try:
                if sdk_adapter:
                    # the SDK adapter waits for its own lock first and calls the
                    # final check immediately before the optimizer starts
                    result = cast(Callable[..., Any], optimize_fn)(
                        call_space,
                        candidate_dir=candidate_dir,
                        admitted_models=admitted,
                        final_check=checked,
                    )
                    if not ran:
                        raise TuningError(
                            E_NO_CANDIDATE_RUN, "the optimizer skipped the check"
                        )
                else:
                    # any other optimizer is never handed the final check: the full
                    # gate runs here, immediately before it is invoked
                    final_check()
                    if accepts_dir:
                        result = cast(Callable[..., Any], optimize_fn)(
                            call_space,
                            candidate_dir=candidate_dir,
                            admitted_models=admitted,
                        )
                    else:
                        result = optimize_fn(call_space)
                return self._map_result(attempt, result)
            except (_GateRefused, _GateFailed):
                raise
            except Exception:  # noqa: BLE001 - a failed run is recorded, content never echoed
                return "error", [], {}, {}

        outcome, refusal, _instant_used = self._gated_effect(
            clock=clock,
            last_persisted=last_persisted,
            authenticate=lambda: self._on_list_now(approver, _approve_context(ref)),
            lapse_at=lapse_at,
            record=record_start,
            effect=run,
        )
        if refusal is not None or outcome is None:
            # nothing started.  A granted human approval already counts as the start
            # (TVL): cancel the attempt so it never holds a later schedule.
            self._cancel_refused(
                report, attempt, ref, marker_path, refusal or "gate_refused"
            )
            return True
        completion, candidates, configs, gates = outcome
        report.started.append(attempt)
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

    def _cancel_refused(
        self,
        report: StepReport,
        attempt: str,
        ref: str,
        marker_path: Path,
        reason: str,
    ) -> None:
        """A spend refused for an approved attempt: say why, and cancel the attempt.

        The cancelled finish is persisted in the marker (the step then records it),
        so a crash in between loses nothing and the optimizer never ran.
        """
        self._refuse_spend(report, attempt, reason)
        _write_json(
            marker_path,
            {
                "attempt": attempt,
                "occurrence": ref,
                "state": "finished",
                "finished_record": self._finished_record(attempt, "cancelled", []),
            },
        )

    def _refuse_spend(self, report: StepReport, attempt: str, reason: str) -> None:
        """Journal a refused spend (a closed controller fact) and report it."""
        self._record_controller_fact(
            {"fact": "optimize_refused", "attempt": attempt, "reason": reason}
        )
        report.refused.append(f"{attempt}:{reason}")

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
            for t in self._module.get("tvars", [])
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
            for t in self._module.get("tvars", [])
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
            for m in self._module.get("monitors", [])
            if m.get("phase") == "promotion"
        )

    def _measure_candidate(
        self,
        attempt: str,
        config: Mapping[str, Any],
        incumbent: Mapping[str, Any],
        loaded: Collection[str] | None = None,
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
            if not _well_formed_digests(loaded):
                # this run did not say, or did not say well, what it loaded: its
                # identity is unknown
                raise EvidenceUnavailable("eval_set_identity_unknown")
            eval_digests, _rows = eval_set_identity(
                getattr(optimized, "eval_dataset", None),
                source if callable(source) else None,
            )
            eval_digests = set(eval_digests) | set(cast(Collection[str], loaded))
            return build_promotion_evidence(
                self._module,
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

        loaded: Collection[str] | None = None
        if isinstance(result, OptimizerRun):
            loaded, result = result.loaded_dataset_digests, result.result
        elif isinstance(result, Mapping):
            loaded = result.get("loaded_dataset_digests")

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
                evidence = self._measure_candidate(attempt, config, incumbent, loaded)
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

    # -- release (planned, not built; not run by this controller) --------------
    def build_acceptance(self, *_args: Any, **_kwargs: Any) -> dict[str, Any]:
        """Release acceptance is not run by this controller: refused before any I/O."""
        raise TuningError(E_UNSUPPORTED_IN_RELEASE, RELEASE_REFUSAL)

    def submit_acceptance(
        self, record: Mapping[str, Any], now: datetime | None = None
    ) -> AcceptanceOutcome:
        """Release acceptance is not run by this controller: refused before any I/O."""
        raise TuningError(E_UNSUPPORTED_IN_RELEASE, RELEASE_REFUSAL)

    def accept(
        self,
        deploy_ref: str,
        approver: str,
        expected_head_generation: int,
        decision: str = "accept",
        now: datetime | None = None,
    ) -> AcceptanceOutcome:
        """Release acceptance is not run by this controller: refused before any I/O."""
        raise TuningError(E_UNSUPPORTED_IN_RELEASE, RELEASE_REFUSAL)

    def _attempt_marker(self, deploy: Mapping[str, Any]) -> dict[str, Any]:
        """The marker of THE deploy's own attempt (by attempt id, never a scan)."""
        ref = occurrence_ref(deploy["occurrence"])
        marker = _read_json(self._marker_path(str(deploy["attempt"]), ref))
        if not isinstance(marker, dict):
            raise TuningError(E_NO_DEPLOY_AWAITING, "candidate evidence not found")
        return marker

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

    def _approval_text(self, ref: str) -> str:
        """Who approved the run, as a person reads it."""
        approval = _read_json(self._p("approvals.json"), {}).get(ref) or {}
        return f"approved by {approval.get('approver', 'an approver')}"

    def status(self) -> dict[str, Any]:
        """Head, open attempts and pending deploys (derived, read-only).

        Each attempt awaiting approval carries ``approve_run`` (what the human
        decision **Approve run** spends and why); each pending deploy carries
        ``release_evidence`` (the candidate evidence a planned release mechanism would release
        from; this controller never installs it).
        """
        self.state.require_validated()
        facts = self._derive()["facts"]
        closed = {f["attempt"] for f in facts if f["fact"] == "attempt_closed"}
        approvals = _read_json(self._p("approvals.json"), {})
        domains = {f["attempt"]: f for f in facts if f["fact"] == "search_domain"}
        attempts = []
        for opened in self._open_attempts(facts):
            if opened["closed"]:
                continue  # draining attempts exist for recovery, not as open work
            ref = occurrence_ref(opened["occurrence"])
            if opened["finished"]:
                state = "deploy_awaiting"
            elif self._approval_recorded(opened["attempt"]) or ref in approvals:
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
                entry["approval"] = self._approval_text(ref)
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
                    "release_evidence": {
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


_DIGEST = re.compile(r"sha256:[0-9a-f]{64}")


def _supported_trigger(trigger: Any) -> bool:
    """A schedule (``every_days`` + ``anchor``): the pilot's only trigger."""
    if not isinstance(trigger, Mapping) or set(trigger) != {"schedule"}:
        return False
    spec = trigger["schedule"]
    return isinstance(spec, Mapping) and set(spec) == {"every_days", "anchor"}


def _well_formed_digests(loaded: Any) -> bool:
    """A real collection of digest strings (never a bare string, which iterates as
    characters); anything else means the run's dataset identity is unknown."""
    return isinstance(loaded, (list, tuple, set, frozenset)) and all(
        isinstance(item, str) and _DIGEST.fullmatch(item) for item in loaded
    )


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


def _pin_local_offline(clone: Any, requested_algorithm: Any = None) -> None:
    """Make the candidate copy run fully local: no backend call, no search-space values out.

    A connected ``optimize()`` sends the raw search space (prompt text included) to
    the backend; the pilot keeps configuration VALUES on the customer side, so the
    copy is pinned to the SDK's offline execution policy.  A function that cannot run
    local (a smart algorithm, a hybrid-API evaluator) is refused, never run connected.
    """
    from traigent.config.types import (
        ExecutionIntent,
        ExecutionMode,
        ResolvedExecutionPolicy,
        is_smart_algorithm,
        resolve_execution_policy,
    )

    refuse = TuningError(
        E_CONNECTED_TUNING_REFUSED,
        "tuning runs only locally: this function is configured for a connected run "
        "(smart algorithm or hybrid-API evaluator), which would send configuration "
        "values to the backend",
    )
    if isinstance(requested_algorithm, str) and is_smart_algorithm(requested_algorithm):
        raise refuse
    if getattr(clone, "external_service_evaluator", None) is not None:
        raise refuse
    if getattr(clone, "execution_mode", None) == ExecutionMode.HYBRID_API.value:
        raise refuse
    policy = getattr(clone, "execution_policy", None)
    if not isinstance(policy, ResolvedExecutionPolicy):
        if hasattr(clone, "execution_policy"):
            raise refuse  # an SDK function whose policy cannot be read: fail closed
        return  # not an SDK function: it has no backend path to pin
    if (
        policy.require_cloud
        or policy.intent is ExecutionIntent.CLOUD_REQUIRED
        or policy.legacy_execution_mode is ExecutionMode.HYBRID_API
    ):
        raise refuse
    clone.execution_policy = resolve_execution_policy(
        algorithm=policy.algorithm,
        offline=True,
        require_cloud=False,
        source_hint="tuning_local_only",
    )
    clone.offline = True
    clone.execution_mode = ExecutionMode.LOCAL.value
    clone._effective_execution_mode = ExecutionMode.LOCAL


# The SDK resolves its artifact roots from the process environment; these are the
# knobs that route the optimization log (best-config artifacts included) and local
# storage.  They are swapped for the duration of one candidate run.
_ARTIFACT_ROOT_ENV = ("TRAIGENT_OPTIMIZATION_LOG_DIR", "TRAIGENT_RESULTS_FOLDER")
_ARTIFACT_ENV_LOCK = threading.Lock()


@contextlib.contextmanager
def _isolated_artifacts(root: Path) -> Iterator[None]:
    """Route every SDK artifact written inside the block under ``root``."""
    _mkdir_durable(root)
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


# The adapters this SDK built.  Only these may be handed the controller's final check:
# membership is by object identity, so an attribute a caller sets on its own optimizer
# cannot claim it.
_SDK_ADAPTERS: weakref.WeakSet[Callable[..., Any]] = weakref.WeakSet()


def _is_sdk_adapter(candidate: Any) -> bool:
    try:
        # Every adapter is a plain function, which compares and hashes by identity, so
        # membership is identity and never iterates the set a registration may grow.
        return type(candidate) is types.FunctionType and candidate in _SDK_ADAPTERS
    except TypeError:  # unhashable: certainly not ours
        return False


def traigent_optimize_fn(fn: Any, **kwargs: Any) -> OptimizeFn:
    """Adapt a ``@traigent.optimize`` function: run it on the search space.

    The optimization runs on an isolated copy (see :func:`_candidate_copy`), so the
    live function's configuration is never changed, and every artifact it writes
    (including best-config files) lands in ``candidate_dir`` -- inside the tuning
    store -- never where a live wrapper's best-config loaders look.  A deploy only
    follows the local head file, never this adapter.
    """

    def run(
        search_space: dict[str, Any],
        *,
        candidate_dir: Path | None = None,
        admitted_models: Sequence[str] = (),
        final_check: Callable[[], Any] | None = None,
    ) -> Any:
        root = (
            Path(candidate_dir)
            if candidate_dir is not None
            else Path(tempfile.mkdtemp(prefix="traigent-candidate-"))
        )
        clone = _candidate_copy(fn)
        _pin_local_offline(clone, kwargs.get("algorithm"))
        # Record the identity of every dataset the run ACTUALLY loads (a per-call
        # dataset or a generated override included), for the holdout-disjointness
        # check at promotion.
        from traigent.tuning.evidence import dataset_digests

        loaded: set[str] = set()
        original_load = clone._load_dataset

        def recording_load(*args: Any, **kwargs: Any) -> Any:
            dataset = original_load(*args, **kwargs)
            loaded.update(dataset_digests(dataset))
            return dataset

        clone._load_dataset = recording_load
        # a model the event admitted may be newer than the price table: it may run
        # with its cost unknown; every other unpriced model still fails the preflight
        clone._unpriced_models_allowed = frozenset(admitted_models)
        with _isolated_artifacts(root):
            # The artifact lock is held now: whatever waited for it is over.  The
            # controller's final check is the last thing before the optimizer starts.
            if final_check is not None:
                final_check()
            result = asyncio.run(
                clone.optimize(configuration_space=search_space, **kwargs)
            )
        return OptimizerRun(result, frozenset(loaded) if loaded else None)

    _SDK_ADAPTERS.add(run)
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
    "HeadLoader",
    "OptimizerRun",
    "StepReport",
    "config_fingerprint",
    "E_STORE_ENVIRONMENT_MISMATCH",
    "default_store_dir",
    "occurrence_ref",
    "parse_occurrence_ref",
    "read_head",
    "traigent_optimize_fn",
]
