"""Local guards: ``block_action`` and ``halt_unit``, in process.

Normative sources (TVL ``origin/main`` 098bcdfba0b7, ``formalization/``):

* The TVL reactive-policy specification: local guards and ``on_check_error``,
  prefix-fixed violations, unit predicates, events (``ref``, ``correlation_id``,
  revocation), and resource limits.
* Formal semantics Definition 12.6 (violation predicates), 12.7 (an evaluator
  profile is outside TVL), 12.8 (classification contract and the local guard
  paragraph).

What a guard does.  Before an effect (a tool call, say) the caller offers the
event to a :class:`UnitGuard`.  The guard acts, by suppressing the event
(``block_action``) or stopping the unit (``halt_unit``), only when it can
establish, from the events that already took effect, both that the unit is in
the monitor's population (``A_q``) and that this event is a prefix-fixed
violation of ``absence``, ``bounded_existence``, ``precedence`` or ``protocol``.
When it cannot establish that, the rule's ``on_check_error`` decides:
``block`` blocks or halts (fail closed), ``allow`` lets the effect proceed and
records the unit as unresolved.  A unit outside the population is never acted on.
Metric, bounded-response and checker predicates cannot be guards.

Everything here is pure and offline: guards keep working when remote authority is
unreachable.  Nothing in this module sends anything.

The evaluator-profile choices the RFC leaves open (Definition 12.7) are stated in
the code where they are made and pinned by the SDK-owned parity vectors in
``tests/fixtures/runtime/local_guard_vectors.json``:

* an event that is irrelevant to a rule's predicate is allowed without consulting
  ``on_check_error`` (there is no violating event to suppress);
* a "violating step" is the offered event itself: the matching ``a`` for
  ``absence``, the (N+1)-th matching ``a`` for ``bounded_existence``, a matching
  ``b`` with no legal authorization for ``precedence``, an in-alphabet event that
  lands in a rejecting state for ``protocol``;
* history is trusted only while each producer's sequence numbers are 0, 1, 2, ...
  with no gap or repeat and (unless the caller attests an order) a single producer.
"""

from __future__ import annotations

import contextvars
import functools
import inspect
import re
import uuid
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, TypeVar

from traigent.runtime.model import RuntimeRequirementError

F = TypeVar("F", bound=Callable[..., Any])

# -- closed vocabularies ---------------------------------------------------------------

ALLOW = "allow"
BLOCK = "block"
HALT = "halt"

LOCAL_FORMATS = ("tvl-runtime-local/1", "tvl-runtime-local/2")
GUARD_ACTIONS = ("block_action", "halt_unit")
ON_CHECK_ERROR = ("block", "allow")
#: The patterns whose violation a finite prefix fixes.
PREFIX_FIXED = ("absence", "bounded_existence", "precedence", "protocol")
EVENT_TYPES = (
    "unit_start",
    "unit_end",
    "tool_call",
    "tool_result",
    "model_call",
    "handoff",
    "error",
    "authorization_granted",
    "authorization_revoked",
)

# decision reasons
R_NOT_RELEVANT = "not_relevant"
R_EXCLUDED = "not_in_population"
R_NO_VIOLATION = "no_violation"
R_VIOLATION = "violation_established"
R_CHECK_ERROR_BLOCK = "check_error_block"
R_CHECK_ERROR_ALLOW = "check_error_allow"
R_UNIT_HALTED = "unit_halted"

# causes of an undecidable moment
C_POPULATION = "population_unknown"
C_INCOMPLETE = "history_incomplete"
C_ORDER = "order_unknown"
C_KEY = "correlation_key_missing"
C_LIMIT = "resource_limit"

# closed error codes
E_LOCAL_INVALID = "local_artifact_invalid"
E_NOT_ENFORCEABLE = "guard_not_enforceable"
E_EVENT_INVALID = "guard_event_invalid"
E_UNIT_MISSING = "guard_unit_missing"
E_TOOL_NAME = "guard_tool_not_code_identifier"
E_BLOCKED = "local_guard_blocked"
E_HALTED = "local_guard_halted"

_CODE_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)*")
_RANK = {ALLOW: 0, BLOCK: 1, HALT: 2}
_REASON_RANK = {
    R_VIOLATION: 5,
    R_CHECK_ERROR_BLOCK: 5,
    R_UNIT_HALTED: 5,
    R_CHECK_ERROR_ALLOW: 4,
    R_EXCLUDED: 3,
    R_NO_VIOLATION: 2,
    R_NOT_RELEVANT: 1,
}


# -- results -----------------------------------------------------------------------------


@dataclass(frozen=True)
class GuardDecision:
    """What the guards decided about one offered event.

    ``verdict`` is ``allow``, ``block`` (suppress this event) or ``halt`` (suppress it
    and stop the unit).  ``reason`` and ``cause`` come from closed vocabularies;
    ``unit_unresolved`` is set when an ``on_check_error: allow`` let the event through
    without the check being established (the unit is then recorded unresolved).
    """

    verdict: str
    reason: str
    cause: str | None = None
    rule_id: str | None = None
    unit_unresolved: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict,
            "reason": self.reason,
            "cause": self.cause,
            "rule_id": self.rule_id,
            "unit_unresolved": self.unit_unresolved,
        }


class LocalGuardError(RuntimeRequirementError):
    """A guard was blocked or halted; carries the closed decision, never event values."""

    def __init__(self, code: str, decision: GuardDecision) -> None:
        super().__init__(code)
        self.decision = decision


# -- the compiled artifact ---------------------------------------------------------------


@dataclass(frozen=True)
class GuardRule:
    """One local guard resolved against its requirement and monitor."""

    rule_id: str
    response: str
    monitor: str
    action: str
    on_check_error: str
    requirement: str
    predicate: Mapping[str, Any]
    among: Mapping[str, Any] | None
    limits: Mapping[str, int]
    single_producer: bool


def _bad(path: str) -> RuntimeRequirementError:
    return RuntimeRequirementError(E_LOCAL_INVALID, path)


def _by_name(items: Any, path: str) -> dict[str, Mapping[str, Any]]:
    if not isinstance(items, list):
        raise _bad(path)
    out: dict[str, Mapping[str, Any]] = {}
    for index, item in enumerate(items):
        if not isinstance(item, Mapping) or not isinstance(item.get("name"), str):
            raise _bad(f"{path}[{index}]")
        out[item["name"]] = item
    return out


def _rule_from_local(
    guard: Any,
    index: int,
    monitors: dict[str, Mapping[str, Any]],
    requirements: dict[str, Mapping[str, Any]],
) -> GuardRule:
    path = f"guards[{index}]"
    if not isinstance(guard, Mapping):
        raise _bad(path)
    action, on_error = guard.get("action"), guard.get("on_check_error")
    if action not in GUARD_ACTIONS or on_error not in ON_CHECK_ERROR:
        raise _bad(path)
    monitor = monitors.get(guard.get("monitor"))  # type: ignore[arg-type]
    if monitor is None:
        raise _bad(f"{path}.monitor")
    requirement = requirements.get(monitor.get("requirement"))  # type: ignore[arg-type]
    if requirement is None:
        raise _bad(f"{path}.monitor.requirement")
    violation = requirement.get("violation")
    evidence = monitor.get("evidence")
    if (
        not isinstance(violation, Mapping)
        or violation.get("pattern") not in PREFIX_FIXED
        or not isinstance(evidence, Mapping)
        or evidence.get("source") != "customer_side"
    ):
        # response_guard_not_enforceable.
        raise RuntimeRequirementError(E_NOT_ENFORCEABLE, path)
    among = requirement.get("among")
    if among is not None and not isinstance(among, Mapping):
        raise _bad(f"{path}.among")
    limits = monitor.get("limits") or {}
    if not isinstance(limits, Mapping):
        raise _bad(f"{path}.limits")
    return GuardRule(
        rule_id=str(guard.get("rule_id")),
        response=str(guard.get("response")),
        monitor=str(guard.get("monitor")),
        action=action,
        on_check_error=on_error,
        requirement=str(requirement["name"]),
        predicate=violation,
        among=among,
        limits={
            k: v
            for k, v in limits.items()
            if k in ("max_active_obligations", "max_tracked_correlations")
            and type(v) is int
        },
        single_producer=bool(monitor.get("single_producer", False)),
    )


def rules_from_local(local: Mapping[str, Any]) -> tuple[GuardRule, ...]:
    """Resolve the ``guards`` of a compiled ``tvl-runtime-local`` artifact (/1 or /2)."""
    if not isinstance(local, Mapping) or local.get("format") not in LOCAL_FORMATS:
        raise _bad("format")
    declarations = local.get("declarations")
    if not isinstance(declarations, Mapping):
        raise _bad("declarations")
    requirements = _by_name(
        declarations.get("requirements"), "declarations.requirements"
    )
    monitors = _by_name(declarations.get("monitors"), "declarations.monitors")
    guards = local.get("guards")
    if not isinstance(guards, list):
        raise _bad("guards")
    rules: list[GuardRule] = []
    for index, guard in enumerate(guards):
        rules.append(_rule_from_local(guard, index, monitors, requirements))
    return tuple(rules)


# -- events --------------------------------------------------------------------------------

_UNIT_KEY = ""


def _event(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Validate an offered event (closed shape; the error never echoes a value)."""
    if not isinstance(raw, Mapping):
        raise RuntimeRequirementError(E_EVENT_INVALID, "event")
    if raw.get("type") not in EVENT_TYPES:
        raise RuntimeRequirementError(E_EVENT_INVALID, "event.type")
    if not isinstance(raw.get("producer"), str) or not raw["producer"]:
        raise RuntimeRequirementError(E_EVENT_INVALID, "event.producer")
    seq = raw.get("sequence")
    if type(seq) is not int or seq < 0:
        raise RuntimeRequirementError(E_EVENT_INVALID, "event.sequence")
    for key in ("tool", "status", "ref", "correlation_id", "tool_call_id"):
        if key in raw and raw[key] is not None and not isinstance(raw[key], str):
            raise RuntimeRequirementError(E_EVENT_INVALID, f"event.{key}")
    return dict(raw)


def _matches(sel: Mapping[str, Any], event: Mapping[str, Any]) -> bool:
    """A selector tests the event type and the optional tool, status and ref fields."""
    if event.get("type") != sel.get("event"):
        return False
    return all(
        event.get(field) == sel[field]
        for field in ("tool", "status", "ref")
        if field in sel
    )


def _key(event: Mapping[str, Any], correlate_by: str) -> str | None:
    """The correlation key of an event; ``None`` when it lacks the chosen key."""
    if correlate_by == "unit":
        return _UNIT_KEY
    value = event.get(correlate_by)
    return value if isinstance(value, str) and value else None


# -- predicate evaluation --------------------------------------------------------------------

_VIOLATION = "violation"
_OK = "ok"
_UNDECIDABLE = "undecidable"
_Outcome = tuple[str, str | None]


def _relevant(predicate: Mapping[str, Any], event: Mapping[str, Any]) -> bool:
    pattern = predicate["pattern"]
    if pattern in ("absence", "bounded_existence"):
        return _matches(predicate["a"], event)
    if pattern == "precedence":
        return _matches(predicate["b"], event)
    return any(_matches(sel, event) for sel in predicate["alphabet"])


def _absence(predicate: Mapping[str, Any]) -> _Outcome:
    # Monotone: the matching event itself is the violation, whatever else happened.
    return _VIOLATION, None


def _bounded_existence(
    predicate: Mapping[str, Any], history: Sequence[Mapping[str, Any]], gap: bool
) -> _Outcome:
    seen = sum(1 for e in history if _matches(predicate["a"], e))
    if (
        seen >= predicate["max_count"]
    ):  # this is the (N+1)-th: established from what we saw
        return _VIOLATION, None
    if gap:  # unseen matching events could already have used up the allowance
        return _UNDECIDABLE, C_INCOMPLETE
    return _OK, None


def _revocation_index(
    timeline: Sequence[Mapping[str, Any]], grant_index: int, upto: int, corr: str
) -> int | None:
    """Index of the first revocation of a grant before position ``upto`` (or ``None``).

    An ``authorization_revoked(ref)`` revokes every earlier ``authorization_granted``
    with the same ``ref`` and correlation key.
    """
    grant = timeline[grant_index]
    grant_key = _key(grant, corr)
    for index in range(grant_index + 1, upto):
        event = timeline[index]
        if (
            event.get("type") == "authorization_revoked"
            and event.get("ref") == grant.get("ref")
            and _key(event, corr) == grant_key
        ):
            return index
    return None


def _assign(
    predicate: Mapping[str, Any], timeline: Sequence[Mapping[str, Any]]
) -> tuple[set[int], dict[int, bool]]:
    """Assign grants to the ``b`` events of ``timeline`` in order (Definition 12.6).

    Returns the consumed grant positions (``single_use``) and, per ``b`` position,
    whether a legal authorizing grant was found.  A grant is legal for a ``b`` when it
    is an earlier matching ``a`` under the same correlation key and was not revoked
    before that ``b``.  Each ``b`` spends the eligible grant that is revoked soonest,
    which keeps later ``b`` events' options open, so a legal assignment is found
    whenever one exists.  A ``b`` that finds none takes nothing.
    """
    corr = predicate.get("correlate_by", "unit")
    reusable = predicate.get("mode", "single_use") == "reusable"
    a_sel, b_sel = predicate["a"], predicate["b"]
    revocable = a_sel.get("event") == "authorization_granted"
    end = len(timeline)

    def revoked_at(grant: int, before: int) -> int | None:
        return _revocation_index(timeline, grant, before, corr) if revocable else None

    consumed: set[int] = set()
    authorized: dict[int, bool] = {}
    for i, event in enumerate(timeline):
        if not _matches(b_sel, event):
            continue
        key = _key(event, corr)
        if key is None:
            authorized[i] = False  # matches nothing (section 6.3)
            continue
        eligible = [
            j
            for j in range(i)
            if _matches(a_sel, timeline[j])
            and _key(timeline[j], corr) == key
            and j not in consumed
            and revoked_at(j, i) is None
        ]
        authorized[i] = bool(eligible)
        if eligible and not reusable:

            def horizon(j: int) -> tuple[int, int]:
                r = revoked_at(j, end)
                return (end + 1 if r is None else r, j)

            consumed.add(min(eligible, key=horizon))
    return consumed, authorized


def _precedence(
    predicate: Mapping[str, Any],
    history: Sequence[Mapping[str, Any]],
    candidate: Mapping[str, Any],
) -> _Outcome:
    timeline = [*history, candidate]
    _consumed, authorized = _assign(predicate, timeline)
    return (_OK, None) if authorized[len(timeline) - 1] else (_VIOLATION, None)


def _precedence_limits_exceeded(
    predicate: Mapping[str, Any],
    history: Sequence[Mapping[str, Any]],
    candidate: Mapping[str, Any],
    limits: Mapping[str, int],
) -> bool:
    """True when a declared resource limit is exceeded.

    ``max_tracked_correlations`` bounds the distinct keys among the events the
    predicate looks at; ``max_active_obligations`` bounds the grants still pending
    (neither consumed nor revoked) once the history is accounted for.
    """
    corr = predicate.get("correlate_by", "unit")
    a_sel, b_sel = predicate["a"], predicate["b"]
    max_keys = limits.get("max_tracked_correlations")
    if max_keys is not None:
        keys = {
            _key(e, corr)
            for e in [*history, candidate]
            if _matches(a_sel, e) or _matches(b_sel, e)
        } - {None}
        if len(keys) > max_keys:
            return True
    max_active = limits.get("max_active_obligations")
    if max_active is not None:
        consumed, _ = _assign(predicate, history)
        revocable = a_sel.get("event") == "authorization_granted"
        pending = [
            j
            for j, e in enumerate(history)
            if _matches(a_sel, e)
            and j not in consumed
            and not (
                revocable
                and _revocation_index(history, j, len(history), corr) is not None
            )
        ]
        if len(pending) > max_active:
            return True
    return False


def _protocol(
    predicate: Mapping[str, Any],
    history: Sequence[Mapping[str, Any]],
    candidate: Mapping[str, Any],
) -> _Outcome:
    alphabet = predicate["alphabet"]
    transitions = predicate["transitions"]
    rejecting = set(predicate["rejecting"])
    ignore_unlisted = predicate.get("unlisted", "reject") == "ignore"
    sink = "\x00sink"
    state = predicate["start"]
    for event in [*history, candidate]:
        entry = next((s for s in alphabet if _matches(s, event)), None)
        if entry is None:
            continue  # outside the alphabet: not a transition
        if state == sink or state in rejecting:
            continue  # rejecting is absorbing (Definition 12.6)
        step = next(
            (t for t in transitions if t["from"] == state and t["when"] == entry), None
        )
        if step is not None:
            state = step["to"]
        elif not ignore_unlisted:
            state = sink
    return (_VIOLATION, None) if state == sink or state in rejecting else (_OK, None)


# -- the per-unit guard ------------------------------------------------------------------------


class UnitGuard:
    """Guard state for one unit: the events that took effect and the producers seen.

    ``among`` answers the population question per requirement name: a mapping
    ``{requirement: True | False | None}`` or one scalar for every requirement
    (``None`` = the checker has not answered).  A requirement without ``among`` is
    always in the population.  ``ordered_attested`` is the caller's attestation that
    events from several producers are in a known order.
    """

    def __init__(
        self,
        rules: Sequence[GuardRule],
        *,
        among: Mapping[str, bool | None] | bool | None = None,
        ordered_attested: bool = False,
        correlation_id: str | None = None,
    ) -> None:
        self._rules = tuple(rules)
        self._among = among
        self._ordered_attested = ordered_attested
        self.correlation_id = correlation_id
        self._history: list[dict[str, Any]] = []
        self._last: dict[str, int] = {}
        self._gap = False
        self._unordered = False
        self.halted = False
        self._halted_by: str | None = None
        self.unresolved = False
        self._counter: dict[str, int] = {}

    # -- bookkeeping ------------------------------------------------------------------
    def set_among(self, requirement: str, answer: bool | None) -> None:
        current = dict(self._among) if isinstance(self._among, Mapping) else {}
        if not isinstance(self._among, Mapping) and self._among is not None:
            current = {r.requirement: self._among for r in self._rules}
        current[requirement] = answer
        self._among = current

    def _answer(self, rule: GuardRule) -> bool | None:
        if rule.among is None:
            return True
        if isinstance(self._among, Mapping):
            return self._among.get(rule.requirement)
        return self._among

    def next_sequence(self, producer: str = "sdk") -> int:
        """The next sequence number for an event this process emits as ``producer``."""
        value = self._counter.get(producer, 0)
        self._counter[producer] = value + 1
        return value

    def _note(self, event: Mapping[str, Any]) -> None:
        producer, seq = event["producer"], event["sequence"]
        expected = self._last.get(producer, -1) + 1
        if seq > expected:
            self._gap = True
        elif seq < expected:
            self._unordered = True
        self._last[producer] = max(seq, self._last.get(producer, -1))

    def _order_cause(self, rule: GuardRule) -> str | None:
        """Why the full ordered history cannot be relied on, or ``None``."""
        if self._gap:
            return C_INCOMPLETE
        if rule.single_producer and len(self._last) > 1:
            return C_ORDER  # the unit contradicts its single_producer declaration
        if self._unordered:
            return C_ORDER
        if len(self._last) > 1 and not self._ordered_attested:
            return C_ORDER
        return None

    # -- the decision -----------------------------------------------------------------
    def _evaluate(self, rule: GuardRule, event: dict[str, Any]) -> GuardDecision:
        predicate = rule.predicate
        if not _relevant(predicate, event):
            return GuardDecision(ALLOW, R_NOT_RELEVANT, rule_id=rule.rule_id)
        answer = self._answer(rule)
        if answer is False:
            return GuardDecision(ALLOW, R_EXCLUDED, rule_id=rule.rule_id)
        outcome: _Outcome
        if answer is None:
            outcome = (_UNDECIDABLE, C_POPULATION)
        else:
            outcome = self._predicate_outcome(rule, event)
        kind, cause = outcome
        if kind == _OK:
            return GuardDecision(ALLOW, R_NO_VIOLATION, rule_id=rule.rule_id)
        acting = BLOCK if rule.action == "block_action" else HALT
        if kind == _VIOLATION:
            return GuardDecision(acting, R_VIOLATION, rule_id=rule.rule_id)
        if rule.on_check_error == "block":
            return GuardDecision(acting, R_CHECK_ERROR_BLOCK, cause, rule.rule_id)
        return GuardDecision(
            ALLOW, R_CHECK_ERROR_ALLOW, cause, rule.rule_id, unit_unresolved=True
        )

    def _predicate_outcome(self, rule: GuardRule, event: dict[str, Any]) -> _Outcome:
        predicate = rule.predicate
        pattern = predicate["pattern"]
        if pattern == "absence":
            return _absence(predicate)
        if pattern == "bounded_existence":
            return _bounded_existence(predicate, self._history, self._gap)
        cause = self._order_cause(rule)
        if cause is not None:
            return _UNDECIDABLE, cause
        if pattern == "precedence":
            corr = predicate.get("correlate_by", "unit")
            if _key(event, corr) is None:
                return _UNDECIDABLE, C_KEY
            if _precedence_limits_exceeded(
                predicate, self._history, event, rule.limits
            ):
                return _UNDECIDABLE, C_LIMIT
            return _precedence(predicate, self._history, event)
        return _protocol(predicate, self._history, event)

    def check(self, event: Mapping[str, Any]) -> GuardDecision:
        """Decide whether ``event`` may take effect; record it when it does."""
        ev = _event(event)
        if self.halted:
            return GuardDecision(HALT, R_UNIT_HALTED, rule_id=self._halted_by)
        self._note(ev)
        decisions = [self._evaluate(rule, ev) for rule in self._rules]
        final = _aggregate(decisions)
        if final.unit_unresolved:
            self.unresolved = True
        if final.verdict == ALLOW:
            self._history.append(ev)
        elif final.verdict == HALT:
            self.halted = True
            self._halted_by = final.rule_id
        return final

    def observe(self, event: Mapping[str, Any]) -> None:
        """Record an event that already happened (a result); nothing to suppress."""
        ev = _event(event)
        self._note(ev)
        self._history.append(ev)

    @contextmanager
    def correlated(self, correlation_id: str) -> Iterator[None]:
        """Stamp ``correlation_id`` on events the decorator builds inside the block."""
        previous = self.correlation_id
        self.correlation_id = correlation_id
        try:
            yield
        finally:
            self.correlation_id = previous


def _aggregate(decisions: Sequence[GuardDecision]) -> GuardDecision:
    if not decisions:
        return GuardDecision(ALLOW, R_NOT_RELEVANT)
    best = max(decisions, key=lambda d: (_RANK[d.verdict], _REASON_RANK[d.reason]))
    unresolved = any(d.unit_unresolved for d in decisions)
    if unresolved and not best.unit_unresolved:
        return GuardDecision(
            best.verdict, best.reason, best.cause, best.rule_id, unit_unresolved=True
        )
    return best


class LocalGuardSet:
    """The local guards of one compiled contract."""

    def __init__(self, rules: Sequence[GuardRule]) -> None:
        self.rules = tuple(rules)

    @classmethod
    def from_local(cls, local: Mapping[str, Any]) -> LocalGuardSet:
        """Build from a compiled ``tvl-runtime-local/1`` or ``/2`` artifact."""
        return cls(rules_from_local(local))

    def new_unit(self, **options: Any) -> UnitGuard:
        return UnitGuard(self.rules, **options)

    @contextmanager
    def unit(self, **options: Any) -> Iterator[UnitGuard]:
        """Run a block as one unit; :func:`guarded` functions inside it consult it."""
        unit = self.new_unit(**options)
        token = _CURRENT.set(unit)
        try:
            yield unit
        finally:
            _CURRENT.reset(token)


_CURRENT: contextvars.ContextVar[UnitGuard | None] = contextvars.ContextVar(
    "traigent_runtime_guard_unit", default=None
)


def current_unit() -> UnitGuard | None:
    return _CURRENT.get()


# -- decorator hook ----------------------------------------------------------------------------


def _guard_begin(name: str) -> tuple[UnitGuard, str]:
    unit = _CURRENT.get()
    if unit is None:
        raise RuntimeRequirementError(E_UNIT_MISSING, "unit")
    call_id = uuid.uuid4().hex
    decision = unit.check(
        {
            "type": "tool_call",
            "tool": name,
            "producer": "sdk",
            "sequence": unit.next_sequence(),
            "tool_call_id": call_id,
            **({"correlation_id": unit.correlation_id} if unit.correlation_id else {}),
        }
    )
    if decision.verdict == BLOCK:
        raise LocalGuardError(E_BLOCKED, decision)
    if decision.verdict == HALT:
        raise LocalGuardError(E_HALTED, decision)
    return unit, call_id


def _guard_finish(name: str, unit: UnitGuard, call_id: str, status: str) -> None:
    unit.observe(
        {
            "type": "tool_result",
            "tool": name,
            "status": status,
            "producer": "sdk",
            "sequence": unit.next_sequence(),
            "tool_call_id": call_id,
            **({"correlation_id": unit.correlation_id} if unit.correlation_id else {}),
        }
    )


def guarded(tool: str | None = None) -> Callable[[F], F]:
    """Guard a function as a ``tool_call`` of the current unit (sync or async).

    The call is offered to the unit's guards before it runs.  A blocked call raises
    :class:`LocalGuardError` (code ``local_guard_blocked``) and the function body does
    not run; a halting guard raises ``local_guard_halted`` and every later call in the
    unit does too.  Outside :meth:`LocalGuardSet.unit` the call is refused
    (``guard_unit_missing``): a guard never silently disappears.  ``tool`` must be a
    code identifier; it defaults to ``module.qualname`` when that is one.
    """

    def decorate(fn: F) -> F:
        name = tool or f"{fn.__module__}.{fn.__qualname__}"
        if len(name) > 128 or not _CODE_IDENT.fullmatch(name):
            raise RuntimeRequirementError(E_TOOL_NAME, "tool")

        if inspect.iscoroutinefunction(fn):

            @functools.wraps(fn)
            async def awrapper(*args: Any, **kwargs: Any) -> Any:
                unit, call_id = _guard_begin(name)
                try:
                    result = await fn(*args, **kwargs)
                except BaseException:
                    _guard_finish(name, unit, call_id, "failed")
                    raise
                _guard_finish(name, unit, call_id, "completed")
                return result

            return awrapper  # type: ignore[return-value]

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            unit, call_id = _guard_begin(name)
            try:
                result = fn(*args, **kwargs)
            except BaseException:
                _guard_finish(name, unit, call_id, "failed")
                raise
            _guard_finish(name, unit, call_id, "completed")
            return result

        return wrapper  # type: ignore[return-value]

    return decorate


__all__ = [
    "ALLOW",
    "BLOCK",
    "HALT",
    "GuardDecision",
    "GuardRule",
    "LocalGuardError",
    "LocalGuardSet",
    "UnitGuard",
    "current_unit",
    "guarded",
    "rules_from_local",
]
