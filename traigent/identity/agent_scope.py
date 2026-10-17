"""Invocation-scoped agent scopes and the per-trace scope ledger.

``@traigent.optimize`` opens one :class:`AgentScope` per call. The scope lives in
a :mod:`contextvars` variable, so it follows sync code, ``await``, tasks that
copy the context, generators and async generators; it is reset with a token in
``finally`` so exceptions and cancellation restore the outer scope.

Each trace owns a :class:`ScopeLedger`. A scope registers itself when it OPENS
inside a trace and records its outcome when the invocation finishes. Scopes that
were already active when the trace started enclose the whole trace and count as
``enclosing`` (they cannot outlive it). At ``end_trace`` the ledger is SEALED and
decides whether the trace may carry a scope-derived ``execution_context.agent_id``;
it does only if the ledger is non-empty, no scope is still open, and every scope
is ``function_derived`` with the same key. Anything else leaves the field unset
("ineligible boundaries poison, never vanish").

Deliberate rule: an ``enclosing`` scope is NOT an ``open`` one.
It wraps the whole trace (``@optimize`` over ``@observe``, the common order), so
it is registered with its key and provenance and takes part in the unanimity
check instead of withholding forever: enclosing A alone -> A; enclosing A plus an
inner B -> withheld; enclosing ineligible -> withheld.

After sealing, any NEW scope registration (a deferred task, a late nested scope)
taints the ledger, and so does a poisoning: the final decision is then "no
identity". The client serializes the trace outside the ledger lock and only then
decides and stamps the detached payload, so no caller code runs under the lock
and a registration either lands before the decision (counted) or after it
(taints).

Nothing here carries a customer name: ``user_named`` and ``ineligible`` scopes
have ``key = None``.
"""

from __future__ import annotations

import contextvars
import inspect
import itertools
import threading
from collections.abc import AsyncIterator, Callable, Generator, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Literal

from traigent.identity.workload import AgentBinding
from traigent.utils.logging import get_logger

logger = get_logger(__name__)

ScopeState = Literal["open", "closed", "enclosing"]
AttributionReason = Literal["attributed", "conflict", "ineligible", "open_scope"]

_scope_ids = itertools.count(1)

_current_scope: contextvars.ContextVar[AgentScope | None] = contextvars.ContextVar(
    "traigent_agent_scope", default=None
)
_current_ledger: contextvars.ContextVar[ScopeLedger | None] = contextvars.ContextVar(
    "traigent_agent_scope_ledger", default=None
)

_anonymous_warned = False


@dataclass(frozen=True)
class AttributionDecision:
    key: str | None
    reason: AttributionReason


class ScopeLedger:
    """Registry of the agent scopes seen inside one trace."""

    def __init__(self) -> None:
        # Re-entrant: ``seal`` calls ``decide`` under it. Never held while
        # caller code (serialization, user callbacks) runs.
        self._lock = threading.RLock()
        # scope_id -> (key, provenance, state)
        self._records: dict[int, tuple[str | None, str, ScopeState]] = {}
        self._sealed: AttributionDecision | None = None
        self._tainted = False
        self._poisoned = False

    def finalize(self, stamp: Callable[[str | None], Any]) -> None:
        """Decide (sealing on the first call) and apply ``stamp(key)`` atomically.

        ``stamp`` must be a pure SDK function over a detached payload: it may
        not run caller code. A registration racing this call lands before it
        (counted) or after it (taints, so the next capture withdraws the id).
        """
        with self._lock:
            stamp(self.seal().key)

    def register_open(self, scope: AgentScope) -> None:
        with self._lock:
            if scope.scope_id not in self._records:
                if self._sealed is not None:
                    self._tainted = True
                self._records[scope.scope_id] = (
                    scope.binding.key,
                    scope.binding.provenance,
                    "open",
                )

    def register_ineligible_boundary(self) -> None:
        """Record a boundary whose scope could not be set up (never vanish)."""
        with self._lock:
            if self._sealed is not None:
                self._tainted = True
            self._records[-next(_scope_ids)] = (None, "ineligible", "closed")

    def poison(self) -> None:
        """Last resort when even recording a failed boundary failed.

        Taken under the lock; after sealing it acts as a taint, so a generated
        id already issued is withdrawn at the next capture.
        """
        with self._lock:
            self._poisoned = True

    def seal(self) -> AttributionDecision:
        """Decide once, at finalization; later registrations only taint."""
        with self._lock:
            if self._sealed is None:
                self._sealed = self.decide()
            if (self._tainted or self._poisoned) and self._sealed.key is not None:
                return AttributionDecision(None, "ineligible")
            return self._sealed

    def register_enclosing(self, scope: AgentScope | None) -> None:
        """Register the scope chain that was active when the trace started.

        These scopes are recorded as ``enclosing``, which is NOT ``open``: they
        join the unanimity check (key + provenance) and do not withhold the
        identity on their own. Treating them as open was deliberately not
        adopted, because ``@optimize`` over ``@observe`` would then never be
        attributed.
        """
        with self._lock:
            while scope is not None:
                self._records.setdefault(
                    scope.scope_id,
                    (scope.binding.key, scope.binding.provenance, "enclosing"),
                )
                scope = scope.parent

    def record_close(self, scope: AgentScope) -> None:
        with self._lock:
            key, provenance, state = self._records.get(
                scope.scope_id,
                (scope.binding.key, scope.binding.provenance, "open"),
            )
            if state == "open":
                self._records[scope.scope_id] = (key, provenance, "closed")

    def decide(self) -> AttributionDecision:
        """Reason precedence: ineligible, conflict, open_scope, attributed."""
        with self._lock:
            records = list(self._records.values())
            poisoned = self._poisoned
        if (
            poisoned
            or not records
            or any(p != "function_derived" for _, p, _ in records)
        ):
            return AttributionDecision(None, "ineligible")
        keys = {key for key, _, _ in records}
        if len(keys) > 1:
            return AttributionDecision(None, "conflict")
        if any(state == "open" for _, _, state in records):
            return AttributionDecision(None, "open_scope")
        (key,) = keys
        return AttributionDecision(key, "attributed")


def record_failed_boundary() -> None:
    """A scope could not be set up: poison the active trace, never let it vanish."""
    ledger = _current_ledger.get()
    if ledger is None:
        return
    try:
        ledger.register_ineligible_boundary()
    except Exception:
        ledger.poison()


def new_trace_ledger() -> ScopeLedger:
    """A ledger for a trace that is starting now (reads the ambient scope)."""
    ledger = ScopeLedger()
    ledger.register_enclosing(_current_scope.get())
    return ledger


def current_scope() -> AgentScope | None:
    return _current_scope.get()


def set_trace_ledger(
    ledger: ScopeLedger | None,
) -> contextvars.Token[ScopeLedger | None]:
    """Make ``ledger`` the registry for scopes opened in this context."""
    return _current_ledger.set(ledger)


def reset_trace_ledger(token: contextvars.Token[ScopeLedger | None]) -> None:
    _current_ledger.reset(token)


@dataclass(eq=False)
class AgentScope:
    """One invocation of an agent-declaring callable."""

    binding: AgentBinding
    parent: AgentScope | None = None
    scope_id: int = field(default_factory=lambda: next(_scope_ids))
    _ledgers: list[ScopeLedger] = field(default_factory=list, repr=False)

    @contextmanager
    def active(self) -> Iterator[None]:
        """Make this the current scope; register with the current trace ledger."""
        ledger = _current_ledger.get()
        if ledger is not None and not any(item is ledger for item in self._ledgers):
            self._ledgers.append(ledger)
            ledger.register_open(self)
        token = _current_scope.set(self)
        try:
            yield
        finally:
            _current_scope.reset(token)

    def finish(self) -> None:
        """Record that the invocation is over (idempotent)."""
        ledgers, self._ledgers = self._ledgers, []
        for ledger in ledgers:
            ledger.record_close(self)


def open_scope(binding: AgentBinding) -> AgentScope:
    """Create the scope for one call; warn once if it is anonymous."""
    global _anonymous_warned
    if binding.provenance == "ineligible" and not _anonymous_warned:
        _anonymous_warned = True
        logger.warning(
            "An @optimize function has an anonymous name (for example a lambda); "
            "traces that include it will not carry an agent id."
        )
    return AgentScope(binding=binding, parent=_current_scope.get())


def run_in_scope(scope: AgentScope, call: Callable[[], Any]) -> Any:
    """Run ``call`` inside ``scope``, keeping it open across async/generator bodies."""
    try:
        with scope.active():
            result = call()
    except BaseException:
        scope.finish()
        raise
    if inspect.iscoroutine(result):
        return _scoped_coroutine(scope, result)
    if inspect.isgenerator(result):
        return _scoped_generator(scope, result)
    if inspect.isasyncgen(result):
        return _scoped_async_generator(scope, result)
    scope.finish()
    return result


async def _scoped_coroutine(scope: AgentScope, coro: Any) -> Any:
    try:
        with scope.active():
            return await coro
    finally:
        scope.finish()


def _scoped_generator(scope: AgentScope, gen: Any) -> Generator[Any, Any, Any]:
    sent: Any = None
    thrown: BaseException | None = None
    try:
        while True:
            with scope.active():
                try:
                    if thrown is not None:
                        pending, thrown = thrown, None
                        item = gen.throw(pending)
                    else:
                        item = gen.send(sent)
                except StopIteration as stop:
                    return stop.value
            try:
                sent = yield item
            except GeneratorExit:
                with scope.active():
                    gen.close()
                raise
            except BaseException as exc:
                thrown = exc
                sent = None
    finally:
        scope.finish()


async def _scoped_async_generator(scope: AgentScope, agen: Any) -> AsyncIterator[Any]:
    sent: Any = None
    thrown: BaseException | None = None
    try:
        while True:
            with scope.active():
                try:
                    if thrown is not None:
                        pending, thrown = thrown, None
                        item = await agen.athrow(pending)
                    else:
                        item = await agen.asend(sent)
                except StopAsyncIteration:
                    return
            try:
                sent = yield item
            except GeneratorExit:
                with scope.active():
                    await agen.aclose()
                raise
            except BaseException as exc:
                thrown = exc
                sent = None
    finally:
        scope.finish()
