"""``respond``: a human-approved fix for an open breach.

The response is a request to run the function's existing ``.optimize()`` under
a declared cost ceiling.  Nothing is deployed.  If the optimize run raises, the occurrence is marked failed
(spend is never undone) and a retry needs a new approval.  Every start is preceded by a
fresh, locally recorded human approval bound to the full-policy digest, rule,
breach, occurrence, attempt, cost ceiling and an expiry, and by a second round of
checks immediately before the run.  Any failed check holds: nothing runs.

Two paths.  With ``agent_ref`` the HOSTED path runs (``respond_hosted``):
a person grants the attempt in the portal, then this controller reserves it, has the
Backend accept the START, and only then runs ``.optimize()``; it heartbeats the
reservation while the run is in flight, reports the result with the reservation
generation, and never asks for a second start.  The local ledger is only a lease
cache there.  Without ``agent_ref`` the legacy local path below runs: occurrences are
reserved in a LOCAL ledger, so two controllers (separate state directories) answering
the same agent are not deduplicated.  Each start still needs its own human approval.
Occurrence numbers start at 0.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, cast

from traigent.runtime import manifest
from traigent.runtime.apply import matching_heads
from traigent.runtime.client import RuntimeClient
from traigent.runtime.hosted import (
    NS_RESPONSE,
    NS_RULE,
    ReactiveClient,
    ReactiveError,
    ReservationHandle,
    TokenMap,
)
from traigent.runtime.lease_cache import LeaseCache
from traigent.runtime.model import (
    E_APPROVAL_EXPIRED,
    E_APPROVAL_REQUIRED,
    E_COMPILER,
    E_FIX_CEILING,
    E_HOLD,
    E_NOT_APPLIED,
    E_NOT_CURRENT,
    E_OCCURRENCE_USED,
    RuntimeRequirement,
    RuntimeRequirementError,
    normalise_environment,
    positive_finite,
)

LOGGER = logging.getLogger("traigent.runtime")
RESTORE_CODE = "RESTORE_PRODUCTION_REQUIREMENT"
OCCURRENCE = 0  # the first firing is occurrence 0; one start per (breach, rule)

E_NO_RULE = "no_optimize_rule"
E_OPTIMIZE_FAILED = "optimize_failed"
E_NO_BREACH = "no_open_breach"
E_NOT_RESTORE = "next_step_not_restore"
E_RETIRED = "requirement_retired"
E_POLICY_CHANGED = "policy_changed"
E_AMBIGUOUS_BREACH = "ambiguous_breach"


class OccurrenceUsed(RuntimeRequirementError):
    """This (breach, rule, occurrence) was already started; carries the earlier run id."""

    def __init__(self, run_id: str | None = None) -> None:
        super().__init__(E_OCCURRENCE_USED)
        self.run_id = run_id


@dataclass(frozen=True)
class RespondResult:
    breach_id: str
    rule_id: str
    occurrence: int
    run_id: str | None
    cost_ceiling_usd: float
    #: Hosted path only: the attempt that ran and the generation that reported it.
    attempt_ref: str | None = None
    reservation_generation: int | None = None
    #: Hosted path only: the Backend recorded the result (else it must be reported later).
    result_recorded: bool = False
    #: Hosted path only: the closed candidate facts that were reported (no content).
    candidates: tuple[dict[str, Any], ...] = ()
    #: Hosted path only: fingerprint -> candidate configuration, for a deploy.  Customer
    #: content kept in memory on the customer side; never sent with these facts.
    configs: dict[str, Any] = field(default_factory=dict, repr=False)


def _optimize_rule(requirement: RuntimeRequirement) -> str:
    for rule in requirement.rules:
        # Only a rule triggered by an OPENED episode may authorize a fix.
        if rule.action == "optimize" and rule.event == "episode_opened":
            return str(rule.id)
    raise RuntimeRequirementError(E_NO_RULE)


def _open_breaches(
    rows: list[dict[str, Any]], requirement: RuntimeRequirement
) -> list[dict[str, Any]]:
    key = requirement.key()
    found = [
        row
        for row in rows
        if row.get("kind") == key["kind"]
        and row.get("latency_threshold_ms") == key["latency_threshold_ms"]
        and isinstance(row.get("environment"), str)
        and normalise_environment(row["environment"]) == key["environment"]
        and row.get("outcome") is None
        and row.get("closed_at") is None
        and isinstance(row.get("breach_id"), str)
    ]
    if not found:
        raise RuntimeRequirementError(E_NO_BREACH)
    return found


def _step_breach_ref(payload: dict[str, Any]) -> str | None:
    """The breach the next-step payload names (``basis.production_breach``), if any."""
    basis = payload.get("basis")
    breach = basis.get("production_breach") if isinstance(basis, dict) else None
    ref = breach.get("open_breach_ref") if isinstance(breach, dict) else None
    return ref if isinstance(ref, str) and ref else None


def _fresh_checks(
    requirement: RuntimeRequirement,
    api: RuntimeClient,
    *,
    project_id: str,
    agent_id: str,
    state: Path,
) -> dict[str, Any]:
    """Every precondition, read fresh; raises on the first one that fails (hold)."""
    record = manifest.read_manifest(state, agent_id)["requirements"].get(
        requirement.name
    )
    if not isinstance(record, dict) or not isinstance(record.get("receipt"), dict):
        raise RuntimeRequirementError(E_NOT_APPLIED)
    if record.get("full_policy_digest") != requirement.policy_digest():
        raise RuntimeRequirementError(E_POLICY_CHANGED)
    receipt = record["receipt"]
    heads = matching_heads(api.list_requirements(), requirement, project_id, agent_id)
    mine = [h for h in heads if h.get("head_id") == receipt.get("head_id")]
    if len(mine) != 1:
        raise RuntimeRequirementError(E_NOT_CURRENT, "head")
    if mine[0].get("status") != "active":
        raise RuntimeRequirementError(E_RETIRED)
    if mine[0].get("revision_id") != receipt.get("revision_id"):
        raise RuntimeRequirementError(E_NOT_CURRENT, "revision")
    rows, _more = api.list_breaches(agent_id)
    candidates = _open_breaches(rows, requirement)
    payload = api.get_next_step(agent_id)
    step = payload.get("step")
    code = step.get("code") if isinstance(step, dict) else None
    if code != RESTORE_CODE:
        raise RuntimeRequirementError(E_NOT_RESTORE)
    ref = _step_breach_ref(payload)
    if ref is not None:
        # The RESTORE step must be for the breach this call will fix: the candidate it names.
        for candidate in candidates:
            if candidate["breach_id"] == ref:
                return candidate
        raise RuntimeRequirementError(E_HOLD, "breach")
    if len(candidates) != 1:
        raise RuntimeRequirementError(E_AMBIGUOUS_BREACH)
    return candidates[0]


def _current_requirement(
    fn: Any, original: RuntimeRequirement
) -> RuntimeRequirement | None:
    """The requirement now declared on ``fn`` under the same name, if any."""
    for item in getattr(fn, "requirements", None) or ():
        if getattr(item, "name", None) == original.name:
            return item  # type: ignore[no-any-return]
    return None


def _text(value: Any) -> str | None:
    return value if isinstance(value, str) else None


def _run_id(result: Any) -> str | None:
    value = getattr(result, "optimization_id", None)
    if isinstance(value, str) and value and not value.startswith("unrestored-legacy:"):
        return value
    return None


def _validate_local_response(
    fn: Any,
    requirement: RuntimeRequirement,
    interactive: bool,
    max_fix_cost_usd: float | None,
) -> tuple[str, float]:
    errors = [i for i in requirement.validate() if i.severity == "error"]
    if errors:
        raise RuntimeRequirementError(errors[0].code, errors[0].path)
    rule_id = _optimize_rule(requirement)
    # Before any approval or reservation: a function without .optimize() consumes nothing.
    if not callable(getattr(fn, "optimize", None)):
        raise RuntimeRequirementError("function_not_optimizable")
    ceiling = (
        max_fix_cost_usd
        if max_fix_cost_usd is not None
        else requirement.max_fix_cost_usd
    )
    if ceiling is None or not positive_finite(ceiling):
        raise RuntimeRequirementError(E_FIX_CEILING, "max_fix_cost_usd")
    if not interactive:
        raise RuntimeRequirementError(E_APPROVAL_REQUIRED)
    return rule_id, ceiling


def _validate_local_approval(
    state: Path,
    agent_id: str,
    key: str,
    policy_digest: str,
    breach_id: str,
    rule_id: str,
    ceiling: float,
    attempt: int,
    clock: Callable[[], datetime],
    fn: Any,
    requirement: RuntimeRequirement,
    declared_ceiling: float | None,
) -> tuple[dict[str, Any], RuntimeRequirement]:
    approval = manifest.read_approval(state, agent_id, key)
    if approval is None or not manifest.approval_matches(
        approval,
        full_policy_digest=policy_digest,
        breach_id=breach_id,
        rule_id=rule_id,
        occurrence=OCCURRENCE,
        cost_ceiling_usd=float(ceiling),
        attempt=attempt,
    ):
        raise RuntimeRequirementError(E_HOLD, "approval")
    if manifest.approval_expired(approval, clock()):
        raise RuntimeRequirementError(E_APPROVAL_EXPIRED)
    # Re-read the CURRENT declaration: it may have changed while the person decided.
    current = _current_requirement(fn, requirement)
    if (
        current is None
        or current.policy_digest() != policy_digest
        or _optimize_rule(current) != rule_id
        or current.max_fix_cost_usd != declared_ceiling
    ):
        raise RuntimeRequirementError(E_POLICY_CHANGED)
    return approval, current


def _reserve_approved_occurrence(
    approval: dict[str, Any],
    state: Path,
    agent_id: str,
    key: str,
    breach_id: str,
    attempt: int,
    clock: Callable[[], datetime],
    occurrence_lease: timedelta,
) -> datetime:
    # Last moment: the second round of checks can take longer than the approval has left.
    if manifest.approval_expired(approval, clock()):
        raise RuntimeRequirementError(E_APPROVAL_EXPIRED)
    # Taken before the reservation, so the verified deadline can only be early, never late.
    reserved_at = clock()
    if not manifest.reserve_occurrence(
        state,
        agent_id,
        key,
        {"run_id": None, "breach_id": breach_id},
        attempt=attempt,
        clock=clock,
        lease=occurrence_lease,
    ):
        raise OccurrenceUsed()
    # Reserving may have waited on the ledger lock across the approval's expiry.
    if manifest.approval_expired(approval, clock()):
        # The fix never started: undo the reservation (no retry consumed).
        manifest.release_occurrence(state, agent_id, key, attempt=attempt)
        raise RuntimeRequirementError(E_APPROVAL_EXPIRED)

    return reserved_at


def _lost_after_wait(run: asyncio.Future[Any], own: _Ownership) -> bool:
    if not run.done():
        return True
    else:
        # A synchronous run can block the loop past the lease with no beat
        # having run: verify ownership once more, strictly (an error is a loss).
        return not own.beat(strict=True)


def _record_occurrence(
    state: Path, agent_id: str, key: str, info: dict[str, Any], attempt: int
) -> None:
    if not manifest.update_occurrence(state, agent_id, key, info, attempt=attempt):
        LOGGER.warning("stale_outcome_dropped")


def _check_earlier_occurrence(earlier: dict[str, Any] | None) -> None:
    if earlier is not None and earlier.get("status") != manifest.STATUS_FAILED:
        raise OccurrenceUsed(_text(earlier.get("run_id")))


async def _run_optimize(
    fn: Any,
    ceiling: Any,
    own: _Ownership,
    heartbeat: asyncio.Task[Any],
    record: Callable[[dict[str, Any]], None],
) -> Any:
    run: asyncio.Future[Any] | None = None
    lost = False
    try:
        try:
            run = asyncio.ensure_future(fn.optimize(cost_limit=float(ceiling)))
            # A heartbeat the ledger refuses ends that task: this controller no longer
            # owns the occurrence, so the run is cancelled and nothing is finished.
            await asyncio.wait({run, heartbeat}, return_when=asyncio.FIRST_COMPLETED)
            lost = _lost_after_wait(run, own)
        finally:
            # Both tasks are drained on every exit, so no exception goes unretrieved.
            await _drain(heartbeat, run)
        if lost:
            raise _LeaseLost
        result = run.result() if run is not None else None
    except _LeaseLost:
        raise RuntimeRequirementError(E_HOLD, "lease") from None
    except BaseException as exc:  # noqa: BLE001 - user code; its text is never surfaced
        # Spend may have occurred and is never undone.  The occurrence is marked
        # failed (kept in the ledger with its attempt count) so that an explicit
        # retry is possible, but only through a NEW human approval.  Cancellation
        # and interrupts mark it too, then propagate natively.
        record({"status": manifest.STATUS_FAILED, "run_id": None})
        if isinstance(exc, Exception):
            raise RuntimeRequirementError(E_OPTIMIZE_FAILED) from None
        raise
    return result


async def respond(
    fn: Any,
    requirement: RuntimeRequirement,
    *,
    project_id: str,
    agent_id: str,
    confirm: Callable[[dict[str, Any]], bool] | None = None,
    interactive: bool = False,
    max_fix_cost_usd: float | None = None,
    client: RuntimeClient | None = None,
    state_dir: Path | None = None,
    clock: Callable[[], datetime] = manifest.utcnow,
    occurrence_lease: timedelta = manifest.OCCURRENCE_LEASE,
    agent_ref: str | None = None,
    attempt_ref: str | None = None,
    reactive: ReactiveClient | None = None,
) -> RespondResult:
    """Ask a person, re-check, record the occurrence, then ``await fn.optimize(...)``.

    With ``agent_ref`` the hosted path runs instead (:func:`respond_hosted`): the
    approval is the human grant recorded by the Backend, not a local prompt.
    Otherwise one controller per agent: cross-controller deduplication is not
    enforced (see the module docstring).
    """
    if agent_ref is not None or reactive is not None:
        return await respond_hosted(
            fn,
            requirement,
            agent_ref=agent_ref or (reactive.agent_ref if reactive else ""),
            attempt_ref=attempt_ref,
            client=reactive,
            max_fix_cost_usd=max_fix_cost_usd,
            state_dir=state_dir,
            clock=clock,
        )
    if confirm is None:
        raise RuntimeRequirementError(E_APPROVAL_REQUIRED)
    rule_id, ceiling = _validate_local_response(
        fn, requirement, interactive, max_fix_cost_usd
    )
    state = state_dir or manifest.default_state_dir()
    api = client or RuntimeClient(project_id)

    breach = _fresh_checks(
        requirement, api, project_id=project_id, agent_id=agent_id, state=state
    )
    breach_id = str(breach["breach_id"])
    key = manifest.occurrence_key(breach_id, rule_id, OCCURRENCE)
    earlier = manifest.read_occurrence(
        state, agent_id, key, now=clock(), lease=occurrence_lease
    )
    _check_earlier_occurrence(earlier)
    # The approval is for THIS attempt: failures recorded so far, plus one.
    attempt = manifest.failed_attempts(earlier) + 1

    facts = {
        "requirement": requirement.name,
        "breach_id": breach_id,
        "opened_at": breach.get("opened_at"),
        "latency_threshold_ms": requirement.latency_threshold_ms,
        "environment": requirement.environment.strip(),
        "rule_id": rule_id,
        "occurrence": OCCURRENCE,
        "cost_ceiling_usd": ceiling,
    }
    # The ceiling is not part of the policy digest, so it is compared on its own.
    declared_ceiling = requirement.max_fix_cost_usd
    if not confirm(facts):
        raise RuntimeRequirementError(E_APPROVAL_REQUIRED)
    policy_digest = requirement.policy_digest()
    manifest.write_approval(
        state,
        agent_id,
        policy_digest=policy_digest,
        rule_id=rule_id,
        breach_id=breach_id,
        occurrence=OCCURRENCE,
        cost_ceiling_usd=float(ceiling),
        attempt=attempt,
        now=clock(),
    )

    # Immediately before starting: approval valid and bound, then every check again.
    approval, current = _validate_local_approval(
        state,
        agent_id,
        key,
        policy_digest,
        breach_id,
        rule_id,
        ceiling,
        attempt,
        clock,
        fn,
        requirement,
        declared_ceiling,
    )
    again = _fresh_checks(
        current, api, project_id=project_id, agent_id=agent_id, state=state
    )
    if again.get("breach_id") != breach_id:
        raise RuntimeRequirementError(E_HOLD, "breach")
    reserved_at = _reserve_approved_occurrence(
        approval, state, agent_id, key, breach_id, attempt, clock, occurrence_lease
    )

    def record(info: dict[str, Any]) -> None:
        _record_occurrence(state, agent_id, key, info, attempt)

    own = _Ownership(
        state, agent_id, key, attempt, clock, occurrence_lease, reserved_at
    )
    heartbeat = asyncio.create_task(_heartbeat(own))
    result = await _run_optimize(fn, ceiling, own, heartbeat, record)
    run_id = _run_id(result)
    record({"run_id": run_id, "finished": True})
    return RespondResult(breach_id, rule_id, OCCURRENCE, run_id, float(ceiling))


class _Ownership:
    """Whether this controller still owns its reservation, from the ledger's own word."""

    def __init__(
        self,
        state: Path,
        agent_id: str,
        key: str,
        attempt: int,
        clock: Callable[[], datetime],
        lease: timedelta,
        reserved_at: datetime,
    ) -> None:
        self.state, self.agent_id, self.key = state, agent_id, key
        self.attempt, self.clock, self.lease = attempt, clock, lease
        #: The last instant the ledger verifiably vouched for (stamp + lease).
        self.verified_until = reserved_at + lease

    def beat(self, *, strict: bool = False) -> bool:
        """One heartbeat; ``False`` when ownership is lost.

        A refused beat is a loss.  A beat that ERRORS (storage trouble) is retried only
        until the last verified deadline, then counts as a loss; ``strict`` (the final
        check before reporting success) treats any error as a loss at once.
        """
        try:
            stamp = manifest.heartbeat_occurrence(
                self.state,
                self.agent_id,
                self.key,
                attempt=self.attempt,
                lease=self.lease,
                clock=self.clock,
            )
        except (OSError, RuntimeRequirementError):
            return not strict and self.clock() < self.verified_until
        if stamp is None:
            return False
        self.verified_until = stamp + self.lease
        return True


async def _heartbeat(own: _Ownership) -> None:
    """Renew the heartbeat every lease/4; RETURNS (ending the task) once the lease is lost."""
    interval = max(own.lease.total_seconds() / 4, 0.001)
    while True:
        await asyncio.sleep(interval)
        if not own.beat():
            return


class _LeaseLost(Exception):
    """Internal signal: this controller no longer owns the occurrence."""


async def _drain(*tasks: asyncio.Future[Any] | None) -> None:
    """Cancel what is still running, then retrieve EVERY task's outcome (even done ones).

    One task failing to finish cleanly never skips the other's cleanup, and no
    exception is left unretrieved for the event loop to log.
    """
    live = [t for t in tasks if t is not None]
    for task in live:
        if not task.done():
            task.cancel()
    await asyncio.gather(*live, return_exceptions=True)


# --- hosted path : reserve -> start -> run -> report -> heartbeat ---------------

E_NO_APPROVED_ATTEMPT = "no_approved_attempt"
E_ATTEMPT_MISMATCH = "attempt_mismatch"
E_AMBIGUOUS_ATTEMPT = "ambiguous_attempt"
#: Longest wait between lease renewals while the optimize run is in flight.
HEARTBEAT_INTERVAL = timedelta(seconds=60)
#: Server and local clocks may differ; the local lease deadline is the server's minus this.
CLOCK_SKEW_MARGIN = timedelta(seconds=5)
#: Transport-ambiguity retries (same idempotency key, so a retry is a replay, never a new act).
RETRY_BOUND = 3
RETRY_DELAY_S = 1.0
_ATTEMPT_PAGES = 5


def _parse_ts(text: str) -> datetime:
    value = datetime.fromisoformat(text.replace("Z", "+00:00"))
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value


def _retryable(exc: ReactiveError) -> bool:
    return bool(exc.ambiguous or exc.code == "rate_limited")


async def _bounded(
    call: Callable[[str], Any],
    *,
    sleep: Callable[[float], Any],
    delay: float,
) -> Any:
    """Call with ONE idempotency key, retrying only ambiguity/rate limits (a replay)."""
    from traigent.runtime.hosted import new_key

    key = new_key()
    for attempt in range(RETRY_BOUND):
        try:
            return call(key)
        except ReactiveError as exc:
            if not _retryable(exc) or attempt == RETRY_BOUND - 1:
                raise
            await sleep(float(exc.retry_after) if exc.retry_after else delay)
    raise AssertionError("unreachable")  # pragma: no cover


class _HostedLease:
    """The reservation is the lease; this renews it and says whether it is still ours."""

    def __init__(
        self,
        client: ReactiveClient,
        reservation: ReservationHandle,
        *,
        clock: Callable[[], datetime],
        skew: timedelta,
        lease_expires_at: str,
    ) -> None:
        self.client, self.reservation, self.clock, self.skew = (
            client,
            reservation,
            clock,
            skew,
        )
        #: The last instant the Backend verifiably vouched for (its stamp minus skew):
        #: the post-start lease from the start receipt, never the pre-start reservation expiry.
        self.verified_until = _parse_ts(lease_expires_at) - skew
        self.lost = False

    def beat(self) -> bool:
        """One renewal; ``False`` once the lease is gone.

        A refusal is a loss.  An ambiguous failure (or a 429) is tolerated only until
        the last deadline the Backend vouched for.
        """
        try:
            doc = self.client.heartbeat(self.reservation)
        except ReactiveError as exc:
            if _retryable(exc):
                self.lost = not self.clock() < self.verified_until
            else:
                self.lost = True
            return not self.lost
        self.verified_until = _parse_ts(doc["lease_expires_at"]) - self.skew
        return True


async def _heartbeat_loop(lease: _HostedLease, interval: float) -> None:
    """Renew every ``interval``; RETURNS (ending the task) once the lease is lost."""
    while True:
        await asyncio.sleep(interval)
        if not await asyncio.to_thread(lease.beat):
            return


def _select_attempt(
    client: ReactiveClient,
    tokens: TokenMap,
    requirement: RuntimeRequirement,
    rule_id: str,
    attempt_ref: str | None,
) -> dict[str, Any]:
    """The approved optimize attempt of this requirement's rule (never guessed)."""
    response = tokens.hosted(NS_RESPONSE, requirement.ladder_name)
    rule = tokens.hosted(NS_RULE, rule_id)

    def mine(item: dict[str, Any]) -> bool:
        return bool(
            item["response"] == response
            and item["rule"] == rule
            and item["effect"] == "optimize"
        )

    if attempt_ref is not None:
        item = client.get_attempt(attempt_ref)
        if not mine(item):
            raise RuntimeRequirementError(E_ATTEMPT_MISMATCH)
        found = [item]
    else:
        found = []
        after: int | None = None
        for _ in range(_ATTEMPT_PAGES):
            page = client.list_attempts(limit=100, after=after)
            found.extend(
                i for i in page["items"] if mine(i) and i["state"] == "approved"
            )
            after = page["next_after"]
            if after is None:
                break
        if len(found) > 1:
            raise RuntimeRequirementError(E_AMBIGUOUS_ATTEMPT)
    if not found or found[0]["state"] != "approved" or not found[0].get("grant_ref"):
        raise RuntimeRequirementError(E_NO_APPROVED_ATTEMPT)
    chosen: dict[str, Any] = found[0]
    return chosen


def _ceilings(
    grant: dict[str, Any], requirement: RuntimeRequirement, override: float | None
) -> tuple[float, int]:
    """Cost (USD) and trial ceilings: the grant's, tightened (never loosened) locally."""
    ceiling = grant.get("ceiling")
    if not isinstance(ceiling, dict):
        raise RuntimeRequirementError(E_FIX_CEILING, "grant")
    usd = ceiling["max_cost_micro_usd"] / 1_000_000
    for local in (override, requirement.max_fix_cost_usd):
        if local is not None and positive_finite(local):
            usd = min(usd, float(local))
    if not positive_finite(usd):
        raise RuntimeRequirementError(E_FIX_CEILING, "grant")
    return usd, int(ceiling["max_trials"])


def _completion(result: Any) -> str:
    return (
        "cost_limit"
        if getattr(result, "stop_reason", None) == "cost_limit"
        else "normal"
    )


def _candidates(
    result: Any,
    extract: Callable[[Any, dict[str, Any]], Any] | None,
    grant: dict[str, Any],
    *,
    requirement: RuntimeRequirement,
    tokens: TokenMap,
) -> tuple[list[Any], dict[str, Any]]:
    """Closed candidate facts and the configurations behind them (customer side only).

    From ``extract`` when given.  Otherwise, a result that carries the optimizer's
    ``promotion_decisions`` (an ``OptimizationResult``) is turned into promotion
    evidence by :func:`evidence_from_result`.  Anything else reports no candidate
    (``optimize_no_candidate``), never a guessed ``Promote``.
    """
    configs: dict[str, Any] = {}
    if extract is not None:
        raw = extract(result, grant)
        configs = dict(getattr(extract, "configs", {}) or {})
    elif hasattr(result, "promotion_decisions"):
        from traigent.runtime.promotion_evidence import (
            candidate_extractor,
            evidence_from_result,
        )

        try:
            reporter = candidate_extractor(
                requirement,
                tokens,
                lambda res: evidence_from_result(requirement, res),
            )
            raw = reporter(result, grant)
            configs = dict(reporter.configs)
        except RuntimeRequirementError as exc:
            if exc.code != E_COMPILER:
                raise
            raw = ()  # no TVL evaluator: nothing can be established, nothing promotes
    else:
        raw = getattr(result, "reactive_candidates", ())
    return (list(raw) if isinstance(raw, (list, tuple)) else []), configs


def _validate_hosted_grant(
    grant: dict[str, Any], attempt: dict[str, Any], clock: Callable[[], datetime]
) -> None:
    if (
        grant["state"] != "granted"
        or grant["effect"] != "optimize"
        or grant["attempt_ref"] != attempt["attempt_ref"]
    ):
        raise RuntimeRequirementError(E_APPROVAL_REQUIRED, "grant")
    if "expires_at" in grant and _parse_ts(grant["expires_at"]) <= clock():
        raise RuntimeRequirementError(E_APPROVAL_EXPIRED)


def _validate_hosted_response(fn: Any, requirement: RuntimeRequirement) -> str:
    errors = [i for i in requirement.validate() if i.severity == "error"]
    if errors:
        raise RuntimeRequirementError(errors[0].code, errors[0].path)
    rule_id = _optimize_rule(requirement)
    if not callable(getattr(fn, "optimize", None)):
        raise RuntimeRequirementError("function_not_optimizable")
    return rule_id


async def _start_hosted_attempt(
    api: ReactiveClient,
    reservation: ReservationHandle,
    cache: LeaseCache,
    sleep: Callable[[float], Any],
    retry_delay: float,
) -> dict[str, Any]:
    try:
        started = await _bounded(
            lambda key: api.start(reservation, key=key), sleep=sleep, delay=retry_delay
        )
    except ReactiveError as exc:
        if _retryable(exc):
            # The start may have been accepted: keep the cache entry, never ask again.
            raise RuntimeRequirementError(E_HOLD, "start_unconfirmed") from None
        cache.release()
        raise
    return cast(dict[str, Any], started)


async def _report_hosted_completion(
    api: ReactiveClient,
    reservation: ReservationHandle,
    result: Any,
    reported: list[Any],
    cache: LeaseCache,
    sleep: Callable[[float], Any],
    retry_delay: float,
) -> dict[str, Any]:
    try:
        receipt = await _bounded(
            lambda key: api.report_finished(
                reservation,
                completion=_completion(result),
                candidates=reported,
                key=key,
            ),
            sleep=sleep,
            delay=retry_delay,
        )
    except ReactiveError as exc:
        cache.update(finished=True, unreported=True)
        if _retryable(exc):
            raise RuntimeRequirementError(E_HOLD, "result_unreported") from None
        raise
    return cast(dict[str, Any], receipt)


async def respond_hosted(
    fn: Any,
    requirement: RuntimeRequirement,
    *,
    agent_ref: str,
    attempt_ref: str | None = None,
    client: ReactiveClient | None = None,
    max_fix_cost_usd: float | None = None,
    state_dir: Path | None = None,
    clock: Callable[[], datetime] = manifest.utcnow,
    heartbeat_interval: timedelta = HEARTBEAT_INTERVAL,
    skew: timedelta = CLOCK_SKEW_MARGIN,
    retry_delay: float = RETRY_DELAY_S,
    sleep: Callable[[float], Any] = asyncio.sleep,
    candidates: Callable[[Any, dict[str, Any]], Any] | None = None,
) -> RespondResult:
    """Run the existing ``.optimize()`` for an attempt a person granted in the portal.

    Order: find the approved attempt -> read its grant -> reserve -> START (the Backend
    accepts before anything runs) -> ``fn.optimize`` while a heartbeat renews the
    reservation -> report the result with the reservation generation.  A second start
    is never requested: a start that cannot be confirmed holds, and the Backend lapses
    the attempt for a person to cancel.  A lost lease cancels the run.  Nothing here
    sends a prompt, an output or any text: only closed facts.
    """
    rule_id = _validate_hosted_response(fn, requirement)
    state = state_dir or manifest.default_state_dir()
    api = client or ReactiveClient(agent_ref, TokenMap(state))
    # The token map is read first: unavailable means nothing is asked or reserved.
    attempt = _select_attempt(api, api.tokens, requirement, rule_id, attempt_ref)
    grant = api.get_grant(attempt["grant_ref"])
    _validate_hosted_grant(grant, attempt, clock)
    usd, trials = _ceilings(grant, requirement, max_fix_cost_usd)

    cache = LeaseCache(
        state, agent_ref, attempt["attempt_ref"], "optimize", clock=clock
    )
    cache.refuse_if_seen()
    reservation = await _bounded(
        lambda key: api.reserve(attempt["attempt_ref"], attempt["grant_ref"], key=key),
        sleep=sleep,
        delay=retry_delay,
    )
    cache.begin()
    started = await _start_hosted_attempt(api, reservation, cache, sleep, retry_delay)
    cache.started()

    lease = _HostedLease(
        api,
        reservation,
        clock=clock,
        skew=skew,
        lease_expires_at=started["lease_expires_at"],
    )
    if not clock() < lease.verified_until:
        # The vouched deadline has already passed: no work under a lease we cannot rely on.
        cache.update(lease_lost=True)
        raise RuntimeRequirementError(E_HOLD, "lease")
    beat = asyncio.create_task(
        _heartbeat_loop(lease, heartbeat_interval.total_seconds())
    )
    run: asyncio.Future[Any] | None = None
    try:
        try:
            run = asyncio.ensure_future(fn.optimize(cost_limit=usd, max_trials=trials))
            await asyncio.wait({run, beat}, return_when=asyncio.FIRST_COMPLETED)
        finally:
            lost = lease.lost or not (run is not None and run.done())
            await _drain(beat, run)
        if lost:
            cache.update(lease_lost=True)
            raise _LeaseLost
        result = run.result() if run is not None else None
    except _LeaseLost:
        raise RuntimeRequirementError(E_HOLD, "lease") from None
    except BaseException as exc:  # noqa: BLE001 - user code; its text is never surfaced
        _report_best_effort(
            api, reservation, "error" if isinstance(exc, Exception) else "cancelled"
        )
        cache.update(finished=True, failed=True)
        if isinstance(exc, Exception):
            raise RuntimeRequirementError(E_OPTIMIZE_FAILED) from None
        raise

    # Once: retries re-send the same facts.
    reported, configs = _candidates(
        result, candidates, grant, requirement=requirement, tokens=api.tokens
    )
    receipt = await _report_hosted_completion(
        api, reservation, result, reported, cache, sleep, retry_delay
    )
    cache.update(finished=True)
    return RespondResult(
        str(attempt["scope"]),
        rule_id,
        int(attempt["occurrence"]),
        _run_id(result),
        usd,
        attempt_ref=attempt["attempt_ref"],
        reservation_generation=reservation.generation,
        result_recorded=bool(receipt["recorded"]),
        candidates=tuple(reported),
        configs=configs,
    )


def _report_best_effort(
    api: ReactiveClient, reservation: ReservationHandle, completion: str
) -> None:
    """Tell the Backend the run ended without a result; a failure here changes nothing."""
    try:
        api.report_finished(reservation, completion=completion, candidates=())
    except (ReactiveError, RuntimeRequirementError):
        LOGGER.warning("unreported_failure")
