"""``respond``: a human-approved fix for an open breach (design section 3.2).

The response is a request to run the function's existing ``.optimize()`` under
a declared cost ceiling.  Nothing is deployed.  If the optimize run raises, the occurrence is marked failed
(spend is never undone) and a retry needs a new approval.  Every start is preceded by a
fresh, locally recorded human approval bound to the full-policy digest, rule,
breach, occurrence, attempt, cost ceiling and an expiry, and by a second round of
checks immediately before the run.  Any failed check holds: nothing runs.

One controller per agent: occurrences are reserved in a LOCAL ledger, so two
controllers (separate state directories) answering the same agent are not
deduplicated until the backend reserves occurrences.  Each start still needs
its own human approval.  Occurrence numbers start at 0 (TVL RFC 0003, 7.4).
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from traigent.runtime import manifest
from traigent.runtime.apply import matching_heads
from traigent.runtime.client import RuntimeClient
from traigent.runtime.model import (
    E_APPROVAL_EXPIRED,
    E_APPROVAL_REQUIRED,
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
OCCURRENCE = (
    0  # RFC 0003 7.4: the first firing is occurrence 0; one start per (breach, rule)
)

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


def _optimize_rule(requirement: RuntimeRequirement) -> str:
    for rule in requirement.rules:
        # Only a rule triggered by an OPENED episode may authorize a fix.
        if rule.action == "optimize" and rule.event == "episode_opened":
            return rule.id
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
        # The RESTORE step must be for the breach this call will fix.
        if ref != candidates[0]["breach_id"]:
            raise RuntimeRequirementError(E_HOLD, "breach")
    elif len(candidates) != 1:
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


async def respond(
    fn: Any,
    requirement: RuntimeRequirement,
    *,
    project_id: str,
    agent_id: str,
    confirm: Callable[[dict[str, Any]], bool],
    interactive: bool,
    max_fix_cost_usd: float | None = None,
    client: RuntimeClient | None = None,
    state_dir: Path | None = None,
    clock: Callable[[], datetime] = manifest.utcnow,
    occurrence_lease: timedelta = manifest.OCCURRENCE_LEASE,
) -> RespondResult:
    """Ask a person, re-check, record the occurrence, then ``await fn.optimize(...)``.

    One controller per agent: cross-controller deduplication is not enforced
    until the backend reserves occurrences (see the module docstring).
    """
    errors = [i for i in requirement.validate() if i.severity == "error"]
    if errors:
        raise RuntimeRequirementError(errors[0].code, errors[0].path)
    rule_id = _optimize_rule(requirement)
    ceiling = (
        max_fix_cost_usd
        if max_fix_cost_usd is not None
        else requirement.max_fix_cost_usd
    )
    if ceiling is None or not positive_finite(ceiling):
        raise RuntimeRequirementError(E_FIX_CEILING, "max_fix_cost_usd")
    if not interactive:
        raise RuntimeRequirementError(E_APPROVAL_REQUIRED)
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
    if earlier is not None and earlier.get("status") != manifest.STATUS_FAILED:
        raise OccurrenceUsed(_text(earlier.get("run_id")))
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
    again = _fresh_checks(
        current, api, project_id=project_id, agent_id=agent_id, state=state
    )
    if again.get("breach_id") != breach_id:
        raise RuntimeRequirementError(E_HOLD, "breach")
    # Last moment: the second round of checks can take longer than the approval has left.
    if manifest.approval_expired(approval, clock()):
        raise RuntimeRequirementError(E_APPROVAL_EXPIRED)
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

    def record(info: dict[str, Any]) -> None:
        if not manifest.update_occurrence(state, agent_id, key, info, attempt=attempt):
            LOGGER.warning("stale_outcome_dropped")

    heartbeat = asyncio.create_task(
        _heartbeat(state, agent_id, key, attempt, clock, occurrence_lease)
    )
    try:
        try:
            result = await fn.optimize(cost_limit=float(ceiling))
        finally:
            # Also on cancellation: no heartbeat may outlive the run.
            await _stop(heartbeat)
    except Exception:  # noqa: BLE001 - user code; its text is never surfaced
        # Spend may have occurred and is never undone.  The occurrence is marked
        # failed (kept in the ledger with its attempt count) so that an explicit
        # retry is possible, but only through a NEW human approval.
        record({"status": manifest.STATUS_FAILED, "run_id": None})
        raise RuntimeRequirementError(E_OPTIMIZE_FAILED) from None
    run_id = _run_id(result)
    record({"run_id": run_id, "finished": True})
    return RespondResult(breach_id, rule_id, OCCURRENCE, run_id, float(ceiling))


async def _heartbeat(
    state: Path,
    agent_id: str,
    key: str,
    attempt: int,
    clock: Callable[[], datetime],
    lease: timedelta,
) -> None:
    """Renew the reservation's heartbeat every lease/4 while the run is awaited."""
    interval = max(lease.total_seconds() / 4, 0.001)
    while True:
        await asyncio.sleep(interval)
        try:
            fenced_in = manifest.update_occurrence(
                state,
                agent_id,
                key,
                {"heartbeat_at": clock().isoformat()},
                attempt=attempt,
            )
        except (OSError, RuntimeRequirementError):
            continue  # a missed beat only shortens the lease; the next one retries
        if not fenced_in:
            return


async def _stop(task: asyncio.Task[None]) -> None:
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task
