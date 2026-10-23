"""Customer-side deploy controller for ``deploy: if_promoted`` (TRAIG-101 plan section 8).

"Deployed" means the agent head was promoted to the candidate configuration.  The flow
for one deploy occurrence:

1. A person granted the DEPLOY attempt in the portal.  The grant binds the candidate
   fingerprint, the gate-result digest and the head generation read AT APPROVAL time;
   that generation, not the one captured when the optimization started, is the
   publish precondition.
2. Reserve, then have the Backend accept the START (``deploy_started``) before any
   write.  One start per occurrence; nothing here asks twice.
3. Publish with the occurrence-bound idempotency key
   ``deploy:{attempt}:{fingerprint}:{gate digest}``.  Only transport-level ambiguity
   is retried (same key, same bindings), at most ``max_attempts`` within ``window``.
   A lookup by key after each ambiguity can show the write already committed.
4. When retries are exhausted the outcome is still unknown: **finalize** (head-lock
   synchronized) returns the stored success or commits a cancellation tombstone.  Only a
   committed tombstone allows ``deploy_failed``; a finalization that cannot be reached
   is a HOLD (no outcome, no replacement publication).  A negative lookup never decides.
5. The terminal ``deploy_result`` fact is reported with the reservation generation and
   the publish-result key; the Backend accepts ``succeeded`` only against its own
   record, so a lost response can never become ``deploy_failed``.

The candidate configuration published here is customer content, stored in Traigent by
this deploy target (plan 8.4); every other field this module sends is a closed fact.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from traigent.runtime import manifest
from traigent.runtime.hosted import (
    PublishResult,
    ReactiveClient,
    ReactiveError,
    ReservationHandle,
    TokenMap,
    deploy_key,
)
from traigent.runtime.lease_cache import LeaseCache
from traigent.runtime.model import (
    E_APPROVAL_EXPIRED,
    E_APPROVAL_REQUIRED,
    E_BACKEND,
    E_HOLD,
    RuntimeRequirementError,
)
from traigent.runtime.respond import (
    CLOCK_SKEW_MARGIN,
    E_NO_APPROVED_ATTEMPT,
    RETRY_DELAY_S,
    _bounded,
    _parse_ts,
    _retryable,
)

LOGGER = logging.getLogger("traigent.runtime")

#: Transport-failure retry bound (plan 8.6): attempts, and the window they must fit in.
MAX_PUBLISH_ATTEMPTS = 3
RETRY_WINDOW = timedelta(minutes=15)
E_BINDING_MISMATCH = "deploy_binding_mismatch"

DEPLOYED = "deployed"
REFUSED = "refused"
FAILED = "failed"

#: Refusals that mean this key did NOT commit and the effect-time check failed.
_REFUSED_CODES = frozenset({"head_generation_stale", "grant_not_valid"})


@dataclass(frozen=True)
class DeployOutcome:
    """Terminal result of one deploy occurrence."""

    state: str  # deployed | refused | failed
    attempt_ref: str
    publish_result_key: str | None
    head_generation: int | None = None
    pinned_version: int | None = None
    #: The Backend recorded the ``deploy_result`` fact (else it must be reported again).
    result_recorded: bool = False


class DeployHold(RuntimeRequirementError):
    """No outcome may be recorded (ambiguity not resolved, or the lease is gone).

    ``outcome`` is set when the outcome IS decided but could not be reported yet.
    """

    def __init__(self, reason: str, outcome: DeployOutcome | None = None) -> None:
        super().__init__(E_HOLD, reason)
        self.outcome = outcome


class DeployController:
    """Run one ``deploy: if_promoted`` occurrence against the agent head."""

    def __init__(
        self,
        client: ReactiveClient,
        *,
        state_dir: Path | None = None,
        clock: Callable[[], datetime] = manifest.utcnow,
        sleep: Callable[[float], Awaitable[Any]] = asyncio.sleep,
        max_attempts: int = MAX_PUBLISH_ATTEMPTS,
        window: timedelta = RETRY_WINDOW,
        retry_delay: float = RETRY_DELAY_S,
        skew: timedelta = CLOCK_SKEW_MARGIN,
    ) -> None:
        self.client = client
        self.state = state_dir or manifest.default_state_dir()
        self.clock, self.sleep = clock, sleep
        self.max_attempts, self.window = max_attempts, window
        self.retry_delay, self.skew = retry_delay, skew

    # -- public ---------------------------------------------------------------------------
    async def run_promoted(
        self,
        attempt_ref: str,
        optimized: Any,
        *,
        config_id: str,
        configs: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> DeployOutcome:
        """Deploy the promoted candidate the deploy grant is bound to.

        ``optimized`` is the :class:`RespondResult` of the optimize attempt; its
        reported candidates supply the fingerprint and gate-result digest, so the caller
        passes neither.  The deploy attempt's grant says which one was approved: exactly
        one promoted candidate must match it, otherwise nothing is reserved.
        ``configs`` (default: the optimize result's own) maps fingerprint to the
        configuration, kept customer-side.
        """
        from traigent.runtime.promotion_evidence import candidate_fingerprint, promoted

        attempt = self.client.get_attempt(attempt_ref)
        if not attempt.get("grant_ref"):
            raise RuntimeRequirementError(E_NO_APPROVED_ATTEMPT)
        binding = self.client.get_grant(attempt["grant_ref"]).get("deploy")
        if not isinstance(binding, dict):
            raise RuntimeRequirementError(E_APPROVAL_REQUIRED, "grant")
        match = [
            c
            for c in promoted(getattr(optimized, "candidates", ()))
            if c["fingerprint"] == binding["candidate_fingerprint"]
            and c["gate_result_digest"] == binding["gate_result_digest"]
        ]
        configs = configs if configs is not None else getattr(optimized, "configs", {})
        config = configs.get(binding["candidate_fingerprint"])
        if len(match) != 1 or config is None:
            raise RuntimeRequirementError(E_BINDING_MISMATCH)
        if candidate_fingerprint(config) != match[0]["fingerprint"]:
            raise RuntimeRequirementError(E_BINDING_MISMATCH)
        return await self.run(
            attempt_ref,
            spec={
                "schema_version": "traigent.best_config.v1",
                "config_id": config_id,
                "config": dict(config),
            },
            candidate_fingerprint=match[0]["fingerprint"],
            gate_result_digest=match[0]["gate_result_digest"],
        )

    async def run(
        self,
        attempt_ref: str,
        *,
        spec: Mapping[str, Any],
        candidate_fingerprint: str,
        gate_result_digest: str,
    ) -> DeployOutcome:
        api = self.client
        attempt = api.get_attempt(attempt_ref)
        if attempt["effect"] != "deploy" or attempt["state"] != "approved":
            raise RuntimeRequirementError(E_NO_APPROVED_ATTEMPT)
        if not attempt.get("grant_ref"):
            raise RuntimeRequirementError(E_NO_APPROVED_ATTEMPT)
        grant = api.get_grant(attempt["grant_ref"])
        binding = grant.get("deploy")
        if (
            grant["state"] != "granted"
            or grant["effect"] != "deploy"
            or grant["attempt_ref"] != attempt_ref
            or not isinstance(binding, dict)
        ):
            raise RuntimeRequirementError(E_APPROVAL_REQUIRED, "grant")
        if "expires_at" in grant and _parse_ts(grant["expires_at"]) <= self.clock():
            raise RuntimeRequirementError(E_APPROVAL_EXPIRED)
        # The grant, not the caller, defines what may be published (nothing is sent if
        # the candidate in hand is not the approved one).
        if (
            binding["candidate_fingerprint"] != candidate_fingerprint
            or binding["gate_result_digest"] != gate_result_digest
        ):
            raise RuntimeRequirementError(E_BINDING_MISMATCH)
        key = deploy_key(attempt_ref, candidate_fingerprint, gate_result_digest)

        cache = LeaseCache(
            self.state, api.agent_ref, attempt_ref, "deploy", clock=self.clock
        )
        cache.refuse_if_seen()
        reservation = await _bounded(
            lambda k: api.reserve(attempt_ref, attempt["grant_ref"], key=k),
            sleep=self.sleep,
            delay=self.retry_delay,
        )
        cache.begin()
        try:
            receipt = await _bounded(
                lambda k: api.start(reservation, key=k),
                sleep=self.sleep,
                delay=self.retry_delay,
            )
        except ReactiveError as exc:
            if _retryable(exc):
                raise DeployHold("start_unconfirmed") from None
            cache.release()
            raise
        if receipt["fact"] != "deploy_started":
            raise RuntimeRequirementError(E_BACKEND, "response")
        cache.started()

        decided = await self._decide(
            reservation,
            key,
            binding["expected_head_generation"],
            grant["grant_ref"],
            spec,
        )
        outcome = await self._report(reservation, key, decided)
        cache.update(finished=True)
        return outcome

    # -- publish / resolve ----------------------------------------------------------------
    async def _decide(
        self,
        reservation: ReservationHandle,
        key: str,
        expected_generation: int,
        grant_ref: str,
        spec: Mapping[str, Any],
    ) -> DeployOutcome:
        """The terminal outcome of the write, or :class:`DeployHold` when it is unknown."""
        api = self.client
        attempt_ref = reservation.attempt_ref
        deadline = self.clock() + self.window
        attempts = 0
        while True:
            attempts += 1
            try:
                result = api.publish_deploy(
                    key=key,
                    reservation=reservation,
                    grant_ref=grant_ref,
                    expected_head_generation=expected_generation,
                    spec=spec,
                )
            except ReactiveError as exc:
                if exc.code in _REFUSED_CODES and not exc.ambiguous:
                    # Nothing committed for this key (a replay would have answered first).
                    return DeployOutcome(REFUSED, attempt_ref, None)
                if exc.code == "publication_cancelled":
                    return await self._finalize(attempt_ref, key)
                if not _retryable(exc):
                    raise DeployHold(exc.code) from None
                retry_after = exc.retry_after
            else:
                if result.state == "succeeded":
                    return _deployed(attempt_ref, result)
                retry_after = (
                    None  # an unexpected success shape is treated as ambiguity
                )
            # Ambiguity: the write may have committed.  A lookup can show it; its absence
            # decides nothing.
            seen = self._lookup(key)
            if seen is not None and seen.state == "succeeded":
                return _deployed(attempt_ref, seen)
            if seen is not None and seen.state == "cancelled":
                return await self._finalize(attempt_ref, key)
            delay = (
                float(retry_after)
                if retry_after
                else self.retry_delay * (2 ** (attempts - 1))
            )
            if (
                attempts >= self.max_attempts
                or self.clock() + timedelta(seconds=delay) > deadline
            ):
                return await self._finalize(attempt_ref, key)
            await self.sleep(delay)
            self._renew(reservation)

    def _lookup(self, key: str) -> PublishResult | None:
        try:
            return self.client.get_publish_result(key)
        except ReactiveError:
            return None  # unknown, never a decision

    def _renew(self, reservation: ReservationHandle) -> None:
        """Keep the lease across the retry window; a refusal means it is not ours any more."""
        try:
            self.client.heartbeat(reservation)
        except ReactiveError as exc:
            if not _retryable(exc):
                raise DeployHold("lease") from None

    async def _finalize(self, attempt_ref: str, key: str) -> DeployOutcome:
        """Resolve the unknown write under the head lock; only a tombstone allows failure."""
        for attempt in range(self.max_attempts):
            try:
                result = self.client.finalize_publish(key)
            except ReactiveError as exc:
                if not _retryable(exc):
                    raise DeployHold(exc.code) from None
                if attempt < self.max_attempts - 1:
                    await self.sleep(self.retry_delay)
                continue
            if result.state == "succeeded":
                return _deployed(attempt_ref, result)
            if result.state == "cancelled":
                return DeployOutcome(FAILED, attempt_ref, key)
            break  # pending: not terminal
        raise DeployHold("finalize_unavailable")

    # -- report ---------------------------------------------------------------------------
    async def _report(
        self, reservation: ReservationHandle, key: str, decided: DeployOutcome
    ) -> DeployOutcome:
        if decided.state == DEPLOYED:
            fields: dict[str, Any] = {
                "checks": "passed",
                "write": "succeeded",
                "retries_exhausted": False,
                "publish_result_key": key,
            }
        elif decided.state == FAILED:
            fields = {
                "checks": "passed",
                "write": "failed",
                "retries_exhausted": True,
                "publish_result_key": key,
            }
        else:
            fields = {
                "checks": "refused",
                "write": "not_attempted",
                "retries_exhausted": False,
            }
        try:
            receipt = await _bounded(
                lambda k: self.client.report_deploy_result(
                    reservation, key=k, **fields
                ),
                sleep=self.sleep,
                delay=self.retry_delay,
            )
        except ReactiveError as exc:
            if _retryable(exc):
                raise DeployHold("result_unreported", decided) from None
            raise
        return DeployOutcome(
            decided.state,
            decided.attempt_ref,
            decided.publish_result_key,
            decided.head_generation,
            decided.pinned_version,
            result_recorded=bool(receipt["recorded"]),
        )


def _deployed(attempt_ref: str, result: PublishResult) -> DeployOutcome:
    return DeployOutcome(
        DEPLOYED,
        attempt_ref,
        result.idempotency_key,
        result.head_generation,
        result.pinned_version,
    )


def controller_for(
    agent_ref: str, state_dir: Path | None = None, **kwargs: Any
) -> DeployController:
    """A controller using the production transport and the persisted token map."""
    state = state_dir or manifest.default_state_dir()
    return DeployController(
        ReactiveClient(agent_ref, TokenMap(state)), state_dir=state, **kwargs
    )
