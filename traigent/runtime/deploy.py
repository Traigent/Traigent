"""Customer-side deploy controller for ``deploy: if_promoted``.

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
this deploy target; every other field this module sends is a closed fact.
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
    E_BACKEND,
    E_HOLD,
    RuntimeRequirementError,
)
from traigent.runtime.respond import (
    CLOCK_SKEW_MARGIN,
    RETRY_DELAY_S,
    _bounded,
    _retryable,
)

LOGGER = logging.getLogger("traigent.runtime")

#: Transport-failure retry bound: attempts, and the window they must fit in.
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


def _publish_error_action(exc: ReactiveError) -> tuple[str, float | None]:
    if exc.code in _REFUSED_CODES and not exc.ambiguous:
        return "refused", None
    if exc.code == "publication_cancelled":
        return "cancelled", None
    if not _retryable(exc):
        raise DeployHold(exc.code) from None
    return "retry", exc.retry_after


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
        grant_ref: str,
        best_config_id: str,
        version: int,
        configs: Mapping[str, Mapping[str, Any]] | None = None,
        candidate_fingerprint: str | None = None,
    ) -> DeployOutcome:
        """Deploy the one promoted candidate of ``optimized`` under the approved grant.

        The backend exposes no attempt or grant read, so the caller names the approved
        ``grant_ref`` and the server-issued ``best_config_id``/``version`` to publish.
        ``candidate_fingerprint`` picks among several promoted candidates; the backend
        refuses a key whose fingerprint and gate digest the grant does not bind.
        """
        from traigent.runtime.promotion_evidence import candidate_fingerprint as fp_of
        from traigent.runtime.promotion_evidence import promoted

        match = [
            c
            for c in promoted(getattr(optimized, "candidates", ()))
            if candidate_fingerprint is None
            or c["fingerprint"] == candidate_fingerprint
        ]
        configs = configs if configs is not None else getattr(optimized, "configs", {})
        if len(match) != 1:
            raise RuntimeRequirementError(E_BINDING_MISMATCH)
        config = configs.get(match[0]["fingerprint"])
        if config is None or fp_of(config) != match[0]["fingerprint"]:
            raise RuntimeRequirementError(E_BINDING_MISMATCH)
        return await self.run(
            attempt_ref,
            grant_ref=grant_ref,
            best_config_id=best_config_id,
            version=version,
            candidate_fingerprint=match[0]["fingerprint"],
            gate_result_digest=match[0]["gate_result_digest"],
        )

    async def run(
        self,
        attempt_ref: str,
        *,
        grant_ref: str,
        best_config_id: str,
        version: int,
        candidate_fingerprint: str,
        gate_result_digest: str,
    ) -> DeployOutcome:
        api = self.client
        key = deploy_key(attempt_ref, candidate_fingerprint, gate_result_digest)

        cache = LeaseCache(
            self.state, api.agent_ref, attempt_ref, "deploy", clock=self.clock
        )
        cache.refuse_if_seen()
        reservation = await _bounded(
            lambda k: api.reserve(attempt_ref, grant_ref, key=k),
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

        decided = await self._decide(reservation, key, best_config_id, version)
        outcome = await self._report(reservation, key, decided)
        cache.update(finished=True)
        return outcome

    # -- publish / resolve ----------------------------------------------------------------
    def _retry_wait(self, retry_after: float | None, attempts: int) -> float:
        return (
            float(retry_after)
            if retry_after
            else self.retry_delay * (2 ** (attempts - 1))
        )

    def _publish_once(
        self, attempt_ref: str, key: str, best_config_id: str, version: int
    ) -> tuple[DeployOutcome | None, bool, float | None]:
        """One write attempt: ``(terminal outcome, cancelled, retry_after)``."""
        try:
            result = self.client.publish_deploy(
                key=key, best_config_id=best_config_id, version=version
            )
        except ReactiveError as exc:
            action, retry_after = _publish_error_action(exc)
            if action == "refused":
                return DeployOutcome(REFUSED, attempt_ref, None), False, None
            return None, action == "cancelled", retry_after
        if result.state == "succeeded":
            return _deployed(attempt_ref, result), False, None
        # an unexpected success shape is treated as ambiguity
        return None, False, None

    async def _decide(
        self,
        reservation: ReservationHandle,
        key: str,
        best_config_id: str,
        version: int,
    ) -> DeployOutcome:
        """The terminal outcome of the write, or :class:`DeployHold` when it is unknown."""
        attempt_ref = reservation.attempt_ref
        deadline = self.clock() + self.window
        attempts = 0
        while True:
            attempts += 1
            outcome, cancelled, retry_after = self._publish_once(
                attempt_ref, key, best_config_id, version
            )
            if outcome is not None:
                return outcome
            if cancelled:
                return await self._finalize(attempt_ref, key)
            # Ambiguity: the write may have committed.  The same key replays the stored
            # result, so retry it; only finalize (under the head lock) decides failure.
            delay = DeployController._retry_wait(self, retry_after, attempts)
            if self._retries_exhausted(attempts, delay, deadline):
                return await self._finalize(attempt_ref, key)
            await self.sleep(delay)
            self._renew(reservation)

    def _retries_exhausted(
        self, attempts: int, delay: float, deadline: datetime
    ) -> bool:
        return (
            attempts >= self.max_attempts
            or self.clock() + timedelta(seconds=delay) > deadline
        )

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
