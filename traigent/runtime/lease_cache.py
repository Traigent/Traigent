"""A local cache of one hosted reservation.

The Backend is the authority for who may start an effect and who still holds the
lease (a reservation is the lease; one start per occurrence).  This cache only lets a
restarted controller notice, without a round trip, that it already began an attempt, so
it does not even ask again.  It never grants anything: a cache hit blocks, a miss
permits nothing.  It is the existing occurrence ledger, keyed by the attempt ref.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from traigent.runtime import manifest
from traigent.runtime.model import E_OCCURRENCE_USED, RuntimeRequirementError

#: A started hosted attempt never expires locally: only the Backend ends a lease.
_NEVER_EXPIRES = timedelta(days=36500)


class LeaseCache:
    """Per-attempt ledger entry: ``begin`` -> ``started`` -> ``finished`` (or ``release``)."""

    def __init__(
        self,
        state_dir: Path,
        agent_ref: str,
        attempt_ref: str,
        effect: str,
        *,
        clock: Callable[[], datetime] = manifest.utcnow,
    ) -> None:
        self._state = state_dir
        self._agent = agent_ref
        self._clock = clock
        self._key = manifest.occurrence_key(attempt_ref, effect, 0)
        self._attempt_ref = attempt_ref
        self._begun = False

    def seen(self) -> dict[str, Any] | None:
        record: dict[str, Any] | None = manifest.read_occurrence(
            self._state, self._agent, self._key, now=self._clock(), lease=_NEVER_EXPIRES
        )
        return record

    def refuse_if_seen(self) -> None:
        """Raise ``occurrence_already_used`` when this controller already began the attempt."""
        if self.seen() is not None:
            raise RuntimeRequirementError(E_OCCURRENCE_USED)

    def begin(self) -> None:
        if not manifest.reserve_occurrence(
            self._state,
            self._agent,
            self._key,
            {"run_id": None, "attempt_ref": self._attempt_ref},
            clock=self._clock,
            lease=_NEVER_EXPIRES,
        ):
            raise RuntimeRequirementError(E_OCCURRENCE_USED)
        self._begun = True

    def update(self, **info: Any) -> None:
        manifest.update_occurrence(self._state, self._agent, self._key, info, attempt=1)

    def started(self) -> None:
        """The Backend accepted the start: from here the entry blocks forever."""
        self.update(run_id=f"hosted:{self._attempt_ref}", start_accepted=True)

    def release(self) -> None:
        """The start was never authorized: undo the entry (nothing ran)."""
        if self._begun:
            manifest.release_occurrence(self._state, self._agent, self._key, attempt=1)
