"""Public contracts for the local cold-start eval-set executor.

``build_cold_start_eval_set()`` calls a CONTENT-FREE backend planning
endpoint (only coarse type shape ever leaves the client) and then does all
candidate generation and scoring LOCALLY, using a generator and a
``LocalVerifier`` the CALLER supplies. The SDK ships no generation technique
and no verification logic of its own -- that is deliberately the customer's
IP, not ours.

Honesty properties these types exist to hold onto:

* A backend cold-start plan is NOT cryptographically signed and NOT
  independently verifiable by the client -- it is trusted the same way any
  other backend response is trusted, no more. Nothing in this module
  pretends otherwise.
* A generated ``(inputs, output)`` pair is never asserted as ground truth
  merely because a ``LocalVerifier`` accepted it -- see
  ``ScoreReceipt.provenance``.
* Synthetic rows this executor writes are never marked as holdout examples.
"""

from __future__ import annotations

import abc
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, ClassVar

from ._contract import VERIFIER_KINDS

__all__ = [
    "ColdStartOutcome",
    "ColdStartResult",
    "DiscoveryGap",
    "LocalVerifier",
    "ScoreReceipt",
]


class ColdStartOutcome(Enum):
    """Terminal outcome of one ``build_cold_start_eval_set()`` call."""

    #: At least one row was generated, locally verified, and written.
    EVAL_SET_BUILT = "eval_set_built"
    #: No eval set was built. See ``ColdStartResult.gap`` for why. No
    #: tuning JSONL or manifest was written on this path -- ever.
    DISCOVERY_ONLY = "discovery_only"


@dataclass(frozen=True, slots=True)
class DiscoveryGap:
    """Why cold start produced no eval set.

    Always paired with ``ColdStartResult.optimizer_eligible=False`` and zero
    artifacts written to disk. ``reason`` is a stable machine-readable code:
    the backend's own 422 reasons (``no_local_scoring_authority``,
    ``no_local_generation_capability``, ``descriptor_arity_mismatch``) pass
    through verbatim; the SDK adds its own client-side reasons for gaps a
    network round trip was never needed to discover (``no_generator_supplied``,
    ``no_verifier_supplied``) and for responses this client refuses to trust
    (``descriptor_digest_mismatch``, ``plan_expired``, ``malformed_response``,
    ``unauthorized``, ``feature_disabled``, ``no_verified_candidates``).
    """

    reason: str
    detail: str
    http_status: int | None = None


@dataclass(frozen=True, slots=True)
class ScoreReceipt:
    """Evidence that a LOCAL verifier actually scored one candidate row.

    A row this executor writes to the tuning JSONL always carries exactly
    one of these. ``provenance`` must never claim more than what happened:

    * ``"oracle_returned"`` -- the output value came out of calling the
      function/generator under test. It is a candidate output, not a
      verified truth, even if it looks obviously correct.
    * ``"independently_verified"`` -- an authority separate from the
      generation path (a property check, a reference oracle, a human)
      confirmed it.

    A ``LocalVerifier`` that merely echoes the generator's own output back
    must report ``"oracle_returned"``, never ``"independently_verified"``.
    """

    verifier_id: str
    verifier_kind: str
    passed: bool
    provenance: str
    evidence: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ColdStartResult:
    """Result of one ``build_cold_start_eval_set()`` call."""

    outcome: ColdStartOutcome
    optimizer_eligible: bool
    plan_id: str | None = None
    eval_set_path: Path | None = None
    manifest_path: Path | None = None
    row_count: int = 0
    receipts: tuple[ScoreReceipt, ...] = field(default_factory=tuple)
    gap: DiscoveryGap | None = None


class LocalVerifier(abc.ABC):
    """Caller-supplied local scoring authority for cold-start candidates.

    The SDK ships no verification logic. A concrete subclass backs exactly
    one ``kind`` -- declared as a class attribute and checked against the
    backend's ``verifier_kinds`` enum at subclass-definition time -- so the
    kind reported in the descriptor sent to the backend is bound to real
    verification code the caller wrote, never a free-form string a caller
    could claim without backing it with anything that actually verifies.
    """

    #: One of the backend's verifier_kinds enum values. Required on every
    #: concrete subclass; enforced by __init_subclass__ below.
    kind: ClassVar[str]

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        declared = getattr(cls, "kind", None)
        if declared not in VERIFIER_KINDS:
            raise TypeError(
                f"{cls.__name__}.kind must be one of {sorted(VERIFIER_KINDS)}; "
                f"got {declared!r}"
            )

    @abc.abstractmethod
    def verify(self, *, inputs: Mapping[str, Any], output: Any) -> ScoreReceipt | None:
        """Score one ``(inputs, output)`` candidate.

        Return a ``ScoreReceipt`` to accept the row, or ``None`` to reject
        it. A row with no receipt is never written to the eval set.
        """
