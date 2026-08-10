"""Local cold-start eval-set executor.

Calls a CONTENT-FREE backend planning endpoint (``POST
/api/v1/guidance/cold-start-plan``) that returns only a coarse-typed,
non-content plan -- never a signed/verifiable token, just a ``plan_id`` +
digest + a candidate-limit grant + an expiry. All candidate GENERATION and
SCORING then happens LOCALLY: the caller supplies a generator callable and a
``LocalVerifier``; this SDK ships neither (that's the customer's IP to
bring). Nothing generated is ever sent back to the backend.

Public surface is deliberately narrow -- descriptor construction, plan
parsing/digest checking, screening/dedup, and artifact writing are all
private (leading-underscore modules) so the wire contract and on-disk
format can evolve without becoming a public API commitment.
"""

from __future__ import annotations

from .executor import build_cold_start_eval_set
from .models import (
    ColdStartOutcome,
    ColdStartResult,
    DiscoveryGap,
    LocalVerifier,
    ScoreReceipt,
)

__all__ = [
    "ColdStartOutcome",
    "ColdStartResult",
    "DiscoveryGap",
    "LocalVerifier",
    "ScoreReceipt",
    "build_cold_start_eval_set",
]
