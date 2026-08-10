"""Local candidate generation + verification loop.

Nothing generated here is ever sent to the backend -- the cold-start-plan
endpoint is content-free and stays that way; only the caller-supplied
generator and ``LocalVerifier`` ever see real inputs/outputs. This module
only screens (structural sanity), dedups, caps at the granted
``candidate_limit``, and calls the verifier -- it never fabricates a
verdict of its own.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Mapping
from typing import Any

from .models import LocalVerifier, ScoreReceipt

#: Caller-supplied generator: given the granted candidate_limit, yield
#: ``(inputs, output)`` pairs for the executor to screen/dedup/verify.
GeneratorFn = Callable[[int], Iterable[tuple[Mapping[str, Any], Any]]]

#: One row this executor is willing to write: the inputs, the candidate
#: output, and the ScoreReceipt that earned it a place in the eval set.
VerifiedRow = tuple[Mapping[str, Any], Any, ScoreReceipt]


def generate_and_score(
    generator: GeneratorFn,
    verifier: LocalVerifier,
    *,
    candidate_limit: int,
) -> list[VerifiedRow]:
    """Pull candidates from ``generator``, verify, dedup, and cap.

    A candidate is written only if all of the following hold: it is
    structurally well-formed, its inputs are not a duplicate of one already
    accepted, the verifier actually returned a ``ScoreReceipt`` (not
    ``None``), the receipt's ``verifier_kind`` matches the verifier's own
    declared ``kind`` (a verifier can't score under a kind it didn't
    declare), and the receipt says ``passed=True``.
    """
    accepted: list[VerifiedRow] = []
    seen: set[str] = set()
    for candidate in generator(candidate_limit):
        if len(accepted) >= candidate_limit:
            break
        inputs, output = candidate
        if not _well_formed(inputs):
            continue
        dedup_key = _canonical_key(inputs)
        if dedup_key in seen:
            continue
        seen.add(dedup_key)

        receipt = verifier.verify(inputs=inputs, output=output)
        if receipt is None:
            # No verifier evidence -> this row is never written.
            continue
        if receipt.verifier_kind != verifier.kind:
            # Defense in depth: a verify() implementation must not report a
            # kind other than the one the class declared.
            continue
        if not receipt.passed:
            continue
        accepted.append((dict(inputs), output, receipt))
    return accepted[:candidate_limit]


def _well_formed(inputs: Any) -> bool:
    if not isinstance(inputs, Mapping):
        return False
    try:
        json.dumps(inputs, sort_keys=True)
    except (TypeError, ValueError):
        return False
    return True


def _canonical_key(inputs: Mapping[str, Any]) -> str:
    return json.dumps(inputs, sort_keys=True, default=str)
