"""Local candidate generation + verification loop.

Nothing generated here is ever sent to the backend -- the cold-start-plan
endpoint is content-free and stays that way; only the caller-supplied
generator and ``LocalVerifier`` ever see real inputs/outputs. This module
only screens (structural sanity), dedups, caps at the granted
``candidate_limit``, and calls the verifier -- it never fabricates a
verdict of its own.
"""

from __future__ import annotations

import inspect
import json
from collections.abc import Callable, Iterable, Mapping
from typing import Any

from ._contract import PROVENANCE_KINDS
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
    func: Callable[..., Any],
) -> list[VerifiedRow]:
    """Pull candidates from ``generator``, verify, dedup, and cap.

    A candidate is written only if all of the following hold: it is
    structurally well-formed, its input keys can actually call ``func``
    (checked against ``func``'s real signature -- ``func`` itself is never
    called), its inputs are not a duplicate of one already accepted, the
    verifier actually returned a real ``ScoreReceipt`` (not ``None`` and not
    a duck-typed lookalike), the receipt's fields are well-formed, the
    receipt's ``verifier_kind`` matches the verifier's own declared ``kind``
    (a verifier can't score under a kind it didn't declare), and the receipt
    says ``passed is True`` exactly.
    """
    target_signature = inspect.signature(func)
    accepted: list[VerifiedRow] = []
    seen: set[str] = set()
    for candidate in generator(candidate_limit):
        if len(accepted) >= candidate_limit:
            break
        inputs, output = candidate
        if not _well_formed(inputs):
            continue
        if not _callable_with(target_signature, inputs):
            # The candidate's input keys don't bind against func's real
            # signature (missing a required parameter, or an unexpected
            # keyword func can't accept) -- it would raise if ever called
            # against the target, so it is not a usable eval-set row.
            continue
        dedup_key = _canonical_key(inputs)
        if dedup_key in seen:
            continue
        seen.add(dedup_key)

        receipt = verifier.verify(inputs=inputs, output=output)
        if receipt is None:
            # No verifier evidence -> this row is never written.
            continue
        if not _is_valid_receipt(receipt):
            # Not a real ScoreReceipt, or one with a malformed field -- a
            # duck-typed lookalike must never be accepted as evidence.
            continue
        if receipt.verifier_kind != verifier.kind:
            # Defense in depth: a verify() implementation must not report a
            # kind other than the one the class declared.
            continue
        if receipt.passed is not True:
            # Exact identity, not truthiness: a non-empty string like
            # "false" is truthy but is not a pass.
            continue
        accepted.append((dict(inputs), output, receipt))
    return accepted[:candidate_limit]


def _callable_with(signature: inspect.Signature, inputs: Mapping[str, Any]) -> bool:
    """Would ``func(**inputs)`` bind without a TypeError?

    Only checks the binding -- a required parameter is missing, or a
    keyword ``func`` cannot accept (no matching parameter and no
    ``**kwargs``) -- never calls ``func`` itself. A row omitting a
    defaulted or keyword-only-with-default parameter still binds fine,
    exactly like a real call would accept.
    """
    try:
        signature.bind(**inputs)
    except TypeError:
        return False
    return True


def _is_valid_receipt(receipt: Any) -> bool:
    """Require a real ``ScoreReceipt`` with well-formed identity/kind/provenance.

    A duck-typed object that merely happens to have ``verifier_kind`` and a
    truthy ``passed`` attribute must never be treated as verifier evidence
    -- only a genuine ``ScoreReceipt`` instance can. ``passed`` itself is
    checked separately, by exact identity against ``True`` (see the
    ``receipt.passed is not True`` check in ``generate_and_score``) rather
    than here, since a non-bool ``passed`` on an otherwise-real
    ``ScoreReceipt`` is exactly the truthiness bug that check exists to
    close.
    """
    if not isinstance(receipt, ScoreReceipt):
        return False
    if not isinstance(receipt.verifier_id, str) or not receipt.verifier_id:
        return False
    if not isinstance(receipt.verifier_kind, str) or not receipt.verifier_kind:
        return False
    # provenance is a CLOSED vocabulary, not free text. The oracle_returned vs
    # independently_verified distinction is the reason receipts exist: the first
    # says "this came out of the generation path", the second says "something
    # separate confirmed it". Left as a free string, a verifier could assert any
    # claim it liked and put arbitrary text into the local manifest. The SDK
    # cannot prove a claim of independence is honest -- only the caller knows --
    # but it can refuse to record a claim it does not recognise.
    if receipt.provenance not in PROVENANCE_KINDS:
        return False
    return True


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
