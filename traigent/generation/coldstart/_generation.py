"""Local candidate generation + verification loop.

Nothing generated here is ever sent to the backend -- the cold-start-plan
endpoint is content-free and stays that way; only the caller-supplied
generator and ``LocalVerifier`` ever see real inputs/outputs. This module
only screens (structural sanity), dedups, caps at the granted
``candidate_limit``, and calls the verifier -- it never fabricates a
verdict of its own.

Every candidate is snapshotted with ``copy.deepcopy`` the instant it is
pulled off the generator -- before screening, before dedup, before the
verifier ever sees it. A caller-supplied generator is a live Python
iterator: its body can hold a mutable object (e.g. the ``output`` half of
an already-yielded pair) and go on mutating it after the executor pulls the
NEXT candidate. Without an independent snapshot taken at the moment of the
pull, a row already accepted -- and the ``ScoreReceipt`` already earned for
it -- can end up describing different content than what this module later
writes. A candidate whose inputs or output cannot be deep-copied is
rejected outright (fail closed), never raised.
"""

from __future__ import annotations

import copy
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

# How many raw candidates this loop will PULL from `generator` while chasing
# one granted `candidate_limit`'s worth of ACCEPTED rows. Screening, dedup,
# and verifier rejection all consume candidates without accepting them, so a
# generator with an imperfect hit rate legitimately needs to be pulled more
# times than `candidate_limit` to fill it -- but a generator that never
# produces an acceptable row (or simply never terminates) must not be able
# to pull this executor into an unbounded, caller-hanging loop. This is a
# bound on THIS LOOP's own resource use, not a generation or verification
# technique, and it is not caller-configurable.
_MAX_PULLS_PER_ACCEPTED = 50


def generate_and_score(
    generator: GeneratorFn,
    verifier: LocalVerifier,
    *,
    candidate_limit: int,
    func: Callable[..., Any],
) -> list[VerifiedRow]:
    """Pull candidates from ``generator``, verify, dedup, and cap.

    A candidate is written only if all of the following hold: its inputs
    and output can be independently snapshotted (``copy.deepcopy``) the
    moment they are pulled, the snapshot is structurally well-formed, the
    output snapshot is itself JSON-serializable (so a row this executor
    could never write is never accepted in the first place), its input
    keys can actually call ``func`` (checked against ``func``'s real
    signature -- ``func`` itself is never called), its inputs are not a
    duplicate of one already ACCEPTED (a candidate a verifier rejected
    does not block a later, differently-scored candidate with the same
    inputs), the verifier actually returned a real ``ScoreReceipt`` (not
    ``None`` and not a duck-typed lookalike), the receipt's fields are
    well-formed, the receipt's ``verifier_kind`` matches the verifier's own
    declared ``kind`` (a verifier can't score under a kind it didn't
    declare), and the receipt says ``passed is True`` exactly.

    Pulling from ``generator`` is bounded two ways: this loop checks
    whether ``candidate_limit`` accepted rows have already been reached
    BEFORE each pull, so it never pulls one candidate more than necessary
    once the limit is filled; and it never pulls more than
    ``candidate_limit * _MAX_PULLS_PER_ACCEPTED`` candidates in total even
    when the limit is never filled, so a generator that never yields an
    acceptable row cannot hang the caller.
    """
    target_signature = inspect.signature(func)
    accepted: list[VerifiedRow] = []
    seen: set[str] = set()
    max_pulls = candidate_limit * _MAX_PULLS_PER_ACCEPTED
    pulls = 0
    iterator = iter(generator(candidate_limit))
    while len(accepted) < candidate_limit and pulls < max_pulls:
        try:
            candidate = next(iterator)
        except StopIteration:
            break
        pulls += 1

        raw_inputs, raw_output = candidate
        # Snapshot BOTH halves independently, right now, before anything
        # else runs -- this is the only point at which the executor still
        # shares the generator's own (possibly still-mutable) objects.
        inputs_snapshot, ok = _safe_deepcopy(raw_inputs)
        if not ok:
            continue
        output_snapshot, ok = _safe_deepcopy(raw_output)
        if not ok:
            continue

        # FREEZE what will be written, through JSON, before the verifier ever
        # sees the candidate.
        #
        # Copying is not enough here. A verifier that checks a value and then
        # mutates it --
        #
        #     assert output["answer"] == "4"
        #     output["answer"] = "5"
        #     return ScoreReceipt(passed=True, ...)
        #
        # -- lands its mutation on whatever object we later write, producing a
        # row whose receipt was earned for different content. Handing it a
        # deepcopy closes that for plain dicts and lists, but NOT in general:
        # copy.deepcopy returns whatever __deepcopy__ says it should, so a
        # JSON-serializable dict subclass whose __deepcopy__ returns self is
        # handed the very object we are about to write, and the mismatch is
        # back. Object identity is not something this executor can rely on when
        # the object comes from the caller.
        #
        # A JSON round-trip does not have that weakness. json.loads always
        # builds fresh plain containers, so the frozen pair is provably
        # independent of anything the caller controls -- and it is exactly the
        # bytes the artifact writer will serialize, so "what was verified" and
        # "what was written" are the same value by construction rather than by
        # a chain of copies each of which has to be trusted.
        #
        # Whether the mutation is malice or a verifier normalising in place
        # (trimming whitespace, coercing a type) does not matter: the receipt
        # must describe the row it is attached to.
        frozen_inputs, frozen_output, ok = _freeze_through_json(
            inputs_snapshot, output_snapshot
        )
        if not ok:
            # Also subsumes the old separate serializability screen: a
            # candidate that cannot survive the round-trip could never have
            # been written, and is rejected before a verifier spends effort.
            continue

        # EVERY gate below runs on the FROZEN value, never on the snapshot.
        #
        # Validating the snapshot and writing the frozen value is the same
        # bug as certifying one value and writing another: a caller-supplied
        # object may answer differently on each read, so a shape that binds
        # to the target on read 1 can be a shape that does not on read 3.
        # Reproduced on the merged code before this change -- a row was
        # written whose only key was one the target cannot accept, having
        # passed the bind check moments earlier.
        #
        # Gating the bytes we are about to write makes "what was checked"
        # and "what was written" the same value by construction.
        if not _well_formed(frozen_inputs):
            continue
        if not _callable_with(target_signature, frozen_inputs):
            # The row's input keys don't bind against func's real signature
            # (missing a required parameter, or an unexpected keyword func
            # can't accept) -- it would raise if ever called against the
            # target, so it is not a usable eval-set row.
            continue
        dedup_key = _canonical_key(frozen_inputs)
        if dedup_key in seen:
            continue
        # The verifier's copy is derived from the ALREADY-FROZEN value, never
        # from the caller's object a second time. Freezing twice from the
        # caller would re-serialize it, and an object is free to serialize
        # differently on each call (a dict subclass whose keys()/items()
        # returns different data on successive reads does exactly that --
        # confirmed, not theoretical). The verifier would then certify one
        # value while a different one was written. frozen_* are plain JSON
        # containers, so this round-trip is deterministic by construction.
        verifier_inputs, verifier_output, ok = _freeze_through_json(
            frozen_inputs, frozen_output
        )
        if not ok:
            continue

        receipt = verifier.verify(inputs=verifier_inputs, output=verifier_output)
        if receipt is None:
            # No verifier evidence -> this row is never written. Its
            # inputs must NOT be added to `seen`: dedup is only against
            # inputs already ACCEPTED, so a later candidate with the same
            # inputs still gets a real chance at verification.
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

        seen.add(dedup_key)
        accepted.append((frozen_inputs, frozen_output, receipt))
    return accepted


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
    return _json_serializable(inputs)


def _json_serializable(value: Any) -> bool:
    """Would ``json.dumps(value)`` succeed?

    Screens a candidate's output BEFORE it can be accepted -- the artifact
    writer (``_artifacts.write_eval_set``) calls ``json.dumps`` on every
    accepted row's output with no further check of its own, so a candidate
    that fails this here would otherwise surface as an uncaught exception
    from deep inside the writer instead of a typed, fail-closed result.
    """
    try:
        json.dumps(value, sort_keys=True)
    except (TypeError, ValueError):
        return False
    return True


def _freeze_through_json(inputs: Any, output: Any) -> tuple[dict[str, Any], Any, bool]:
    """Rebuild both halves as fresh plain JSON containers.

    Returns ``(inputs, output, ok)``; ``ok`` is False when either half cannot
    survive the round-trip, in which case the row is dropped.

    Why not ``copy.deepcopy``: deepcopy honours ``__deepcopy__``, so a
    caller-supplied JSON-serializable ``dict`` subclass can legally return
    itself and defeat the isolation entirely. ``json.loads`` cannot do that --
    it only ever constructs new dicts, lists and scalars.
    """
    try:
        return (
            json.loads(json.dumps(inputs, sort_keys=True)),
            json.loads(json.dumps(output, sort_keys=True)),
            True,
        )
    except (TypeError, ValueError, RecursionError):
        return {}, None, False


def _safe_deepcopy(value: Any) -> tuple[Any, bool]:
    """``copy.deepcopy(value)``, but fail closed instead of raising.

    Some objects a generator might yield (a lock, a file handle, a
    self-referential structure ``copy`` can't handle, ...) cannot be
    deep-copied at all. Rather than letting that raise out of the whole
    executor, the candidate is simply not usable -- return ``ok=False`` so
    the caller can reject the row the same way it rejects any other
    malformed candidate.
    """
    try:
        return copy.deepcopy(value), True
    except Exception:
        return None, False


def _canonical_key(inputs: Mapping[str, Any]) -> str:
    return json.dumps(inputs, sort_keys=True, default=str)
