"""Local cold-start eval-set executor.

``build_cold_start_eval_set()`` is the SDK's answer to "I have no evaluation
dataset yet." It:

1. Builds a CONTENT-FREE descriptor from the target callable's signature
   (coarse arity/type shape only -- see ``_descriptor.build_descriptor``).
2. POSTs that descriptor to the backend's cold-start-plan endpoint via a
   caller-injected transport, and gets back a plan: a ``plan_id``, a
   digest of the descriptor it planned against, a granted
   ``candidate_limit`` (which may be lower than requested), and an expiry.
   This plan is NOT cryptographically signed -- it is not independently
   verifiable, only digest-consistent with what this process sent.
3. Runs a caller-supplied ``generator`` and ``LocalVerifier`` LOCALLY to
   produce and score candidate rows. The SDK ships neither -- that is
   deliberately the customer's IP to bring, not ours.
4. Writes a tuning JSONL + manifest locally. Nothing generated is ever sent
   back to the backend.

Every failure mode -- a 422 from the backend, a missing generator/verifier,
a digest mismatch, an expired plan, a malformed/unauthorized response, or a
generator that produced nothing a verifier would accept -- fails CLOSED:
``ColdStartOutcome.DISCOVERY_ONLY``, ``optimizer_eligible=False``, a typed
``DiscoveryGap``, and zero files written.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from datetime import datetime, UTC
from pathlib import Path
from typing import Any

from ._artifacts import write_eval_set
from ._contract import (
    GENERATION_CAPABILITIES,
    MAX_CANDIDATE_LIMIT,
    MIN_CANDIDATE_LIMIT,
    VERIFIER_KINDS,
)
from ._descriptor import build_descriptor
from ._generation import GeneratorFn, generate_and_score
from ._plan import (
    Plan,
    Transport,
    build_request,
    check_digest,
    check_not_expired,
    parse_response,
    validate_descriptor_arity,
)
from .models import ColdStartOutcome, ColdStartResult, DiscoveryGap, LocalVerifier

_DEFAULT_GENERATION_CAPABILITIES: tuple[str, ...] = ("customer_llm",)


def build_cold_start_eval_set(
    func: Callable[..., Any],
    *,
    generator: GeneratorFn | None,
    verifier: LocalVerifier | None,
    transport: Transport,
    output_dir: str | Path,
    dataset_name: str = "cold_start",
    requested_candidate_limit: int = 12,
    generation_capabilities: Sequence[str] = _DEFAULT_GENERATION_CAPABILITIES,
    containment_root: str | Path | None = None,
    clock: Callable[[], datetime] | None = None,
) -> ColdStartResult:
    """Build a first evaluation dataset for ``func`` with no built-in generation/scoring.

    Args:
        func: The target callable cold start is building an eval set for.
            Only its signature is inspected -- never called by this
            function itself.
        generator: Caller-supplied candidate producer. Given the granted
            ``candidate_limit``, yields ``(inputs, output)`` pairs. The SDK
            ships no generation technique; a missing generator is a
            fail-closed gap, not a fallback to a built-in one.
        verifier: Caller-supplied ``LocalVerifier``. A missing verifier is a
            fail-closed gap for the same reason.
        transport: Caller-injected POST to the backend's cold-start-plan
            endpoint. Takes the JSON request body, returns a
            ``_plan.TransportResponse(status_code, body)``.
        output_dir: Where to write the tuning JSONL + manifest. Only
            touched when an eval set is actually built.
        dataset_name: Base filename (sanitized) for the two artifacts.
        requested_candidate_limit: Upper bound this call asks for; the
            backend's grant (``Plan.candidate_limit``) may be lower and
            always wins.
        generation_capabilities: What the supplied ``generator`` represents,
            drawn from the backend's ``generation_capabilities`` enum.
            Defaults to ``("customer_llm",)`` -- true for any arbitrary
            caller-supplied generator by construction.
        containment_root: Optional root ``output_dir`` must stay under.
        clock: Injectable now() for expiry checks; defaults to real UTC now.

    Returns:
        A ``ColdStartResult``. ``outcome=EVAL_SET_BUILT`` with
        ``optimizer_eligible=True`` on success; otherwise
        ``outcome=DISCOVERY_ONLY`` with ``optimizer_eligible=False`` and a
        populated ``gap``, and NO artifacts written.
    """
    # bool is an int subclass, so `True`/`False` would otherwise sail through
    # the numeric range check below and end up serialized as a JSON boolean
    # (`"candidate_limit": true`) -- the same shape the response parser
    # already refuses to accept coming back. Reject it symmetrically here.
    if isinstance(requested_candidate_limit, bool) or not isinstance(
        requested_candidate_limit, int
    ):
        raise ValueError(
            f"requested_candidate_limit must be an int, got "
            f"{type(requested_candidate_limit).__name__}"
        )
    if not (MIN_CANDIDATE_LIMIT <= requested_candidate_limit <= MAX_CANDIDATE_LIMIT):
        raise ValueError(
            f"requested_candidate_limit must be within "
            f"[{MIN_CANDIDATE_LIMIT}, {MAX_CANDIDATE_LIMIT}]; "
            f"got {requested_candidate_limit}"
        )

    # Requirement 3: generator/verifier are caller-injected; the SDK ships
    # neither. Absence fails closed before any network call.
    if generator is None:
        return _discovery_only(
            DiscoveryGap(
                reason="no_generator_supplied",
                detail="build_cold_start_eval_set requires a caller-supplied generator",
            )
        )
    if verifier is None:
        return _discovery_only(
            DiscoveryGap(
                reason="no_verifier_supplied",
                detail="build_cold_start_eval_set requires a caller-supplied LocalVerifier",
            )
        )

    resolved_capabilities = _validate_generation_capabilities(generation_capabilities)

    # Defense in depth: __init_subclass__ only validates `kind` against the
    # enum at CLASS-definition time. `kind` is a plain instance attribute,
    # so a caller can still overwrite it on an instance
    # (`v = MyVerifier(); v.kind = "<anything>"`) after the class check has
    # already run. Re-validate the INSTANCE value here, at the point where
    # it is actually read and placed on the wire, and fail closed if it has
    # drifted from the enum -- never send an unvalidated string.
    if verifier.kind not in VERIFIER_KINDS:
        return _discovery_only(
            DiscoveryGap(
                reason="invalid_verifier_kind",
                detail=(
                    f"verifier.kind must be one of {sorted(VERIFIER_KINDS)}; "
                    f"got {verifier.kind!r}"
                ),
            )
        )

    # Requirement 4: verifier_kinds is derived from the verifier object
    # actually supplied (its declared, class-bound `kind`), never a
    # free-form claim passed in separately.
    descriptor = build_descriptor(
        func,
        verifier_kinds=(verifier.kind,),
        generation_capabilities=resolved_capabilities,
    )

    # Requirement 2: len(input_kinds) == input_arity, checked client-side --
    # the JSON schema alone cannot express this cross-field invariant.
    arity_gap = validate_descriptor_arity(descriptor)
    if arity_gap is not None:
        return _discovery_only(arity_gap)

    request = build_request(descriptor, requested_candidate_limit)
    response = transport(request)
    parsed = parse_response(response)
    if isinstance(parsed, DiscoveryGap):
        # Requirement 5: 422 (and any other non-200) fails closed here.
        return _discovery_only(parsed)
    plan: Plan = parsed

    # Requirement 7: recompute and compare the descriptor digest.
    digest_gap = check_digest(descriptor, plan)
    if digest_gap is not None:
        return _discovery_only(digest_gap)

    # Requirement 8: reject an expired plan before generating anything.
    now = clock() if clock is not None else datetime.now(UTC)
    expiry_gap = check_not_expired(plan, now=now)
    if expiry_gap is not None:
        return _discovery_only(expiry_gap)

    # Requirement 6: honour the grant, never the ask. A defensive min() also
    # guards a hypothetical backend bug that grants more than was asked.
    granted_limit = min(plan.candidate_limit, requested_candidate_limit)

    # Requirement 9: only rows a LocalVerifier actually scored (dedup +
    # screening -- including a signature-callability check against `func` --
    # happen here too) ever become candidates for writing.
    accepted_rows = generate_and_score(
        generator, verifier, candidate_limit=granted_limit, func=func
    )
    if not accepted_rows:
        return _discovery_only(
            DiscoveryGap(
                reason="no_verified_candidates",
                detail="no generated candidate passed local verification",
            )
        )

    # Requirement 10: write tuning JSONL + manifest locally. Nothing
    # generated is ever sent back to the backend.
    eval_set_path, manifest_path = write_eval_set(
        output_dir,
        dataset_name,
        accepted_rows,
        plan_id=plan.plan_id,
        descriptor=descriptor,
        containment_root=containment_root,
    )

    receipts = tuple(receipt for _, _, receipt in accepted_rows)
    return ColdStartResult(
        outcome=ColdStartOutcome.EVAL_SET_BUILT,
        optimizer_eligible=True,
        plan_id=plan.plan_id,
        eval_set_path=eval_set_path,
        manifest_path=manifest_path,
        row_count=len(accepted_rows),
        receipts=receipts,
    )


def _discovery_only(gap: DiscoveryGap) -> ColdStartResult:
    return ColdStartResult(
        outcome=ColdStartOutcome.DISCOVERY_ONLY,
        optimizer_eligible=False,
        gap=gap,
    )


def _validate_generation_capabilities(values: Sequence[str]) -> tuple[str, ...]:
    resolved: list[str] = []
    for value in values:
        if value not in GENERATION_CAPABILITIES:
            raise ValueError(
                f"unknown generation capability {value!r}; must be one of "
                f"{sorted(GENERATION_CAPABILITIES)}"
            )
        if value not in resolved:
            resolved.append(value)
    if not resolved:
        raise ValueError(
            "generation_capabilities must not be empty when a generator is supplied"
        )
    return tuple(resolved)
