"""Regression tests for the seven adversarial-review findings (F1-F7).

Each ``test_fN_*`` below reproduces the exact defect the review described and
asserts the fail-closed/content-free behaviour that should hold once it is
fixed. Every one of these was confirmed to FAIL against the pre-fix code and
PASS against the post-fix code (see the session report for the exact
stash/revert-per-fix evidence) -- that is the point of this file, not just
passing once the fixes already exist.
"""

from __future__ import annotations

from datetime import datetime, timedelta, UTC
from pathlib import Path
from typing import Any

import pytest

from traigent.generation.coldstart import (
    ColdStartOutcome,
    LocalVerifier,
    ScoreReceipt,
    build_cold_start_eval_set,
)
from traigent.generation.coldstart._artifacts import write_eval_set
from traigent.generation.coldstart._contract import compute_descriptor_digest
from traigent.generation.coldstart._plan import TransportResponse


def target(a: str, b: int) -> bool:
    return True


class _AcceptingVerifier(LocalVerifier):
    kind = "executable_property"

    def verify(self, *, inputs: Any, output: Any) -> ScoreReceipt | None:
        return ScoreReceipt(
            verifier_id="v1",
            verifier_kind=self.kind,
            passed=True,
            provenance="oracle_returned",
        )


def _generator(limit: int):
    for i in range(limit):
        yield ({"a": f"row-{i}", "b": i}, True)


def _transport(candidate_limit: int = 10):
    def transport(request):
        return TransportResponse(
            200,
            {
                "plan_id": "csp_ok",
                "protocol_version": "cold-start.v1",
                "descriptor_digest": compute_descriptor_digest(request["descriptor"]),
                "candidate_limit": candidate_limit,
                "expires_at": (datetime.now(UTC) + timedelta(hours=1)).isoformat(),
            },
        )

    return transport


class _RecordingTransport:
    """Spies on outbound requests; returns a well-formed 422 so any call
    that does reach it completes without raising (letting the leak-detection
    assertion, not an incidental crash, be what fails the test)."""

    def __init__(self) -> None:
        self.requests: list[Any] = []

    def __call__(self, request: Any) -> TransportResponse:
        self.requests.append(request)
        return TransportResponse(
            422, {"error": "declined", "reason": "no_local_scoring_authority"}
        )


# --- F1: verifier.kind is re-validated at the point of use, not just at ----
# --- class-definition time -------------------------------------------------


def test_f1_instance_level_kind_mutation_never_reaches_the_network(
    tmp_path: Path,
) -> None:
    """__init_subclass__ only checks `kind` when the class is DEFINED. A
    caller can mutate the INSTANCE attribute afterward to any string --
    including customer content -- and that string was being placed straight
    into descriptor["verifier_kinds"] and handed to the transport. The fix
    must catch this at the point verifier.kind is actually read/used, before
    any network call."""
    verifier = _AcceptingVerifier()
    verifier.kind = "this string could be arbitrary customer content"
    transport = _RecordingTransport()

    result = build_cold_start_eval_set(
        target,
        generator=_generator,
        verifier=verifier,
        transport=transport,
        output_dir=tmp_path,
    )

    assert transport.requests == [], (
        "verifier.kind must be revalidated before it is ever placed on the "
        f"wire; leaked request(s): {transport.requests}"
    )
    assert result.outcome is ColdStartOutcome.DISCOVERY_ONLY
    assert result.optimizer_eligible is False
    assert list(tmp_path.iterdir()) == []


# --- F2: passed must be checked by identity, not truthiness ----------------


def test_f2_truthy_non_bool_passed_value_is_not_accepted(tmp_path: Path) -> None:
    """`passed="false"` is a non-empty string -- truthy in Python -- but it
    is not a pass. `if not receipt.passed` would let it through; only an
    exact `is True` check closes this."""

    class _StringPassedVerifier(LocalVerifier):
        kind = "executable_property"

        def verify(self, *, inputs: Any, output: Any) -> ScoreReceipt | None:
            return ScoreReceipt(
                verifier_id="v1",
                verifier_kind=self.kind,
                passed="false",  # type: ignore[arg-type]  # deliberately not a bool
                provenance="oracle_returned",
            )

    result = build_cold_start_eval_set(
        target,
        generator=_generator,
        verifier=_StringPassedVerifier(),
        transport=_transport(),
        output_dir=tmp_path,
    )

    assert result.outcome is ColdStartOutcome.DISCOVERY_ONLY
    assert result.gap is not None
    assert result.gap.reason == "no_verified_candidates"
    assert list(tmp_path.iterdir()) == []


# --- F3: verify()'s return value must be a real ScoreReceipt ---------------


def test_f3_duck_typed_receipt_lookalike_is_not_accepted(tmp_path: Path) -> None:
    """A non-ScoreReceipt object that merely happens to expose
    `verifier_kind` and a truthy `passed` must not be treated as verifier
    evidence -- only a real, isinstance-checked ScoreReceipt can accept a
    row."""

    class _FakeReceipt:
        def __init__(self, kind: str) -> None:
            self.verifier_id = "v1"
            self.verifier_kind = kind
            self.passed = True
            self.provenance = "oracle_returned"

    class _DuckTypedVerifier(LocalVerifier):
        kind = "executable_property"

        def verify(self, *, inputs: Any, output: Any) -> Any:
            return _FakeReceipt(self.kind)

    result = build_cold_start_eval_set(
        target,
        generator=_generator,
        verifier=_DuckTypedVerifier(),  # type: ignore[arg-type]
        transport=_transport(),
        output_dir=tmp_path,
    )

    assert result.outcome is ColdStartOutcome.DISCOVERY_ONLY
    assert result.gap is not None
    assert result.gap.reason == "no_verified_candidates"
    assert list(tmp_path.iterdir()) == []


# --- F4: JSONL + manifest are written atomically; never a JSONL alone ------


def test_f4_manifest_write_failure_leaves_no_partial_jsonl(tmp_path: Path) -> None:
    """If writing the manifest fails (here: its final path is already an
    existing directory, so the write raises IsADirectoryError), the JSONL
    that was already "written" must not be left behind either -- and no
    temp files should linger."""
    receipt = ScoreReceipt(
        verifier_id="v1",
        verifier_kind="executable_property",
        passed=True,
        provenance="oracle_returned",
    )
    rows = [({"a": "row-0", "b": 0}, True, receipt)]

    # Sabotage the manifest's final path by pre-creating it as a directory.
    (tmp_path / "cold_start.manifest.json").mkdir()

    with pytest.raises(OSError):
        write_eval_set(
            tmp_path,
            "cold_start",
            rows,
            plan_id="csp_ok",
            descriptor={"input_arity": 2},
        )

    assert not (tmp_path / "cold_start.jsonl").exists(), (
        "a JSONL was left on disk without its manifest -- property B violated"
    )
    leftover = {p.name for p in tmp_path.iterdir()}
    assert leftover == {"cold_start.manifest.json"}, (
        f"unexpected leftover files (temp files not cleaned up?): {leftover}"
    )


# --- F5: generated inputs must actually be callable against the target -----


def _target_two_required(question: str, context: str) -> bool:
    return True


def test_f5_inputs_that_cannot_call_the_target_are_rejected(tmp_path: Path) -> None:
    """A generator can yield input keys with nothing to do with the target
    callable's real parameters -- `{"x": 1}` against
    `def answer(question: str, context: str)`. Such a row is uncallable
    against the target and must never be written, even if a verifier passes
    it."""

    def mismatched_generator(limit: int):
        for i in range(limit):
            yield ({"x": i}, True)  # neither "question" nor "context"

    result = build_cold_start_eval_set(
        _target_two_required,
        generator=mismatched_generator,
        verifier=_AcceptingVerifier(),
        transport=_transport(),
        output_dir=tmp_path,
    )

    assert result.outcome is ColdStartOutcome.DISCOVERY_ONLY
    assert result.gap is not None
    assert result.gap.reason == "no_verified_candidates"
    assert list(tmp_path.iterdir()) == []


def test_f5_positive_control_defaults_keyword_only_and_kwargs_still_accepted(
    tmp_path: Path,
) -> None:
    """Not part of the seven regressions -- a sanity check that the F5 fix
    is not overly strict: omitting a defaulted keyword-only parameter, or
    supplying extra keys a **kwargs catch-all can absorb, must still bind
    against the target and be accepted."""

    def target_flexible(a: str, *, b: int = 5, **extra: Any) -> bool:
        return True

    def generator(limit: int):
        yield ({"a": "hi"}, True)  # omits defaulted keyword-only b
        yield ({"a": "hi2", "b": 9, "anything": "ok"}, True)  # extra -> **extra

    result = build_cold_start_eval_set(
        target_flexible,
        generator=generator,
        verifier=_AcceptingVerifier(),
        transport=_transport(candidate_limit=100),
        output_dir=tmp_path,
        requested_candidate_limit=100,
    )

    assert result.outcome is ColdStartOutcome.EVAL_SET_BUILT
    assert result.row_count == 2


# --- F6: a naive clock must fail closed deterministically, never raise -----


def test_f6_naive_clock_fails_closed_instead_of_raising(tmp_path: Path) -> None:
    """`check_not_expired` compares a tz-aware `plan.expires_at` against a
    caller-injected `clock`. A naive clock -- e.g.
    `lambda: datetime.utcnow()` -- must not blow up the whole call with
    `TypeError: can't compare offset-naive and offset-aware datetimes`; it
    must produce the proper closed outcome."""

    def naive_clock() -> datetime:
        return datetime.utcnow()  # deliberately tz-naive

    def transport(request):
        return TransportResponse(
            200,
            {
                "plan_id": "csp_ok",
                "protocol_version": "cold-start.v1",
                "descriptor_digest": compute_descriptor_digest(request["descriptor"]),
                "candidate_limit": 5,
                # Already expired relative to real UTC now (and thus also
                # relative to naive_clock(), once normalized).
                "expires_at": (datetime.now(UTC) - timedelta(minutes=1)).isoformat(),
            },
        )

    result = build_cold_start_eval_set(
        target,
        generator=_generator,
        verifier=_AcceptingVerifier(),
        transport=transport,
        output_dir=tmp_path,
        clock=naive_clock,
    )

    assert result.outcome is ColdStartOutcome.DISCOVERY_ONLY
    assert result.gap is not None
    assert result.gap.reason == "plan_expired"
    assert list(tmp_path.iterdir()) == []


# --- F7: requested_candidate_limit must reject bool the same way the -------
# --- response parser already rejects it for candidate_limit ----------------


def test_f7_bool_requested_candidate_limit_is_rejected(tmp_path: Path) -> None:
    """bool is an int subclass, so `requested_candidate_limit=True` (== 1)
    satisfies the numeric range check and would end up serialized on the
    wire as `"candidate_limit": true` -- the exact shape
    `_plan._parse_plan` already refuses to accept coming back. The request
    path must reject it just as strictly, before any network call."""
    transport = _RecordingTransport()

    with pytest.raises(ValueError):
        build_cold_start_eval_set(
            target,
            generator=_generator,
            verifier=_AcceptingVerifier(),
            transport=transport,
            output_dir=tmp_path,
            requested_candidate_limit=True,
        )

    assert transport.requests == [], (
        f"a bool candidate_limit reached the transport: {transport.requests}"
    )
    assert list(tmp_path.iterdir()) == []
