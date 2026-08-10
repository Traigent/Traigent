"""Regression tests for a second adversarial review's six findings (S1-S6).

Each ``test_sN_*`` below reproduces the exact defect the review described
and asserts the fail-closed/honest behaviour that should hold once it is
fixed. Every one of these was confirmed to FAIL against the pre-fix code
and PASS against the post-fix code -- that is the point of this file, not
just passing once the fixes already exist. See the session report for the
exact revert-per-fix evidence.
"""

from __future__ import annotations

import inspect
import json
import os
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
from traigent.generation.coldstart import _artifacts
from traigent.generation.coldstart._contract import compute_descriptor_digest
from traigent.generation.coldstart._plan import TransportResponse


def target(a: str, b: int) -> bool:
    return True


def _question_target(question: str) -> str:
    return question


def _x_target(x: int) -> str:
    return str(x)


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


# --- S1: a receipt must never end up describing content different from ----
# --- what was actually written ----------------------------------------------


def test_s1_generator_resumption_cannot_corrupt_an_already_accepted_row(
    tmp_path: Path,
) -> None:
    """Isolated variant of the finding's exact reproducer.

    Uses ``candidate_limit=2`` (not 1) so this test isolates the deep-copy
    snapshot fix from the SEPARATE pull-bounding fix (S2): with limit=2 the
    executor legitimately needs a second pull regardless of S2 (only 1 of 2
    candidates is acceptable), so the generator's post-yield mutation
    happens either way -- only the deep-copy snapshot decides whether the
    row already accepted survives that mutation unharmed.
    """

    def generator(limit: int):
        out = {"answer": "4"}
        yield {"question": "2+2"}, out
        out["answer"] = "5"  # runs when the loop resumes to pull candidate 2
        yield {"question": "unused"}, {"answer": "unused"}  # verifier rejects this

    class _OnlyAcceptsFour(LocalVerifier):
        kind = "executable_property"

        def verify(self, *, inputs: Any, output: Any) -> ScoreReceipt | None:
            if output.get("answer") != "4":
                return None
            return ScoreReceipt(
                verifier_id="v1",
                verifier_kind=self.kind,
                passed=True,
                provenance="oracle_returned",
            )

    result = build_cold_start_eval_set(
        _question_target,
        generator=generator,
        verifier=_OnlyAcceptsFour(),
        transport=_transport(candidate_limit=2),
        output_dir=tmp_path,
        requested_candidate_limit=2,
        generation_capabilities=("customer_llm",),
    )

    assert result.outcome is ColdStartOutcome.EVAL_SET_BUILT, result
    assert result.row_count == 1
    rows = [json.loads(line) for line in result.eval_set_path.read_text().splitlines()]
    assert rows[0]["output"] == {"answer": "4"}, (
        f"wrote {rows[0]['output']!r}, but the ScoreReceipt on record was earned "
        "for {'answer': '4'} -- a written row must be exactly the content its "
        "receipt was earned for, immune to the generator mutating the output "
        "object after it was already accepted"
    )


def test_s1_task_literal_reproducer_candidate_limit_1(tmp_path: Path) -> None:
    """The finding's reproducer verbatim (candidate_limit=1). Exercises S1
    and S2 together against the actual pre-fix code -- with limit=1,
    stopping the pull early (S2) also happens to prevent the mutating
    resumption, so this one does not isolate S1 the way the test above
    does, but it does pin the literal example from the review."""

    def generator(limit: int):
        out = {"answer": "4"}
        yield {"question": "2+2"}, out
        out["answer"] = "5"  # runs when the loop pulls candidate 2
        yield {"question": "unused"}, {"answer": "unused"}

    class _OnlyAcceptsFour(LocalVerifier):
        kind = "executable_property"

        def verify(self, *, inputs: Any, output: Any) -> ScoreReceipt | None:
            if output.get("answer") != "4":
                return None
            return ScoreReceipt(
                verifier_id="v1",
                verifier_kind=self.kind,
                passed=True,
                provenance="oracle_returned",
            )

    result = build_cold_start_eval_set(
        _question_target,
        generator=generator,
        verifier=_OnlyAcceptsFour(),
        transport=_transport(candidate_limit=1),
        output_dir=tmp_path,
        requested_candidate_limit=1,
        generation_capabilities=("customer_llm",),
    )

    assert result.outcome is ColdStartOutcome.EVAL_SET_BUILT, result
    rows = [json.loads(line) for line in result.eval_set_path.read_text().splitlines()]
    assert rows[0]["output"] == {"answer": "4"}


# --- S2: the granted limit must bound candidates PULLED, not just accepted -


def test_s2_bounds_total_candidates_pulled_not_just_accepted(tmp_path: Path) -> None:
    """A generator that never yields an acceptable row must not be able to
    pull this executor into unbounded work. The generator below is finite
    (so a pre-fix run terminates instead of hanging pytest forever) but
    large enough that "pulled everything" and "pulled a bounded amount" are
    unmistakably different outcomes."""
    pulls = {"count": 0}

    def hostile_generator(limit: int):
        for i in range(5000):
            pulls["count"] += 1
            yield ({"a": f"row-{i}", "b": i}, False)

    class _AlwaysRejects(LocalVerifier):
        kind = "executable_property"

        def verify(self, *, inputs: Any, output: Any) -> ScoreReceipt | None:
            return None

    result = build_cold_start_eval_set(
        target,
        generator=hostile_generator,
        verifier=_AlwaysRejects(),
        transport=_transport(candidate_limit=1),
        output_dir=tmp_path,
        requested_candidate_limit=1,
        generation_capabilities=("customer_llm",),
    )

    assert result.outcome is ColdStartOutcome.DISCOVERY_ONLY
    assert pulls["count"] < 5000, (
        f"generator was pulled {pulls['count']} times chasing a granted limit of "
        "1 accepted row -- a generator that never produces an acceptable row "
        "must not be able to pull this executor into unbounded work"
    )


# --- S3: dedup blocks only inputs already ACCEPTED, never merely SEEN ------


def test_s3_dedup_only_blocks_inputs_that_were_actually_accepted(
    tmp_path: Path,
) -> None:
    def generator(limit: int):
        yield ({"x": 1}, "wrong")  # verifier rejects this candidate
        yield ({"x": 1}, "right")  # same inputs; verifier would accept this one

    class _OnlyAcceptsRight(LocalVerifier):
        kind = "executable_property"

        def verify(self, *, inputs: Any, output: Any) -> ScoreReceipt | None:
            if output != "right":
                return None
            return ScoreReceipt(
                verifier_id="v1",
                verifier_kind=self.kind,
                passed=True,
                provenance="oracle_returned",
            )

    result = build_cold_start_eval_set(
        _x_target,
        generator=generator,
        verifier=_OnlyAcceptsRight(),
        transport=_transport(candidate_limit=2),
        output_dir=tmp_path,
        requested_candidate_limit=2,
        generation_capabilities=("customer_llm",),
    )

    assert result.outcome is ColdStartOutcome.EVAL_SET_BUILT, (
        "a valid candidate must not be blocked by dedup merely because an "
        f"earlier REJECTED candidate had the same inputs; got {result.gap}"
    )
    assert result.row_count == 1
    rows = [json.loads(line) for line in result.eval_set_path.read_text().splitlines()]
    assert rows[0]["output"] == "right"


# --- S4: a non-serializable output is screened BEFORE acceptance -----------


def test_s4_non_serializable_output_is_screened_before_acceptance(
    tmp_path: Path,
) -> None:
    """Before the fix this candidate sailed through generate_and_score and
    only blew up later as an uncaught json.dumps() TypeError deep inside
    write_eval_set() -- not the typed fail-closed result required here."""

    def generator(limit: int):
        yield ({"a": "row-0", "b": 0}, object())  # not JSON-serializable

    result = build_cold_start_eval_set(
        target,
        generator=generator,
        verifier=_AcceptingVerifier(),
        transport=_transport(),
        output_dir=tmp_path,
        generation_capabilities=("customer_llm",),
    )

    assert result.outcome is ColdStartOutcome.DISCOVERY_ONLY
    assert result.gap is not None
    assert result.gap.reason == "no_verified_candidates"
    assert list(tmp_path.iterdir()) == []


# --- S5: the atomicity docstring must not claim more than it delivers ------


def test_s5_docstring_is_honest_about_atomicity_limits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # First, demonstrate the window the docstring talks about is real: right
    # after the JSONL's os.replace lands, and before the manifest's runs,
    # the directory legitimately contains the JSONL without a manifest.
    observed: dict[str, bool] = {}
    real_replace = os.replace

    def spying_replace(src: Any, dst: Any) -> None:
        real_replace(src, dst)
        if str(dst).endswith(".jsonl"):
            observed["window_seen"] = (
                Path(dst).exists()
                and not (tmp_path / "cold_start.manifest.json").exists()
            )

    monkeypatch.setattr(_artifacts.os, "replace", spying_replace)

    receipt = ScoreReceipt(
        verifier_id="v1",
        verifier_kind="executable_property",
        passed=True,
        provenance="oracle_returned",
    )
    rows = [({"a": "row-0", "b": 0}, True, receipt)]
    _artifacts.write_eval_set(
        tmp_path, "cold_start", rows, plan_id="csp_ok", descriptor={}
    )

    assert observed.get("window_seen") is True, (
        "expected a real window between the two os.replace calls to be "
        "observable -- if this fails, the fixture needs re-checking, not "
        "the docstring assertions below"
    )

    # Now pin that the docstring is honest about that window instead of
    # promising a caller can never observe it.
    doc = (_artifacts._write_pair_atomically.__doc__ or "").lower()
    assert (
        "must never observe the jsonl at its final path without a manifest" not in doc
    ), (
        "the docstring claims a guarantee -- isolation from a concurrent "
        "observer, or crash-atomicity across the two os.replace calls -- "
        "that is not actually achievable with two separate files on POSIX"
    )
    assert "crash" in doc and "not guarantee" in doc, (
        "the docstring must explicitly say what is NOT guaranteed "
        "(crash-atomicity, concurrent-observer isolation), not just what is"
    )

    module_doc = (_artifacts.__doc__ or "").lower()
    assert "both land, or neither does" not in module_doc, (
        "the module docstring must not repeat the same overclaim"
    )


# --- S6: generation_capabilities must not default to a claim the SDK -------
# --- cannot actually verify -------------------------------------------------


def test_s6_generation_capabilities_has_no_default(tmp_path: Path) -> None:
    """Before the fix, this parameter defaulted to ("customer_llm",) --
    even for a caller-supplied generator that, say, reads hand-authored
    rows from a file. The SDK has no way to know what the generator
    actually is, so it must not guess on the caller's behalf."""
    signature = inspect.signature(build_cold_start_eval_set)
    param = signature.parameters["generation_capabilities"]
    assert param.default is inspect.Parameter.empty, (
        "generation_capabilities must have NO default -- a default lets an "
        "unstated (and possibly false) capability claim reach the backend"
    )


def test_s6_omitting_generation_capabilities_is_rejected_before_any_claim_is_made(
    tmp_path: Path,
) -> None:
    with pytest.raises(TypeError):
        build_cold_start_eval_set(
            target,
            generator=_generator,
            verifier=_AcceptingVerifier(),
            transport=_transport(),
            output_dir=tmp_path,
            # generation_capabilities intentionally omitted
        )
    assert list(tmp_path.iterdir()) == []
