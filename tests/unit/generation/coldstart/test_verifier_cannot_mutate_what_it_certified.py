"""A receipt must describe exactly the content that gets written.

Snapshotting the candidate against the GENERATOR was not sufficient. ``verify()``
still received live references to those snapshots and the same objects were then
written, so a verifier that checks a value and then changes it:

    assert output["answer"] == "4"
    output["answer"] = "5"
    return ScoreReceipt(passed=True, ...)

produced a JSONL row of ``{"answer": "5"}`` carrying a receipt earned for
``{"answer": "4"}``. Reproduced before the fix; these tests keep it dead.

This is not only a malice scenario. A verifier that normalises in place --
trimming whitespace, coercing a type, filling a default -- silently does the
same thing while believing it is being helpful. Either way the receipt stops
describing the row, and a receipt that does not describe its row is worse than
no receipt: it carries the authority of verification with none of the content.

The verifier now gets its own copy. What we write is the copy it never touched.
"""

from __future__ import annotations

import json
import tempfile
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any


from traigent.generation.coldstart import (
    ColdStartOutcome,
    LocalVerifier,
    ScoreReceipt,
    build_cold_start_eval_set,
)
from traigent.generation.coldstart._contract import compute_descriptor_digest
from traigent.generation.coldstart._plan import TransportResponse


def target(question: str) -> dict:
    return {}


def _transport(request: Any) -> TransportResponse:
    return TransportResponse(
        200,
        {
            "plan_id": "csp_x",
            "protocol_version": "cold-start.v1",
            "descriptor_digest": compute_descriptor_digest(request["descriptor"]),
            "candidate_limit": 1,
            "expires_at": (datetime.now(UTC) + timedelta(hours=1)).isoformat(),
        },
    )


def _generator(limit: int):
    yield ({"question": "2+2"}, {"answer": "4"})


class _MutatesOutputAfterChecking(LocalVerifier):
    kind = "executable_property"

    def verify(self, *, inputs: Any, output: Any) -> ScoreReceipt | None:
        assert output["answer"] == "4"  # certifies the ORIGINAL value
        output["answer"] = "5"  # then changes it
        return ScoreReceipt(
            verifier_id="v1",
            verifier_kind=self.kind,
            passed=True,
            provenance="independently_verified",
        )


class _MutatesInputsAfterChecking(LocalVerifier):
    kind = "executable_property"

    def verify(self, *, inputs: Any, output: Any) -> ScoreReceipt | None:
        inputs["question"] = "TAMPERED"
        return ScoreReceipt(
            verifier_id="v1",
            verifier_kind=self.kind,
            passed=True,
            provenance="independently_verified",
        )


def _build(verifier: LocalVerifier) -> dict:
    with tempfile.TemporaryDirectory() as directory:
        result = build_cold_start_eval_set(
            target,
            generator=_generator,
            verifier=verifier,
            transport=_transport,
            output_dir=directory,
            generation_capabilities=("deterministic_contract",),
        )
        assert result.outcome is ColdStartOutcome.EVAL_SET_BUILT
        assert result.eval_set_path is not None
        return json.loads(
            Path(result.eval_set_path).read_text(encoding="utf-8").splitlines()[0]
        )


def test_output_mutated_by_the_verifier_does_not_reach_the_eval_set() -> None:
    assert _build(_MutatesOutputAfterChecking())["output"] == {"answer": "4"}


def test_inputs_mutated_by_the_verifier_do_not_reach_the_eval_set() -> None:
    assert _build(_MutatesInputsAfterChecking())["input"] == {"question": "2+2"}


def test_a_wellbehaved_verifier_is_unaffected() -> None:
    """The isolation must not change the ordinary path."""

    class _Clean(LocalVerifier):
        kind = "executable_property"

        def verify(self, *, inputs: Any, output: Any) -> ScoreReceipt | None:
            return ScoreReceipt(
                verifier_id="v1",
                verifier_kind=self.kind,
                passed=True,
                provenance="independently_verified",
            )

    row = _build(_Clean())
    assert row["input"] == {"question": "2+2"}
    assert row["output"] == {"answer": "4"}


def test_the_verifier_still_sees_the_real_content() -> None:
    """Isolation must not degrade into handing the verifier something else.

    A copy that did not equal the candidate would make verification meaningless
    -- the verifier would be certifying a different row than the one written.
    """
    seen: list[tuple[Any, Any]] = []

    class _Recording(LocalVerifier):
        kind = "executable_property"

        def verify(self, *, inputs: Any, output: Any) -> ScoreReceipt | None:
            seen.append((dict(inputs), dict(output)))
            return ScoreReceipt(
                verifier_id="v1",
                verifier_kind=self.kind,
                passed=True,
                provenance="independently_verified",
            )

    row = _build(_Recording())

    assert seen == [({"question": "2+2"}, {"answer": "4"})]
    assert (row["input"], row["output"]) == seen[0]
