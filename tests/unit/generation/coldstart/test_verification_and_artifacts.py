"""LocalVerifier is the sole scoring authority: its declared `kind` backs
what the descriptor claims, a row with no verifier evidence never reaches
the eval set, and honesty fields (provenance, holdout) are never inflated.
"""

from __future__ import annotations

import json
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
from traigent.generation.coldstart._contract import compute_descriptor_digest
from traigent.generation.coldstart._plan import TransportResponse


def target(a: str, b: int) -> bool:
    return True


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


# --- LocalVerifier.kind is derived, not a free-form claim ------------------


def test_local_verifier_subclass_must_declare_a_valid_kind() -> None:
    with pytest.raises(TypeError):

        class BadVerifier(LocalVerifier):
            kind = "made_up_kind"

            def verify(self, *, inputs: Any, output: Any) -> ScoreReceipt | None:
                return None


def test_local_verifier_subclass_missing_kind_entirely_is_rejected() -> None:
    with pytest.raises(TypeError):

        class NoKindVerifier(LocalVerifier):
            def verify(self, *, inputs: Any, output: Any) -> ScoreReceipt | None:
                return None


def test_local_verifier_cannot_be_instantiated_without_implementing_verify() -> None:
    with pytest.raises(TypeError):
        LocalVerifier()  # type: ignore[abstract]


def test_descriptor_verifier_kinds_comes_from_the_verifier_object() -> None:
    """There is no way to pass verifier_kinds directly -- it is always
    derived from verifier.kind."""
    import inspect

    params = inspect.signature(build_cold_start_eval_set).parameters
    assert "verifier_kinds" not in params


# --- rows require actual verifier evidence ----------------------------------


class _RejectingVerifier(LocalVerifier):
    kind = "executable_property"

    def verify(self, *, inputs: Any, output: Any) -> ScoreReceipt | None:
        return None  # never scores anything


class _FailingVerifier(LocalVerifier):
    kind = "executable_property"

    def verify(self, *, inputs: Any, output: Any) -> ScoreReceipt | None:
        return ScoreReceipt(
            verifier_id="v1",
            verifier_kind=self.kind,
            passed=False,
            provenance="oracle_returned",
        )


class _LyingKindVerifier(LocalVerifier):
    """Declares one kind but its verify() reports scoring under a different
    kind -- must be rejected (defense in depth, requirement 4)."""

    kind = "executable_property"

    def verify(self, *, inputs: Any, output: Any) -> ScoreReceipt | None:
        return ScoreReceipt(
            verifier_id="v1",
            verifier_kind="calibrated_judge",  # does not match self.kind
            passed=True,
            provenance="oracle_returned",
        )


class _AcceptingVerifier(LocalVerifier):
    kind = "executable_property"

    def verify(self, *, inputs: Any, output: Any) -> ScoreReceipt | None:
        return ScoreReceipt(
            verifier_id="v1",
            verifier_kind=self.kind,
            passed=True,
            provenance="oracle_returned",
        )


def _row_generator(limit: int):
    for i in range(limit):
        yield ({"a": f"row-{i}", "b": i}, True)


def test_row_with_no_verifier_evidence_never_reaches_the_jsonl(tmp_path: Path) -> None:
    result = build_cold_start_eval_set(
        target,
        generator=_row_generator,
        verifier=_RejectingVerifier(),
        transport=_transport(),
        output_dir=tmp_path,
        generation_capabilities=("customer_llm",),
    )
    assert result.outcome is ColdStartOutcome.DISCOVERY_ONLY
    assert result.gap.reason == "no_verified_candidates"
    assert list(tmp_path.iterdir()) == []


def test_row_that_fails_verification_never_reaches_the_jsonl(tmp_path: Path) -> None:
    result = build_cold_start_eval_set(
        target,
        generator=_row_generator,
        verifier=_FailingVerifier(),
        transport=_transport(),
        output_dir=tmp_path,
        generation_capabilities=("customer_llm",),
    )
    assert result.outcome is ColdStartOutcome.DISCOVERY_ONLY
    assert result.gap.reason == "no_verified_candidates"
    assert list(tmp_path.iterdir()) == []


def test_verifier_reporting_a_mismatched_kind_is_rejected(tmp_path: Path) -> None:
    result = build_cold_start_eval_set(
        target,
        generator=_row_generator,
        verifier=_LyingKindVerifier(),
        transport=_transport(),
        output_dir=tmp_path,
        generation_capabilities=("customer_llm",),
    )
    assert result.outcome is ColdStartOutcome.DISCOVERY_ONLY
    assert result.gap.reason == "no_verified_candidates"
    assert list(tmp_path.iterdir()) == []


def test_mixed_batch_only_writes_rows_with_passing_receipts(tmp_path: Path) -> None:
    class _MixedVerifier(LocalVerifier):
        kind = "executable_property"

        def verify(self, *, inputs: Any, output: Any) -> ScoreReceipt | None:
            index = int(inputs["b"])
            if index % 3 == 0:
                return None  # no evidence
            if index % 3 == 1:
                return ScoreReceipt(
                    verifier_id="v1",
                    verifier_kind=self.kind,
                    passed=False,
                    provenance="oracle_returned",
                )
            return ScoreReceipt(
                verifier_id="v1",
                verifier_kind=self.kind,
                passed=True,
                provenance="oracle_returned",
            )

    def generator(limit: int):
        for i in range(9):
            yield ({"a": f"row-{i}", "b": i}, True)

    result = build_cold_start_eval_set(
        target,
        generator=generator,
        verifier=_MixedVerifier(),
        transport=_transport(candidate_limit=100),
        output_dir=tmp_path,
        requested_candidate_limit=100,
        generation_capabilities=("customer_llm",),
    )
    assert result.outcome is ColdStartOutcome.EVAL_SET_BUILT
    # indices 2,5,8 are the only "passed" rows out of 9
    assert result.row_count == 3
    rows = [json.loads(line) for line in result.eval_set_path.read_text().splitlines()]
    assert {row["input"]["b"] for row in rows} == {2, 5, 8}


# --- dedup -------------------------------------------------------------------


def test_duplicate_inputs_are_deduplicated(tmp_path: Path) -> None:
    def generator(limit: int):
        for _ in range(5):
            yield ({"a": "same", "b": 1}, True)
        yield ({"a": "different", "b": 2}, True)

    result = build_cold_start_eval_set(
        target,
        generator=generator,
        verifier=_AcceptingVerifier(),
        transport=_transport(candidate_limit=100),
        output_dir=tmp_path,
        requested_candidate_limit=100,
        generation_capabilities=("customer_llm",),
    )
    assert result.outcome is ColdStartOutcome.EVAL_SET_BUILT
    assert result.row_count == 2


# --- honesty: provenance and holdout ----------------------------------------


def test_oracle_returned_provenance_is_preserved_verbatim_never_upgraded(
    tmp_path: Path,
) -> None:
    class _OracleOnlyVerifier(LocalVerifier):
        kind = "executable_property"

        def verify(self, *, inputs: Any, output: Any) -> ScoreReceipt | None:
            return ScoreReceipt(
                verifier_id="v1",
                verifier_kind=self.kind,
                passed=True,
                provenance="oracle_returned",
            )

    result = build_cold_start_eval_set(
        target,
        generator=_row_generator,
        verifier=_OracleOnlyVerifier(),
        transport=_transport(),
        output_dir=tmp_path,
        generation_capabilities=("customer_llm",),
    )
    assert result.outcome is ColdStartOutcome.EVAL_SET_BUILT
    for receipt in result.receipts:
        assert receipt.provenance == "oracle_returned"
    manifest = json.loads(result.manifest_path.read_text())
    assert all(r["provenance"] == "oracle_returned" for r in manifest["receipts"])
    assert "independently_verified" not in result.manifest_path.read_text()


def test_synthetic_rows_are_never_marked_as_holdout(tmp_path: Path) -> None:
    class _EvidenceClaimsHoldoutVerifier(LocalVerifier):
        """Tries to sneak holdout=True in via evidence -- must be ignored;
        there is no field on ScoreReceipt or the generator's return value
        that can set holdout at all."""

        kind = "executable_property"

        def verify(self, *, inputs: Any, output: Any) -> ScoreReceipt | None:
            return ScoreReceipt(
                verifier_id="v1",
                verifier_kind=self.kind,
                passed=True,
                provenance="oracle_returned",
                evidence={"holdout": True, "is_holdout": True},
            )

    result = build_cold_start_eval_set(
        target,
        generator=_row_generator,
        verifier=_EvidenceClaimsHoldoutVerifier(),
        transport=_transport(),
        output_dir=tmp_path,
        generation_capabilities=("customer_llm",),
    )
    assert result.outcome is ColdStartOutcome.EVAL_SET_BUILT
    rows = [json.loads(line) for line in result.eval_set_path.read_text().splitlines()]
    assert all(row["holdout"] is False for row in rows)
    manifest = json.loads(result.manifest_path.read_text())
    assert manifest["holdout"] is False
