"""``ScoreReceipt.provenance`` is a closed vocabulary, not a free string.

The distinction it carries is the reason receipts exist at all:

* ``oracle_returned`` -- the expected output came out of the generation path.
  A candidate, not verified truth, however obviously right it looks.
* ``independently_verified`` -- something SEPARATE from generation confirmed it.

The discovery report is explicit that a generator-supplied value must not be
allowed to imply independently-computed truth. That rule was stated in a
docstring and enforced by nothing: any non-empty string was accepted, so a
verifier could assert whatever it liked and write arbitrary text into the local
manifest.

The SDK cannot prove that a claim of independence is HONEST -- only the caller
knows whether their verifier really consulted a separate authority. It can
refuse to record a claim it does not recognise, which is what these tests pin.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from traigent.generation.coldstart import (
    ColdStartOutcome,
    LocalVerifier,
    ScoreReceipt,
    build_cold_start_eval_set,
)
from traigent.generation.coldstart._contract import (
    PROVENANCE_INDEPENDENTLY_VERIFIED,
    PROVENANCE_KINDS,
    PROVENANCE_ORACLE_RETURNED,
)
from traigent.generation.coldstart._plan import TransportResponse


def target(a: str, b: int) -> bool:
    return True


def _transport(request: Any) -> TransportResponse:
    from traigent.generation.coldstart._contract import compute_descriptor_digest

    return TransportResponse(
        200,
        {
            "plan_id": "csp_ok",
            "protocol_version": "cold-start.v1",
            "descriptor_digest": compute_descriptor_digest(request["descriptor"]),
            "candidate_limit": 4,
            "expires_at": (datetime.now(UTC) + timedelta(hours=1)).isoformat(),
        },
    )


def _row_generator(limit: int):
    for i in range(limit):
        yield ({"a": f"row-{i}", "b": i}, True)


def _verifier_claiming(provenance: str) -> LocalVerifier:
    class _Claiming(LocalVerifier):
        kind = "executable_property"

        def verify(self, *, inputs: Any, output: Any) -> ScoreReceipt | None:
            return ScoreReceipt(
                verifier_id="v1",
                verifier_kind=self.kind,
                passed=True,
                provenance=provenance,
            )

    return _Claiming()


def _build(tmp_path: Path, provenance: str):
    return build_cold_start_eval_set(
        target,
        generator=_row_generator,
        verifier=_verifier_claiming(provenance),
        transport=_transport,
        output_dir=tmp_path,
        generation_capabilities=("customer_llm",),
    )


def test_the_vocabulary_is_exactly_two_values() -> None:
    assert PROVENANCE_KINDS == {
        PROVENANCE_ORACLE_RETURNED,
        PROVENANCE_INDEPENDENTLY_VERIFIED,
    }


@pytest.mark.parametrize(
    "provenance", [PROVENANCE_ORACLE_RETURNED, PROVENANCE_INDEPENDENTLY_VERIFIED]
)
def test_a_recognised_claim_is_accepted(tmp_path: Path, provenance: str) -> None:
    result = _build(tmp_path, provenance)

    assert result.outcome is ColdStartOutcome.EVAL_SET_BUILT
    assert result.row_count > 0


@pytest.mark.parametrize(
    "provenance",
    [
        "verified",  # plausible-looking but not the vocabulary
        "INDEPENDENTLY_VERIFIED",  # case matters; this is not the value
        "trust me",
        "ground_truth",  # the exact overclaim the vocabulary exists to prevent
        "SECRET-CUSTOMER-PROMPT-a1b2c3",  # free text is a content channel
        "",
    ],
)
def test_an_unrecognised_claim_never_reaches_the_eval_set(
    tmp_path: Path, provenance: str
) -> None:
    """Rows whose evidence carries an unrecognised claim are not written.

    Fail closed: an unreadable provenance claim means we do not know what the
    verifier actually did, so the row is not admissible evidence.
    """
    result = _build(tmp_path, provenance)

    assert result.outcome is ColdStartOutcome.DISCOVERY_ONLY
    assert result.optimizer_eligible is False
    assert result.row_count == 0
    assert not list(tmp_path.glob("*.jsonl"))


def test_free_text_provenance_cannot_reach_the_local_manifest(tmp_path: Path) -> None:
    """A free-string field would have been a channel into a shareable artifact."""
    secret = "SECRET-CUSTOMER-PROMPT-a1b2c3"
    _build(tmp_path, secret)

    for path in tmp_path.rglob("*"):
        if path.is_file():
            assert secret not in path.read_text(encoding="utf-8")
