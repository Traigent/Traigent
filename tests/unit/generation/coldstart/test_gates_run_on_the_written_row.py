"""Every admission gate must run on the bytes that get written.

Validating one value and writing another is the same defect as certifying one
value and writing another -- just on a different gate. A caller-supplied object
may answer differently on each read, so a shape that binds to the target on read
1 can be a shape that does not on read 3.

Reproduced on merged develop before the fix: with the bind check running on the
snapshot and the frozen value written, a row was written whose only key was one
the target cannot accept, having passed the bind check moments earlier. Sweeping
the read threshold, 2 of 5 positions produced an uncallable row.

The gates now run on the frozen value, so "what was checked" and "what was
written" are the same bytes by construction.
"""

from __future__ import annotations

import json
import tempfile
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
from traigent.generation.coldstart._contract import compute_descriptor_digest
from traigent.generation.coldstart._plan import TransportResponse


def target(message: str) -> str:
    return "ok"


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


class _Accepting(LocalVerifier):
    kind = "executable_property"

    def verify(self, *, inputs: Any, output: Any) -> ScoreReceipt | None:
        return ScoreReceipt(
            verifier_id="v1",
            verifier_kind=self.kind,
            passed=True,
            provenance="oracle_returned",
        )


def _shifting_inputs(threshold: int):
    """Binds to target(message=...) for the first `threshold` reads, then does not."""

    class _Shifting(dict):
        def __init__(self) -> None:
            super().__init__(message="hi")
            self._reads = 0

        def __deepcopy__(self, memo: Any) -> _Shifting:
            return self

        def _shape(self):
            self._reads += 1
            return (
                [("message", "hi")]
                if self._reads <= threshold
                else [("NOT_A_PARAM", "hi")]
            )

        def items(self):  # noqa: ANN201 - dict protocol
            return self._shape()

        def keys(self):  # noqa: ANN201 - dict protocol
            return [k for k, _ in self._shape()]

        def __getitem__(self, key: str) -> str:
            return "hi"

    return _Shifting


@pytest.mark.parametrize("threshold", [1, 2, 3, 4, 5])
def test_a_row_that_stops_binding_never_reaches_the_eval_set(threshold: int) -> None:
    """Whatever read the object shifts on, the written row must bind to the target."""
    shifting = _shifting_inputs(threshold)

    def generator(limit: int):
        yield (shifting(), "ok")

    with tempfile.TemporaryDirectory() as directory:
        result = build_cold_start_eval_set(
            target,
            generator=generator,
            verifier=_Accepting(),
            transport=_transport,
            output_dir=directory,
            generation_capabilities=("deterministic_contract",),
        )

        if result.outcome is ColdStartOutcome.DISCOVERY_ONLY:
            # Rejecting outright is also correct -- what must never happen is a
            # written row the target cannot be called with.
            assert not list(Path(directory).glob("*.jsonl"))
            return

        written = json.loads(
            Path(result.eval_set_path).read_text(encoding="utf-8").splitlines()[0]
        )["input"]

    assert set(written) == {"message"}, (
        f"wrote a row keyed {set(written)}, which target(message=...) cannot accept"
    )
