"""Every failure mode fails CLOSED: DISCOVERY_ONLY, optimizer_eligible=False,
a typed DiscoveryGap, and -- critically -- zero artifact files on disk.
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
from traigent.generation.coldstart._contract import compute_descriptor_digest
from traigent.generation.coldstart._descriptor import build_descriptor
from traigent.generation.coldstart._plan import TransportResponse


def target(a: str, b: int) -> bool:
    return True


class _Verifier(LocalVerifier):
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


def _valid_descriptor() -> dict:
    return build_descriptor(
        target,
        verifier_kinds=("executable_property",),
        generation_capabilities=("customer_llm",),
    )


def _valid_body(
    *, candidate_limit: int = 5, expires_in: timedelta = timedelta(hours=1)
) -> dict:
    descriptor = _valid_descriptor()
    return {
        "plan_id": "csp_ok",
        "protocol_version": "cold-start.v1",
        "descriptor_digest": compute_descriptor_digest(descriptor),
        "candidate_limit": candidate_limit,
        "expires_at": (datetime.now(UTC) + expires_in).isoformat(),
    }


def _run(transport, tmp_path: Path, **kwargs):
    kwargs.setdefault("generation_capabilities", ("customer_llm",))
    return build_cold_start_eval_set(
        target,
        generator=_generator,
        verifier=_Verifier(),
        transport=transport,
        output_dir=tmp_path,
        **kwargs,
    )


def _assert_failed_closed(result, tmp_path: Path, expected_reason: str) -> None:
    assert result.outcome is ColdStartOutcome.DISCOVERY_ONLY
    assert result.optimizer_eligible is False
    assert result.eval_set_path is None
    assert result.manifest_path is None
    assert result.row_count == 0
    assert result.receipts == ()
    assert result.gap is not None
    assert result.gap.reason == expected_reason
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    "reason",
    [
        "no_local_scoring_authority",
        "no_local_generation_capability",
        "descriptor_arity_mismatch",
    ],
)
def test_422_each_backend_reason_fails_closed(tmp_path: Path, reason: str) -> None:
    def transport(request):
        return TransportResponse(422, {"error": "declined", "reason": reason})

    result = _run(transport, tmp_path)
    _assert_failed_closed(result, tmp_path, reason)
    assert result.gap.http_status == 422


def test_422_unrecognized_reason_is_malformed_response(tmp_path: Path) -> None:
    def transport(request):
        return TransportResponse(422, {"error": "declined", "reason": "something_new"})

    result = _run(transport, tmp_path)
    _assert_failed_closed(result, tmp_path, "malformed_response")


def test_501_feature_disabled_fails_closed(tmp_path: Path) -> None:
    def transport(request):
        return TransportResponse(501, {})

    result = _run(transport, tmp_path)
    _assert_failed_closed(result, tmp_path, "feature_disabled")


@pytest.mark.parametrize("status", [401, 403])
def test_auth_failures_fail_closed(tmp_path: Path, status: int) -> None:
    def transport(request):
        return TransportResponse(status, {})

    result = _run(transport, tmp_path)
    _assert_failed_closed(result, tmp_path, "unauthorized")
    assert result.gap.http_status == status


@pytest.mark.parametrize(
    "body",
    [
        {},
        {"plan_id": "x"},
        {**_valid_body(), "extra_field": "not allowed"},
    ],
)
def test_malformed_200_body_fails_closed(tmp_path: Path, body: dict) -> None:
    def transport(request):
        return TransportResponse(200, body)

    result = _run(transport, tmp_path)
    _assert_failed_closed(result, tmp_path, "malformed_response")


def test_descriptor_digest_mismatch_is_refused(tmp_path: Path) -> None:
    def transport(request):
        body = _valid_body()
        body["descriptor_digest"] = "0" * 64  # wrong digest
        return TransportResponse(200, body)

    result = _run(transport, tmp_path)
    _assert_failed_closed(result, tmp_path, "descriptor_digest_mismatch")


def test_expired_plan_is_rejected_before_generating(tmp_path: Path) -> None:
    calls = []

    def spying_generator(limit: int):
        calls.append(limit)
        yield from _generator(limit)

    def transport(request):
        return TransportResponse(200, _valid_body(expires_in=-timedelta(minutes=1)))

    result = build_cold_start_eval_set(
        target,
        generator=spying_generator,
        verifier=_Verifier(),
        transport=transport,
        output_dir=tmp_path,
        generation_capabilities=("customer_llm",),
    )
    _assert_failed_closed(result, tmp_path, "plan_expired")
    assert calls == []  # generation must never run against an expired plan


def test_expiry_uses_injected_clock(tmp_path: Path) -> None:
    """A plan expiring 30 minutes from real now is already expired per an
    injected clock set an hour into the future -- proves the clock, not
    wall time, drives the check."""

    def transport(request):
        return TransportResponse(200, _valid_body(expires_in=timedelta(minutes=30)))

    def future_clock() -> datetime:
        return datetime.now(UTC) + timedelta(hours=1)

    result = _run(transport, tmp_path, clock=future_clock)
    _assert_failed_closed(result, tmp_path, "plan_expired")


def test_grant_below_ask_is_honoured(tmp_path: Path) -> None:
    def transport(request):
        assert request["budget"]["candidate_limit"] == 12
        return TransportResponse(200, _valid_body(candidate_limit=3))

    result = _run(transport, tmp_path, requested_candidate_limit=12)

    assert result.outcome is ColdStartOutcome.EVAL_SET_BUILT
    assert result.row_count == 3
    lines = result.eval_set_path.read_text().strip().splitlines()
    assert len(lines) == 3


def test_no_verified_candidates_fails_closed(tmp_path: Path) -> None:
    def transport(request):
        return TransportResponse(200, _valid_body())

    def empty_generator(limit: int):
        return iter(())

    result = build_cold_start_eval_set(
        target,
        generator=empty_generator,
        verifier=_Verifier(),
        transport=transport,
        output_dir=tmp_path,
        generation_capabilities=("customer_llm",),
    )
    _assert_failed_closed(result, tmp_path, "no_verified_candidates")
