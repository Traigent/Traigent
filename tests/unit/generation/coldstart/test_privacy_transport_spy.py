"""The wire payload must be content-free: no names, annotation text,
docstrings, module paths, or default values -- only coarse type shape.

This is the required "transport spy" test: a distinctive signature is used
so any leak (parameter name, annotation text, docstring, file path, default
value) shows up as a literal substring match against the serialized outbound
request.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from traigent.generation.coldstart import (
    LocalVerifier,
    ScoreReceipt,
    build_cold_start_eval_set,
)
from traigent.generation.coldstart._plan import TransportResponse


def customer_scoring_function(customer_ssn: str, secret_threshold: int = 700) -> bool:
    """This docstring names a sensitive_business_rule and must never be sent."""
    return len(customer_ssn) > secret_threshold


class _RecordingTransport:
    def __init__(self) -> None:
        self.requests: list[Any] = []

    def __call__(self, request: Any) -> TransportResponse:
        self.requests.append(request)
        return TransportResponse(
            status_code=422,
            body={"error": "no local verifier", "reason": "no_local_scoring_authority"},
        )


class _PassthroughVerifier(LocalVerifier):
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
        yield ({"customer_ssn": f"ssn-{i}", "secret_threshold": i}, True)


def test_outbound_payload_excludes_names_annotations_docstrings_paths_and_defaults(
    tmp_path: Path,
) -> None:
    transport = _RecordingTransport()

    build_cold_start_eval_set(
        customer_scoring_function,
        generator=_generator,
        verifier=_PassthroughVerifier(),
        transport=transport,
        output_dir=tmp_path,
        generation_capabilities=("customer_llm",),
    )

    assert len(transport.requests) == 1
    serialized = json.dumps(transport.requests[0]).lower()

    forbidden = [
        "customer_ssn",
        "secret_threshold",
        "700",  # the default value
        "sensitive_business_rule",  # the docstring
        "customer_scoring_function",  # the function name
        __file__.lower(),  # the module's file path
        "self",
        "docstring",
    ]
    for marker in forbidden:
        assert marker not in serialized, (
            f"leaked {marker!r} into the wire payload: {serialized}"
        )

    # Only the coarse-shape fields may appear.
    assert set(transport.requests[0]) == {"protocol_version", "descriptor", "budget"}
    assert set(transport.requests[0]["descriptor"]) == {
        "input_arity",
        "input_kinds",
        "output_kind",
        "verifier_kinds",
        "generation_capabilities",
    }
    assert transport.requests[0]["budget"] == {"candidate_limit": 12}


def test_no_network_call_when_generator_is_missing(tmp_path: Path) -> None:
    transport = _RecordingTransport()

    result = build_cold_start_eval_set(
        customer_scoring_function,
        generator=None,
        verifier=_PassthroughVerifier(),
        transport=transport,
        output_dir=tmp_path,
        generation_capabilities=("customer_llm",),
    )

    assert transport.requests == []
    assert result.gap is not None
    assert result.gap.reason == "no_generator_supplied"
    assert result.optimizer_eligible is False
    assert list(tmp_path.iterdir()) == []


def test_no_network_call_when_verifier_is_missing(tmp_path: Path) -> None:
    transport = _RecordingTransport()

    result = build_cold_start_eval_set(
        customer_scoring_function,
        generator=_generator,
        verifier=None,
        transport=transport,
        output_dir=tmp_path,
        generation_capabilities=("customer_llm",),
    )

    assert transport.requests == []
    assert result.gap is not None
    assert result.gap.reason == "no_verifier_supplied"
    assert result.optimizer_eligible is False
    assert list(tmp_path.iterdir()) == []
