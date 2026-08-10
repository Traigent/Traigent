"""The cold-start package must stay free of generation TECHNIQUE and credentials.

## Why this file exists, and why it is narrower than the test it replaces

PRs #2082/#2083 (2026-08-03) pulled the cold-start SDK boundary back after a
public draft exposed a broad architecture whose built-in generator was only a
placeholder emitting marker strings — we had shipped the maintenance commitment
and none of the technique. The guard installed then, `test_implementation_has_no
_forbidden_markers`, banned every one of:

    inspect. verifier candidate plan_ score prompt threshold
    urllib http api_key write_text jsonl

That guard was calibrated to an interim state in which the package was a
96-line opaque stub that did nothing. Seven of those markers describe work the
APPROVED design explicitly assigns to the customer's machine — the discovery
report's boundary table puts "decorated-function inspection, local target
execution, Oracle/verifier execution, input screening and deduplication,
score-receipt validation, and all generated examples/manifests" on the CLIENT
side. A local executor cannot inspect a signature without `inspect.`, verify a
row without a `verifier`, record a `score`, or write a tuning set without
`jsonl`. Keeping the ban would have forbidden the feature, not protected it.

So the ban is kept for the markers that still mean "proprietary technique or a
credential leaked into the public SDK", and dropped for the ones the approved
design requires. Deleting the guard outright — which is what happens by default
when an implementation collides with it — would have thrown away the real
protection along with the over-broad part.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from traigent.generation import coldstart

# Each entry is a marker that must NOT appear, with the reason it is banned.
# A reason is required: a banned string with no stated harm is how a guard rots
# into cargo cult and gets deleted wholesale the next time it is inconvenient.
FORBIDDEN_MARKERS = {
    "prompt": "a built-in prompt is generation technique; prompts belong to the caller",
    "threshold": "a tuned constant is technique and would leak calibration",
    "api_key": "this package never handles credentials; the transport is injected",
    "urllib": "the SDK must not open its own connections; the transport is injected",
    "requests.": "same — no direct HTTP client in the cold-start package",
    "httpx": "same — no direct HTTP client in the cold-start package",
    "openai": "no built-in model call; generation runs on the customer's own LLM",
    "anthropic": "no built-in model call; generation runs on the customer's own LLM",
    "litellm": "no built-in model call; generation runs on the customer's own LLM",
}

# Dropped from the original ban, with the reason each is now legitimate. Kept as
# data (not a comment) so a reader can see exactly what changed and why.
DELIBERATELY_ALLOWED = {
    "inspect.": "signature inspection is local and is how the content-free descriptor is built",
    "verifier": "the LocalVerifier runs on the customer machine by design",
    "candidate": "candidate rows are generated and screened locally",
    "plan_": "plan_id/plan parsing is the documented wire contract",
    "score": "score receipts are issued locally and are a public contract type",
    "write_text": "tuning JSONL and the manifest are written locally, never uploaded",
    "jsonl": "the local tuning artifact format",
}


def _package_source() -> str:
    package = Path(coldstart.__file__).parent
    return "\n".join(
        path.read_text(encoding="utf-8").lower() for path in package.glob("*.py")
    )


@pytest.mark.parametrize(("marker", "reason"), sorted(FORBIDDEN_MARKERS.items()))
def test_package_carries_no_technique_or_credential_marker(
    marker: str, reason: str
) -> None:
    assert marker not in _package_source(), f"{marker!r} must not appear here: {reason}"


def test_the_relaxation_is_explicit_and_disjoint() -> None:
    """Nothing may be both banned and allowed — that would make the ban unreadable."""
    assert not (set(FORBIDDEN_MARKERS) & set(DELIBERATELY_ALLOWED))


def test_generation_technique_is_not_shipped() -> None:
    """The caller supplies the generator. The SDK must not default one in.

    This is the substance the marker ban is a proxy for: `build_cold_start_eval_set`
    must have no default generator, so a caller cannot accidentally run Traigent's
    technique instead of their own.
    """
    import inspect as _inspect

    signature = _inspect.signature(coldstart.build_cold_start_eval_set)
    generator = signature.parameters.get("generator")

    assert generator is not None, "generator must be an explicit parameter"
    assert generator.default is _inspect.Parameter.empty, (
        "generator must have NO default — a default generator would be shipped technique"
    )
