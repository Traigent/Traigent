"""Per-example signals derived locally so evaluation content never leaves the client.

The platform can assess evaluator quality from a run's per-example record, but the
assessment needs three things it cannot get from aggregate metrics: a stable identity
for the example, a stable identity for the output, and whether the output actually
matched the expected answer.

Sending the text itself would answer all three -- and would ship every prompt,
completion and gold label off the machine. It is also unnecessary: the destination
stores only digests, booleans and floats, never content. So the SDK computes the three
signals here, where the content already is, and sends fixed-width digests plus one
number.

What leaves the client per example:

============================  =========================================================
``example_digest``            64-hex digest of (input, expected output)
``output_digest``             64-hex digest of the produced output
``verified_match``            ``1.0`` / ``0.0`` -- did the output match the expected
                              answer under the SDK's own comparison? Omitted entirely
                              when the example has no usable expected answer.
============================  =========================================================

A digest is one-way: it identifies an example across runs so the same example can be
recognised, and reveals nothing about its content. Two runs over the same dataset
produce the same digests; a changed prompt produces a different one.

**The comparison is deliberately not a new one.** ``verified_match`` reuses
:func:`~traigent.evaluators.base._accuracy_values_match`, the same predicate the
built-in scorer already applies, gated by the same empty-expected-output rule. A second
implementation would drift from the first and the two would disagree on exactly the
examples that matter.

**It is also independent of any judge.** The comparison never consults an evaluator's
verdict, only the recorded output and the dataset's expected answer, so it stays usable
as a reference point for assessing the evaluator itself.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

#: Domain separator, so a digest computed for one purpose can never collide with a
#: digest computed for another even on identical bytes.
_EXAMPLE_DOMAIN = "traigent.example.v1"
_OUTPUT_DOMAIN = "traigent.output.v1"

#: Sibling keys attached to a per-example record. Neutral, outcome-shaped names: they
#: describe the user's own data, not how the platform uses them.
EXAMPLE_DIGEST_KEY = "example_digest"
OUTPUT_DIGEST_KEY = "output_digest"
VERIFIED_MATCH_KEY = "verified_match"


def _canonical(value: Any) -> str:
    """Stable text for a JSON-ish value, so equal values always digest equally.

    Sorted keys and tight separators remove dict-ordering and whitespace as sources
    of difference. Values that are not JSON-serialisable fall back to ``repr``, which
    keeps the digest defined rather than raising in the middle of a run -- an
    undigestable output is still a real output that must be counted.
    """
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
            default=repr,
        )
    except (TypeError, ValueError):
        return repr(value)


def _digest(domain: str, value: Any) -> str:
    payload = f"{domain}\x00{_canonical(value)}".encode()
    return hashlib.sha256(payload).hexdigest()


def example_digest(input_data: Any, expected_output: Any) -> str:
    """Stable identity for an example, from its input and expected answer."""
    return _digest(_EXAMPLE_DOMAIN, {"input": input_data, "expected": expected_output})


def output_digest(actual_output: Any) -> str:
    """Stable identity for a produced output."""
    return _digest(_OUTPUT_DOMAIN, actual_output)


def verified_match(
    actual_output: Any, expected_output: Any, *, errored: bool = False
) -> float | None:
    """``1.0``/``0.0`` if the output matched the expected answer, else ``None``.

    ``None`` means "no usable expected answer, so this example cannot be checked" --
    which is materially different from ``0.0`` ("checked, and it did not match"). The
    caller must omit the key rather than coerce ``None`` to a number: recording an
    uncheckable example as a failure would understate quality on exactly the datasets
    that lack gold labels.

    An errored call counts as a non-match rather than as uncheckable, matching the
    built-in scorer: a config that fails on an example did not get it right.
    """
    from traigent.evaluators.base import (
        _accuracy_values_match,
        _is_empty_expected_output,
    )

    if _is_empty_expected_output(expected_output):
        return None
    if errored:
        return 0.0
    return 1.0 if _accuracy_values_match(actual_output, expected_output) else 0.0


def build_example_signals(example_result: Any) -> dict[str, Any]:
    """The signal sibling keys for one example result.

    Returns only keys that are meaningful for this example: ``verified_match`` is
    absent when there is no usable expected answer. Never raises -- a signal that
    cannot be computed is omitted, because failing to describe an example must not
    fail the run that produced it.
    """
    signals: dict[str, Any] = {}
    try:
        input_data = getattr(example_result, "input_data", None)
        expected = getattr(example_result, "expected_output", None)
        actual = getattr(example_result, "actual_output", None)
        errored = getattr(example_result, "error_message", None) is not None

        signals[EXAMPLE_DIGEST_KEY] = example_digest(input_data, expected)
        signals[OUTPUT_DIGEST_KEY] = output_digest(actual)

        match = verified_match(actual, expected, errored=errored)
        if match is not None:
            signals[VERIFIED_MATCH_KEY] = match
    except Exception:  # noqa: BLE001 - signals are diagnostic, never load-bearing
        return {}
    return signals
