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
import re
import threading
from collections.abc import Mapping
from typing import Any

from traigent.utils.logging import get_logger

logger = get_logger(__name__)

#: Domain separator, so a digest computed for one purpose can never collide with a
#: digest computed for another even on identical bytes.
_EXAMPLE_DOMAIN = "traigent.example.v1"
_OUTPUT_DOMAIN = "traigent.output.v1"

#: Sibling keys attached to a per-example record. Neutral, outcome-shaped names: they
#: describe the user's own data, not how the platform uses them.
EXAMPLE_DIGEST_KEY = "example_digest"
OUTPUT_DIGEST_KEY = "output_digest"
VERIFIED_MATCH_KEY = "verified_match"

#: The default ``object.__repr__`` embeds the object's memory address
#: (``<Foo object at 0x7f...>``), which differs across processes and even across
#: runs within one process (ASLR). A repr matching this is not a stable identity
#: and must never be digested.
_MEMORY_ADDRESS_PATTERN = re.compile(r"0x[0-9a-fA-F]{4,}")


class _Unstable(Exception):
    """Internal signal: a value has no deterministic canonical form.

    Never escapes this module -- callers see ``None`` (digest omitted), not an
    exception.
    """


def _example_field(example_result: Any, name: str, default: Any = None) -> Any:
    """Read a field from an example result object OR its dict payload form.

    Trial metadata stores example results as redacted ``to_dict()`` payloads
    (see ``trial_result_factory._to_redactable_payloads``), so callers must
    read plain dicts as well as ``ExampleResult`` objects.
    """
    if isinstance(example_result, Mapping):
        return example_result.get(name, default)
    return getattr(example_result, name, default)


def _stable_repr(value: Any) -> str:
    """``repr(value)``, rejected if it embeds a memory address."""
    try:
        text = repr(value)
    except Exception as exc:  # noqa: BLE001 - repr() itself is untrusted here
        raise _Unstable from exc
    if _MEMORY_ADDRESS_PATTERN.search(text):
        raise _Unstable
    return text


def _canonicalize(value: Any) -> Any:
    """Recursively convert ``value`` into a structure ``json.dumps`` renders the
    same way every time.

    Sets and dict ordering are otherwise sources of run-to-run difference for an
    otherwise-identical example: dict key order is normalised by
    ``json.dumps(sort_keys=True)`` at the caller, and set/frozenset members are
    sorted here (Python's set iteration order depends on hash randomisation,
    which varies across processes). Anything left over (an arbitrary object) is
    canonicalised via its ``repr`` -- but only when that ``repr`` does not embed
    a memory address, since an address-bearing repr is not a stable identity.
    """
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {
            (key if isinstance(key, str) else _stable_repr(key)): _canonicalize(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_canonicalize(item) for item in value]
    if isinstance(value, (set, frozenset)):
        canonicalized = [_canonicalize(item) for item in value]
        return sorted(
            canonicalized,
            key=lambda item: json.dumps(
                item, sort_keys=True, separators=(",", ":"), ensure_ascii=False
            ),
        )
    return _stable_repr(value)


def _canonical(value: Any) -> str | None:
    """Stable text for a value, so equal values always digest equally and
    unstable ones never digest at all.

    Returns ``None`` -- never a value that merely looks stable -- when no
    deterministic canonical form exists, so the caller omits the signal rather
    than emit a digest that would silently vary across processes.
    """
    try:
        structure = _canonicalize(value)
    except _Unstable:
        return None
    try:
        return json.dumps(
            structure,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError):
        return None


def _digest(domain: str, value: Any) -> str | None:
    canonical = _canonical(value)
    if canonical is None:
        return None
    payload = f"{domain}\x00{canonical}".encode()
    return hashlib.sha256(payload).hexdigest()


def example_digest(input_data: Any, expected_output: Any) -> str | None:
    """Stable identity for an example, from its input and expected answer.

    ``None`` when no deterministic digest exists for this content (see
    :func:`_canonical`) -- never an unstable one.
    """
    return _digest(_EXAMPLE_DOMAIN, {"input": input_data, "expected": expected_output})


def output_digest(actual_output: Any) -> str | None:
    """Stable identity for a produced output. ``None`` if it can't be made stable."""
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


#: Counts total signal-build failures process-wide, so a systemic failure (every
#: example silently producing ``{}``) is observable instead of indistinguishable
#: from "this dataset has no expected outputs". Never reset -- it's a lifetime
#: counter for the log line's own "count so far" context, not a rolling window.
_failure_count = 0
_failure_count_lock = threading.Lock()


def _note_signal_failure(exc: Exception) -> None:
    """Rate-limited, content-free observability for a failed signal build.

    Logs the exception TYPE only, never ``str(exc)`` -- a message can echo
    interpolated data (e.g. a comparison failure embedding a value) even for
    exception types that look innocuous. Logs the first few failures immediately
    (a run-starting misconfiguration should surface fast) then falls back to
    every 100th, so a systemic failure across a large run does not flood logs
    but also never goes silent.
    """
    with _failure_count_lock:
        global _failure_count
        _failure_count += 1
        count = _failure_count
    if count <= 3 or count % 100 == 0:
        logger.warning(
            "outcome_signals: could not derive per-example signals (%s); "
            "%d failure(s) so far this process",
            type(exc).__name__,
            count,
        )


def build_example_signals(example_result: Any) -> dict[str, Any]:
    """The signal sibling keys for one example result.

    Returns only keys that are meaningful for this example: ``verified_match`` is
    absent when there is no usable expected answer, and either digest is absent
    when its content has no deterministic canonical form (see ``_canonical``).
    Never raises -- a signal that cannot be computed is omitted, because failing
    to describe an example must not fail the run that produced it. A failure is
    still recorded (content-free) via ``_note_signal_failure`` so a systemic
    failure is visible rather than indistinguishable from "no expected outputs".
    """
    signals: dict[str, Any] = {}
    try:
        input_data = _example_field(example_result, "input_data")
        expected = _example_field(example_result, "expected_output")
        actual = _example_field(example_result, "actual_output")
        errored = _example_field(example_result, "error_message") is not None

        digest = example_digest(input_data, expected)
        if digest is not None:
            signals[EXAMPLE_DIGEST_KEY] = digest

        out_digest = output_digest(actual)
        if out_digest is not None:
            signals[OUTPUT_DIGEST_KEY] = out_digest

        match = verified_match(actual, expected, errored=errored)
        if match is not None:
            signals[VERIFIED_MATCH_KEY] = match
    except Exception as exc:  # noqa: BLE001 - signals are diagnostic, never load-bearing
        _note_signal_failure(exc)
        return {}
    return signals
