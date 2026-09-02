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
``example_digest``            64-hex keyed digest of (input, expected output)
``output_digest``             64-hex keyed digest of the produced output
``verified_match``            ``1.0`` / ``0.0`` -- did the output match the expected
                              answer under the SDK's own comparison? Omitted entirely
                              when the example has no usable expected answer.
``signal_key_id``             12-hex tag identifying which key version the two digests
                              above were computed under (see "Keying", below).
============================  =========================================================

**Keying, and what it does and does not claim.** The digests are HMAC-SHA256, not bare
SHA-256, keyed with material derived from the project's own API key
(:func:`traigent.config.backend_config.BackendConfig.get_api_key`). This is an identity
mechanism, not a secrecy guarantee: a bare digest over a short, low-entropy value (a
"4", a class label, "yes") is a confirmation oracle -- anyone holding the wire payload
can hash every plausible candidate and see which one matches, recovering the gold label
without ever seeing it directly. Keying closes that off for anyone who does NOT hold the
project's key material: they cannot enumerate candidates against a digest they cannot
reproduce. It does NOT close it off for Traigent itself -- Traigent holds the project key
and could, in principle, recompute a candidate digest and compare. Two runs over the same
dataset under the same key produce the same digests; a changed prompt, or a rotated key,
produces a different one. ``signal_key_id`` lets the backend tell those two causes apart
(new example vs. rotated key) without itself revealing the key -- it is a one-way tag of
the key material, not the key or a way to derive it.

Every one of the four keys is fail-closed on missing key material: if no project API key
is configured, this module emits NONE of them, never an unkeyed digest.

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

import functools
import hashlib
import hmac
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

#: Salts for deriving the HMAC key and the non-reversible key-id tag from the project
#: API key via PBKDF2 (see ``_pbkdf2``). Distinct from each other so neither derivation
#: can be used to reconstruct the other, and distinct from the two digest domains above
#: so nothing here can collide with an example/output digest on identical input bytes.
_KEY_DERIVATION_SALT = b"traigent.signal.key.v1"
_KEY_ID_SALT = b"traigent.signal.keyid.v1"

#: PBKDF2-HMAC-SHA256 iteration count for deriving signal key material from the
#: project API key. THIS IS PART OF THE IDENTITY CONTRACT, not a tunable perf knob:
#: this count, the two salts above, and the algorithm together determine every
#: ``example_digest``/``output_digest``/``signal_key_id`` this module has ever
#: produced. Changing this value changes every derived key and id for every
#: project -- silently breaking cross-run digest joins for data already on the
#: backend, which is exactly the class of silent breakage this module exists to
#: remove. A genuine strengthening (e.g. raising this as hardware improves) must
#: ship as a new, distinctly-salted derivation (bump the salts to ``...v2``), not
#: an edit to this constant.
_KDF_ITERATIONS = 600_000

#: Sibling keys attached to a per-example record. Neutral, outcome-shaped names: they
#: describe the user's own data, not how the platform uses them.
EXAMPLE_DIGEST_KEY = "example_digest"
OUTPUT_DIGEST_KEY = "output_digest"
VERIFIED_MATCH_KEY = "verified_match"
SIGNAL_KEY_ID_KEY = "signal_key_id"

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


def _digest(domain: str, value: Any, hmac_key: bytes) -> str | None:
    canonical = _canonical(value)
    if canonical is None:
        return None
    payload = f"{domain}\x00{canonical}".encode()
    return hmac.new(hmac_key, payload, hashlib.sha256).hexdigest()


def example_digest(
    input_data: Any, expected_output: Any, hmac_key: bytes
) -> str | None:
    """Stable identity for an example, from its input and expected answer.

    ``hmac_key`` is REQUIRED -- see :func:`_resolve_signal_key`. There is no
    unkeyed fallback: a bare digest over short, low-entropy content (a "4", a
    class label) is a confirmation oracle, so a digest with no key is not a
    degraded-but-safe result, it is the exact thing this keying exists to
    prevent.

    ``None`` when no deterministic digest exists for this content (see
    :func:`_canonical`) -- never an unstable one.
    """
    return _digest(
        _EXAMPLE_DOMAIN, {"input": input_data, "expected": expected_output}, hmac_key
    )


def output_digest(actual_output: Any, hmac_key: bytes) -> str | None:
    """Stable identity for a produced output. ``None`` if it can't be made stable.

    ``hmac_key`` is REQUIRED; see :func:`example_digest` for why there is no
    unkeyed fallback.
    """
    return _digest(_OUTPUT_DOMAIN, actual_output, hmac_key)


def _pbkdf2(api_key: str, salt: bytes) -> bytes:
    """PBKDF2-HMAC-SHA256 over the project API key, one genuine KDF pass.

    A single fast hash (even domain-separated, even HMAC-keyed elsewhere) is
    the wrong primitive for turning a credential-shaped secret into derived key
    material: it is exactly what CodeQL's ``py/weak-sensitive-data-hashing``
    flags, because a fast hash makes offline guessing of a low-entropy or
    reused API key cheap. PBKDF2 with a real iteration count (``_KDF_ITERATIONS``)
    makes each guess expensive instead.
    """
    return hashlib.pbkdf2_hmac("sha256", api_key.encode(), salt, _KDF_ITERATIONS)


def _derive_signal_key(api_key: str) -> bytes:
    """Derive HMAC key material from the project API key. Never the raw key
    itself -- a derived value so this module never uses (or could leak) the
    key used to authenticate to the backend as a digest key directly."""
    return _pbkdf2(api_key, _KEY_DERIVATION_SALT)


def _signal_key_id(api_key: str) -> str:
    """Non-reversible tag for the current key version.

    A distinct salt from :func:`_derive_signal_key` so the id can never be used
    to reconstruct the HMAC key, or vice versa. Truncated to 12 hex chars --
    enough to distinguish key versions for a single project, not an independent
    secret.
    """
    return _pbkdf2(api_key, _KEY_ID_SALT).hex()[:12]


@functools.lru_cache(maxsize=8)
def _cached_key_pair(api_key: str) -> tuple[bytes, str]:
    """(HMAC key, key id) for one API key, computed once per process per key.

    ``_KDF_ITERATIONS`` is deliberately expensive (that is the point of a KDF),
    so paying it once per example across a large evaluation run would be a real
    performance regression. ``lru_cache`` keys strictly on the ``api_key``
    VALUE passed in -- a key rotation within one process gets its own,
    independent cache entry rather than reusing another key's derived
    material, so the cache cannot leak identity across different API keys.
    """
    return _derive_signal_key(api_key), _signal_key_id(api_key)


def _resolve_signal_key() -> tuple[bytes, str] | None:
    """The current (HMAC key, key id) pair, or ``None`` if unavailable.

    ``None`` means "no project API key is configured" -- every caller in this
    module must treat that as fail-closed: omit every signal, never fall back
    to an unkeyed digest. A silent downgrade to the unkeyed construction would
    quietly reintroduce the confirmation-oracle problem the keying exists to
    close.
    """
    from traigent.config.backend_config import BackendConfig

    api_key = BackendConfig.get_api_key()
    if not api_key:
        return None
    return _cached_key_pair(api_key)


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

    Fail-closed on key material: if no project API key is configured (see
    :func:`_resolve_signal_key`), this returns ``{}`` -- NONE of
    ``example_digest``/``output_digest``/``verified_match``/``signal_key_id`` are
    emitted. There is no unkeyed fallback; see the module docstring for why an
    unkeyed digest is a confirmation oracle, not a degraded-but-safe signal.

    Otherwise returns only keys that are meaningful for this example:
    ``verified_match`` is absent when there is no usable expected answer, and
    either digest is absent when its content has no deterministic canonical form
    (see ``_canonical``). Never raises -- a signal that cannot be computed is
    omitted, because failing to describe an example must not fail the run that
    produced it. A failure is still recorded (content-free) via
    ``_note_signal_failure`` so a systemic failure is visible rather than
    indistinguishable from "no expected outputs".
    """
    signals: dict[str, Any] = {}
    try:
        key_material = _resolve_signal_key()
        if key_material is None:
            return {}
        hmac_key, key_id = key_material

        input_data = _example_field(example_result, "input_data")
        expected = _example_field(example_result, "expected_output")
        actual = _example_field(example_result, "actual_output")
        errored = _example_field(example_result, "error_message") is not None

        digest = example_digest(input_data, expected, hmac_key)
        if digest is not None:
            signals[EXAMPLE_DIGEST_KEY] = digest

        out_digest = output_digest(actual, hmac_key)
        if out_digest is not None:
            signals[OUTPUT_DIGEST_KEY] = out_digest

        match = verified_match(actual, expected, errored=errored)
        if match is not None:
            signals[VERIFIED_MATCH_KEY] = match

        if signals:
            signals[SIGNAL_KEY_ID_KEY] = key_id
    except Exception as exc:  # noqa: BLE001 - signals are diagnostic, never load-bearing
        _note_signal_failure(exc)
        return {}
    return signals
