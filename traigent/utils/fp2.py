"""fp2 canonicalization/digest algorithm, ported from TraigentSchema.

Normative document: TraigentSchema `docs/fingerprints/fp2.md`.
Reference implementation: TraigentSchema `traigent_schema/fp2.py`.

Vendored, not imported: `pyproject.toml` forbids a direct git dependency on
TraigentSchema (PyPI rejects git deps even inside an extra), so this module
is a line-for-line port rather than an import. Ported from TraigentSchema
commit ``fbf8d734f865073f9ce52744377516c7a94a8295`` (``origin/develop``,
2026-08-10). The conformance corpus vendored alongside this module's tests
(``tests/unit/utils/fixtures/fp2/agent_lifecycle_cases.json``) was pulled
from the same commit -- see that file's sibling ``SOURCE.md`` for the exact
provenance. If TraigentSchema's `fp2.py` or its corpus changes, re-diff
against that commit and re-vendor; nothing here re-checks it automatically.

**This module is `fp2`, not `fp1`.** `traigent.utils.artifact_fingerprints`
(`fp1`) is a different, incompatible algorithm: it canonicalizes with
``json.dumps(..., default=str)``, which fp2's own spec explicitly forbids
("Never coerce what you cannot represent"). Do not extend that module with
fp2 behavior and do not import from it here; the two must stay independent
until fp1 is deliberately retired.

Scope note (as of this port): fp2 digests never cross the wire yet. Every
agent-lifecycle Route 1-4 request/response schema in TraigentSchema has zero
`fp2`/`afp2`/`dfp2o`/`efp2`/`cfp2`/`ArtifactVersion` fields (only a
forbid-list mentions the names), and all four artifact-version slots on
Route 5 are unconditional ``state: "UNKNOWN", ref: null``. This module is
therefore local computation only: nothing here sends a digest anywhere, and
nothing reads `state`/`ref` off a version slot (that value cannot vary yet).

Two rules exist purely so Python and JavaScript agree byte-for-byte, and
both are easy to get wrong in the obvious way:

* Object keys are ordered by **UTF-16 code unit**, not by code point.
  Python's bare ``sorted()`` compares code points, which inverts against
  JavaScript for any key containing an astral character.
* Numbers are serialized with the **ECMAScript ``Number::toString``**
  algorithm, not with Python's ``repr``. ``repr(1e16)`` is ``'1e+16'``
  where JavaScript emits ``10000000000000000``.

Both rules follow RFC 8785 (JSON Canonicalization Scheme).
"""

from __future__ import annotations

import hashlib
import math
from typing import Any

__all__ = ["MAX_DEPTH", "Fp2UnsupportedValue", "canonicalize", "digest"]

# IEEE-754 doubles represent every integer up to this magnitude exactly.
# A Python int beyond it cannot round-trip through a JavaScript Number, so it
# is unsupported rather than silently divergent (see fp2.md).
_MAX_SAFE_INTEGER = 2**53 - 1

# Normative nesting limit, identical in every fp2 implementation. It is part of
# the spec rather than a Python detail on purpose: if each SDK simply nested
# until its own runtime gave out, Python (~332 levels at the default recursion
# limit) and JavaScript (thousands) would disagree about which manifests are
# digestible, which is the cross-language divergence fp2 exists to prevent.
# 100 is far above any real manifest and far below every target runtime's
# capacity, so a plain recursive implementation can comply without tricks.
MAX_DEPTH = 100


class Fp2UnsupportedValue(TypeError):
    """A value that fp2 cannot represent; the manifest is incomplete."""


_ESCAPES = {
    '"': '\\"',
    "\\": "\\\\",
    "\n": "\\n",
    "\r": "\\r",
    "\t": "\\t",
    "\b": "\\b",
    "\f": "\\f",
}


def _utf16_sort_key(value: str) -> bytes:
    """Return the UTF-16 code-unit sort key for ``value``.

    Comparing UTF-16 big-endian bytes lexicographically is exactly comparing
    UTF-16 code-unit sequences numerically, which is what JavaScript's ``<``
    does. Python's own string comparison uses code points and disagrees for
    astral characters, so the sort key cannot be the string itself.

    Raises:
        Fp2UnsupportedValue: ``value`` is not encodable text (a lone
            surrogate), so no two implementations could agree on its bytes.
    """
    try:
        return value.encode("utf-16-be")
    except UnicodeEncodeError as error:
        # The offending text is NOT echoed. fp2 runs over dataset rows and agent
        # bound state, so any value here may be user content, and an exception
        # message is one of the most reliably logged strings in a system.
        raise Fp2UnsupportedValue(
            f"string is not encodable text (lone surrogate at index {error.start})"
        ) from error


def _encode_string(value: str) -> str:
    # Reject un-encodable text (lone surrogates) here rather than letting a
    # UnicodeEncodeError escape from digest(): callers catch Fp2UnsupportedValue
    # to record state="unknown", and an unexpected exception type would crash
    # the run instead of degrading to unknown.
    _utf16_sort_key(value)
    out = ['"']
    for character in value:
        if character in _ESCAPES:
            out.append(_ESCAPES[character])
        elif character < " ":
            out.append(f"\\u{ord(character):04x}")
        else:
            out.append(character)
    out.append('"')
    return "".join(out)


def _shortest_digits(value: float) -> tuple[str, int]:
    """Decompose a positive finite float into ECMAScript's ``(s, n)``.

    Returns the shortest round-trip significant digits ``s`` and the exponent
    ``n`` such that ``0.s * 10**n == value``, matching the variables named in
    ECMA-262 Number::toString.
    """
    text = repr(value)
    mantissa, _, exponent_text = text.partition("e")
    exponent = int(exponent_text) if exponent_text else 0
    integer_part, _, fraction_part = mantissa.partition(".")

    digits = integer_part + fraction_part
    n = len(integer_part) + exponent

    stripped = digits.lstrip("0")
    n -= len(digits) - len(stripped)
    digits = stripped.rstrip("0") or "0"
    return digits, n


def _ecmascript_number_to_string(value: float) -> str:
    """Serialize a finite float exactly as ECMAScript ``Number::toString``.

    ECMA-262 section 6.1.6.1.20, which RFC 8785 adopts as the canonical JSON
    number format. Python's ``repr`` round-trips but picks different notation
    thresholds and pads exponents, so it cannot be used directly.
    """
    if value == 0:  # covers -0.0, which canonicalizes to 0
        return "0"
    if value < 0:
        return "-" + _ecmascript_number_to_string(-value)

    digits, n = _shortest_digits(value)
    k = len(digits)

    if k <= n <= 21:
        return digits + "0" * (n - k)
    if 0 < n <= 21:
        return digits[:n] + "." + digits[n:]
    if -6 < n <= 0:
        return "0." + "0" * (-n) + digits
    exponent = n - 1
    sign = "+" if exponent >= 0 else "-"
    significand = digits if k == 1 else digits[0] + "." + digits[1:]
    return f"{significand}e{sign}{abs(exponent)}"


def _encode_float(value: float) -> str:
    # Reached only for an exact float. A subclass would carry a caller-defined
    # __repr__/__float__ (numpy.float64 renders as "np.float64(0.1)") and is
    # rejected by the exact-type dispatch before it gets here.
    if math.isnan(value) or math.isinf(value):
        raise Fp2UnsupportedValue("non-finite number (NaN or Infinity)")
    return _ecmascript_number_to_string(value)


def _encode_int(value: int) -> str:
    # Exact int only; a subclass could override __int__ and choose its own digits.
    if abs(value) > _MAX_SAFE_INTEGER:
        raise Fp2UnsupportedValue(
            "integer outside the IEEE-754 safe range, not representable as a "
            f"JavaScript Number (limit {_MAX_SAFE_INTEGER})"
        )
    return str(value)


# Work-stack entry tags for the iterative encoder.
_EMIT = 0  # append literal text to the output
_VALUE = 1  # encode this value
_CLOSE = 2  # leave a container: drop it from the open-path set


def _encode(root: Any) -> str:
    """Encode ``root`` with an explicit work stack rather than recursion.

    Recursion would make the result depend on how much interpreter stack the
    *caller* happened to have left, so identical data could canonicalize from
    one call site and raise RecursionError from a deeper one. An explicit stack
    makes the outcome a property of the data alone; MAX_DEPTH is then the only
    thing that rejects a manifest, and it rejects it identically everywhere.
    """
    out: list[str] = []
    open_containers: set[int] = set()
    work: list[tuple[int, Any, int]] = [(_VALUE, root, 1)]

    while work:
        kind, payload, depth = work.pop()

        if kind == _EMIT:
            out.append(payload)
            continue
        if kind == _CLOSE:
            open_containers.discard(payload)
            continue

        value = payload
        if value is None:
            out.append("null")
            continue
        if value is True:
            out.append("true")
            continue
        if value is False:
            out.append("false")
            continue
        # Exact types only. isinstance() would admit subclasses, and a subclass
        # decides its own __iter__/items()/__int__/__float__ -- so user code,
        # not this function, would choose the canonical bytes. That is the same
        # hole as formatting through repr(), reached through a different door.
        # The spec already calls class instances unsupported; this makes the
        # implementation agree with it.
        if type(value) is str:
            out.append(_encode_string(value))
            continue
        if type(value) is int:
            out.append(_encode_int(value))
            continue
        if type(value) is float:
            out.append(_encode_float(value))
            continue

        if type(value) in (list, dict):
            if depth > MAX_DEPTH:
                raise Fp2UnsupportedValue(
                    f"manifest nests deeper than the fp2 limit of {MAX_DEPTH}"
                )
            identity = id(value)
            if identity in open_containers:
                raise Fp2UnsupportedValue("circular reference")
            open_containers.add(identity)

            # Children are pushed reversed so they pop in emission order.
            pending: list[tuple[int, Any, int]] = []
            if type(value) is dict:
                for key in value:
                    # Exact type for KEYS as well as values. A str subclass can
                    # override encode(), and encode() produces the UTF-16 sort
                    # key -- so a subclassed key chooses the ordering of the
                    # whole object, and the result is a *verified* digest.
                    if type(key) is not str:
                        raise Fp2UnsupportedValue(
                            f"object key is not a plain string: {type(key).__name__}"
                        )
                items = sorted(value.items(), key=lambda item: _utf16_sort_key(item[0]))
                out.append("{")
                for index, (key, item) in enumerate(items):
                    if index:
                        pending.append((_EMIT, ",", 0))
                    pending.append((_EMIT, _encode_string(key) + ":", 0))
                    pending.append((_VALUE, item, depth + 1))
                pending.append((_EMIT, "}", 0))
            else:
                out.append("[")
                for index, item in enumerate(value):
                    if index:
                        pending.append((_EMIT, ",", 0))
                    pending.append((_VALUE, item, depth + 1))
                pending.append((_EMIT, "]", 0))
            pending.append((_CLOSE, identity, 0))
            work.extend(reversed(pending))
            continue

        raise Fp2UnsupportedValue(f"unsupported type: {type(value).__name__}")

    return "".join(out)


def canonicalize(value: Any) -> str:
    """Return the fp2 canonical JSON text for ``value``.

    Raises:
        Fp2UnsupportedValue: the manifest is incomplete and the caller must
            record ``state="unknown"`` with no digest. This is the ONLY
            exception type this function raises. A manifest either canonicalizes
            or is incomplete; there is no third outcome, so any other failure is
            translated rather than allowed to escape. Letting a foreign type out
            crashes a caller that is documented to catch one type, turning an
            honest "unknown" into a dead run.
    """
    try:
        return _encode(value)
    except Fp2UnsupportedValue:
        raise
    except Exception as error:  # noqa: BLE001 - deliberate, see docstring
        # Only the exception TYPE is reported. A foreign exception's message can
        # embed caller data (a KeyError naming a user key, for instance), and
        # fp2 messages must stay content-free. The original is kept on __cause__
        # for a developer holding the traceback.
        raise Fp2UnsupportedValue(
            f"manifest could not be canonicalized ({type(error).__name__})"
        ) from error


def digest(value: Any) -> str:
    """Return the algorithm-prefixed fp2 digest for ``value``.

    Raises:
        Fp2UnsupportedValue: as for :func:`canonicalize`, and for the same
            reason -- hashing is inside the guarantee, not outside it.
    """
    canonical = canonicalize(value)
    try:
        return "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    except Fp2UnsupportedValue:
        raise
    except Exception as error:  # noqa: BLE001 - deliberate, see canonicalize
        raise Fp2UnsupportedValue(
            f"manifest could not be digested ({type(error).__name__})"
        ) from error
