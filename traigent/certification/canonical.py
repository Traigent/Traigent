"""Client-local canonicalization helpers for Agent Certificate v0.

The certificate contract uses the SDK's vendored :mod:`traigent.utils.fp2`
implementation as its JCS reference.  This module deliberately exposes no
fallback serializer: values which fp2 cannot represent are rejected.
"""

from __future__ import annotations

from typing import Any

from traigent.utils import fp2

__all__ = ["canonicalize_artifact_document"]


def _reject_floats(value: Any) -> None:
    """Apply the certificate contract's stricter no-float preimage rule."""

    pending = [value]
    seen: set[int] = set()
    while pending:
        current = pending.pop()
        if type(current) is float:
            raise fp2.Fp2UnsupportedValue("artifact_document must not contain floats")
        if type(current) is dict or type(current) is list:
            identity = id(current)
            if identity in seen:
                continue
            seen.add(identity)
            pending.extend(current.values() if type(current) is dict else current)


def canonicalize_artifact_document(artifact_document: dict[str, Any]) -> str:
    """Return the fp2/JCS bytes (as text) for a client-local JSON object.

    Only a plain built-in ``dict`` is accepted at the document boundary.  fp2
    then rejects subclasses, tuples, floats, unsafe integers, cycles, lone
    surrogates, and every other value outside its cross-language contract.
    Error text is intentionally content-free because this function runs over
    potentially sensitive client-local evidence.
    """

    if type(artifact_document) is not dict:
        raise TypeError("artifact_document must be a plain JSON object")
    try:
        _reject_floats(artifact_document)
        return fp2.canonicalize(artifact_document)
    except fp2.Fp2UnsupportedValue:
        raise fp2.Fp2UnsupportedValue(
            "artifact_document is not fp2-compatible"
        ) from None
