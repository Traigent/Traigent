"""Offline verifier for signed Pillar Legend claim statements (v0).

Mirrors the TraigentSchema reference verifier (``traigent_schema/legend_claim_verifier.py``,
Schema commit 2244144) so a customer can verify a Legend certificate without the Backend and
without any network call. A statement is ``{"subject": ..., "signature": ...}``. The relying
party pins a trusted-key set; nothing is fetched.

Checks, in order: signature domain, declared ``digest_alg`` (only ``traigent-fp2-sha256``:
RFC 8785 canonical JSON via fp2, SHA-256), the closed statement schema, the pinned
``(key_id, algorithm)``, then the signature over ``domain || 0x00 || canonical_json(subject)``
recomputed from the presented subject (ed25519, or ecdsa_p256_sha256 with low-S enforced).
Failures raise :class:`LegendClaimVerificationError` carrying only a fixed content-free code.
"""

from __future__ import annotations

import base64
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, cast

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ec, ed25519, utils
from jsonschema import Draft7Validator, FormatChecker
from referencing import Registry, Resource

from traigent.utils import fp2

LEGEND_SIGNATURE_DOMAIN = b"traigent.legend_claim_subject.signature.v0"
LEGEND_SIGNATURE_ALGORITHMS = frozenset({"ed25519", "ecdsa_p256_sha256"})
LEGEND_DIGEST_ALGORITHMS = frozenset({"traigent-fp2-sha256"})

LEGEND_CLAIM_VERIFICATION_ERROR_CODES = frozenset(
    {
        "STRUCTURE",
        "DOMAIN",
        "DIGEST_ALG",
        "SCHEMA",
        "CANONICALIZATION",
        "ALGORITHM",
        "UNKNOWN_KEY",
        "KEY_ALGORITHM",
        "SIGNATURE_BASE64",
        "SIGNATURE_SCALAR",
        "SIGNATURE_HIGH_S",
        "SIGNATURE_INVALID",
        "VERIFICATION_FAILED",
    }
)

_SCHEMAS_DIR = Path(__file__).resolve().parent / "schemas"
_STATEMENT_FILE = "legend_claim_statement_v0.json"
_SIG_RE = re.compile(r"^[A-Za-z0-9+/]{86}==$")
_ECDSA_ORDER = int(
    "FFFFFFFF00000000FFFFFFFFFFFFFFFFBCE6FAADA7179E84F3B9CAC2FC632551", 16
)
_ECDSA_HALF_ORDER = _ECDSA_ORDER // 2

_RFC3339_RE = re.compile(
    r"^(?P<date>\d{4}-\d{2}-\d{2})[Tt]"
    r"(?P<time>\d{2}:\d{2}:\d{2})"
    r"(?P<fraction>\.\d+)?"
    r"(?P<tz>[Zz]|[+-]\d{2}:\d{2})$"
)
_FORMAT_CHECKER = FormatChecker()


@_FORMAT_CHECKER.checks("date-time")
def _is_rfc3339_date_time(value: object) -> bool:
    """Traigent's RFC-3339 subset (same rule as the Schema validator)."""
    if not isinstance(value, str):
        return True
    match = _RFC3339_RE.fullmatch(value)
    if match is None or match.group("time").endswith(":60"):
        return False
    fraction = match.group("fraction") or ""
    if fraction:
        fraction = "." + fraction[1:7]
    tz = match.group("tz")
    if tz in {"Z", "z"}:
        tz = "+00:00"
    try:
        datetime.fromisoformat(
            f"{match.group('date')}T{match.group('time')}{fraction}{tz}"
        )
    except ValueError:
        return False
    return True


class LegendClaimVerificationError(ValueError):
    """A fixed-code verification failure that never includes statement data."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


def _fail(code: str) -> Any:
    raise LegendClaimVerificationError(code) from None


@dataclass(frozen=True)
class TrustedLegendKey:
    """One pinned signer key: the single algorithm it is trusted for, and its public key."""

    algorithm: str
    public_key: Any  # ed25519.Ed25519PublicKey | ec.EllipticCurvePublicKey (P-256)


@dataclass(frozen=True)
class LegendClaimVerification:
    """Successful result; only the verified, schema-valid subject is returned."""

    subject: Mapping[str, Any]
    key_id: str
    algorithm: str
    canonical_bytes: bytes


def canonical_subject_json(subject: Any) -> str:
    """RFC 8785 canonical subject text via the fp2 canonicalizer."""
    return fp2.canonicalize(subject)


def legend_signing_material(subject: Any) -> bytes:
    """``domain || 0x00 || canonical_json(subject)`` as UTF-8 bytes (the signed input)."""
    return (
        LEGEND_SIGNATURE_DOMAIN
        + b"\x00"
        + canonical_subject_json(subject).encode("utf-8")
    )


@lru_cache(maxsize=1)
def _statement_validator() -> Draft7Validator:
    resources = []
    statement: Any = None
    for path in sorted(_SCHEMAS_DIR.glob("*.json")):
        document = json.loads(path.read_text(encoding="utf-8"))
        resources.append((document["$id"], Resource.from_contents(document)))
        if path.name == _STATEMENT_FILE:
            statement = document
    return Draft7Validator(
        statement,
        registry=Registry().with_resources(resources),
        format_checker=_FORMAT_CHECKER,
    )


def _decode_signature(encoded: object) -> bytes:
    if type(encoded) is not str or not _SIG_RE.fullmatch(encoded):
        _fail("SIGNATURE_BASE64")
    try:
        raw = base64.b64decode(cast(str, encoded).encode("ascii"), validate=True)
    except ValueError:
        _fail("SIGNATURE_BASE64")
    if len(raw) != 64 or base64.b64encode(raw).decode("ascii") != encoded:
        _fail("SIGNATURE_BASE64")
    return raw


def _verify(key: TrustedLegendKey, material: bytes, raw: bytes) -> None:
    try:
        if key.algorithm == "ed25519":
            if not isinstance(key.public_key, ed25519.Ed25519PublicKey):
                _fail("KEY_ALGORITHM")
            key.public_key.verify(raw, material)
            return
        pub = key.public_key
        if not isinstance(pub, ec.EllipticCurvePublicKey) or not isinstance(
            pub.curve, ec.SECP256R1
        ):
            _fail("KEY_ALGORITHM")
        r = int.from_bytes(raw[:32], "big")
        s = int.from_bytes(raw[32:], "big")
        if not 1 <= r < _ECDSA_ORDER or not 1 <= s < _ECDSA_ORDER:
            _fail("SIGNATURE_SCALAR")
        if s > _ECDSA_HALF_ORDER:
            _fail("SIGNATURE_HIGH_S")
        pub.verify(
            utils.encode_dss_signature(r, s), material, ec.ECDSA(hashes.SHA256())
        )
    except LegendClaimVerificationError:
        raise
    except (InvalidSignature, ValueError):
        _fail("SIGNATURE_INVALID")
    except Exception:
        _fail("VERIFICATION_FAILED")


def verify_legend_claim_statement(
    statement: object, trusted_keys: Mapping[str, TrustedLegendKey]
) -> LegendClaimVerification:
    """Verify one signed Legend claim statement against a pinned trusted-key set.

    ``trusted_keys`` maps ``key_id`` to the key and the single algorithm it is trusted for.
    Raises :class:`LegendClaimVerificationError` (closed, content-free code) on any failure.
    """
    if type(statement) is not dict or set(statement) != {"subject", "signature"}:
        _fail("STRUCTURE")
    stmt = cast(dict[str, Any], statement)
    sig = stmt["signature"]
    if type(sig) is not dict:
        _fail("STRUCTURE")
    if sig.get("domain") != LEGEND_SIGNATURE_DOMAIN.decode("ascii"):
        _fail("DOMAIN")
    subject = stmt["subject"]
    alg_id = subject.get("digest_alg") if type(subject) is dict else None
    if type(alg_id) is not str or alg_id not in LEGEND_DIGEST_ALGORITHMS:
        _fail("DIGEST_ALG")
    if _statement_validator().is_valid(stmt) is not True:
        _fail("SCHEMA")
    algorithm, key_id = sig["algorithm"], sig["key_id"]
    if algorithm not in LEGEND_SIGNATURE_ALGORITHMS:
        _fail("ALGORITHM")
    key = trusted_keys.get(key_id)
    if key is None:
        _fail("UNKNOWN_KEY")
    if key.algorithm != algorithm:
        _fail("ALGORITHM")
    raw = _decode_signature(sig["value"])
    try:
        material = legend_signing_material(stmt["subject"])
    except (TypeError, ValueError, RecursionError):
        _fail("CANONICALIZATION")
    _verify(key, material, raw)
    return LegendClaimVerification(
        subject=stmt["subject"],
        key_id=key_id,
        algorithm=algorithm,
        canonical_bytes=material,
    )
