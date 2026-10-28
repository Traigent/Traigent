"""Trusted local pins for G1 co-attestation.

One 0600 JSON file holds everything the client must already trust before it
signs: the ``prj_...`` project reference, the issuer's public key and key
reference, the trust ring, the issuer algorithm, the compiler register versions,
the G1 verifier id and version, and the disclosure profile. It is provisioned
out of band by ``traigent certify pin --from <file>`` (the file comes from the
customer's own trust channel) and is never fetched from the Backend call being
verified. A missing or incomplete pin refuses with ``PIN_MISSING``.

Only ed25519 issuers are accepted for now; ``ecdsa_p256_sha256`` is refused.
"""

from __future__ import annotations

import base64
import json
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ed25519

__all__ = [
    "PIN_FIELDS",
    "PinError",
    "Pins",
    "default_pin_path",
    "load_pins",
    "write_pins",
]

PIN_FIELDS: Final = (
    "project_ref",
    "issuer_algorithm",
    "issuer_public_key_der_b64",
    "issuer_key_ref",
    "trust_ring_ref",
    "compiler_register_versions",
    "g1_verifier_id",
    "g1_verifier_version",
    "disclosure_profile_id",
)
_PROJECT_REF_RE = re.compile(r"[A-Za-z0-9_.-]{1,128}")
_REF_RE = re.compile(r"[a-z][a-z0-9_.-]{1,63}:[A-Za-z0-9_-]{8,128}")
_SUPPORTED_ISSUER_ALGORITHM: Final = "ed25519"


class PinError(Exception):
    """Closed failure; ``code`` is a fixed token."""

    CODES: Final = frozenset(
        {"PIN_MISSING", "PIN_INVALID", "PIN_FILE_UNSAFE", "PIN_PERSIST_FAILED"}
    )

    def __init__(self, code: str) -> None:
        if code not in self.CODES:
            raise ValueError("unknown pin error code")
        self.code = code
        super().__init__(code)


@dataclass(frozen=True)
class Pins:
    project_ref: str
    issuer_algorithm: str
    issuer_public_key: ed25519.Ed25519PublicKey
    issuer_key_ref: str
    trust_ring_ref: str
    compiler_register_versions: tuple[tuple[str, str], ...]
    g1_verifier_id: str
    g1_verifier_version: str
    disclosure_profile_id: str


def default_pin_path() -> Path:
    return Path.home() / ".traigent" / "certification" / "pins.json"


def _parse(document: Any) -> Pins:
    try:
        if not isinstance(document, dict) or set(document) != set(PIN_FIELDS):
            raise ValueError
        values = {name: document[name] for name in PIN_FIELDS}
        project_ref = values["project_ref"]
        if type(project_ref) is not str or not _PROJECT_REF_RE.fullmatch(project_ref):
            raise ValueError
        if values["issuer_algorithm"] != _SUPPORTED_ISSUER_ALGORITHM:
            raise ValueError
        for name in ("issuer_key_ref", "trust_ring_ref"):
            if type(values[name]) is not str or not _REF_RE.fullmatch(values[name]):
                raise ValueError
        for name in ("g1_verifier_id", "g1_verifier_version", "disclosure_profile_id"):
            if type(values[name]) is not str or not values[name]:
                raise ValueError
        registers = tuple(
            (str(row[0]), str(row[1])) for row in values["compiler_register_versions"]
        )
        if not registers or any(
            len(row) != 2 for row in values["compiler_register_versions"]
        ):
            raise ValueError
        public = serialization.load_der_public_key(
            base64.b64decode(values["issuer_public_key_der_b64"], validate=True)
        )
        if not isinstance(public, ed25519.Ed25519PublicKey):
            raise ValueError
    except (ValueError, TypeError, KeyError, IndexError):
        raise PinError("PIN_INVALID") from None
    return Pins(
        project_ref,
        values["issuer_algorithm"],
        public,
        values["issuer_key_ref"],
        values["trust_ring_ref"],
        registers,
        values["g1_verifier_id"],
        values["g1_verifier_version"],
        values["disclosure_profile_id"],
    )


def load_pins(path: str | os.PathLike[str] | None = None) -> Pins:
    """Load the pin file; refuses a missing file and group/world-accessible or foreign-owned files."""
    target = Path(path) if path is not None else default_pin_path()
    try:
        info = os.stat(target)
    except FileNotFoundError:
        raise PinError("PIN_MISSING") from None
    except OSError:
        raise PinError("PIN_INVALID") from None
    if (info.st_mode & 0o077) != 0 or info.st_uid != os.getuid():
        raise PinError("PIN_FILE_UNSAFE")
    try:
        document = json.loads(target.read_bytes())
    except (OSError, ValueError):
        raise PinError("PIN_INVALID") from None
    return _parse(document)


def write_pins(
    source: str | os.PathLike[str], destination: str | os.PathLike[str] | None = None
) -> Pins:
    """Validate ``source`` and install it atomically as the 0600 pin file."""
    target = Path(destination) if destination is not None else default_pin_path()
    try:
        document = json.loads(Path(source).read_bytes())
    except (OSError, ValueError):
        raise PinError("PIN_INVALID") from None
    pins = _parse(document)
    data = json.dumps(document, sort_keys=True, separators=(",", ":")).encode("utf-8")
    try:
        target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        fd, temp_name = tempfile.mkstemp(dir=target.parent, prefix=".pins-")
        try:
            os.fchmod(fd, 0o600)
            with os.fdopen(fd, "wb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_name, target)
        except BaseException:
            Path(temp_name).unlink(missing_ok=True)
            raise
    except OSError:
        raise PinError("PIN_PERSIST_FAILED") from None
    return pins
