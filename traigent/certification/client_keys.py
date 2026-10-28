"""Client signing key for G1 co-attestation: generate, store, enrol, pin.

The private key never leaves the key file (mode 0600, directory 0700) and is
never sent anywhere. Only the SPKI public key is enrolled, and the Backend's
answer is accepted only if its ``client_key_ref`` equals the reference derived
locally with the Schema helper from the pinned project reference.

Only ed25519 is supported by this producer.
"""

from __future__ import annotations

import base64
import json
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ed25519
from traigent_schema.certification import derive_client_key_ref

from traigent.certification.transport import G1RequestError, G1Transport

__all__ = [
    "ALGORITHM",
    "CLIENT_KEYS_PATH",
    "ClientKey",
    "ClientKeyError",
    "enroll_client_key",
    "generate_client_key",
    "load_client_key",
]

ALGORITHM: Final = "ed25519"
CLIENT_KEYS_PATH: Final = "/api/v1beta/certificate-client-keys"
_KEY_FILE_VERSION: Final = 1


class ClientKeyError(Exception):
    """Closed failure; ``code`` is a fixed token, never key material."""

    CODES: Final = frozenset(
        {
            "KEY_EXISTS",
            "KEY_FILE_UNSAFE",
            "KEY_FILE_INVALID",
            "KEY_PERSIST_FAILED",
            "ENROLL_REFUSED",
            "KEY_REF_MISMATCH",
            "KEY_NOT_PINNED",
        }
    )

    def __init__(self, code: str) -> None:
        if code not in self.CODES:
            raise ValueError("unknown client key error code")
        self.code = code
        super().__init__(code)


def default_key_directory() -> Path:
    return Path.home() / ".traigent" / "certification" / "keys"


@dataclass
class ClientKey:
    """An ed25519 client key plus the key references pinned for it, by project reference."""

    path: Path
    _private: ed25519.Ed25519PrivateKey = field(repr=False)
    pins: dict[str, str] = field(default_factory=dict)

    @property
    def public_key(self) -> ed25519.Ed25519PublicKey:
        return self._private.public_key()

    @property
    def public_key_der_b64(self) -> str:
        der = self.public_key.public_bytes(
            serialization.Encoding.DER, serialization.PublicFormat.SubjectPublicKeyInfo
        )
        return base64.b64encode(der).decode("ascii")

    def derive_ref(self, project_ref: str) -> str:
        return str(derive_client_key_ref(project_ref, ALGORITHM, self.public_key))

    def sign(self, message: bytes) -> str:
        """Raw 64-byte ed25519 signature, standard base64 (the certificate's SignatureBytes)."""
        return base64.b64encode(self._private.sign(message)).decode("ascii")

    def pinned_ref(self, project_ref: str) -> str:
        """The enrolled reference for ``project_ref``; raises ``KEY_NOT_PINNED`` if not enrolled."""
        ref = self.pins.get(project_ref)
        if ref is None or ref != self.derive_ref(project_ref):
            raise ClientKeyError("KEY_NOT_PINNED")
        return ref

    def pin(self, project_ref: str, key_ref: str) -> None:
        if key_ref != self.derive_ref(project_ref):
            raise ClientKeyError("KEY_REF_MISMATCH")
        self.pins[project_ref] = key_ref
        _write(self.path, self._document(), replace=True)

    def _document(self) -> dict[str, Any]:
        pkcs8 = self._private.private_bytes(
            serialization.Encoding.DER,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
        return {
            "version": _KEY_FILE_VERSION,
            "algorithm": ALGORITHM,
            "private_key_pkcs8_b64": base64.b64encode(pkcs8).decode("ascii"),
            "pins": dict(sorted(self.pins.items())),
        }

    def __repr__(self) -> str:
        return f"ClientKey(path={str(self.path)!r})"


def _write(path: Path, document: dict[str, Any], *, replace: bool) -> None:
    data = json.dumps(document, sort_keys=True, separators=(",", ":")).encode("utf-8")
    try:
        directory = path.parent
        directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(directory, 0o700)
        fd, temp_name = tempfile.mkstemp(dir=directory, prefix=".key-")
        try:
            os.fchmod(fd, 0o600)
            with os.fdopen(fd, "wb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            if replace:
                os.replace(temp_name, path)
            else:
                try:
                    os.link(temp_name, path)  # atomic create-if-absent
                finally:
                    os.unlink(temp_name)
        except FileExistsError:
            raise ClientKeyError("KEY_EXISTS") from None
        except BaseException:
            Path(temp_name).unlink(missing_ok=True)
            raise
    except ClientKeyError:
        raise
    except OSError:
        raise ClientKeyError("KEY_PERSIST_FAILED") from None


def generate_client_key(path: str | os.PathLike[str] | None = None) -> ClientKey:
    """Create a new ed25519 key file (mode 0600). Refuses to overwrite an existing key."""
    target = Path(path) if path is not None else None
    private = ed25519.Ed25519PrivateKey.generate()
    if target is None:
        raw = private.public_key().public_bytes(
            serialization.Encoding.Raw, serialization.PublicFormat.Raw
        )
        target = default_key_directory() / f"{raw.hex()[:32]}.json"
    key = ClientKey(target, private)
    _write(target, key._document(), replace=False)
    return key


def load_client_key(path: str | os.PathLike[str]) -> ClientKey:
    """Load a key file; refuses group/world-accessible files or files not owned by the caller."""
    target = Path(path)
    try:
        info = os.stat(target)
    except OSError:
        raise ClientKeyError("KEY_FILE_INVALID") from None
    if (info.st_mode & 0o077) != 0 or info.st_uid != os.getuid():
        raise ClientKeyError("KEY_FILE_UNSAFE")
    try:
        document = json.loads(target.read_bytes())
        if (
            document["version"] != _KEY_FILE_VERSION
            or document["algorithm"] != ALGORITHM
        ):
            raise ValueError
        private = serialization.load_der_private_key(
            base64.b64decode(document["private_key_pkcs8_b64"], validate=True),
            password=None,
        )
        pins = {str(k): str(v) for k, v in dict(document["pins"]).items()}
    except (ValueError, KeyError, TypeError):
        raise ClientKeyError("KEY_FILE_INVALID") from None
    if not isinstance(private, ed25519.Ed25519PrivateKey):
        raise ClientKeyError("KEY_FILE_INVALID")
    return ClientKey(target, private, pins)


def enroll_client_key(transport: G1Transport, key: ClientKey, project_ref: str) -> str:
    """Enrol the public key and pin the derived reference.

    Sends only ``{algorithm, public_key_der_b64}``. The Backend's returned
    ``client_key_ref`` must equal the locally derived one; otherwise nothing is
    pinned (``KEY_REF_MISMATCH``).
    """
    expected = key.derive_ref(project_ref)
    try:
        response = transport.request(
            "POST",
            CLIENT_KEYS_PATH,
            {"algorithm": ALGORITHM, "public_key_der_b64": key.public_key_der_b64},
        )
    except G1RequestError:
        raise ClientKeyError("ENROLL_REFUSED") from None
    body = response.body
    if response.status not in (200, 201) or not isinstance(body, dict):
        raise ClientKeyError("ENROLL_REFUSED")
    if body.get("client_key_ref") != expected or body.get("algorithm") != ALGORITHM:
        raise ClientKeyError("KEY_REF_MISMATCH")
    key.pin(project_ref, expected)
    return expected
