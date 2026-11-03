"""Opt-in client side of the C1 commitment-linkage sidecar.

``finalize_process_record(..., link_commitments=True)`` asks the Backend to mint
a process-record certificate together with a signed *commitment-linkage*
sidecar. The sidecar commits to the dataset and configuration-space
fingerprints the session declared at start, hiding each behind a fresh random
256-bit *opening* that only this client keeps.

Security properties enforced here:

* Openings exist ONLY in the client bundle file (mode 0600) and in the one
  finalize request body. They are never logged, printed, or put in an
  exception message; every object that holds one has a redacted ``repr``.
* The bundle is written (atomic rename + fsync) BEFORE the request is sent, so
  an ambiguous failure never loses an opening. A retry reuses the openings
  already in the bundle for the same build session; it never regenerates them.
* Only the ``dataset`` and ``config_space`` roles are opened. The agent and
  evaluator roles are withheld by the Backend contract.
* Backend refusals map to closed codes. No response body, key or value is
  echoed.

This is client attestation: the sidecar proves the issuer checked equality
with the session's recorded declaration. It is not execution evidence.
"""

from __future__ import annotations

import base64
import json
import os
import re
import secrets
import stat
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol, cast

BUNDLE_SCHEMA = "traigent.commitment_bundle.v1"

#: The only roles the Backend accepts openings for (agent/evaluator are withheld).
OPENING_ROLES: tuple[str, ...] = ("dataset", "config_space")

#: Printed verbatim by ``traigent certify verify --open-commitments``.
DISCLOSURE_LINE = (
    "Dataset and configuration-space declarations only; agent, evaluator and "
    "build definition are not covered. This is not execution evidence."
)

_OPENING_BYTES = 32
_SHA256_RE = re.compile(r"^sha256:[a-f0-9]{64}$")
_SESSION_ID_RE = re.compile(r"^[a-z][a-z0-9_]{1,31}_[0-9a-f]{32}$")

#: Closed client-side codes -> fixed, value-free text.
ERROR_TEXT: dict[str, str] = {
    "COMMITMENT_LINKAGE_DISABLED": "the backend has commitment linkage turned off",
    "COMMITMENT_LINKAGE_OPENING_SHAPE": "the backend rejected the opening shape",
    "COMMITMENT_LINKAGE_UNAVAILABLE": (
        "the backend could not build the linkage for this session"
    ),
    "COMMITMENT_LINKAGE_CORRUPT": "the backend's stored linkage failed re-verification",
    "DUPLICATE_ISSUANCE": "a certificate already exists for this build session",
    "ISSUANCE_IN_PROGRESS": "a finalize for this build session is already running; retry",
    "BACKEND_REFUSED": "the backend refused the request",
    "TRANSPORT_ERROR": "the backend could not be reached",
    "BUNDLE_SESSION_MISMATCH": "the bundle file belongs to a different build session",
    "BUNDLE_CORRUPT": "the bundle file is not a valid commitment bundle",
    "BUNDLE_PERMISSIONS": "the bundle file is readable by other users; expected mode 0600",
    "BUNDLE_EXISTS": "the bundle file already exists",
    "BUNDLE_WRITE_FAILED": "the bundle file could not be written; nothing was sent",
    "BUNDLE_REQUIRED": "a bundle path is required",
    "LINKAGE_UNSUPPORTED_PLATFORM": (
        "commitment linking needs POSIX file permissions and is not supported "
        "on this platform yet"
    ),
    "REQUEST_HAS_OPENINGS": "the request payload must not carry openings",
    "BAD_SESSION_ID": "the build session id is not a well-formed row id",
    "LINKAGE_ABSENT": "the backend returned a record without a commitment linkage",
    "EGRESS_DISABLED": "backend egress is disabled",
    "BAD_FP_DIGEST": "fp digests must be dataset or config_space as sha256:<64 hex>",
    "SCHEMA_PACKAGE_TOO_OLD": (
        "the installed traigent-schema lacks certification.commitment_linkage; "
        "install a traigent-schema version that includes it; nothing was sent"
    ),
}


class CommitmentLinkageClientError(Exception):
    """A closed-code failure. ``str()`` is fixed text, never a request/response value."""

    def __init__(self, code: str) -> None:
        self.code = code if code in ERROR_TEXT else "BACKEND_REFUSED"
        super().__init__(f"{self.code}: {ERROR_TEXT[self.code]}")


class ProcessRecordRequestError(Exception):
    """Transport-level refusal carrying only the HTTP status and a plain error code."""

    def __init__(self, status_code: int | None, error_code: str | None) -> None:
        self.status_code = status_code
        self.error_code = error_code
        super().__init__(f"process record request failed (status={status_code})")


class _Opening(bytes):
    """32 secret bytes whose ``repr``/``str`` never reveal them."""

    __slots__ = ()

    def __repr__(self) -> str:
        return "<opening redacted>"

    __str__ = __repr__


class ProcessRecordTransport(Protocol):
    """What ``BackendIntegratedClient`` provides (mockable in tests)."""

    def finalize_process_record_sync(
        self, build_session_row_id: str, payload: dict[str, Any]
    ) -> dict[str, Any]: ...

    def get_process_record_sync(self, build_session_row_id: str) -> dict[str, Any]: ...


def _is_posix() -> bool:
    return os.name == "posix"


def require_posix_platform() -> None:
    """Refuse commitment linking off POSIX, before any secret is generated or read.

    The bundle relies on POSIX owner/mode checks and directory fsync; there is
    no equivalent guarantee on other platforms, so linking is restricted.
    """
    if not _is_posix():
        raise CommitmentLinkageClientError("LINKAGE_UNSUPPORTED_PLATFORM")


def _fsync_dir(directory: Path) -> None:
    dir_fd = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(dir_fd)
    finally:
        os.close(dir_fd)


def generate_openings() -> dict[str, bytes]:
    """Fresh openings for the dataset and config_space roles (Schema ``new_opening``)."""
    try:
        from traigent_schema.certification.commitment_linkage import new_opening
    except ImportError:
        # Older traigent-schema: raised outside the except block so the import
        # failure is not chained into the user-visible error.
        new_opening = None
    if new_opening is None:
        raise CommitmentLinkageClientError("SCHEMA_PACKAGE_TOO_OLD")

    return {role: _Opening(new_opening()) for role in OPENING_ROLES}


def _b64(opening: bytes) -> str:
    return base64.b64encode(bytes(opening)).decode("ascii")


def _decode_opening(value: Any) -> bytes:
    raw = b""
    ok = type(value) is str
    if ok:
        try:
            raw = base64.b64decode(value, validate=True)
        except ValueError:
            ok = False
    # Raised OUTSIDE the except block so no decoder exception stays as __context__.
    if not ok or len(raw) != _OPENING_BYTES:
        raise CommitmentLinkageClientError("BUNDLE_CORRUPT")
    return _Opening(raw)


class CommitmentBundle:
    """The client-side bundle file: openings, the returned record and its sidecar."""

    def __init__(self, path: Path, data: dict[str, Any]) -> None:
        self.path = path
        self._data = data

    def __repr__(self) -> str:  # never include openings
        return f"CommitmentBundle(path={str(self.path)!r}, openings=<redacted>)"

    __str__ = __repr__

    @property
    def build_session_row_id(self) -> str:
        return str(self._data["build_session_row_id"])

    @property
    def openings(self) -> dict[str, bytes]:
        raw = self._data.get("openings")
        if not isinstance(raw, dict):
            return {}
        return {
            role: _decode_opening(raw[role]) for role in OPENING_ROLES if role in raw
        }

    @property
    def record(self) -> dict[str, Any] | None:
        value = self._data.get("record")
        return value if isinstance(value, dict) else None

    @property
    def commitment_linkage(self) -> dict[str, Any] | None:
        value = self._data.get("commitment_linkage")
        return value if isinstance(value, dict) else None

    def set_result(
        self, record: Mapping[str, Any], linkage: Mapping[str, Any] | None
    ) -> None:
        self._data["record"] = dict(record)
        if linkage is not None:
            self._data["commitment_linkage"] = dict(linkage)
        self.save()

    def set_fp_digests(self, digests: Mapping[str, str]) -> None:
        for role, digest in digests.items():
            if role not in OPENING_ROLES or not _SHA256_RE.fullmatch(digest):
                raise CommitmentLinkageClientError("BAD_FP_DIGEST")
        self._data["fp_digests"] = dict(digests)
        self.save()

    def save(self) -> None:
        _atomic_write_private(self.path, self._data)

    def ensure_durable(self) -> None:
        """fsync the bundle file and its directory entry; call before any send.

        A caller that lost the creation race may see the winner's link before
        the winner has flushed the directory, so every sender does this itself.
        """
        failed = False
        try:
            fd = os.open(self.path, os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
            _fsync_dir(self.path.parent)
        except OSError:
            failed = True
        if failed:
            raise CommitmentLinkageClientError("BUNDLE_WRITE_FAILED")

    @classmethod
    def create(
        cls, path: Path, build_session_row_id: str, *, with_openings: bool
    ) -> CommitmentBundle:
        """Create the bundle EXCLUSIVELY; raises ``BUNDLE_EXISTS`` if one is already there.

        The finished (fsynced, 0600) file is hard-linked into place, so the
        destination either does not exist or is complete, and an existing
        bundle is never replaced.
        """
        data: dict[str, Any] = {
            "schema": BUNDLE_SCHEMA,
            "build_session_row_id": build_session_row_id,
        }
        if with_openings:
            data["openings"] = {
                role: _b64(o) for role, o in generate_openings().items()
            }
        _atomic_write_private(path, data, exclusive=True)
        return cls(path, data)

    @classmethod
    def load(cls, path: Path) -> CommitmentBundle:
        data = read_private_json(path)
        if data.get("schema") != BUNDLE_SCHEMA or not isinstance(
            data.get("build_session_row_id"), str
        ):
            raise CommitmentLinkageClientError("BUNDLE_CORRUPT")
        bundle = cls(path, data)
        bundle.openings  # noqa: B018 - validate shape eagerly; value is never kept in a message
        return bundle


def read_private_json(path: Path) -> dict[str, Any]:
    """Read a bundle file that must be private (exactly mode 0600, owned by us).

    Every failure is a closed ``CommitmentLinkageClientError`` raised OUTSIDE
    any ``except`` block: no ``__cause__``/``__context__`` survives, so a
    ``JSONDecodeError.doc`` holding the file's contents never escapes.
    """
    code: str | None = None
    data: Any = None
    try:
        st = path.stat()
    except OSError:
        code = "BUNDLE_CORRUPT"
    else:
        if stat.S_IMODE(st.st_mode) != 0o600 or st.st_uid != os.geteuid():
            code = "BUNDLE_PERMISSIONS"
        else:
            try:
                with path.open(encoding="utf-8") as handle:
                    data = json.load(handle)
            except (OSError, ValueError):
                code = "BUNDLE_CORRUPT"
    if code is None and not isinstance(data, dict):
        code = "BUNDLE_CORRUPT"
    if code is not None:
        raise CommitmentLinkageClientError(code)
    return cast("dict[str, Any]", data)


def _atomic_write_private(
    path: Path, data: Mapping[str, Any], *, exclusive: bool = False
) -> None:
    """Write ``data`` as JSON to ``path`` with mode 0600: temp file, fsync, publish, fsync dir.

    ``exclusive=True`` publishes with ``os.link`` (fails if ``path`` exists)
    instead of ``os.replace``; the loser gets ``BUNDLE_EXISTS``.
    """
    payload = json.dumps(data, sort_keys=True, indent=2).encode("utf-8")
    tmp = path.with_name(f".{path.name}.{secrets.token_hex(6)}.tmp")
    fd = -1
    exists = False
    failed = False
    try:
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        view = memoryview(payload)
        while view:
            view = view[os.write(fd, view) :]
        os.fsync(fd)
        os.close(fd)
        fd = -1
        if exclusive:
            os.link(tmp, path)
            tmp.unlink()
        else:
            os.replace(tmp, path)
        _fsync_dir(path.parent)
    except FileExistsError:
        exists = True
    except OSError:
        failed = True
    if exists or failed:
        if fd >= 0:
            os.close(fd)
        try:
            tmp.unlink()
        except OSError:  # silent-ok: best-effort cleanup of our own temp file
            pass
    if exists:
        raise CommitmentLinkageClientError("BUNDLE_EXISTS")
    if failed:
        raise CommitmentLinkageClientError("BUNDLE_WRITE_FAILED")


def _closed_error(exc: ProcessRecordRequestError) -> CommitmentLinkageClientError:
    code = (exc.error_code or "").upper()
    if code in ERROR_TEXT and code != "BACKEND_REFUSED":
        return CommitmentLinkageClientError(code)
    return CommitmentLinkageClientError("BACKEND_REFUSED")


def finalize_process_record(
    client: ProcessRecordTransport,
    build_session_row_id: str,
    request_payload: Mapping[str, Any],
    *,
    link_commitments: bool = False,
    bundle_path: str | os.PathLike[str] | None = None,
    fp_digests: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Finalize a process record; optionally with the commitment-linkage sidecar.

    ``request_payload`` is the Backend finalize body (refs, base certificate,
    verification materials); it must not contain ``openings``. Returns
    ``{"record": ..., "commitment_linkage": ... or None}``.

    With ``link_commitments=True`` a ``bundle_path`` is required. The bundle is
    created (or an existing one for this session reused) and flushed to disk
    before anything is sent.
    """
    _validate_finalize_request(
        build_session_row_id, request_payload, link_commitments, bundle_path
    )
    if bundle_path is None:
        if link_commitments:
            raise CommitmentLinkageClientError("BUNDLE_REQUIRED")
        return _send(
            client, build_session_row_id, dict(request_payload), None, link=False
        )

    path = Path(bundle_path)
    bundle: CommitmentBundle | None = None
    if not path.exists():
        try:
            bundle = CommitmentBundle.create(
                path, build_session_row_id, with_openings=link_commitments
            )
        except CommitmentLinkageClientError as err:
            if err.code != "BUNDLE_EXISTS":
                raise
            # Lost the creation race: fall through and reuse the winner's bundle.
    if bundle is None:
        bundle = _load_finalize_bundle(path, build_session_row_id, link_commitments)
    if fp_digests:
        bundle.set_fp_digests(fp_digests)

    if bundle.record is not None and (
        bundle.commitment_linkage is not None or not link_commitments
    ):
        return {
            "record": bundle.record,
            "commitment_linkage": bundle.commitment_linkage,
        }

    # Winner and loser alike: the bundle must be durable before anything is sent.
    bundle.ensure_durable()
    return _send(
        client,
        build_session_row_id,
        dict(request_payload),
        bundle,
        link=link_commitments,
    )


def _validate_finalize_request(
    build_session_row_id: str,
    request_payload: Mapping[str, Any],
    link_commitments: bool,
    bundle_path: str | os.PathLike[str] | None,
) -> None:
    if link_commitments or bundle_path is not None:
        # A bundle is written for either flag; refuse off POSIX before any work.
        require_posix_platform()
    if not _SESSION_ID_RE.fullmatch(build_session_row_id):
        raise CommitmentLinkageClientError("BAD_SESSION_ID")
    if "openings" in request_payload:
        raise CommitmentLinkageClientError("REQUEST_HAS_OPENINGS")


def _load_finalize_bundle(
    path: Path, build_session_row_id: str, link_commitments: bool
) -> CommitmentBundle:
    bundle = CommitmentBundle.load(path)
    if bundle.build_session_row_id != build_session_row_id:
        raise CommitmentLinkageClientError("BUNDLE_SESSION_MISMATCH")
    if link_commitments and set(bundle.openings) != set(OPENING_ROLES):
        raise CommitmentLinkageClientError("BUNDLE_CORRUPT")
    return bundle


def _send(
    client: ProcessRecordTransport,
    build_session_row_id: str,
    payload: dict[str, Any],
    bundle: CommitmentBundle | None,
    *,
    link: bool,
) -> dict[str, Any]:
    if link and bundle is not None:
        payload["openings"] = {role: _b64(o) for role, o in bundle.openings.items()}
    failure, body = _finalize_with_closed_failure(client, build_session_row_id, payload)
    if failure == "DUPLICATE_ISSUANCE" and link:
        # A row already exists: an earlier attempt may have committed. GET it,
        # never re-mint; the bundle's openings are the ones that attempt used.
        body, failure = _get(client, build_session_row_id)
    if failure is not None:
        raise CommitmentLinkageClientError(failure)
    return _record_result(body, bundle, link=link)


def _finalize_with_closed_failure(
    client: ProcessRecordTransport,
    build_session_row_id: str,
    payload: dict[str, Any],
) -> tuple[str | None, dict[str, Any]]:
    """Send the finalize request; return ``(closed failure code or None, body)``.

    Failures are translated to closed codes OUTSIDE the except blocks so no
    original exception (which could embed the request body) stays attached
    as ``__context__``.
    """
    failure: str | None = None
    body: dict[str, Any] = {}
    try:
        body = client.finalize_process_record_sync(build_session_row_id, payload)
    except ProcessRecordRequestError as exc:
        failure = _closed_error(exc).code
    except Exception:  # noqa: BLE001 - never let a request/response value escape
        failure = "TRANSPORT_ERROR"
    finally:
        payload.pop("openings", None)
    return failure, body


def _record_result(
    body: dict[str, Any], bundle: CommitmentBundle | None, *, link: bool
) -> dict[str, Any]:
    """Validate a successful response body and record it on the bundle."""
    record = body.get("data") if isinstance(body, dict) else None
    linkage = body.get("commitment_linkage") if isinstance(body, dict) else None
    if not isinstance(record, dict):
        raise CommitmentLinkageClientError("BACKEND_REFUSED")
    if link and not isinstance(linkage, dict):
        if bundle is not None:
            bundle.set_result(record, None)
        raise CommitmentLinkageClientError("LINKAGE_ABSENT")
    if bundle is not None:
        bundle.set_result(record, linkage if link else None)
    return {"record": record, "commitment_linkage": linkage if link else None}


def _get(
    client: ProcessRecordTransport, build_session_row_id: str
) -> tuple[dict[str, Any], str | None]:
    try:
        return client.get_process_record_sync(build_session_row_id), None
    except ProcessRecordRequestError as exc:
        return {}, _closed_error(exc).code
    except Exception:  # noqa: BLE001
        return {}, "TRANSPORT_ERROR"


__all__ = [
    "BUNDLE_SCHEMA",
    "DISCLOSURE_LINE",
    "ERROR_TEXT",
    "OPENING_ROLES",
    "CommitmentBundle",
    "CommitmentLinkageClientError",
    "ProcessRecordRequestError",
    "finalize_process_record",
    "generate_openings",
    "read_private_json",
]
