"""Durable typed token store and issuance client (content-less round 1b-2a, SDK half).

The Backend issues opaque ``knobref_`` / ``choiceref_`` / ``metricref_`` / ``templateref_``
values (``POST /api/v1/optimization/token-refs``, Schema ``optimization/token_refs_schema.json``).
Names (knob names, choices, metric names, templates) are labels: they never travel on that
route. This module keeps the local mapping *name -> issued ref* so a name keeps one token for
the life of an agent, and journals every issuance so a crash at any point resumes without an
orphan or a duplicate identity.

Design (PHASE1b-DESIGN-v2 sections 3.2, 3.3, 10.1', 10.2, 10.6, 10.7):

* **Inert unless enabled.** ``TRAIGENT_TOKEN_STORE_ENABLED`` is read at call time; only
  ``true`` / ``1`` enable. Off: no file, no directory, no request.
* **Directory trust is strict.** Every component of the state-root chain (default
  ``~/.traigent``, or ``TRAIGENT_RESULTS_FOLDER``) must be owned by the user or root and must not
  be group- or world-writable unless sticky (``/tmp``); group exclusivity is never inferred from
  the group database. A violation is refused as ``token_store_insecure_permissions``; the remedy
  is ``chmod g-w,o-w <dir>`` (the SDK never changes a mode it did not create). Symlinked
  components are followed only from a trusted directory and when owned by the user or root.
* **Namespace** = (normalized backend origin, agent_ref); one store file per namespace named by
  the sha256 of the canonical namespace (never a plaintext name or agent id). The same
  (kind, parent token ref, name) maps to the same token across specs of one agent; nothing is
  shared across agents or backends.
* **Key** = (kind, parent token ref | None, local name). A choice's parent is a knob token that
  must already be in the store (the server only accepts a parent issued by an EARLIER request).
* **Journal.** S0 intent (operation_key + the exact ordered batch) is made durable BEFORE the
  network call; S1 (refs received, ``pending_label``) after; S2 (label acked, revision >= 1) and
  S3 (spec committed) are store-level transitions here; S2 is driven by
  ``LabelSender.send_new`` on a negotiated label write (1b-2b), S3 is wired in 1b-4. A
  restart from S0 re-sends the SAME operation_key and the SAME ordered batch.
* **Durability.** Every write is exclusive lock + temp file + fsync + ``os.replace`` + directory
  fsync, file mode 0600 in a 0700 directory, verified on open (a wrong mode is refused, never
  repaired). A leading version byte guards the format. A store that does not parse and validate
  completely is refused as a whole (``token_map_not_persisted``); a partial map is never returned.
* **Closed errors.** Everything that leaves a public entry point is a :class:`TokenStoreError`
  carrying one code from :data:`CODES`; no name, ref, path, URL or server text is ever put in an
  exception, a log record or a request (this module logs nothing).

The file holds plaintext names by design (it is the customer's own machine); that is the only
place they exist.
"""

# Traceability: CONC-Layer-Core CONC-Quality-Security FUNC-SECURITY

from __future__ import annotations

import contextlib
import errno
import functools
import hashlib
import json
import math
import os
import re
import secrets
import stat
import threading
import time
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

ENV_FLAG = "TRAIGENT_TOKEN_STORE_ENABLED"
KINDS = ("knob", "choice", "metric", "template")
MAX_ITEMS_PER_OPERATION = 200
MAX_NAME_CHARS = 4096
MAX_SPEC_HISTORY = 256
#: Longest a caller waits for a namespace lock held by someone else (then ``token_store_busy``).
LOCK_TIMEOUT_SECONDS = 10.0
#: Wall-clock bound for everything done while the lock is held (auth + every request).
OPERATION_DEADLINE_SECONDS = 60.0
MAX_COUNT = 2**53 - 1
FORMAT_VERSION = 1

# --- closed error codes -------------------------------------------------------------------

TOKEN_STORE_DISABLED = "token_store_disabled"
TOKEN_MAP_NOT_PERSISTED = "token_map_not_persisted"
TOKEN_STORE_VERSION_UNSUPPORTED = "token_store_version_unsupported"
TOKEN_STORE_INSECURE_PERMISSIONS = "token_store_insecure_permissions"
TOKEN_NAMESPACE_INVALID = "token_namespace_invalid"
TOKEN_REQUEST_INVALID = "token_request_invalid"
TOKEN_PARENT_MISSING = "token_parent_missing"
TOKEN_STATE_INVALID = "token_state_invalid"
TOKEN_REVISION_STALE = "token_revision_stale"
TOKEN_REFS_UNAVAILABLE = "token_refs_unavailable"
TOKEN_REFS_UNAUTHENTICATED = "token_refs_unauthenticated"
TOKEN_REFS_FORBIDDEN = "token_refs_forbidden"
TOKEN_REFS_INVALID_REQUEST = "token_refs_invalid_request"
TOKEN_REFS_REFERENCE_NOT_FOUND = "token_refs_reference_not_found"
OPERATION_KEY_CONFLICT = "operation_key_conflict"
TOKEN_REFS_QUOTA_EXCEEDED = "token_refs_quota_exceeded"
TOKEN_REFS_SERVER_ERROR = "token_refs_server_error"
TOKEN_REFS_TRANSPORT_FAILED = "token_refs_transport_failed"
#: 503 ``service_unavailable`` answered until the operation deadline (retryable by the caller).
TOKEN_REFS_SERVICE_UNAVAILABLE = "token_refs_service_unavailable"
TOKEN_REFS_RESPONSE_INVALID = "token_refs_response_invalid"
LABEL_CONFLICT_UNVERIFIED = "label_conflict_unverified"
TOKEN_STORE_BUSY = "token_store_busy"
TOKEN_STORE_DEADLINE_EXCEEDED = "token_store_deadline_exceeded"
TOKEN_STORE_INTERNAL = "token_store_internal"

CODES = frozenset(
    {
        TOKEN_STORE_DISABLED,
        TOKEN_MAP_NOT_PERSISTED,
        TOKEN_STORE_VERSION_UNSUPPORTED,
        TOKEN_STORE_INSECURE_PERMISSIONS,
        TOKEN_NAMESPACE_INVALID,
        TOKEN_REQUEST_INVALID,
        TOKEN_PARENT_MISSING,
        TOKEN_STATE_INVALID,
        TOKEN_REVISION_STALE,
        TOKEN_REFS_UNAVAILABLE,
        TOKEN_REFS_UNAUTHENTICATED,
        TOKEN_REFS_FORBIDDEN,
        TOKEN_REFS_INVALID_REQUEST,
        TOKEN_REFS_REFERENCE_NOT_FOUND,
        OPERATION_KEY_CONFLICT,
        TOKEN_REFS_QUOTA_EXCEEDED,
        TOKEN_REFS_SERVER_ERROR,
        TOKEN_REFS_TRANSPORT_FAILED,
        TOKEN_REFS_SERVICE_UNAVAILABLE,
        TOKEN_REFS_RESPONSE_INVALID,
        LABEL_CONFLICT_UNVERIFIED,
        TOKEN_STORE_BUSY,
        TOKEN_STORE_DEADLINE_EXCEEDED,
        TOKEN_STORE_INTERNAL,
    }
)


#: Codes a caller may retry later without changing anything (the intent keeps its key).
RETRYABLE_CODES = frozenset({TOKEN_REFS_SERVICE_UNAVAILABLE})


class TokenStoreError(Exception):
    """A closed refusal: carries exactly one code from :data:`CODES` and nothing else."""

    def __init__(self, code: str) -> None:
        if code not in CODES:
            code = TOKEN_STORE_INTERNAL
        super().__init__(code)
        self.code = code

    @property
    def retryable(self) -> bool:
        return self.code in RETRYABLE_CODES

    def __str__(self) -> str:
        return self.code

    def __repr__(self) -> str:
        return f"TokenStoreError({self.code!r})"


def _closed_boundary[F: Callable[..., Any]](func: F) -> F:
    """Whatever leaves a public entry point is a :class:`TokenStoreError` without a cause.

    An exception nobody anticipated (a dependency's, an ``OSError`` quoting a path, a
    transport error quoting a URL) is replaced by ``token_store_internal`` with no
    ``__cause__`` / ``__context__``. ``KeyboardInterrupt`` and ``SystemExit`` pass through.
    """

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        error: TokenStoreError
        try:
            return func(*args, **kwargs)
        except TokenStoreError as exc:
            error = exc
        except Exception:  # noqa: BLE001 - closed boundary: nothing else may escape
            error = TokenStoreError(TOKEN_STORE_INTERNAL)
        # Raised outside the ``except`` block, then cleared AFTER the raise has chained: when
        # a caller invokes this inside its own ``except``, the interpreter attaches that
        # handled exception as ``__context__`` during the raise, so clearing beforehand is
        # not enough (same pattern as best_config_diagnostics.fs_guard); bare re-raise last.
        try:
            raise error from None
        except TokenStoreError as raised:
            raised.__cause__ = None
            raised.__context__ = None
            raised.__suppress_context__ = True
            raise

    return wrapper  # type: ignore[return-value]


# --- flag ---------------------------------------------------------------------------------


def is_enabled() -> bool:
    """``TRAIGENT_TOKEN_STORE_ENABLED``, read at call time; only ``true`` / ``1`` enable."""
    return os.environ.get(ENV_FLAG, "").strip().lower() in ("true", "1")


def _require_enabled() -> None:
    if not is_enabled():
        raise TokenStoreError(TOKEN_STORE_DISABLED)


# --- namespace ----------------------------------------------------------------------------

_AGENT_REF = re.compile(r"^agentref_[0-9a-f]{32}$")
_KNOB_REF = re.compile(r"^knobref_[0-9a-f]{32}$")
_SPEC_REF = re.compile(r"^searchspecref_[0-9a-f]{32}$")
_OP_KEY = re.compile(r"^[0-9a-f]{32}$")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_REF = {kind: re.compile(rf"^{kind}ref_[0-9a-f]{{32}}$") for kind in KINDS}
_DEFAULT_PORT = {"http": 80, "https": 443}


def normalize_origin(url: str) -> str:
    """``scheme://host[:port]`` of ``url``: scheme and host lower-cased, default port elided.

    Path, query, fragment and userinfo are dropped. Anything that is not an http(s) URL with a
    host is refused ``token_namespace_invalid`` (the value is never echoed).
    """
    if not isinstance(url, str):
        raise TokenStoreError(TOKEN_NAMESPACE_INVALID)
    try:
        parts = urlsplit(url.strip())
        scheme = parts.scheme.lower()
        host = parts.hostname
        port = parts.port
    except ValueError:
        raise TokenStoreError(TOKEN_NAMESPACE_INVALID) from None
    if scheme not in _DEFAULT_PORT or not host:
        raise TokenStoreError(TOKEN_NAMESPACE_INVALID)
    shown = f"[{host}]" if ":" in host else host
    if port is not None and port != _DEFAULT_PORT[scheme]:
        shown = f"{shown}:{port}"
    return f"{scheme}://{shown}"


@dataclass(frozen=True)
class TokenNamespace:
    """(normalized backend origin, agent_ref): one store file, one token space."""

    origin: str
    agent_ref: str

    @classmethod
    def for_backend(cls, backend_url: str, agent_ref: str) -> TokenNamespace:
        if not isinstance(agent_ref, str) or not _AGENT_REF.fullmatch(agent_ref):
            raise TokenStoreError(TOKEN_NAMESPACE_INVALID)
        return cls(normalize_origin(backend_url), agent_ref)

    def __post_init__(self) -> None:
        if (
            not isinstance(self.origin, str)
            or self.origin != _renormalized(self.origin)
            or not isinstance(self.agent_ref, str)
            or not _AGENT_REF.fullmatch(self.agent_ref)
        ):
            raise TokenStoreError(TOKEN_NAMESPACE_INVALID)

    @property
    def digest(self) -> str:
        """sha256 hex of the canonical namespace; the store file's name and a check inside it."""
        canonical = json.dumps(
            ["traigent-token-store-ns-v1", self.origin, self.agent_ref],
            separators=(",", ":"),
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _renormalized(origin: str) -> str | None:
    try:
        return normalize_origin(origin)
    except TokenStoreError:
        return None


# --- request digest (pinned against the Backend's frozen preimage) ------------------------


def request_digest(agent_id: str, items: Sequence[tuple[str, str | None]]) -> bytes:
    """sha256 of ``{"agent_id": <id>, "items": [[ordinal, kind, parent_ref|null], ...]}``.

    The Backend's receipt digest (PHASE1b-DESIGN-v2 section 10.7: RFC 8785 JSON over this
    fixed shape, which is ``json.dumps(sort_keys, compact separators, ensure_ascii=False)``).
    The client does not send it; it stores it per journalled batch so a persisted batch that
    no longer matches its recorded digest is refused, and a test pins our ordering and JSON
    against the Backend's vectors.
    """
    body = {
        "agent_id": agent_id,
        "items": [[i, kind, parent] for i, (kind, parent) in enumerate(items)],
    }
    text = json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(text.encode("utf-8")).digest()


# --- on-disk layout -----------------------------------------------------------------------

_STORE_DIRNAME = "token_store"

S0_INTENT = "intent"
S1_PENDING_LABEL = "pending_label"
S2_LABEL_ACKED = "label_acked"
S3_SPEC_COMMITTED = "spec_committed"
S_LABEL_CONFLICT = "label_conflict_unverified"
STATES = (
    S0_INTENT,
    S1_PENDING_LABEL,
    S2_LABEL_ACKED,
    S3_SPEC_COMMITTED,
    S_LABEL_CONFLICT,
)


def _state_root() -> Path:
    """The SDK's existing per-user state root (same rule as ``labels/transport.py``)."""
    base = os.getenv("TRAIGENT_RESULTS_FOLDER")
    return Path(base).expanduser() if base else Path.home() / ".traigent"


def _store_dir() -> Path:
    return _state_root() / _STORE_DIRNAME


def store_path(namespace: TokenNamespace) -> Path:
    """Where the namespace's store file lives (the name is a hash; nothing is created)."""
    return _store_dir() / f"{namespace.digest}.json"


def _key(kind: str, parent_ref: str | None, name: str) -> str:
    return json.dumps(
        [kind, parent_ref, name], separators=(",", ":"), ensure_ascii=False
    )


def _fault_point(name: str) -> None:
    """Named boundaries inside the filesystem layer; a no-op (tests inject faults here)."""


class _Anchor:
    """A validated directory descriptor for the store directory.

    Every lock, read, temp, replace and unlink goes through ``fd`` (``dir_fd=``), so they
    all act on the one directory that passed validation even if a path component is
    swapped afterwards. ``path`` is only used to re-check that the name still denotes it.
    """

    def __init__(self, fd: int, path: str) -> None:
        self.fd = fd
        self.path = path

    def close(self) -> None:
        with contextlib.suppress(OSError):
            os.close(self.fd)

    def verify(self) -> None:
        """Refuse unless ``path`` still names the directory ``fd`` was validated as."""
        try:
            mine = os.fstat(self.fd)
            named = os.lstat(self.path)
        except OSError:
            raise TokenStoreError(TOKEN_STORE_INSECURE_PERMISSIONS) from None
        if (
            not stat.S_ISDIR(named.st_mode)
            or (named.st_ino, named.st_dev) != (mine.st_ino, mine.st_dev)
            or mine.st_nlink == 0
        ):
            raise TokenStoreError(TOKEN_STORE_INSECURE_PERMISSIONS)


@dataclass(frozen=True)
class _Place:
    """A file name inside an :class:`_Anchor` directory."""

    anchor: _Anchor
    name: str


def _require_dirfd_support() -> None:
    """Fail closed where descriptor-relative operations are unavailable."""
    supported = (
        hasattr(os, "O_DIRECTORY")
        and hasattr(os, "O_NOFOLLOW")
        and {os.open, os.mkdir, os.unlink, os.rename, os.stat} <= os.supports_dir_fd
        and os.scandir in os.supports_fd
    )
    if not supported:
        raise TokenStoreError(TOKEN_MAP_NOT_PERSISTED)


def _trusted_dir(info: os.stat_result) -> bool:
    """Owned by us or root and not writable by anyone else.

    Strict: a group-writable or world-writable directory is refused unless it is sticky
    (``/tmp``-style: only an entry's owner may rename or remove it). Group exclusivity is never
    inferred from the group database (NSS/SSSD may hide members), so a group-writable component
    is refused even if the group looks private. Remedy for the operator: ``chmod g-w,o-w <dir>``
    (the SDK never changes the mode of a directory it did not create).
    """
    if not stat.S_ISDIR(info.st_mode) or info.st_uid not in (0, os.geteuid()):
        return False
    if info.st_mode & stat.S_ISVTX:
        return True
    return not info.st_mode & 0o022


class _LinkComponent(Exception):
    """Internal: the component is a symlink (or not a directory); the caller decides."""


def _open_dir_nofollow(name: str, parent_fd: int) -> int:
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
    try:
        return os.open(name, flags, dir_fd=parent_fd)
    except OSError as exc:
        if exc.errno in (errno.ELOOP, errno.ENOTDIR):  # a symlink or a non-directory
            raise _LinkComponent from None
        raise


def _read_trusted_link(name: str, parent_fd: int) -> str:
    """Target of symlink ``name`` in ``parent_fd``, refusing an untrusted link or container."""
    try:
        if not _trusted_dir(os.fstat(parent_fd)):
            raise TokenStoreError(TOKEN_STORE_INSECURE_PERMISSIONS)
        info = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        if not stat.S_ISLNK(info.st_mode) or info.st_uid not in (0, os.geteuid()):
            raise TokenStoreError(TOKEN_STORE_INSECURE_PERMISSIONS)
        return os.readlink(name, dir_fd=parent_fd)
    except OSError:
        raise TokenStoreError(TOKEN_STORE_INSECURE_PERMISSIONS) from None


_MAX_LINK_HOPS = 16


def _configured_root() -> str:
    """The configured state root as an absolute path, WITHOUT normalizing it.

    ``os.path.abspath`` would collapse ``link/../state`` to ``<parent of link>/state`` before the
    walk, but the filesystem resolves it through the link's target; the store would then look in
    a different place than the operator configured (and could mint fresh identities). A ``..``
    component in the configured value is therefore refused outright
    (``token_store_insecure_permissions``, the code every other path-trust refusal uses; no new
    closed code is needed). A relative value is joined to the working directory as-is.
    """
    configured = os.fspath(_state_root())
    if not os.path.isabs(configured):
        configured = os.path.join(os.getcwd(), configured)
    if ".." in configured.split(os.sep):
        raise TokenStoreError(TOKEN_STORE_INSECURE_PERMISSIONS)
    return configured


def _open_root() -> int:
    """Open ``/`` and hold it to the same trust predicate as every other directory."""
    try:
        fd = os.open(os.sep, os.O_RDONLY | os.O_DIRECTORY)
    except OSError:
        raise TokenStoreError(TOKEN_MAP_NOT_PERSISTED) from None
    try:
        trusted = _trusted_dir(os.fstat(fd))
    except OSError:
        trusted = False
    if not trusted:
        os.close(fd)
        raise TokenStoreError(TOKEN_STORE_INSECURE_PERMISSIONS)
    return fd


def _open_anchor() -> _Anchor:
    """Walk the ORIGINAL configured state-root chain by descriptors, creating what is missing.

    The path is never ``realpath``-ed: it is walked component by component. Every directory is
    opened with ``O_DIRECTORY | O_NOFOLLOW`` relative to its parent and must be trusted (owned by
    this user or root; group/world-writable only if sticky). A symlink component is followed
    only if the directory CONTAINING it is trusted and the link is owned by this user or root;
    the walk then continues over the link target (absolute or relative; ``..`` in a target is
    refused) with the same checks on every component, at most ``_MAX_LINK_HOPS`` links in all
    (a loop is refused). The final ``token_store`` directory is never a link, must be ours and
    ``0700``, and is never repaired. A component created here is ``0700`` and its parent is
    fsynced so the new entry survives power loss BEFORE anything depends on it.
    """
    _require_dirfd_support()
    configured = _configured_root()
    pending: list[tuple[str, bool]] = [
        (p, False) for p in configured.split(os.sep) if p
    ]
    pending.append((_STORE_DIRNAME, True))
    pending.reverse()  # pop() takes the next component
    hops = 0
    fd = _open_root()
    try:
        while pending:
            part, is_store = pending.pop()
            if _skip_anchor_part(part):
                continue
            created = False
            try:
                child = _open_dir_nofollow(part, fd)
            except _LinkComponent:
                hops, target = _anchor_link_target(part, fd, is_store, hops)
                if target.startswith(os.sep):
                    os.close(fd)
                    fd = -1  # never close a stale descriptor if the reopen is refused
                    fd = _open_root()
                pending.extend((p, False) for p in reversed(target.split(os.sep)) if p)
                continue
            except FileNotFoundError:
                try:
                    os.mkdir(part, 0o700, dir_fd=fd)
                    created = True
                except FileExistsError:
                    pass
                child = _open_dir_nofollow(part, fd)
            _validate_anchor_child(child, fd, created, is_store)
            os.close(fd)
            fd = child
        anchor = _Anchor(fd, os.path.join(configured, _STORE_DIRNAME))
        fd = -1
        return anchor
    except TokenStoreError:
        raise
    except OSError:
        raise TokenStoreError(TOKEN_MAP_NOT_PERSISTED) from None
    finally:
        _close_anchor_fd(fd)


def _close_anchor_fd(fd: int) -> None:
    if fd != -1:
        os.close(fd)


def _skip_anchor_part(part: str) -> bool:
    if part in (".", ""):
        return True
    if part == "..":
        raise TokenStoreError(TOKEN_STORE_INSECURE_PERMISSIONS)
    return False


def _validate_anchor_child(child: int, fd: int, created: bool, is_store: bool) -> None:
    try:
        _validate_opened_child(child, fd, created, is_store)
    except BaseException:
        os.close(child)
        raise


def _validate_opened_child(child: int, fd: int, created: bool, is_store: bool) -> None:
    info = os.fstat(child)
    if created:
        os.fchmod(child, 0o700)  # our own new directory: defeat a restrictive umask
        os.fsync(child)
        os.fsync(fd)  # link the new entry into its parent durably
        info = os.fstat(child)
    if is_store:
        if info.st_uid != os.geteuid() or stat.S_IMODE(info.st_mode) & 0o077:
            raise TokenStoreError(TOKEN_STORE_INSECURE_PERMISSIONS)
    elif not _trusted_dir(info):
        raise TokenStoreError(TOKEN_STORE_INSECURE_PERMISSIONS)


def _anchor_link_target(
    part: str, fd: int, is_store: bool, hops: int
) -> tuple[int, str]:
    if is_store:
        raise TokenStoreError(TOKEN_STORE_INSECURE_PERMISSIONS) from None
    hops += 1
    target = _read_trusted_link(part, fd)
    if hops > _MAX_LINK_HOPS:
        raise TokenStoreError(TOKEN_STORE_INSECURE_PERMISSIONS) from None
    return hops, target


def _write_atomic(place: _Place, payload: bytes) -> None:
    """Durably replace the file: temp (0600) + fsync + ``os.replace`` + directory fsync.

    All steps are relative to the validated directory descriptor.
    """
    anchor, name = place.anchor, place.name
    tmp = f"{name}.tmp{os.getpid()}.{secrets.token_hex(4)}"
    try:
        anchor.verify()
        fd = os.open(
            tmp,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o600,
            dir_fd=anchor.fd,
        )
        try:
            os.fchmod(fd, 0o600)
            with os.fdopen(fd, "wb") as handle:
                fd = -1
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
        finally:
            if fd != -1:
                os.close(fd)
        _fault_point("after_payload_flush")
        anchor.verify()
        os.replace(tmp, name, src_dir_fd=anchor.fd, dst_dir_fd=anchor.fd)
        os.fsync(anchor.fd)
    except BaseException as exc:
        with contextlib.suppress(OSError):
            os.unlink(tmp, dir_fd=anchor.fd)
        if isinstance(exc, Exception):
            if isinstance(exc, TokenStoreError):
                raise
            raise TokenStoreError(TOKEN_MAP_NOT_PERSISTED) from None
        raise


def _remove_stale_temps(place: _Place) -> None:
    """Delete abandoned temp files of THIS namespace (a writer died between flush and replace).

    Called under the namespace lock, so no live writer of this namespace exists. Only the
    exact temp naming of this store file matches; nothing else in the directory is touched.
    """
    pattern = re.compile(rf"^{re.escape(place.name)}\.tmp[0-9]+\.[0-9a-f]{{8}}$")
    try:
        with os.scandir(place.anchor.fd) as entries:
            stale = [e.name for e in entries if pattern.fullmatch(e.name)]
        for name in stale:
            os.unlink(name, dir_fd=place.anchor.fd)
    except OSError:
        raise TokenStoreError(TOKEN_MAP_NOT_PERSISTED) from None


@contextlib.contextmanager
def _exclusive_lock(anchor: _Anchor, name: str, timeout: float) -> Iterator[None]:
    """Cross-process exclusive lock on ``<name>`` in the anchored directory.

    Bounded: tries without blocking until ``timeout`` elapses, then ``token_store_busy``.
    Fails closed without ``fcntl``. After acquiring, the anchor is re-verified.
    """
    try:
        import fcntl
    except ImportError:  # no advisory locks on this platform: fail closed
        raise TokenStoreError(TOKEN_MAP_NOT_PERSISTED) from None
    try:
        fd = os.open(
            name, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600, dir_fd=anchor.fd
        )
    except OSError:
        raise TokenStoreError(TOKEN_MAP_NOT_PERSISTED) from None
    try:
        info = os.fstat(fd)
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_uid != os.geteuid()
            or stat.S_IMODE(info.st_mode) & 0o077
        ):
            raise TokenStoreError(TOKEN_STORE_INSECURE_PERMISSIONS)
        give_up = time.monotonic() + max(0.0, timeout)
        delay = 0.005
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= give_up:
                    raise TokenStoreError(TOKEN_STORE_BUSY) from None
                time.sleep(delay)
                delay = min(delay * 2, 0.1)
            except OSError:
                raise TokenStoreError(TOKEN_MAP_NOT_PERSISTED) from None
        _fault_point("after_lock")
        anchor.verify()
        yield
    finally:
        os.close(fd)  # closing the descriptor releases the lock


def _empty_state(namespace: TokenNamespace) -> dict[str, Any]:
    return {
        "namespace": namespace.digest,
        "operations": {},
        "tokens": {},
        "revisions": {},
    }


def _read_state(place: _Place, namespace: TokenNamespace) -> dict[str, Any]:
    """The validated state, a fresh empty one if the file does not exist, or a closed refusal."""
    try:
        fd = os.open(place.name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=place.anchor.fd)
    except FileNotFoundError:
        return _empty_state(namespace)
    except OSError:
        raise TokenStoreError(TOKEN_MAP_NOT_PERSISTED) from None
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_uid != os.geteuid():
            raise TokenStoreError(TOKEN_MAP_NOT_PERSISTED)
        if stat.S_IMODE(info.st_mode) != 0o600:
            raise TokenStoreError(TOKEN_STORE_INSECURE_PERMISSIONS)
        with os.fdopen(fd, "rb") as handle:
            fd = -1
            raw = handle.read()
    except OSError:
        raise TokenStoreError(TOKEN_MAP_NOT_PERSISTED) from None
    finally:
        if fd != -1:
            os.close(fd)
    if not raw:
        raise TokenStoreError(TOKEN_MAP_NOT_PERSISTED)
    version = raw[0]
    if version != FORMAT_VERSION:
        # 0x01..0x1f is the version space; anything else is not a version byte at all
        if 0 < version < 0x20:
            raise TokenStoreError(TOKEN_STORE_VERSION_UNSUPPORTED)
        raise TokenStoreError(TOKEN_MAP_NOT_PERSISTED)
    try:
        state = json.loads(raw[1:].decode("utf-8"))
    except ValueError:
        raise TokenStoreError(TOKEN_MAP_NOT_PERSISTED) from None
    _validate_state(state, namespace)
    return state  # type: ignore[no-any-return]


def _serialize(state: dict[str, Any]) -> bytes:
    return bytes([FORMAT_VERSION]) + json.dumps(
        state, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


# --- state validation (any deviation refuses the WHOLE store) -----------------------------


def _bad() -> TokenStoreError:
    return TokenStoreError(TOKEN_MAP_NOT_PERSISTED)


def _is_int(value: Any, low: int = 0) -> bool:
    return (
        isinstance(value, int)
        and not isinstance(value, bool)
        and low <= value <= MAX_COUNT
    )


def _valid_name(value: Any) -> bool:
    if not isinstance(value, str) or not 1 <= len(value) <= MAX_NAME_CHARS:
        return False
    try:
        value.encode("utf-8")
    except UnicodeEncodeError:
        return False
    return True


def _valid_item(item: Any) -> bool:
    if not isinstance(item, list) or len(item) != 3:
        return False
    kind, parent, name = item
    if kind not in KINDS or not _valid_name(name):
        return False
    if kind == "choice":
        return isinstance(parent, str) and bool(_KNOB_REF.fullmatch(parent))
    return parent is None


def _validate_state_shape(
    state: Any, namespace: TokenNamespace
) -> tuple[dict[Any, Any], dict[Any, Any], dict[Any, Any]]:
    if not isinstance(state, dict) or set(state) != {
        "namespace",
        "operations",
        "tokens",
        "revisions",
    }:
        raise _bad()
    if state["namespace"] != namespace.digest:
        raise _bad()
    operations, tokens, revisions = (
        state["operations"],
        state["tokens"],
        state["revisions"],
    )
    if not (
        isinstance(operations, dict)
        and isinstance(tokens, dict)
        and isinstance(revisions, dict)
    ):
        raise _bad()
    # revision values are typed before any comparison below (a null / string / object value
    # must refuse as an unreadable store, not surface as a TypeError)
    if not all(isinstance(r, str) and _is_int(v, 1) for r, v in revisions.items()):
        raise _bad()
    return operations, tokens, revisions


def _validate_state(state: Any, namespace: TokenNamespace) -> None:
    operations, tokens, revisions = _validate_state_shape(state, namespace)
    _validate_operations(operations, namespace)

    claimed: set[tuple[str, int]] = set()
    token_refs: set[str] = set()
    for key, entry in tokens.items():
        _validate_token_entry(key, entry, operations, revisions, claimed, token_refs)
    # every journalled item has its token entry (no half-written batch)
    for op_key, op in operations.items():
        for ordinal in range(len(op["items"])):
            if (op_key, ordinal) not in claimed:
                raise _bad()
    for ref, revision in revisions.items():
        if ref not in token_refs or not _is_int(revision, 1):
            raise _bad()


def _validate_operations(operations: dict[Any, Any], namespace: TokenNamespace) -> None:
    seen_refs: set[str] = set()
    for op_key, op in operations.items():
        _validate_operation(op_key, op, namespace, seen_refs)


def _validate_operation(
    op_key: Any, op: Any, namespace: TokenNamespace, seen_refs: set[str]
) -> None:
    if not isinstance(op_key, str) or not _OP_KEY.fullmatch(op_key):
        raise _bad()
    if not isinstance(op, dict) or set(op) != {"items", "digest", "refs"}:
        raise _bad()
    items, refs = op["items"], op["refs"]
    if not isinstance(items, list) or not 1 <= len(items) <= MAX_ITEMS_PER_OPERATION:
        raise _bad()
    if not all(_valid_item(i) for i in items):
        raise _bad()
    if not isinstance(op["digest"], str) or not _HEX64.fullmatch(op["digest"]):
        raise _bad()
    if (
        request_digest(namespace.agent_ref, [(i[0], i[1]) for i in items]).hex()
        != op["digest"]
    ):
        raise _bad()
    if refs is not None:
        _validate_operation_refs(items, refs, seen_refs)


def _validate_operation_refs(items: list[Any], refs: Any, seen_refs: set[str]) -> None:
    if not isinstance(refs, list) or len(refs) != len(items):
        raise _bad()
    for item, ref in zip(items, refs, strict=True):
        if not isinstance(ref, str) or not _REF[item[0]].fullmatch(ref):
            raise _bad()
        if ref in seen_refs:
            raise _bad()
        seen_refs.add(ref)


_TOKEN_ENTRY_KEYS = {
    "kind",
    "parent",
    "name",
    "ref",
    "state",
    "op",
    "ordinal",
    "spec",
    "spec_history",
}


def _validate_token_entry(
    key: Any,
    entry: Any,
    operations: dict[str, Any],
    revisions: dict[str, Any],
    claimed: set[tuple[str, int]],
    token_refs: set[str],
) -> None:
    if not isinstance(key, str) or not isinstance(entry, dict):
        raise _bad()
    if set(entry) != _TOKEN_ENTRY_KEYS:
        raise _bad()
    if not _valid_item([entry["kind"], entry["parent"], entry["name"]]):
        raise _bad()
    if key != _key(entry["kind"], entry["parent"], entry["name"]):
        raise _bad()
    op_key, ordinal = entry["op"], entry["ordinal"]
    if not isinstance(op_key, str) or op_key not in operations or not _is_int(ordinal):
        raise _bad()
    op = operations[op_key]
    if ordinal >= len(op["items"]) or op["items"][ordinal] != [
        entry["kind"],
        entry["parent"],
        entry["name"],
    ]:
        raise _bad()
    if (op_key, ordinal) in claimed:
        raise _bad()
    claimed.add((op_key, ordinal))
    _validate_entry_status(entry, op, revisions, token_refs)


def _validate_entry_status(
    entry: dict[str, Any],
    op: dict[str, Any],
    revisions: dict[str, Any],
    token_refs: set[str],
) -> None:
    status = entry["state"]
    if status not in STATES:
        raise _bad()
    if status == S0_INTENT:
        if (
            entry["ref"] is not None
            or op["refs"] is not None
            or entry["spec"] is not None
            or entry["spec_history"] != []
        ):
            raise _bad()
        return
    if op["refs"] is None or entry["ref"] != op["refs"][entry["ordinal"]]:
        raise _bad()
    token_refs.add(entry["ref"])
    if (status == S3_SPEC_COMMITTED) != (entry["spec"] is not None):
        raise _bad()
    if entry["spec"] is not None and not (
        isinstance(entry["spec"], str) and _SPEC_REF.fullmatch(entry["spec"])
    ):
        raise _bad()
    _validate_spec_history(entry["spec_history"], entry["spec"])
    if (
        status in (S2_LABEL_ACKED, S3_SPEC_COMMITTED)
        and revisions.get(entry["ref"], 0) < 1
    ):
        raise _bad()


def _validate_spec_history(history: Any, spec: Any) -> None:
    if (
        not isinstance(history, list)
        or len(history) > MAX_SPEC_HISTORY
        or len(set(map(str, history))) != len(history)
        or not all(isinstance(h, str) and _SPEC_REF.fullmatch(h) for h in history)
        or (history[-1:] or [None])[0] != spec
    ):
        raise _bad()


# --- the store ----------------------------------------------------------------------------


class _Locked:
    """The namespace's state while its lock is held; ``write`` persists durably."""

    def __init__(self, place: _Place, state: dict[str, Any], deadline: float) -> None:
        self.place = place
        self.state = state
        #: ``time.monotonic()`` instant by which everything under this lock must be done
        self.deadline = deadline

    def write(self) -> None:
        _write_atomic(self.place, _serialize(self.state))


class TokenStore:
    """One namespace's durable token map, journal and revision maxima."""

    def __init__(self, namespace: TokenNamespace) -> None:
        if not isinstance(namespace, TokenNamespace):
            raise TokenStoreError(TOKEN_NAMESPACE_INVALID)
        self.namespace = namespace
        self.path = store_path(namespace)

    @contextlib.contextmanager
    def locked(self, *, deadline_seconds: float | None = None) -> Iterator[_Locked]:
        _require_enabled()
        started = time.monotonic()
        anchor = _open_anchor()
        try:
            name = self.path.name
            _fault_point("after_validate")
            with _exclusive_lock(
                anchor, f"{self.namespace.digest}.lock", LOCK_TIMEOUT_SECONDS
            ):
                place = _Place(anchor, name)
                _remove_stale_temps(place)
                yield _Locked(
                    place,
                    _read_state(place, self.namespace),
                    started
                    + (
                        OPERATION_DEADLINE_SECONDS
                        if deadline_seconds is None
                        else deadline_seconds
                    ),
                )
        finally:
            anchor.close()

    # -- reads -------------------------------------------------------------------------

    @_closed_boundary
    def lookup(self, kind: str, parent_ref: str | None, name: str) -> str | None:
        """The ref issued for this key, or ``None`` (also while still only an intent)."""
        with self.locked() as tx:
            entry = tx.state["tokens"].get(_key(kind, parent_ref, name))
            return None if entry is None else entry["ref"]

    @_closed_boundary
    def state_of(self, ref: str) -> str:
        with self.locked() as tx:
            return str(_find_by_ref(tx.state, ref)["state"])

    @_closed_boundary
    def state_if_known(self, ref: str) -> str | None:
        """The journal state of ``ref``, or ``None`` when it is not a token of this namespace."""
        with self.locked() as tx:
            if isinstance(ref, str):
                for entry in tx.state["tokens"].values():
                    if entry["ref"] == ref:
                        return str(entry["state"])
            return None

    @_closed_boundary
    def revision_of(self, ref: str) -> int:
        """The durable revision maximum recorded for ``ref`` (0 if none)."""
        with self.locked() as tx:
            _find_by_ref(tx.state, ref)
            return int(tx.state["revisions"].get(ref, 0))

    @_closed_boundary
    def conflicted_refs(self) -> list[str]:
        """Refs whose label write lost to another writer and cannot be verified (S1->S2)."""
        with self.locked() as tx:
            return sorted(
                e["ref"]
                for e in tx.state["tokens"].values()
                if e["state"] == S_LABEL_CONFLICT
            )

    @_closed_boundary
    def require_no_label_conflict(self) -> None:
        """Refuse ``label_conflict_unverified`` while any token is in that state."""
        if self.conflicted_refs():
            raise TokenStoreError(LABEL_CONFLICT_UNVERIFIED)

    # -- transitions (S2 / S3 and the revision maximum) --------------------------------

    @_closed_boundary
    def record_revision(self, ref: str, revision: int) -> None:
        """Raise the durable per-ref revision maximum; a lower revision is refused."""
        with self.locked() as tx:
            _find_by_ref(tx.state, ref)
            _raise_revision(tx.state, ref, revision)
            tx.write()

    @_closed_boundary
    def mark_label_acked(self, ref: str, revision: int) -> None:
        """S1 -> S2: a ``send_new`` ack with revision >= 1; persists the revision maximum.

        Idempotent for an already acked ref at the same or a higher revision.
        """
        with self.locked() as tx:
            entry = _find_by_ref(tx.state, ref)
            if entry["state"] not in (S1_PENDING_LABEL, S2_LABEL_ACKED):
                raise TokenStoreError(TOKEN_STATE_INVALID)
            _raise_revision(tx.state, ref, revision)
            entry["state"] = S2_LABEL_ACKED
            tx.write()

    @_closed_boundary
    def mark_label_stale(self, ref: str) -> None:
        """S1 -> ``label_conflict_unverified``: ``stale_revision`` with no approved reader.

        Another writer won and the plaintext cannot be compared, so the run must refuse to
        start (:meth:`require_no_label_conflict`); a new spec is never created silently.
        """
        with self.locked() as tx:
            entry = _find_by_ref(tx.state, ref)
            if entry["state"] not in (S1_PENDING_LABEL, S_LABEL_CONFLICT):
                raise TokenStoreError(TOKEN_STATE_INVALID)
            entry["state"] = S_LABEL_CONFLICT
            tx.write()

    @_closed_boundary
    def mark_spec_committed(self, ref: str, search_spec_ref: str) -> None:
        """S2 -> S3, and every later spec of the same token: record the ``searchspecref_``.

        A token reused by a successor spec keeps its identity, label state and revision
        maximum; each spec commit is recorded separately (the latest one in ``spec`` and the
        bounded history in ``spec_history``), so successive specs never remint a token or lower
        a revision. Committing the spec that is already the latest is idempotent.
        """
        if not isinstance(search_spec_ref, str) or not _SPEC_REF.fullmatch(
            search_spec_ref
        ):
            raise TokenStoreError(TOKEN_STATE_INVALID)
        with self.locked() as tx:
            entry = _find_by_ref(tx.state, ref)
            if entry["state"] == S3_SPEC_COMMITTED and entry["spec"] == search_spec_ref:
                return
            if entry["state"] not in (S2_LABEL_ACKED, S3_SPEC_COMMITTED):
                raise TokenStoreError(TOKEN_STATE_INVALID)
            history = [h for h in entry["spec_history"] if h != search_spec_ref]
            history.append(search_spec_ref)
            entry["state"] = S3_SPEC_COMMITTED
            entry["spec"] = search_spec_ref
            entry["spec_history"] = history[-MAX_SPEC_HISTORY:]
            tx.write()

    @_closed_boundary
    def spec_commits(self, ref: str) -> list[str]:
        """The recorded spec commits of ``ref``, oldest first (the last is the latest)."""
        with self.locked() as tx:
            return list(_find_by_ref(tx.state, ref)["spec_history"])


def _find_by_ref(state: dict[str, Any], ref: str) -> dict[str, Any]:
    if isinstance(ref, str):
        for entry in state["tokens"].values():
            if entry["ref"] == ref:
                return entry  # type: ignore[no-any-return]
    raise TokenStoreError(TOKEN_STATE_INVALID)


def _raise_revision(state: dict[str, Any], ref: str, revision: int) -> None:
    if not _is_int(revision, 1):
        raise TokenStoreError(TOKEN_STATE_INVALID)
    current = state["revisions"].get(ref, 0)
    if revision < current:
        raise TokenStoreError(TOKEN_REVISION_STALE)
    state["revisions"][ref] = revision


# --- issuance client ----------------------------------------------------------------------

#: ``transport(body) -> (http_status, parsed_json_or_None)`` or, when the transport can read
#: it, ``(http_status, parsed_json_or_None, retry_after_seconds_or_None)``; may raise on a
#: network failure.
TokenRefsTransport = Callable[[dict[str, Any]], "tuple[Any, ...]"]

#: Retry pacing for ``service_unavailable``: used when no ``Retry-After`` is given.
RETRY_BACKOFF_SECONDS = (0.5, 1.0, 2.0, 4.0, 8.0)
MAX_SERVICE_UNAVAILABLE_ATTEMPTS = 8
_sleep = time.sleep  # indirection so tests can run the pacing without waiting

#: Closed refusal envelope of the route (Schema ``TokenRefsRefusal``): code -> (status, message).
_REFUSALS: dict[str, tuple[int, str, str]] = {
    "invalid_request": (400, "Request is invalid.", TOKEN_REFS_INVALID_REQUEST),
    "reference_not_found": (
        404,
        "Reference not found.",
        TOKEN_REFS_REFERENCE_NOT_FOUND,
    ),
    "operation_key_conflict": (
        409,
        "Operation key conflicts with an earlier request.",
        OPERATION_KEY_CONFLICT,
    ),
    "quota_exceeded": (429, "Quota exceeded.", TOKEN_REFS_QUOTA_EXCEEDED),
    "internal_error": (500, "Internal error.", TOKEN_REFS_SERVER_ERROR),
    "service_unavailable": (
        503,
        "Service is temporarily unavailable.",
        TOKEN_REFS_SERVICE_UNAVAILABLE,
    ),
}
_ENVELOPE_KEYS = frozenset({"success", "error", "error_code", "message"})


def backend_transport(client: Any) -> TokenRefsTransport:
    """Adapt a ``BackendIntegratedClient`` (its auth, origin gate and egress switches)."""
    return lambda body: client.issue_token_refs_with_retry_after_sync(body)  # type: ignore[no-any-return]


def _refusal_code(status: Any, payload: Any) -> str:
    """The SDK code for a non-200 answer; nothing of the payload is kept."""
    if isinstance(payload, dict) and set(payload) == _ENVELOPE_KEYS:
        raw = payload.get("error_code")
        spec = _REFUSALS.get(raw) if isinstance(raw, str) else None
        if (
            spec is not None
            and payload.get("success") is False
            and payload.get("error") == raw
            and payload.get("message") == spec[1]
            and status == spec[0]
        ):
            return spec[2]
    if status == 404:
        return TOKEN_REFS_UNAVAILABLE  # the flag-off uniform unknown-route 404
    if status == 401:
        return TOKEN_REFS_UNAUTHENTICATED
    if status == 403:
        return TOKEN_REFS_FORBIDDEN
    if status == 429:
        return TOKEN_REFS_QUOTA_EXCEEDED
    if (
        isinstance(status, int)
        and not isinstance(status, bool)
        and 500 <= status <= 599
    ):
        return TOKEN_REFS_SERVER_ERROR
    return TOKEN_REFS_RESPONSE_INVALID


def _wire_body(
    namespace: TokenNamespace, op_key: str, items: list[list[Any]]
) -> dict[str, Any]:
    """The request body exactly per Schema: kinds and parent refs only, never a name."""
    wire = []
    for kind, parent, _name in items:
        item: dict[str, Any] = {"kind": kind}
        if parent is not None:
            item["parent_ref"] = parent
        wire.append(item)
    return {
        "schema_version": 1,
        "agent_ref": namespace.agent_ref,
        "operation_key": op_key,
        "items": wire,
    }


def _call_with_deadline(
    transport: TokenRefsTransport, body: dict[str, Any], deadline: float
) -> tuple[Any, ...]:
    """Run ``transport(body)`` (auth included) but give up at ``deadline`` (monotonic).

    The call runs in a daemon thread so a stalled transport cannot hold the namespace lock
    past the deadline; on timeout the call is abandoned (its result, if it ever arrives, is
    discarded) and ``token_store_deadline_exceeded`` is raised with S0 intact, so a retry
    re-sends the same operation_key. Any ``BaseException`` the callable raises is re-raised
    here unchanged, exactly as if it had been called inline.
    """
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TokenStoreError(TOKEN_STORE_DEADLINE_EXCEEDED)
    box: list[Any] = []

    def run() -> None:
        try:
            box.append((True, transport(body)))
        except BaseException as exc:  # noqa: BLE001 - relayed to the caller below
            box.append((False, exc))

    worker = threading.Thread(target=run, name="token-refs-issue", daemon=True)
    worker.start()
    worker.join(remaining)
    if worker.is_alive() or not box:
        raise TokenStoreError(TOKEN_STORE_DEADLINE_EXCEEDED)
    ok, value = box[0]
    if not ok:
        raise value
    return value  # type: ignore[no-any-return]


def _answer(value: Any) -> tuple[Any, Any, float | None]:
    """``(status, payload, retry_after)`` from a 2- or 3-tuple transport answer.

    ``retry_after`` is kept only when it is a finite non-negative number; anything else is
    ``None`` (the pacing default applies). A malformed answer raises ``ValueError``, which the
    caller treats like any other transport failure.
    """
    if not isinstance(value, tuple) or len(value) not in (2, 3):
        raise ValueError("transport answer")
    retry_after = value[2] if len(value) == 3 else None
    if (
        isinstance(retry_after, bool)
        or not isinstance(retry_after, (int, float))
        or not math.isfinite(retry_after)
        or retry_after < 0
    ):
        retry_after = None
    return value[0], value[1], retry_after


def _issue(
    tx: _Locked,
    namespace: TokenNamespace,
    transport: TokenRefsTransport,
    op_key: str,
) -> None:
    """Send operation ``op_key`` (S0) with its journalled batch and persist S1, or refuse."""
    op = tx.state["operations"][op_key]
    body = _wire_body(namespace, op_key, op["items"])
    attempt = 0
    while True:
        outcome: TokenStoreError | None = None
        try:
            status, payload, retry_after = _answer(
                _call_with_deadline(transport, body, tx.deadline)
            )
        except TokenStoreError as exc:
            outcome = exc
        except Exception:  # noqa: BLE001 - a transport error can quote a URL or header
            outcome = TokenStoreError(TOKEN_REFS_TRANSPORT_FAILED)
        if outcome is not None:
            raise outcome from None
        if status == 200:
            break
        # S0 is NEVER deleted on a refusal. The Backend authorizes the agent before it looks
        # up the receipt, so a 400/404 on a resumed key can follow an earlier attempt that
        # DID commit (response lost, then access lost or revoked): dropping the key would let
        # a later call mint a new one and double-issue. The intent keeps failing with its
        # closed code until it succeeds or the caller deliberately abandons it.
        delay, attempt = _issue_retry(
            status, payload, retry_after, attempt, tx.deadline
        )
        _sleep(delay)
    refs = _validated_issue_refs(payload, op, tx.state)
    op["refs"] = list(refs)
    for entry in tx.state["tokens"].values():
        if entry["op"] == op_key:
            entry["ref"] = refs[entry["ordinal"]]
            entry["state"] = S1_PENDING_LABEL
    tx.write()


def _issue_retry(
    status: Any, payload: Any, retry_after: float | None, attempt: int, deadline: float
) -> tuple[float, int]:
    code = _refusal_code(status, payload)
    if code != TOKEN_REFS_SERVICE_UNAVAILABLE:
        raise TokenStoreError(code)
    # 503 service_unavailable: the SAME operation_key is re-sent (idempotent), paced by
    # Retry-After when the server gave one, within the operation deadline; once the next
    # wait would not fit (or the attempts are used up) the closed retryable code stands.
    delay = (
        retry_after
        if retry_after is not None
        else RETRY_BACKOFF_SECONDS[min(attempt, len(RETRY_BACKOFF_SECONDS) - 1)]
    )
    attempt += 1
    if attempt >= MAX_SERVICE_UNAVAILABLE_ATTEMPTS or delay >= (
        deadline - time.monotonic()
    ):
        raise TokenStoreError(code)
    return delay, attempt


def _validated_issue_refs(
    payload: Any, op: dict[str, Any], state: dict[str, Any]
) -> list[str]:
    refs = (
        payload.get("refs")
        if isinstance(payload, dict) and set(payload) == {"refs"}
        else None
    )
    if (
        not isinstance(refs, list)
        or len(refs) != len(op["items"])
        # every element is a well-formed ref string BEFORE any set() (an unhashable nested
        # value would otherwise raise a TypeError and surface as token_store_internal)
        or any(
            not isinstance(ref, str) or not _REF[item[0]].fullmatch(ref)
            for item, ref in zip(op["items"], refs, strict=True)
        )
        or len(set(refs)) != len(refs)
        or any(ref in _all_refs(state) for ref in refs)
    ):
        raise TokenStoreError(TOKEN_REFS_RESPONSE_INVALID)
    return refs


def _all_refs(state: dict[str, Any]) -> set[str]:
    return {ref for op in state["operations"].values() for ref in (op["refs"] or [])}


def _abandon(tx: _Locked, op_key: str) -> None:
    del tx.state["operations"][op_key]
    for key in [k for k, e in tx.state["tokens"].items() if e["op"] == op_key]:
        del tx.state["tokens"][key]
    tx.write()


@_closed_boundary
def pending_operation_keys(namespace: TokenNamespace) -> list[str]:
    """Operation keys still at S0 (sent or not, never answered), oldest-insertion first."""
    _require_enabled()
    with TokenStore(namespace).locked() as tx:
        return [k for k, op in tx.state["operations"].items() if op["refs"] is None]


@_closed_boundary
def abandon_pending_operation(
    namespace: TokenNamespace, *, operation_key: str | None = None
) -> int:
    """Deliberately give up S0 intents (all of them, or only ``operation_key``); returns how many.

    WARNING: this may create DUPLICATE identities. If the earlier attempt reached the server
    and committed (its response lost), the refs it issued are orphaned and the names are
    issued again under a new operation_key. Use it only when the operator has decided that
    the pending intent will never succeed. Nothing in the SDK calls it automatically; a
    pending intent that keeps failing raises its closed code on every ``ensure_tokens``.
    """
    _require_enabled()
    if operation_key is not None and (
        not isinstance(operation_key, str) or not _OP_KEY.fullmatch(operation_key)
    ):
        raise TokenStoreError(TOKEN_REQUEST_INVALID)
    with TokenStore(namespace).locked() as tx:
        pending = [
            k
            for k, op in tx.state["operations"].items()
            if op["refs"] is None and (operation_key is None or k == operation_key)
        ]
        if operation_key is not None and not pending:
            raise TokenStoreError(TOKEN_STATE_INVALID)
        for key in pending:
            _abandon(tx, key)
        return len(pending)


def _journal(tx: _Locked, namespace: TokenNamespace, items: list[list[Any]]) -> str:
    """S0: make a new operation (fresh 128-bit key, exact ordered batch) durable."""
    op_key = secrets.token_hex(16)
    tx.state["operations"][op_key] = {
        "items": items,
        "digest": request_digest(
            namespace.agent_ref, [(i[0], i[1]) for i in items]
        ).hex(),
        "refs": None,
    }
    for ordinal, (kind, parent, name) in enumerate(items):
        tx.state["tokens"][_key(kind, parent, name)] = {
            "kind": kind,
            "parent": parent,
            "name": name,
            "ref": None,
            "state": S0_INTENT,
            "op": op_key,
            "ordinal": ordinal,
            "spec": None,
            "spec_history": [],
        }
    tx.write()
    return op_key


TokenRequest = tuple[str, "str | None", str]


def _validate_requests(requests: Sequence[TokenRequest]) -> list[TokenRequest]:
    if isinstance(requests, (str, bytes)) or not isinstance(requests, Sequence):
        raise TokenStoreError(TOKEN_REQUEST_INVALID)
    out: list[TokenRequest] = []
    seen: set[TokenRequest] = set()
    for request in requests:
        triple = _validate_token_request(request)
        if triple not in seen:
            seen.add(triple)
            out.append(triple)
    return out


def _validate_token_request(request: TokenRequest) -> TokenRequest:
    if not isinstance(request, (tuple, list)) or len(request) != 3:
        raise TokenStoreError(TOKEN_REQUEST_INVALID)
    kind, parent_name, name = request
    if kind not in KINDS or not _valid_name(name):
        raise TokenStoreError(TOKEN_REQUEST_INVALID)
    if kind == "choice":
        if not _valid_name(parent_name):
            raise TokenStoreError(TOKEN_REQUEST_INVALID)
    elif parent_name is not None:
        raise TokenStoreError(TOKEN_REQUEST_INVALID)
    triple = (kind, parent_name, name)
    return triple


@_closed_boundary
def ensure_tokens(
    namespace: TokenNamespace,
    requests: Sequence[TokenRequest],
    *,
    transport: TokenRefsTransport,
    deadline_seconds: float | None = None,
) -> dict[TokenRequest, str]:
    """Return the issued ref for each ``(kind, parent_knob_name | None, local_name)``.

    Keys already in the store are answered locally (no request). Missing keys are journalled
    (S0, durable BEFORE the call), issued in operations of at most 200 items, and persisted
    (S1). An operation left at S0 by a crash is re-sent first with the SAME operation_key and
    the SAME ordered batch; if the server reports ``operation_key_conflict`` the call is
    refused, a new key is never minted silently. A ``choice`` needs its parent knob to be in
    the store already (issue knobs in an earlier call). The namespace lock is held for the
    whole call, so concurrent callers of one namespace produce one operation and one token.
    Acquiring the lock is bounded (``token_store_busy``) and so is everything done under it
    (``deadline_seconds``, default :data:`OPERATION_DEADLINE_SECONDS`, else
    ``token_store_deadline_exceeded``); S0 survives a timeout.
    """
    _require_enabled()
    wanted = _validate_requests(requests)
    if not callable(transport):
        raise TokenStoreError(TOKEN_REQUEST_INVALID)
    store = TokenStore(namespace)
    with store.locked(deadline_seconds=deadline_seconds) as tx:
        for op_key in [
            k for k, op in tx.state["operations"].items() if op["refs"] is None
        ]:
            _issue(tx, namespace, transport, op_key)

        resolved: list[tuple[TokenRequest, str, str | None]] = []
        for request in wanted:
            kind, parent_name, name = request
            parent_ref = None
            if kind == "choice":
                parent = tx.state["tokens"].get(_key("knob", None, str(parent_name)))
                if parent is None or parent["ref"] is None:
                    raise TokenStoreError(TOKEN_PARENT_MISSING)
                parent_ref = parent["ref"]
            resolved.append((request, _key(kind, parent_ref, name), parent_ref))

        missing = [
            [req[0], parent_ref, req[2]]
            for req, key, parent_ref in resolved
            if key not in tx.state["tokens"]
        ]
        for start in range(0, len(missing), MAX_ITEMS_PER_OPERATION):
            chunk = missing[start : start + MAX_ITEMS_PER_OPERATION]
            _issue(tx, namespace, transport, _journal(tx, namespace, chunk))

        result: dict[TokenRequest, str] = {}
        for request, key, _parent in resolved:
            ref = tx.state["tokens"][key]["ref"]
            if ref is None:
                raise TokenStoreError(TOKEN_MAP_NOT_PERSISTED)
            result[request] = ref
        return result
