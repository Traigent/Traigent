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
* **Namespace** = (normalized backend origin, agent_ref); one store file per namespace named by
  the sha256 of the canonical namespace (never a plaintext name or agent id). The same
  (kind, parent token ref, name) maps to the same token across specs of one agent; nothing is
  shared across agents or backends.
* **Key** = (kind, parent token ref | None, local name). A choice's parent is a knob token that
  must already be in the store (the server only accepts a parent issued by an EARLIER request).
* **Journal.** S0 intent (operation_key + the exact ordered batch) is made durable BEFORE the
  network call; S1 (refs received, ``pending_label``) after; S2 (label acked, revision >= 1) and
  S3 (spec committed) are store-level transitions here and are wired in 1b-2b / 1b-4. A
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
import functools
import hashlib
import json
import os
import re
import secrets
import stat
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar
from urllib.parse import urlsplit

ENV_FLAG = "TRAIGENT_TOKEN_STORE_ENABLED"
KINDS = ("knob", "choice", "metric", "template")
MAX_ITEMS_PER_OPERATION = 200
MAX_NAME_CHARS = 4096
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
TOKEN_REFS_RESPONSE_INVALID = "token_refs_response_invalid"
LABEL_CONFLICT_UNVERIFIED = "label_conflict_unverified"
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
        TOKEN_REFS_RESPONSE_INVALID,
        LABEL_CONFLICT_UNVERIFIED,
        TOKEN_STORE_INTERNAL,
    }
)


class TokenStoreError(Exception):
    """A closed refusal: carries exactly one code from :data:`CODES` and nothing else."""

    def __init__(self, code: str) -> None:
        if code not in CODES:
            code = TOKEN_STORE_INTERNAL
        super().__init__(code)
        self.code = code

    def __str__(self) -> str:
        return self.code

    def __repr__(self) -> str:
        return f"TokenStoreError({self.code!r})"


_F = TypeVar("_F", bound=Callable[..., Any])


def _closed_boundary(func: _F) -> _F:
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


def _ensure_dir(path: Path) -> None:
    """Create the 0700 store directory, or verify an existing one (never repaired)."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.mkdir(path, 0o700)
        except FileExistsError:
            pass
        else:
            os.chmod(
                path, 0o700
            )  # our own new directory: defeat a restrictive umask only
        info = os.lstat(path)
    except OSError:
        raise TokenStoreError(TOKEN_MAP_NOT_PERSISTED) from None
    if (
        not stat.S_ISDIR(info.st_mode)
        or info.st_uid != os.geteuid()
        or stat.S_IMODE(info.st_mode) & 0o077
    ):
        raise TokenStoreError(TOKEN_STORE_INSECURE_PERMISSIONS)


def _fsync_dir(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _write_atomic(path: Path, payload: bytes) -> None:
    """Durably replace ``path``: temp (0600) + fsync + ``os.replace`` + directory fsync."""
    tmp = path.with_name(f"{path.name}.tmp{os.getpid()}.{secrets.token_hex(4)}")
    try:
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
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
        os.replace(tmp, path)
        _fsync_dir(path.parent)
    except BaseException as exc:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        if isinstance(exc, Exception):
            raise TokenStoreError(TOKEN_MAP_NOT_PERSISTED) from None
        raise


@contextlib.contextmanager
def _exclusive_lock(path: Path) -> Iterator[None]:
    """Cross-process exclusive lock on ``<store>.lock``; fails closed without ``fcntl``."""
    try:
        import fcntl
    except ImportError:  # no advisory locks on this platform: fail closed
        raise TokenStoreError(TOKEN_MAP_NOT_PERSISTED) from None
    lock_path = path.with_suffix(".lock")
    try:
        fd = os.open(lock_path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
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
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
        except OSError:
            raise TokenStoreError(TOKEN_MAP_NOT_PERSISTED) from None
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


def _read_state(path: Path, namespace: TokenNamespace) -> dict[str, Any]:
    """The validated state, a fresh empty one if the file does not exist, or a closed refusal."""
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
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
    except (ValueError, UnicodeDecodeError):
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


def _validate_state(state: Any, namespace: TokenNamespace) -> None:
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

    seen_refs: set[str] = set()
    for op_key, op in operations.items():
        if not isinstance(op_key, str) or not _OP_KEY.fullmatch(op_key):
            raise _bad()
        if not isinstance(op, dict) or set(op) != {"items", "digest", "refs"}:
            raise _bad()
        items, refs = op["items"], op["refs"]
        if (
            not isinstance(items, list)
            or not 1 <= len(items) <= MAX_ITEMS_PER_OPERATION
        ):
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
            if not isinstance(refs, list) or len(refs) != len(items):
                raise _bad()
            for item, ref in zip(items, refs, strict=True):
                if not isinstance(ref, str) or not _REF[item[0]].fullmatch(ref):
                    raise _bad()
                if ref in seen_refs:
                    raise _bad()
                seen_refs.add(ref)

    claimed: set[tuple[str, int]] = set()
    token_refs: set[str] = set()
    for key, entry in tokens.items():
        if not isinstance(key, str) or not isinstance(entry, dict):
            raise _bad()
        if set(entry) != {
            "kind",
            "parent",
            "name",
            "ref",
            "state",
            "op",
            "ordinal",
            "spec",
        }:
            raise _bad()
        if not _valid_item([entry["kind"], entry["parent"], entry["name"]]):
            raise _bad()
        if key != _key(entry["kind"], entry["parent"], entry["name"]):
            raise _bad()
        op_key, ordinal = entry["op"], entry["ordinal"]
        if (
            not isinstance(op_key, str)
            or op_key not in operations
            or not _is_int(ordinal)
        ):
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
        status = entry["state"]
        if status not in STATES:
            raise _bad()
        if status == S0_INTENT:
            if (
                entry["ref"] is not None
                or op["refs"] is not None
                or entry["spec"] is not None
            ):
                raise _bad()
        else:
            if op["refs"] is None or entry["ref"] != op["refs"][ordinal]:
                raise _bad()
            token_refs.add(entry["ref"])
            if (status == S3_SPEC_COMMITTED) != (entry["spec"] is not None):
                raise _bad()
            if entry["spec"] is not None and not (
                isinstance(entry["spec"], str) and _SPEC_REF.fullmatch(entry["spec"])
            ):
                raise _bad()
            if (
                status in (S2_LABEL_ACKED, S3_SPEC_COMMITTED)
                and revisions.get(entry["ref"], 0) < 1
            ):
                raise _bad()
    # every journalled item has its token entry (no half-written batch)
    for op_key, op in operations.items():
        for ordinal in range(len(op["items"])):
            if (op_key, ordinal) not in claimed:
                raise _bad()
    for ref, revision in revisions.items():
        if ref not in token_refs or not _is_int(revision, 1):
            raise _bad()


# --- the store ----------------------------------------------------------------------------


class _Locked:
    """The namespace's state while its lock is held; ``write`` persists durably."""

    def __init__(self, path: Path, state: dict[str, Any]) -> None:
        self.path = path
        self.state = state

    def write(self) -> None:
        _write_atomic(self.path, _serialize(self.state))


class TokenStore:
    """One namespace's durable token map, journal and revision maxima."""

    def __init__(self, namespace: TokenNamespace) -> None:
        if not isinstance(namespace, TokenNamespace):
            raise TokenStoreError(TOKEN_NAMESPACE_INVALID)
        self.namespace = namespace
        self.path = store_path(namespace)

    @contextlib.contextmanager
    def locked(self) -> Iterator[_Locked]:
        _require_enabled()
        _ensure_dir(self.path.parent)
        with _exclusive_lock(self.path):
            yield _Locked(self.path, _read_state(self.path, self.namespace))

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
        """S2 -> S3: the spec commit's ``searchspecref_`` is known. Done."""
        if not isinstance(search_spec_ref, str) or not _SPEC_REF.fullmatch(
            search_spec_ref
        ):
            raise TokenStoreError(TOKEN_STATE_INVALID)
        with self.locked() as tx:
            entry = _find_by_ref(tx.state, ref)
            if entry["state"] == S3_SPEC_COMMITTED and entry["spec"] == search_spec_ref:
                return
            if entry["state"] != S2_LABEL_ACKED:
                raise TokenStoreError(TOKEN_STATE_INVALID)
            entry["state"] = S3_SPEC_COMMITTED
            entry["spec"] = search_spec_ref
            tx.write()


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

#: ``transport(body) -> (http_status, parsed_json_or_None)``; may raise on a network failure.
TokenRefsTransport = Callable[[dict[str, Any]], "tuple[int, Any]"]

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
}
_ENVELOPE_KEYS = frozenset({"success", "error", "error_code", "message"})
#: Codes after which the server is known to have issued NOTHING for this operation.
_NOTHING_ISSUED = frozenset(
    {TOKEN_REFS_INVALID_REQUEST, TOKEN_REFS_REFERENCE_NOT_FOUND}
)


def backend_transport(client: Any) -> TokenRefsTransport:
    """Adapt a ``BackendIntegratedClient`` (its auth, origin gate and egress switches)."""
    return lambda body: client.issue_token_refs_sync(body)  # type: ignore[no-any-return]


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


def _issue(
    tx: _Locked,
    namespace: TokenNamespace,
    transport: TokenRefsTransport,
    op_key: str,
) -> None:
    """Send operation ``op_key`` (S0) with its journalled batch and persist S1, or refuse."""
    op = tx.state["operations"][op_key]
    body = _wire_body(namespace, op_key, op["items"])
    outcome: TokenStoreError | None = None
    try:
        status, payload = transport(body)
    except TokenStoreError as exc:
        outcome = exc
    except Exception:  # noqa: BLE001 - a transport error can quote a URL or header
        outcome = TokenStoreError(TOKEN_REFS_TRANSPORT_FAILED)
    if outcome is not None:
        raise outcome from None
    if status != 200:
        code = _refusal_code(status, payload)
        if code in _NOTHING_ISSUED:
            _abandon(
                tx, op_key
            )  # the server rolled back; the intent would only fail forever
        raise TokenStoreError(code)
    refs = (
        payload.get("refs")
        if isinstance(payload, dict) and set(payload) == {"refs"}
        else None
    )
    if (
        not isinstance(refs, list)
        or len(refs) != len(op["items"])
        or len(set(refs)) != len(refs)
        or any(
            not isinstance(ref, str) or not _REF[item[0]].fullmatch(ref)
            for item, ref in zip(op["items"], refs, strict=True)
        )
        or any(ref in _all_refs(tx.state) for ref in refs)
    ):
        raise TokenStoreError(TOKEN_REFS_RESPONSE_INVALID)
    op["refs"] = list(refs)
    for entry in tx.state["tokens"].values():
        if entry["op"] == op_key:
            entry["ref"] = refs[entry["ordinal"]]
            entry["state"] = S1_PENDING_LABEL
    tx.write()


def _all_refs(state: dict[str, Any]) -> set[str]:
    return {ref for op in state["operations"].values() for ref in (op["refs"] or [])}


def _abandon(tx: _Locked, op_key: str) -> None:
    del tx.state["operations"][op_key]
    for key in [k for k, e in tx.state["tokens"].items() if e["op"] == op_key]:
        del tx.state["tokens"][key]
    tx.write()


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
        if triple not in seen:
            seen.add(triple)
            out.append(triple)
    return out


@_closed_boundary
def ensure_tokens(
    namespace: TokenNamespace,
    requests: Sequence[TokenRequest],
    *,
    transport: TokenRefsTransport,
) -> dict[TokenRequest, str]:
    """Return the issued ref for each ``(kind, parent_knob_name | None, local_name)``.

    Keys already in the store are answered locally (no request). Missing keys are journalled
    (S0, durable BEFORE the call), issued in operations of at most 200 items, and persisted
    (S1). An operation left at S0 by a crash is re-sent first with the SAME operation_key and
    the SAME ordered batch; if the server reports ``operation_key_conflict`` the call is
    refused, a new key is never minted silently. A ``choice`` needs its parent knob to be in
    the store already (issue knobs in an earlier call). The namespace lock is held for the
    whole call, so concurrent callers of one namespace produce one operation and one token.
    """
    _require_enabled()
    wanted = _validate_requests(requests)
    if not callable(transport):
        raise TokenStoreError(TOKEN_REQUEST_INVALID)
    store = TokenStore(namespace)
    with store.locked() as tx:
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
