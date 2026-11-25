"""Append-only JSON-lines log of what passed through the egress relay.

One JSON object per line, ``schema_version`` 1.

Request line::

    seq, ts (UTC ISO), kind="request", method, route, route_source, bytes, sha256

``route_source`` is ``"allowlist"``: the route keeps only words from a fixed list (logs
written by the first version say ``"heuristic"``; readers do not use the field, so both
stay readable and ``schema_version`` is still 1).

Response line (same ``seq``)::

    seq, ts, kind="response", status, bytes   [, outcome]

``route`` is a shape, never a raw path (see :func:`route_shape`). Query strings,
header names or values, body content and JSON key names are never logged.
``outcome`` appears only when the relay itself ended the exchange:
``upstream_unreachable``, ``upstream_error``, ``upstream_unsupported`` (the upstream
answered 101 Switching Protocols, which the relay does not support) or
``relay_stopped`` (the relay was stopped while the exchange was still running).
A request the relay refuses before forwarding (malformed, unsupported
Transfer-Encoding) gets no log line: nothing was sent upstream.

The file is created with mode 0600. Writes are single ``write`` calls under a lock,
followed by ``fsync``, so lines are never interleaved and ``seq`` is strictly
increasing. After the first failed write the log refuses every further request
line (``failed`` is True), because it can no longer be complete. One relay at a time holds an advisory lock on the file.
"""

# Traceability: CONC-Layer-Core CONC-Quality-Security FUNC-SECURITY

from __future__ import annotations

import json
import os
import re
import threading
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from traigent._fsutil import open_verified
from traigent.egress.route_words import ROUTE_WORDS

try:  # POSIX
    import fcntl
except ImportError:  # pragma: no cover - Windows
    fcntl = None  # type: ignore[assignment]

SCHEMA_VERSION = 1
ROUTE_SOURCE_ALLOWLIST = "allowlist"
_LOG_UNOPENABLE_MESSAGE = "the egress relay log cannot be opened"
DEFAULT_LOG_PATH = Path(".traigent") / "egress" / "relay.jsonl"
METHODS = frozenset({"GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"})
OTHER_METHOD = "OTHER"
OUTCOMES = frozenset(
    {"upstream_unreachable", "upstream_error", "relay_stopped", "upstream_unsupported"}
)

_VERSION_SEGMENT = re.compile(r"^v\d+((alpha|beta)\d*)?$")
_MAX_ROUTE_LEN = 512


MSG_TORN_LOG = (
    "the egress relay log ends with an incomplete line; move the file aside "
    "or repair it before starting the relay"
)


class RelayLogError(RuntimeError):
    """The log line could not be written. Carries no path or content."""


class RelayLogLockedError(RelayLogError):
    """Another relay already holds this log."""


def route_shape(path: str) -> str:
    """Return the route shape of a request target.

    The query string and fragment are dropped. A path segment is kept only when it is
    in the fixed allowlist :data:`traigent.egress.route_words.ROUTE_WORDS` or is a
    version marker such as ``v1`` or ``v1beta``; every other segment becomes ``{}``,
    including a lowercase word the customer chose (an id or a name).
    """
    target = path.split("?", 1)[0].split("#", 1)[0]
    shaped: list[str] = []
    for segment in target.split("/"):
        if (
            segment == ""
            or _VERSION_SEGMENT.fullmatch(segment)
            or segment in ROUTE_WORDS
        ):
            shaped.append(segment)
        else:
            shaped.append("{}")
    route = "/".join(shaped) or "/"
    if len(route) > _MAX_ROUTE_LEN:
        route = route[:_MAX_ROUTE_LEN]
    return route


def _now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def operator_log_path(path: str | os.PathLike[str]) -> Path:
    """Resolve the log path the operator chose, refusing anything that is not a plain file.

    The path comes from the operator (``--log`` on the egress CLI or the caller of
    :class:`RelayLog`), never from a request or a server response, so its *location* is
    trusted: any directory the operator can write is acceptable.  What is refused is a
    path that is not what it looks like: a NUL byte, a symlink at the file itself (it
    could redirect an append-only audit log elsewhere), or an existing entry that is not
    a regular file.  Errors carry no path.
    """
    if "\0" in os.fspath(path):
        raise RelayLogError(_LOG_UNOPENABLE_MESSAGE)
    candidate = Path(path)
    try:
        if candidate.is_symlink():
            raise RelayLogError(_LOG_UNOPENABLE_MESSAGE)
        # Only the directory is resolved; the final component is never followed.
        resolved = candidate.parent.resolve() / candidate.name
        if resolved.exists() and not resolved.is_file():
            raise RelayLogError(_LOG_UNOPENABLE_MESSAGE)
    except (OSError, RuntimeError, ValueError):
        raise RelayLogError(_LOG_UNOPENABLE_MESSAGE) from None
    return resolved


def _open_log_fd(path: Path, flags: int, mode: int = 0o600) -> int:
    """Open ``path`` as given, without following a symlink; verify the descriptor.

    ``O_NOFOLLOW`` is used where available, and the opened descriptor is always
    checked to be a regular file that is the one the path names now, so a swap after
    ``operator_log_path`` ran is refused on every platform.  The checks that follow
    (lock, ``fchmod``) act on this verified descriptor only.
    """
    return open_verified(path, flags, mode)


class RelayLog:
    """Writer for the relay log. Thread-safe."""

    def __init__(self, path: str | os.PathLike[str]) -> None:
        self.path = operator_log_path(path)
        self._lock = threading.Lock()
        self._failed = False
        self._closed = False
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            self._fd = _open_log_fd(
                self.path, os.O_RDWR | os.O_APPEND | os.O_CREAT, 0o600
            )
        except OSError:
            raise RelayLogError(_LOG_UNOPENABLE_MESSAGE) from None
        try:
            if fcntl is not None:
                try:
                    fcntl.flock(self._fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except OSError:
                    raise RelayLogLockedError(
                        "another egress relay is already writing this log"
                    ) from None
            if hasattr(os, "fchmod"):  # an existing file may be looser than 0600
                try:
                    os.fchmod(self._fd, 0o600)
                except OSError:
                    raise RelayLogError(_LOG_UNOPENABLE_MESSAGE) from None
            self._require_clean_tail()
            self._seq = max((e["seq"] for e in iter_lines(self.path)), default=0)
        except BaseException:
            os.close(self._fd)
            raise

    def _require_clean_tail(self) -> None:
        """Refuse a log whose last line is torn (no final newline).

        Appending to it would glue the next record onto the torn bytes and produce an
        invalid line for a request that was forwarded.
        """
        try:
            size = os.fstat(self._fd).st_size
            if size == 0:
                return
            os.lseek(self._fd, size - 1, os.SEEK_SET)  # appends still go to the end
            last = os.read(self._fd, 1)
        except OSError:
            raise RelayLogError(_LOG_UNOPENABLE_MESSAGE) from None
        if last != b"\n":
            raise RelayLogError(MSG_TORN_LOG)

    @property
    def failed(self) -> bool:
        """True once any write failed: the log is no longer complete."""
        return self._failed

    def _append(self, entry: dict[str, Any]) -> None:
        line = (json.dumps(entry, separators=(",", ":")) + "\n").encode("utf-8")
        if self._closed:  # never write to a closed (and possibly reused) descriptor
            raise RelayLogError("the egress relay log is closed")
        try:
            written = os.write(self._fd, line)
            if written != len(line):  # pragma: no cover - O_APPEND regular file
                raise OSError("short write")
            os.fsync(self._fd)
        except (OSError, ValueError):
            self._failed = True
            raise RelayLogError("the egress relay log cannot be written") from None

    def log_request(self, method: str, target: str, size: int, sha256: str) -> int:
        """Write the request line and return its ``seq``. Raises RelayLogError."""
        with self._lock:
            if self._failed:  # an earlier write failed: the log is incomplete
                raise RelayLogError("the egress relay log cannot be written")
            seq = self._seq + 1
            self._append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "seq": seq,
                    "ts": _now(),
                    "kind": "request",
                    "method": method if method in METHODS else OTHER_METHOD,
                    "route": route_shape(target),
                    "route_source": ROUTE_SOURCE_ALLOWLIST,
                    "bytes": size,
                    "sha256": sha256,
                }
            )
            self._seq = seq
            return seq

    def log_response(
        self, seq: int, status: int, size: int, outcome: str | None = None
    ) -> None:
        """Write the response line for ``seq``. Raises RelayLogError."""
        entry: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "seq": seq,
            "ts": _now(),
            "kind": "response",
            "status": status,
            "bytes": size,
        }
        if outcome in OUTCOMES:
            entry["outcome"] = outcome
        with self._lock:
            self._append(entry)

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            try:
                os.close(self._fd)
            except OSError:
                pass


def iter_lines(path: str | os.PathLike[str]) -> Iterator[dict[str, Any]]:
    """Yield the well-formed entries of a log file; torn or foreign lines are skipped."""
    try:
        handle = os.fdopen(
            _open_log_fd(operator_log_path(path), os.O_RDONLY),
            encoding="utf-8",
        )
    except (OSError, RelayLogError):
        return
    with handle:
        for raw in handle:
            try:
                entry = json.loads(raw)
            except ValueError:
                continue
            if (
                isinstance(entry, dict)
                and isinstance(entry.get("seq"), int)
                and entry.get("schema_version") == SCHEMA_VERSION
                and entry.get("kind") in ("request", "response")
            ):
                yield entry
