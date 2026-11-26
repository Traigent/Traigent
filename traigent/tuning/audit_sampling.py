"""Local execution of the committed sampling a TVL module declares.

``RandomAudit(rate)`` serializes as ``evidence.sampling = {mode:
"hash_bernoulli_committed_v1", rate}``.  This module executes it on the
customer's machine.  The keep-or-drop choice is a keyed hash of
(window id, unit id): it is fixed by a key whose digest is published before the
window starts, cannot depend on any unit's outcome, and can be re-checked after
the key is revealed.  Nothing here touches the network; unit ids and records
stay on this machine.

Selection probability is ``ceil(rate * 2**64) / 2**64`` (absolute error below
``2**-64`` from ``rate``).  Use a fresh key per window.  Unit ids must be assigned
before an outcome is known and must not be re-drawn: re-drawing an id until it is
kept defeats the commitment.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
import stat
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any

from traigent._fsutil import open_verified
from traigent.tuning.errors import E_INVALID_TUNING, TuningError
from traigent.tuning.policy import _rate

SAMPLING_MODE = "hash_bernoulli_committed_v1"
_MIN_KEY_BYTES = 32


@dataclass(frozen=True)
class SamplingCommitment:
    """What is published before the window starts."""

    mode: str
    window_id: str
    rate: str
    key_digest: str


def _open_rate(value: Any, field: str) -> Decimal:
    """A rate strictly inside (0, 1), as TVL's prob_open requires."""
    rate: Decimal = _rate(value, field)
    if not Decimal(0) < rate < Decimal(1):
        raise TuningError(E_INVALID_TUNING, f"{field}: must be strictly within (0, 1)")
    return rate


def _check_id(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise TuningError(
            E_INVALID_TUNING, f"{field}: expected a non-empty string without NUL"
        )
    return value


def _hash_int(key: bytes, window_id: str, unit_id: str) -> int:
    window = window_id.encode("utf-8")
    message = len(window).to_bytes(8, "big") + window + unit_id.encode("utf-8")
    mac = hmac.new(key, message, hashlib.sha256).digest()
    return int.from_bytes(mac[:8], "big")


def _decide(rate: Decimal, key: bytes, window_id: str, unit_id: str) -> bool:
    numerator, denominator = rate.as_integer_ratio()
    return _hash_int(key, window_id, unit_id) * denominator < numerator * 2**64


class CommittedSampler:
    """Keep a unit iff HMAC(key, window, unit) falls below ``rate`` (exact, no floats).

    Selection probability is ``ceil(rate * 2**64) / 2**64`` (absolute error below
    ``2**-64``).  ``rate`` must be strictly within (0, 1).  Use a fresh key per
    window; unit ids must be assigned before the outcome is known and must not be
    re-drawn.  A caller-supplied key must be ``bytes`` of at least 32 bytes.
    """

    def __init__(self, rate: Any, key: bytes | None = None, *, window_id: str) -> None:
        self._rate = _open_rate(rate, "CommittedSampler.rate")
        if key is None:
            key = secrets.token_bytes(_MIN_KEY_BYTES)
        elif not isinstance(key, bytes) or len(key) < _MIN_KEY_BYTES:
            raise TuningError(
                E_INVALID_TUNING,
                f"key must be bytes of at least {_MIN_KEY_BYTES} bytes",
            )
        self._key = key
        self._window_id = _check_id(window_id, "window_id")

    def commitment(self) -> SamplingCommitment:
        return SamplingCommitment(
            mode=SAMPLING_MODE,
            window_id=self._window_id,
            rate=format(self._rate, "f"),
            key_digest=hashlib.sha256(self._key).hexdigest(),
        )

    def keep(self, unit_id: str) -> bool:
        _check_id(unit_id, "unit_id")
        return _decide(self._rate, self._key, self._window_id, unit_id)

    def reveal(self) -> bytes:
        return self._key


def _unique_ids(ids: Iterable[Any]) -> set[str] | None:
    seen: set[str] = set()
    for unit in ids:
        if not isinstance(unit, str) or not unit or "\x00" in unit or unit in seen:
            return None
        seen.add(unit)
    return seen


def verify(
    commitment: SamplingCommitment,
    key: bytes,
    eligible_unit_ids: Iterable[str],
    kept_unit_ids: Iterable[str],
) -> bool:
    """True iff the kept set is exactly what the committed sampler selects.

    Checks the mode, that SHA-256 of ``key`` equals the commitment, that ids are
    unique ``str`` values, that kept is a subset of eligible, and that kept equals
    ``{u in eligible : decide(u)}``.  The caller must record the commitment
    somewhere it cannot later change before the window opens, and must keep the
    COMPLETE list of eligible unit ids (not only the kept ones).  The key stays
    customer-side until the window closes.
    """
    if commitment.mode != SAMPLING_MODE or not isinstance(key, bytes):
        return False
    if not hmac.compare_digest(hashlib.sha256(key).hexdigest(), commitment.key_digest):
        return False
    try:
        rate = _open_rate(commitment.rate, "commitment.rate")
        _check_id(commitment.window_id, "commitment.window_id")
    except TuningError:
        return False
    eligible = _unique_ids(eligible_unit_ids)
    kept = _unique_ids(kept_unit_ids)
    if eligible is None or kept is None or not kept <= eligible:
        return False
    return kept == {u for u in eligible if _decide(rate, key, commitment.window_id, u)}


def sampler_from_monitor(
    monitor: Mapping[str, Any],
    *,
    window_id: str,
    key: bytes | None = None,
) -> CommittedSampler | None:
    """Build the sampler a monitor declares; ``None`` when it declares none (census)."""
    evidence = monitor.get("evidence") or {}
    sampling = evidence.get("sampling")
    if sampling is None:
        return None
    if not isinstance(sampling, Mapping) or sampling.get("mode") != SAMPLING_MODE:
        raise TuningError(
            E_INVALID_TUNING, f"evidence.sampling.mode must be {SAMPLING_MODE}"
        )
    rate = sampling.get("rate")
    if isinstance(rate, float):
        # TVL carries JSON numbers; go through repr so 0.25 stays exact.
        rate = Decimal(repr(rate))
    return CommittedSampler(rate, key, window_id=window_id)


_QUEUE_ERR = "audit queue path is not safe"
_POSIX = os.name == "posix"


def _queue_error() -> TuningError:
    return TuningError(E_INVALID_TUNING, _QUEUE_ERR)


class LocalAuditQueue:
    """Append-only per-window JSON-lines queue on local disk (0700 dir, 0600 files)."""

    def __init__(self, directory: str | os.PathLike[str]) -> None:
        self._dir = Path(directory)

    def _path(self, window_id: str) -> Path:
        if (
            not window_id
            or "/" in window_id
            or "\\" in window_id
            or "\x00" in window_id
            or window_id.startswith(".")
        ):
            raise TuningError(E_INVALID_TUNING, "window_id is not a safe file name")
        return self._dir / f"{window_id}.jsonl"

    def _check_dir(self) -> None:
        try:
            self._dir.mkdir(mode=0o700, parents=True, exist_ok=True)
            info = self._dir.lstat()
        except OSError:
            raise _queue_error() from None
        if not stat.S_ISDIR(info.st_mode):  # a symlink or file is not a directory
            raise _queue_error()
        if _POSIX and (info.st_uid != os.getuid() or info.st_mode & 0o077):
            raise _queue_error()

    def _open(self, path: Path, flags: int) -> int:
        try:
            fd = open_verified(path, flags, 0o600)
        except FileNotFoundError:
            raise
        except OSError:
            raise _queue_error() from None
        if _POSIX and os.fstat(fd).st_mode & 0o077:
            os.close(fd)
            raise _queue_error()
        return int(fd)

    def append(
        self, unit_id: str, record: Mapping[str, Any], *, window_id: str
    ) -> None:
        path = self._path(window_id)
        self._check_dir()
        data = (
            json.dumps({"unit_id": unit_id, "record": dict(record)}, sort_keys=True)
            + "\n"
        ).encode("utf-8")
        fd = self._open(path, os.O_RDWR | os.O_APPEND | os.O_CREAT)
        try:
            size = os.fstat(fd).st_size
            if size and os.pread(fd, 1, size - 1) != b"\n":
                data = b"\n" + data  # repair the framing of a torn final line
            view = memoryview(data)
            while view:
                view = view[os.write(fd, view) :]
        finally:
            os.close(fd)

    def list(self, window_id: str) -> list[dict[str, Any]]:
        path = self._path(window_id)
        try:
            fd = self._open(path, os.O_RDONLY)
        except FileNotFoundError:
            return []
        with os.fdopen(fd, "rb") as handle:
            raw = handle.read()
        pieces = raw.split(b"\n")
        pieces.pop()  # after the last newline: empty, or a torn line to ignore
        out: list[dict[str, Any]] = []
        for piece in pieces:
            if not piece.strip():
                continue
            try:
                out.append(json.loads(piece))
            except ValueError:
                # a torn line that a later append sealed with a newline
                continue
        return out
