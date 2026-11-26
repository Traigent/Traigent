"""Local execution of the committed sampling a TVL module declares.

``RandomAudit(rate)`` serializes as ``evidence.sampling = {mode:
"hash_bernoulli_committed_v1", rate}``.  This module executes it on the
customer's machine.  The keep-or-drop choice is a keyed hash of
(window id, unit id): it is fixed by a key whose digest is published before the
window starts, cannot depend on any unit's outcome, and can be re-checked after
the key is revealed.  Nothing here touches the network; unit ids and records
stay on this machine.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any

from traigent.tuning.errors import E_INVALID_TUNING, TuningError
from traigent.tuning.policy import _rate, decimal_to_str

SAMPLING_MODE = "hash_bernoulli_committed_v1"


@dataclass(frozen=True)
class SamplingCommitment:
    """What is published before the window starts."""

    mode: str
    window_id: str
    rate: str
    key_digest: str


def _hash_int(key: bytes, window_id: str, unit_id: str) -> int:
    mac = hmac.new(key, f"{window_id}\x00{unit_id}".encode(), hashlib.sha256).digest()
    return int.from_bytes(mac[:8], "big")


def _decide(rate: Decimal, key: bytes, window_id: str, unit_id: str) -> bool:
    numerator, denominator = rate.as_integer_ratio()
    return _hash_int(key, window_id, unit_id) * denominator < numerator * 2**64


class CommittedSampler:
    """Keep a unit iff HMAC(key, window, unit) falls below ``rate`` (exact, no floats)."""

    def __init__(
        self, rate: Any, key: bytes | None = None, window_id: str = ""
    ) -> None:
        self._rate = _rate(rate, "CommittedSampler.rate")
        self._key = secrets.token_bytes(32) if key is None else bytes(key)
        self._window_id = window_id

    def commitment(self) -> SamplingCommitment:
        return SamplingCommitment(
            mode=SAMPLING_MODE,
            window_id=self._window_id,
            rate=decimal_to_str(self._rate),
            key_digest=hashlib.sha256(self._key).hexdigest(),
        )

    def keep(self, unit_id: str) -> bool:
        return _decide(self._rate, self._key, self._window_id, unit_id)

    def reveal(self) -> bytes:
        return self._key


def verify(
    commitment: SamplingCommitment, key: bytes, decisions: Mapping[str, bool]
) -> bool:
    """True iff ``key`` matches the commitment and every decision recomputes."""
    if not hmac.compare_digest(hashlib.sha256(key).hexdigest(), commitment.key_digest):
        return False
    rate = _rate(commitment.rate, "commitment.rate")
    return all(
        _decide(rate, key, commitment.window_id, unit) is bool(kept)
        for unit, kept in decisions.items()
    )


def sampler_from_monitor(
    monitor: Mapping[str, Any],
    window_id: str = "",
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
    return CommittedSampler(rate, key, window_id)


class LocalAuditQueue:
    """Append-only per-window JSON-lines queue on local disk (0700 dir, 0600 files)."""

    def __init__(self, directory: str | os.PathLike[str]) -> None:
        self._dir = Path(directory)

    def _path(self, window_id: str) -> Path:
        if (
            not window_id
            or "/" in window_id
            or "\\" in window_id
            or window_id.startswith(".")
        ):
            raise TuningError(E_INVALID_TUNING, "window_id is not a safe file name")
        return self._dir / f"{window_id}.jsonl"

    def append(
        self, unit_id: str, record: Mapping[str, Any], *, window_id: str
    ) -> None:
        path = self._path(window_id)
        self._dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        line = json.dumps({"unit_id": unit_id, "record": dict(record)}, sort_keys=True)
        fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
        with os.fdopen(fd, "a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    def list(self, window_id: str) -> list[dict[str, Any]]:
        path = self._path(window_id)
        if not path.exists():
            return []
        with path.open(encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]
