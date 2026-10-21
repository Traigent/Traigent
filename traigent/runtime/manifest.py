"""Local, customer-side state: token map, manifest, occurrences, approvals, watch state.

Everything here stays on the customer's machine.  The token map holds the
only link between local names and the opaque refs that cross to the backend,
so it is written with owner-only permissions and random refs (never a hash of
the name).  All writes are atomic (``os.replace``).
"""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import tempfile
from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager, suppress
from datetime import datetime, timedelta, UTC
from pathlib import Path
from typing import Any

from traigent.runtime.model import E_HOLD, E_STATE, RuntimeRequirementError

try:  # POSIX only; elsewhere the lock degrades to a no-op
    import fcntl
except ImportError:  # pragma: no cover
    fcntl = None  # type: ignore[assignment]

STATE_DIRNAME = ".traigent"
TOKEN_MAP_FORMAT = "tvl-token-map/1"
MANIFEST_FORMAT = "traigent-runtime-manifest/1"
APPROVAL_TTL = timedelta(hours=24)
#: A reserved occurrence with no outcome (no run id, not failed) older than this is
#: treated as failed when read (INV-33), so a crashed controller cannot block forever.
OCCURRENCE_LEASE = timedelta(hours=24)


STATE_DIR_ENV = "TRAIGENT_RUNTIME_STATE_DIR"


def default_state_dir() -> Path:
    """``$TRAIGENT_RUNTIME_STATE_DIR`` if set, else ``./.traigent``."""
    override = os.environ.get(STATE_DIR_ENV)
    return Path(override) if override else Path.cwd() / STATE_DIRNAME


def agent_dir(state_dir: Path, agent_id: str) -> Path:
    """A per-agent directory whose name never depends on the id's characters."""
    return (
        state_dir
        / "runtime"
        / ("agent_" + hashlib.sha256(agent_id.encode()).hexdigest()[:16])
    )


def _write_json(path: Path, doc: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(
        dir=str(path.parent), prefix=path.name + ".", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(doc, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.chmod(tmp, 0o600)
        os.replace(tmp, path)
    except BaseException:
        with suppress(OSError):
            os.unlink(tmp)
        raise


def _read_json(path: Path, default: Any) -> Any:
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return default
    except OSError:
        raise RuntimeRequirementError(E_STATE, path.name) from None
    try:
        return json.loads(text)
    except ValueError:
        raise RuntimeRequirementError(E_STATE, path.name) from None


# --- token map -----------------------------------------------------------------------


def ensure_tokens(
    state_dir: Path, wanted: Iterable[tuple[str, str]], *, persist: bool = True
) -> dict[str, Any]:
    """Return the token map covering ``wanted``, minting random refs for new names.

    ``persist=False`` (dry runs) mints in memory and writes nothing.
    """
    path = state_dir / "tvl-token-map.json"
    doc = _read_json(path, {"format": TOKEN_MAP_FORMAT, "tokens": []})
    if (
        not isinstance(doc, dict)
        or doc.get("format") != TOKEN_MAP_FORMAT
        or not isinstance(doc.get("tokens"), list)
    ):
        raise RuntimeRequirementError(E_STATE, path.name)
    tokens: list[dict[str, str]] = doc["tokens"]
    known = {(t["namespace"], t["local"]) for t in tokens}
    used = {(t["namespace"], t["hosted"]) for t in tokens}
    changed = False
    for namespace, local in wanted:
        if (namespace, local) in known:
            continue
        while True:
            hosted = "t" + secrets.token_urlsafe(12)
            if (namespace, hosted) not in used:
                break
        tokens.append({"namespace": namespace, "local": local, "hosted": hosted})
        known.add((namespace, local))
        used.add((namespace, hosted))
        changed = True
    if changed and persist:
        _write_json(path, doc)
    return doc


def token_for(token_map: Mapping[str, Any], namespace: str, local: str) -> str:
    for entry in token_map["tokens"]:
        if entry["namespace"] == namespace and entry["local"] == local:
            return str(entry["hosted"])
    raise RuntimeRequirementError(E_STATE, "tokens")


def local_for(state_dir: Path, namespace: str, hosted: str) -> str | None:
    """Resolve an opaque ref back to a local name (``traigent requirements explain``)."""
    doc = _read_json(state_dir / "tvl-token-map.json", {"tokens": []})
    for entry in doc.get("tokens", []) if isinstance(doc, dict) else []:
        if entry.get("hosted") == hosted and (
            namespace == "*" or entry.get("namespace") == namespace
        ):
            return str(entry.get("local"))
    return None


# --- manifest ----------------------------------------------------------------------


def manifest_path(state_dir: Path, agent_id: str) -> Path:
    return agent_dir(state_dir, agent_id) / "manifest.json"


def read_manifest(state_dir: Path, agent_id: str) -> dict[str, Any]:
    doc = _read_json(
        manifest_path(state_dir, agent_id),
        {"format": MANIFEST_FORMAT, "requirements": {}},
    )
    if not isinstance(doc, dict) or not isinstance(doc.get("requirements"), dict):
        raise RuntimeRequirementError(E_STATE, "manifest.json")
    return doc


def record_apply(
    state_dir: Path,
    agent_id: str,
    *,
    name: str,
    policy_digest: str,
    hosted_digest: str,
    key: Mapping[str, Any],
    route: str,
    receipt: Mapping[str, Any],
) -> None:
    with _ledger_lock(manifest_path(state_dir, agent_id)):
        _record_apply_locked(
            state_dir, agent_id, name, policy_digest, hosted_digest, key, route, receipt
        )


def _record_apply_locked(
    state_dir: Path,
    agent_id: str,
    name: str,
    policy_digest: str,
    hosted_digest: str,
    key: Mapping[str, Any],
    route: str,
    receipt: Mapping[str, Any],
) -> None:
    doc = read_manifest(state_dir, agent_id)
    doc["requirements"][name] = {
        "full_policy_digest": policy_digest,
        "hosted_digest": hosted_digest,
        "key": dict(key),
        "route": route,
        "receipt": dict(receipt),
    }
    _write_json(manifest_path(state_dir, agent_id), doc)


# --- occurrences ---------------------------------------------------------------------


def _occ_path(state_dir: Path, agent_id: str) -> Path:
    return agent_dir(state_dir, agent_id) / "occurrences.json"


def occurrence_key(breach_id: str, rule_id: str, occurrence: int) -> str:
    return f"{breach_id}|{rule_id}|{occurrence}"


def read_occurrence(
    state_dir: Path,
    agent_id: str,
    key: str,
    *,
    now: datetime | None = None,
    lease: timedelta = OCCURRENCE_LEASE,
) -> dict[str, Any] | None:
    """The occurrence record, with an expired lease read as ``failed`` (INV-33)."""
    doc = _read_json(_occ_path(state_dir, agent_id), {})
    record = doc.get(key) if isinstance(doc, dict) else None
    return effective_record(
        record if isinstance(record, dict) else None, now=now, lease=lease
    )


@contextmanager
def _ledger_lock(path: Path) -> Iterator[None]:
    """Exclusive, process-safe lock around a read-modify-write of one ledger file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(str(path) + ".lock", "a", encoding="utf-8") as handle:
        if fcntl is not None:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            if fcntl is not None:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


STATUS_STARTED = "started"
STATUS_FAILED = "failed"


def _lease_expired(record: Mapping[str, Any], now: datetime, lease: timedelta) -> bool:
    """A started occurrence with no run id and no outcome, older than ``lease``."""
    if (
        record.get("status") != STATUS_STARTED
        or record.get("run_id")
        or record.get("finished")
    ):
        return False
    try:
        started = datetime.fromisoformat(str(record["started_at"]))
    except (KeyError, ValueError):
        return False  # unreadable start time: fail closed, keep blocking
    if started.tzinfo is None:
        started = started.replace(tzinfo=UTC)
    return now - started >= lease


def effective_record(
    record: dict[str, Any] | None,
    *,
    now: datetime | None = None,
    lease: timedelta = OCCURRENCE_LEASE,
) -> dict[str, Any] | None:
    """``record`` with an expired lease read as ``failed`` (INV-33); never persisted."""
    if record is None or not _lease_expired(record, now or utcnow(), lease):
        return record
    return {**record, "status": STATUS_FAILED}


def failed_attempts(record: Mapping[str, Any] | None) -> int:
    """Failed attempts recorded for an occurrence (``record`` already lease-resolved)."""
    if record is None:
        return 0
    attempts = int(record.get("attempts", 1))
    # A failed record's last attempt failed; an in-flight or finished one did not.
    return attempts if record.get("status") == STATUS_FAILED else attempts - 1


def reserve_occurrence(
    state_dir: Path,
    agent_id: str,
    key: str,
    run_info: Mapping[str, Any],
    *,
    attempt: int | None = None,
    now: datetime | None = None,
    lease: timedelta = OCCURRENCE_LEASE,
) -> bool:
    """Atomically record an occurrence; ``False`` when it is already recorded.

    A record whose status is ``failed`` (the optimize run raised, or its lease
    expired) is released for an explicit retry with a NEW approval.  The
    approval carries ``attempt`` = failed attempts at approval time + 1; the
    reservation succeeds only while the ledger still holds ``attempt - 1``
    failures (compare-and-set under the ledger lock), so an approval issued
    before a failure can never authorize the retry after it.  A generation
    mismatch raises ``hold`` at ``approval``.  ``attempt=None`` (a low-level
    caller with no approval) skips the generation check; ``respond`` always
    passes it.
    """
    path = _occ_path(state_dir, agent_id)
    with _ledger_lock(path):
        doc = _read_json(path, {})
        if not isinstance(doc, dict):
            raise RuntimeRequirementError(E_STATE, "occurrences.json")
        earlier = effective_record(doc.get(key), now=now, lease=lease)
        if isinstance(earlier, dict) and earlier.get("status") != STATUS_FAILED:
            return False
        failed = failed_attempts(earlier)
        if attempt is not None and failed != attempt - 1:
            raise RuntimeRequirementError(E_HOLD, "approval")
        doc[key] = {
            "started_at": (now or utcnow()).isoformat(),
            **dict(run_info),
            "status": STATUS_STARTED,
            "attempts": failed + 1,
        }
        _write_json(path, doc)
    return True


def update_occurrence(
    state_dir: Path, agent_id: str, key: str, run_info: Mapping[str, Any]
) -> None:
    path = _occ_path(state_dir, agent_id)
    with _ledger_lock(path):
        doc = _read_json(path, {})
        if isinstance(doc, dict) and key in doc:
            doc[key].update(run_info)
            _write_json(path, doc)


# --- approvals ------------------------------------------------------------------------


def _approval_path(state_dir: Path, agent_id: str) -> Path:
    return agent_dir(state_dir, agent_id) / "approvals.json"


def utcnow() -> datetime:
    return datetime.now(UTC)


def write_approval(
    state_dir: Path,
    agent_id: str,
    *,
    policy_digest: str,
    rule_id: str,
    breach_id: str,
    occurrence: int,
    cost_ceiling_usd: float,
    attempt: int = 1,
    now: datetime | None = None,
) -> dict[str, Any]:
    issued = now or utcnow()
    record = {
        "full_policy_digest": policy_digest,
        "rule_id": rule_id,
        "breach_id": breach_id,
        "occurrence": occurrence,
        "cost_ceiling_usd": cost_ceiling_usd,
        "attempt": attempt,
        "issued_at": issued.isoformat(),
        "expires_at": (issued + APPROVAL_TTL).isoformat(),
    }
    path = _approval_path(state_dir, agent_id)
    with _ledger_lock(path):
        doc = _read_json(path, {})
        if not isinstance(doc, dict):
            raise RuntimeRequirementError(E_STATE, "approvals.json")
        doc[occurrence_key(breach_id, rule_id, occurrence)] = record
        _write_json(path, doc)
    return record


def read_approval(state_dir: Path, agent_id: str, key: str) -> dict[str, Any] | None:
    doc = _read_json(_approval_path(state_dir, agent_id), {})
    record = doc.get(key) if isinstance(doc, dict) else None
    return record if isinstance(record, dict) else None


def _same(left: Any, right: Any) -> bool:
    if isinstance(left, bool) or isinstance(right, bool):
        return left is right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return left == right
    return isinstance(left, str) and isinstance(right, str) and left == right


def approval_matches(record: Mapping[str, Any], **expected: Any) -> bool:
    """Every named bound field of ``record`` equals the value this start will use."""
    return all(
        name in record and _same(record[name], value)
        for name, value in expected.items()
    )


def approval_expired(record: Mapping[str, Any], now: datetime | None = None) -> bool:
    try:
        expires = datetime.fromisoformat(str(record["expires_at"]))
    except (KeyError, ValueError):
        return True
    return (now or utcnow()) >= expires


# --- watch state --------------------------------------------------------------------------


def _watch_path(state_dir: Path, agent_id: str) -> Path:
    return agent_dir(state_dir, agent_id) / "state.json"


@contextmanager
def watch_lock(state_dir: Path, agent_id: str) -> Iterator[None]:
    """Exclusive lock across one watch poll's read, diff, emit and write."""
    with _ledger_lock(_watch_path(state_dir, agent_id)):
        yield


def read_watch_state(state_dir: Path, agent_id: str) -> dict[str, Any]:
    doc = _read_json(_watch_path(state_dir, agent_id), {"breaches": {}})
    if not isinstance(doc, dict) or not isinstance(doc.get("breaches"), dict):
        raise RuntimeRequirementError(E_STATE, "state.json")
    return doc


def write_watch_state(state_dir: Path, agent_id: str, doc: Mapping[str, Any]) -> None:
    _write_json(_watch_path(state_dir, agent_id), dict(doc))
