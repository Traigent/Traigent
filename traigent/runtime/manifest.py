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
import threading
import time
from collections.abc import Callable, Iterable, Iterator, Mapping
from contextlib import contextmanager, suppress
from datetime import datetime, timedelta, UTC
from pathlib import Path
from typing import Any

from traigent.runtime.model import (
    E_EPISODE_FIX_RUNNING,
    E_HOLD,
    E_LOCK_TIMEOUT,
    E_LOCK_UNAVAILABLE,
    E_STATE,
    E_STATE_OWNER,
    E_STATE_VERSION,
    E_WATCH_REENTRY,
    RuntimeRequirementError,
)

try:  # POSIX only; elsewhere every locked section fails closed (never runs unlocked)
    import fcntl
except ImportError:  # pragma: no cover
    fcntl = None  # type: ignore[assignment]

STATE_DIRNAME = ".traigent"
SDK_NAME = "python"
#: Version 2: occurrence numbers start at 0, canonical plain-number digests, heartbeat
#: leases.  A state directory written by an earlier version is refused, never converted.
STATE_VERSION = 2
STATE_MARKER = "state-owner.json"
#: The compiler's contract (tvl-spec accepts only /1); local state versioning is on the marker.
TOKEN_MAP_FORMAT = "tvl-token-map/1"
MANIFEST_FORMAT = "traigent-runtime-manifest/2"
#: Longest wait for a ledger lock (same default as the JS SDK).
LOCK_TIMEOUT_S = 60.0
APPROVAL_TTL = timedelta(hours=24)
#: A reserved occurrence with no outcome (no run id, not failed) older than this is
#: treated as failed when read (INV-33), so a crashed controller cannot block forever.
OCCURRENCE_LEASE = timedelta(hours=24)


STATE_DIR_ENV = "TRAIGENT_RUNTIME_STATE_DIR"


def default_state_dir() -> Path:
    """``$TRAIGENT_RUNTIME_STATE_DIR`` if set, else ``./.traigent/python``.

    One SDK per state directory: the default includes the SDK name.
    """
    override = os.environ.get(STATE_DIR_ENV)
    return Path(override) if override else Path.cwd() / STATE_DIRNAME / SDK_NAME


def _has_state(state_dir: Path) -> bool:
    try:
        return any(state_dir.iterdir())
    except OSError:
        return False


def _write_marker(state_dir: Path, marker: Path) -> bool:
    """Publish the marker atomically (temp file hard-linked into place); False if it exists."""
    state_dir.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(
        dir=str(state_dir), prefix=STATE_MARKER + ".", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(json.dumps({"sdk": SDK_NAME, "state_version": STATE_VERSION}))
        try:
            os.link(tmp, str(marker))
        except FileExistsError:
            return False
        return True
    finally:
        with suppress(OSError):
            os.unlink(tmp)


def check_state_dir(state_dir: Path, *, create: bool = False) -> None:
    """Refuse a directory owned by another SDK or written by an earlier version.

    The marker is published atomically on first use when ``create``.  A
    directory holding state but no marker predates versioning and is refused.
    """
    marker = state_dir / STATE_MARKER
    text: str | None = None
    for _attempt in range(2):
        try:
            text = marker.read_text(encoding="utf-8")
            break
        except FileNotFoundError:
            if _has_state(state_dir):
                continue  # a concurrent first use may have just published it: re-read
            if not create:
                return
            if _write_marker(state_dir, marker):
                return
        except OSError:
            raise RuntimeRequirementError(E_STATE, STATE_MARKER) from None
    if text is None:
        raise RuntimeRequirementError(E_STATE_VERSION, "state_dir")
    try:
        doc = json.loads(text)
    except ValueError:
        raise RuntimeRequirementError(E_STATE, STATE_MARKER) from None
    if not isinstance(doc, dict):
        raise RuntimeRequirementError(E_STATE, STATE_MARKER)
    if doc.get("sdk") != SDK_NAME:
        raise RuntimeRequirementError(E_STATE_OWNER, "state_dir")
    if doc.get("state_version") != STATE_VERSION:
        raise RuntimeRequirementError(E_STATE_VERSION, "state_dir")


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
    check_state_dir(state_dir, create=persist)
    path = state_dir / "tvl-token-map.json"
    if not persist:
        return _mint_tokens(path, wanted, persist=False)
    # Read-mint-replace is one critical section: atomic replace alone loses updates.
    with _ledger_lock(state_dir, path, reentry_code=E_STATE):
        return _mint_tokens(path, wanted, persist=True)


def _mint_tokens(
    path: Path, wanted: Iterable[tuple[str, str]], *, persist: bool
) -> dict[str, Any]:
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
    check_state_dir(state_dir)
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
    check_state_dir(state_dir)
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
    with _ledger_lock(state_dir, manifest_path(state_dir, agent_id)):
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
    check_state_dir(state_dir)
    doc = _read_json(_occ_path(state_dir, agent_id), {})
    record = doc.get(key) if isinstance(doc, dict) else None
    return effective_record(
        record if isinstance(record, dict) else None, now=now, lease=lease
    )


_HELD = threading.local()


@contextmanager
def _ledger_lock(
    state_dir: Path,
    path: Path,
    *,
    reentry_code: str | None = None,
    timeout: float | None = None,
) -> Iterator[None]:
    """Exclusive, process-safe lock around a read-modify-write of one ledger file.

    Waiting is bounded (``E_STATE`` ``lock`` on timeout).  Re-entering a lock
    this thread already holds would self-deadlock; with ``reentry_code`` it
    raises that code at once instead.
    """
    if fcntl is None:
        # No exclusive lock on this platform: refuse before entering the protected section.
        raise RuntimeRequirementError(E_LOCK_UNAVAILABLE, path.name)
    check_state_dir(state_dir, create=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    held: set[str] = _HELD.__dict__.setdefault("paths", set())
    ident = str(path)
    if ident in held:
        raise RuntimeRequirementError(reentry_code or E_STATE, path.name)
    with open(str(path) + ".lock", "a", encoding="utf-8") as handle:
        deadline = time.monotonic() + (LOCK_TIMEOUT_S if timeout is None else timeout)
        while True:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError:
                if time.monotonic() >= deadline:
                    raise RuntimeRequirementError(
                        E_LOCK_TIMEOUT, str(path) + ".lock"
                    ) from None
                time.sleep(0.01)
        held.add(ident)
        try:
            yield
        finally:
            held.discard(ident)
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


STATUS_STARTED = "started"
STATUS_FAILED = "failed"


def _lease_expired(record: Mapping[str, Any], now: datetime, lease: timedelta) -> bool:
    """A started occurrence with no run id and no outcome whose last heartbeat is older than ``lease``.

    A running controller renews ``heartbeat_at``; the lease is measured from it
    (or from ``started_at`` when no heartbeat was ever recorded).
    """
    if (
        record.get("status") != STATUS_STARTED
        or record.get("run_id")
        or record.get("finished")
    ):
        return False
    try:
        started = datetime.fromisoformat(
            str(record.get("heartbeat_at") or record["started_at"])
        )
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
    clock: Callable[[], datetime] | None = None,
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

    ``clock``, when given, is read again once the lock is held: the lease of
    the new reservation starts at that moment, never at a time captured
    before a long lock wait.
    """
    path = _occ_path(state_dir, agent_id)
    with _ledger_lock(state_dir, path):
        if clock is not None:
            now = clock()
        doc = _read_json(path, {})
        if not isinstance(doc, dict):
            raise RuntimeRequirementError(E_STATE, "occurrences.json")
        earlier = effective_record(doc.get(key), now=now, lease=lease)
        if isinstance(earlier, dict) and earlier.get("status") != STATUS_FAILED:
            return False
        failed = failed_attempts(earlier)
        if attempt is not None and failed != attempt - 1:
            raise RuntimeRequirementError(E_HOLD, "approval")
        breach_id = _breach_of_key(key)
        for other_key, other in doc.items():
            # Episode-wide exclusion: any rule's non-final, live record blocks.
            if other_key == key or _breach_of_key(other_key) != breach_id:
                continue
            live = effective_record(
                other if isinstance(other, dict) else None, now=now, lease=lease
            )
            if (
                isinstance(live, dict)
                and live.get("status") == STATUS_STARTED
                and not live.get("finished")
            ):
                raise RuntimeRequirementError(E_EPISODE_FIX_RUNNING, "breach")
        started_at = (now or utcnow()).isoformat()
        # A caller-supplied start time (low-level, no ``clock``) is honoured;
        # ``respond`` supplies none, so its lease starts at the stamp above.
        started_at = str(dict(run_info).get("started_at", started_at))
        doc[key] = {
            **dict(run_info),
            "started_at": started_at,
            "heartbeat_at": started_at,
            "status": STATUS_STARTED,
            "attempts": failed + 1,
        }
        _write_json(path, doc)
    return True


def _breach_of_key(key: str) -> str:
    """The breach id of an occurrence key (``breach|rule|occurrence``)."""
    return key.rsplit("|", 2)[0] if key.count("|") >= 2 else key


def update_occurrence(
    state_dir: Path,
    agent_id: str,
    key: str,
    run_info: Mapping[str, Any],
    *,
    attempt: int,
    lease: timedelta | None = None,
    now: datetime | None = None,
) -> bool:
    """Merge ``run_info`` into the record only if it still belongs to ``attempt``.

    A heartbeat passes ``lease``: an expired lease is read as failed (INV-33) and a
    beat never resurrects it.  Outcome writes pass none and keep the attempt fence.

    A writer whose reservation was superseded (its lease expired and a newer
    approved attempt took over) is fenced out: nothing is written, ``False``.
    The record must also still be an active reservation (``started``, not
    finished): a restored ``failed`` record rejects late writes.
    """
    path = _occ_path(state_dir, agent_id)
    with _ledger_lock(state_dir, path):
        doc = _read_json(path, {})
        record = doc.get(key) if isinstance(doc, dict) else None
        if (
            not isinstance(record, dict)
            or record.get("attempts") != attempt
            or record.get("status") != STATUS_STARTED
            or record.get("finished")
        ):
            return False
        if lease is not None:
            resolved = effective_record(record, now=now, lease=lease)
            if resolved is None or resolved.get("status") != STATUS_STARTED:
                return False
        record.update(run_info)
        _write_json(path, doc)
    return True


def release_occurrence(
    state_dir: Path, agent_id: str, key: str, *, attempt: int
) -> bool:
    """Undo a reservation whose fix never started (only if it is still ``attempt``'s).

    The first attempt's record is deleted; a retry's record goes back to the
    ``failed`` record it replaced, so the recorded failures are kept and the
    unstarted fix consumes no retry.
    """
    path = _occ_path(state_dir, agent_id)
    with _ledger_lock(state_dir, path):
        doc = _read_json(path, {})
        record = doc.get(key) if isinstance(doc, dict) else None
        if not isinstance(record, dict) or record.get("attempts") != attempt:
            return False
        if attempt <= 1:
            del doc[key]
        else:
            doc[key] = {**record, "status": STATUS_FAILED, "attempts": attempt - 1}
            doc[key].pop("heartbeat_at", None)
        _write_json(path, doc)
    return True


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
    with _ledger_lock(state_dir, path):
        doc = _read_json(path, {})
        if not isinstance(doc, dict):
            raise RuntimeRequirementError(E_STATE, "approvals.json")
        doc[occurrence_key(breach_id, rule_id, occurrence)] = record
        _write_json(path, doc)
    return record


def read_approval(state_dir: Path, agent_id: str, key: str) -> dict[str, Any] | None:
    check_state_dir(state_dir)
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
    with _ledger_lock(
        state_dir, _watch_path(state_dir, agent_id), reentry_code=E_WATCH_REENTRY
    ):
        yield


def read_watch_state(state_dir: Path, agent_id: str) -> dict[str, Any]:
    check_state_dir(state_dir)
    doc = _read_json(_watch_path(state_dir, agent_id), {"breaches": {}})
    if not isinstance(doc, dict) or not isinstance(doc.get("breaches"), dict):
        raise RuntimeRequirementError(E_STATE, "state.json")
    return doc


def write_watch_state(state_dir: Path, agent_id: str, doc: Mapping[str, Any]) -> None:
    _write_json(_watch_path(state_dir, agent_id), dict(doc))
