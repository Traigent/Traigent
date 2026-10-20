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
from collections.abc import Iterable, Mapping
from contextlib import suppress
from datetime import datetime, timedelta, UTC
from pathlib import Path
from typing import Any

from traigent.runtime.model import E_STATE, RuntimeRequirementError

STATE_DIRNAME = ".traigent"
TOKEN_MAP_FORMAT = "tvl-token-map/1"
MANIFEST_FORMAT = "traigent-runtime-manifest/1"
APPROVAL_TTL = timedelta(hours=24)


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


def read_occurrence(state_dir: Path, agent_id: str, key: str) -> dict[str, Any] | None:
    doc = _read_json(_occ_path(state_dir, agent_id), {})
    record = doc.get(key) if isinstance(doc, dict) else None
    return record if isinstance(record, dict) else None


def reserve_occurrence(
    state_dir: Path, agent_id: str, key: str, run_info: Mapping[str, Any]
) -> bool:
    """Record an occurrence durably; ``False`` when it was already recorded."""
    path = _occ_path(state_dir, agent_id)
    doc = _read_json(path, {})
    if not isinstance(doc, dict):
        raise RuntimeRequirementError(E_STATE, "occurrences.json")
    if key in doc:
        return False
    doc[key] = dict(run_info)
    _write_json(path, doc)
    return True


def update_occurrence(
    state_dir: Path, agent_id: str, key: str, run_info: Mapping[str, Any]
) -> None:
    path = _occ_path(state_dir, agent_id)
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
    now: datetime | None = None,
) -> dict[str, Any]:
    issued = now or utcnow()
    record = {
        "full_policy_digest": policy_digest,
        "rule_id": rule_id,
        "breach_id": breach_id,
        "occurrence": occurrence,
        "cost_ceiling_usd": cost_ceiling_usd,
        "issued_at": issued.isoformat(),
        "expires_at": (issued + APPROVAL_TTL).isoformat(),
    }
    path = _approval_path(state_dir, agent_id)
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


def approval_expired(record: Mapping[str, Any], now: datetime | None = None) -> bool:
    try:
        expires = datetime.fromisoformat(str(record["expires_at"]))
    except (KeyError, ValueError):
        return True
    return (now or utcnow()) >= expires


# --- watch state --------------------------------------------------------------------------


def _watch_path(state_dir: Path, agent_id: str) -> Path:
    return agent_dir(state_dir, agent_id) / "state.json"


def read_watch_state(state_dir: Path, agent_id: str) -> dict[str, Any]:
    doc = _read_json(_watch_path(state_dir, agent_id), {"breaches": {}})
    if not isinstance(doc, dict) or not isinstance(doc.get("breaches"), dict):
        raise RuntimeRequirementError(E_STATE, "state.json")
    return doc


def write_watch_state(state_dir: Path, agent_id: str, doc: Mapping[str, Any]) -> None:
    _write_json(_watch_path(state_dir, agent_id), dict(doc))
