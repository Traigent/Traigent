"""Ratchet pins, loaded from the checked-in manifest.

leak_manifest.json pins leak *identities* (route, location, field_path,
canary kind), not counts: a leak that moves to another field or location on
the same route is a removal plus an addition and fails.  register only
shrinks; each removal needs an entry in leak_dispositions.json.
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

Identity = tuple[str, str, str, str]

_HERE = Path(__file__).parent


def _ids(rows: list[dict[str, str]]) -> set[Identity]:
    return {(r["route"], r["location"], r["field_path"], r["kind"]) for r in rows}


def load_manifest() -> tuple[set[Identity], set[Identity]]:
    data = json.loads((_HERE / "leak_manifest.json").read_text(encoding="utf-8"))
    return _ids(data["baseline"]), _ids(data["register"])


def load_dispositions() -> list[dict[str, str]]:
    data = json.loads((_HERE / "leak_dispositions.json").read_text(encoding="utf-8"))
    return list(data["dispositions"])


BASELINE, REGISTER = load_manifest()

#: The 43 (route, kind) pairs measured before the coverage expansion (default
#: config, identity fields, offline replay, failure paths). Kept so the report
#: can state which pairs are new.
PRE_EXPANSION_PAIRS: tuple[tuple[str, str], ...] = (
    ("GET /api/v1/best-configs/{id}", "agent_name"),
    ("POST /api/v1/best-configs", "agent_name"),
    ("POST /api/v1/best-configs", "choice_value"),
    ("POST /api/v1/best-configs", "knob_name"),
    ("POST /api/v1/best-configs", "metric_name"),
    ("POST /api/v1/example-sets", "agent_name"),
    ("POST /api/v1/example-sets", "dataset_file_name"),
    ("POST /api/v1/example-sets/{id}/upload", "example_id"),
    ("POST /api/v1/example-sets/{id}/upload", "input_text"),
    ("POST /api/v1/example-sets/{id}/upload", "metadata_key"),
    ("POST /api/v1/example-sets/{id}/upload", "metadata_value"),
    ("POST /api/v1/example-sets/{id}/upload", "output_text"),
    ("POST /api/v1/hybrid/optimizer/{id}/batch", "metadata_key"),
    ("POST /api/v1/hybrid/optimizer/{id}/batch", "metadata_value"),
    ("POST /api/v1/hybrid/optimizer/{id}/batch", "metric_name"),
    ("POST /api/v1/hybrid/sessions", "agent_name"),
    ("POST /api/v1/hybrid/sessions", "choice_value"),
    ("POST /api/v1/hybrid/sessions", "knob_name"),
    ("POST /api/v1/hybrid/sessions", "metadata_key"),
    ("POST /api/v1/hybrid/sessions", "metadata_value"),
    ("POST /api/v1/hybrid/sessions", "metric_name"),
    ("POST /api/v1/hybrid/sessions", "prompt_template"),
    ("POST /api/v1/hybrid/sessions", "run_title"),
    ("POST /api/v1/sessions", "agent_name"),
    ("POST /api/v1/sessions", "choice_value"),
    ("POST /api/v1/sessions", "evaluation_set"),
    ("POST /api/v1/sessions", "knob_name"),
    ("POST /api/v1/sessions", "metric_name"),
    ("POST /api/v1/sessions", "prompt_template"),
    ("POST /api/v1/sessions", "run_description"),
    ("POST /api/v1/sessions", "run_title"),
    ("POST /api/v1/sessions/{id}/finalize", "metric_name"),
    ("POST /api/v1/sessions/{id}/intermediate-report", "metric_name"),
    ("POST /api/v1/sessions/{id}/results", "choice_value"),
    ("POST /api/v1/sessions/{id}/results", "exception_message"),
    ("POST /api/v1/sessions/{id}/results", "knob_name"),
    ("POST /api/v1/sessions/{id}/results", "metadata_key"),
    ("POST /api/v1/sessions/{id}/results", "metadata_value"),
    ("POST /api/v1/sessions/{id}/results", "metric_name"),
    ("POST /api/v1/sessions/{id}/results", "model_output"),
    ("POST /api/v1/sessions/{id}/results", "prompt_template"),
    ("PUT /api/v1/configuration-runs/{id}/measures", "metric_name"),
    ("PUT /api/v1/configuration-runs/{id}/status", "exception_message"),
)


# --- anchored ratchet -------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[2]
_ANCHOR_REF = "origin/develop"
_MANIFEST_REL = "tests/egress_canary/leak_manifest.json"


class AnchorUnavailable(RuntimeError):
    """The authority (origin/develop) cannot be read: the ratchet must fail."""


def _git(*args: str) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(  # noqa: S603
            ["git", "-C", str(_REPO_ROOT), *args],  # noqa: S607
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise AnchorUnavailable(f"git unavailable: {type(exc).__name__}") from exc


def load_anchor() -> set[Identity] | None:
    """Register on ``origin/develop``; ``None`` when the file is not there yet.

    Raises ``AnchorUnavailable`` when git or the ref cannot be read -- never
    silently falls back to the editable working-tree copy.
    """

    if _git("rev-parse", "--verify", "--quiet", f"{_ANCHOR_REF}^{{commit}}").returncode:
        raise AnchorUnavailable(f"{_ANCHOR_REF} is not a readable ref (git fetch?)")
    if _git("cat-file", "-e", f"{_ANCHOR_REF}:{_MANIFEST_REL}").returncode:
        return None
    shown = _git("show", f"{_ANCHOR_REF}:{_MANIFEST_REL}")
    if shown.returncode:
        raise AnchorUnavailable("git show of the anchor manifest failed")
    return _ids(json.loads(shown.stdout)["register"])


def inventory_ids() -> set[str]:
    data = json.loads((_HERE / "inventory_ids.json").read_text(encoding="utf-8"))
    return set(data["ids"])


def reference_problem(
    reference: str, repo_root: Path = _REPO_ROOT, ids: set[str] | None = None
) -> str | None:
    """None when ``reference`` is ``INV:<known id>`` or an existing ``file:line``."""

    ref = reference.strip()
    if ref.startswith("INV:"):
        known = inventory_ids() if ids is None else ids
        return None if ref[4:] in known else f"unknown inventory id {ref[4:]!r}"
    m = re.fullmatch(r"([\w./-]+):(\d+)", ref)
    if not m:
        return "reference must be 'INV:<id>' or '<repo-relative file>:<line>'"
    path = (repo_root / m.group(1)).resolve()
    if repo_root.resolve() not in path.parents or not path.is_file():
        return f"no such file in the repo: {m.group(1)}"
    lines = len(path.read_text(encoding="utf-8", errors="replace").splitlines())
    if not 1 <= int(m.group(2)) <= lines:
        return f"{m.group(1)} has {lines} lines, not {m.group(2)}"
    return None


def ratchet_problems(
    live: set[Identity],
    anchor: set[Identity] | None,
    register: set[Identity],
    dispositions: list[dict[str, str]],
    repo_root: Path = _REPO_ROOT,
    ids: set[str] | None = None,
) -> list[str]:
    """Every reason the live leaks are not acceptable against the authority."""

    problems: list[str] = []
    if anchor is None:  # first introduction: manifest must equal the live result
        if live != register:
            problems.append(
                f"first introduction: manifest != live; added={sorted(live - register)} "
                f"missing={sorted(register - live)}"
            )
        return problems
    for added in sorted(live - anchor):
        problems.append(f"new leak not in {_ANCHOR_REF} manifest: {added}")
    disposed = {
        (d["route"], d["location"], d["field_path"], d["kind"]): d for d in dispositions
    }
    for gone in sorted(anchor - live):
        entry = disposed.get(gone)
        if entry is None:
            problems.append(f"leak gone without disposition: {gone}")
            continue
        bad = reference_problem(entry.get("reference", ""), repo_root, ids)
        if bad or not entry.get("reason", "").strip():
            problems.append(f"invalid disposition for {gone}: {bad or 'empty reason'}")
    return problems
