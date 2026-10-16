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
#: Paths the ratchet reads from the anchor ref. Module constants so that renaming a file
#: cannot silently re-enter first-introduction mode; ``test_anchor_paths_*`` pins them.
_MANIFEST_REL = "tests/egress_canary/leak_manifest.json"
_DISPOSITIONS_REL = "tests/egress_canary/leak_dispositions.json"


class AnchorUnavailable(RuntimeError):
    """The authority (origin/develop) cannot be read: the ratchet must fail."""


def _git(*args: str, repo: Path = _REPO_ROOT) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(  # noqa: S603
            ["git", "-C", str(repo), *args],  # noqa: S607
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise AnchorUnavailable(f"git unavailable: {type(exc).__name__}") from exc


def anchor_blob_state(
    rel: str, repo: Path = _REPO_ROOT, ref: str = _ANCHOR_REF
) -> bool:
    """True if ``rel`` is a blob on ``ref``, False if cleanly absent.

    ``git ls-tree`` distinguishes the two: exit 0 and empty output is "absent". A
    nonzero exit, any stderr, or output that is not exactly one blob entry for ``rel``
    (unreadable or corrupt object, bad ref) raises: never treated as "absent".
    """

    res = _git("ls-tree", ref, "--", rel, repo=repo)
    if res.returncode or res.stderr.strip():
        raise AnchorUnavailable(f"git ls-tree {ref} -- {rel} failed")
    out = res.stdout.strip()
    if not out:
        return False
    meta, _, name = out.partition("\t")
    if name != rel or meta.split()[:2] != ["100644", "blob"]:
        raise AnchorUnavailable(f"unexpected tree entry for {rel}")
    return True


def load_anchor(
    repo: Path = _REPO_ROOT, ref: str = _ANCHOR_REF, rel: str = _MANIFEST_REL
) -> set[Identity] | None:
    """Register on ``ref``; ``None`` when the file is cleanly absent (first introduction).

    Raises ``AnchorUnavailable`` when git or the ref cannot be read -- never
    silently falls back to the editable working-tree copy.
    """

    if _git(
        "rev-parse", "--verify", "--quiet", f"{ref}^{{commit}}", repo=repo
    ).returncode:
        raise AnchorUnavailable(f"{ref} is not a readable ref (git fetch?)")
    if not anchor_blob_state(rel, repo, ref):
        return None
    shown = _git("show", f"{ref}:{rel}", repo=repo)
    if shown.returncode:
        raise AnchorUnavailable("git show of the anchor manifest failed")
    return _ids(json.loads(shown.stdout)["register"])


#: Inventory sections (INVENTORY-v1) each canary route belongs to. A disposition citing
#: ``INV:<id>`` must cite one of its own route's sections.
ROUTE_INVENTORY_IDS: dict[str, frozenset[str]] = {
    "POST /api/v1/sessions": frozenset({"S1"}),
    "POST /api/v1/sessions/{id}/next-trial": frozenset({"S2"}),
    "POST /api/v1/sessions/{id}/results": frozenset({"S3"}),
    "POST /api/v1/sessions/{id}/intermediate-report": frozenset({"S4"}),
    "POST /api/v1/sessions/{id}/finalize": frozenset({"S5"}),
    "POST /api/v1/optimize": frozenset({"S12"}),
    "PUT /api/v1/configuration-runs/{id}/status": frozenset({"C1"}),
    "PUT /api/v1/configuration-runs/{id}/measures": frozenset({"C2"}),
    "POST /api/v1/hybrid/sessions": frozenset({"H1"}),
    "POST /api/v1/hybrid/optimizer/{id}/batch": frozenset({"H8"}),
    "POST /api/v1/best-configs": frozenset({"B1"}),
    "GET /api/v1/best-configs/{id}": frozenset({"B2"}),
    "POST /api/v1/example-sets": frozenset({"E1"}),
    "POST /api/v1/example-sets/{id}/upload": frozenset({"E1"}),
}
_WINDOW = 5


def inventory_ids() -> set[str]:
    data = json.loads((_HERE / "inventory_ids.json").read_text(encoding="utf-8"))
    return set(data["ids"])


def reference_problem(
    reference: str,
    repo_root: Path = _REPO_ROOT,
    ids: set[str] | None = None,
    identity: Identity | None = None,
) -> str | None:
    """None when ``reference`` is valid and, given ``identity``, is about THAT identity.

    ``INV:<id>`` must be a known id and one of the identity's own route sections.
    ``<file>:<line>`` must exist and the cited line (+/- a few lines) must mention a
    token of the identity's own field path: an unrelated existing line fails.
    """

    ref = reference.strip()
    if ref.startswith("INV:"):
        known = inventory_ids() if ids is None else ids
        if ref[4:] not in known:
            return f"unknown inventory id {ref[4:]!r}"
        if identity is not None and ref[4:] not in ROUTE_INVENTORY_IDS.get(
            identity[0], frozenset()
        ):
            return f"{ref[4:]} is not an inventory section of {identity[0]}"
        return None
    m = re.fullmatch(r"([\w./-]+):(\d+)", ref)
    if not m:
        return "reference must be 'INV:<id>' or '<repo-relative file>:<line>'"
    path = (repo_root / m.group(1)).resolve()
    if repo_root.resolve() not in path.parents or not path.is_file():
        return f"no such file in the repo: {m.group(1)}"
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    line_no = int(m.group(2))
    if not 1 <= line_no <= len(lines):
        return f"{m.group(1)} has {len(lines)} lines, not {m.group(2)}"
    if identity is not None:
        tokens = [t for t in re.split(r"[^A-Za-z0-9_]+", identity[2]) if len(t) >= 3]
        window = "\n".join(lines[max(0, line_no - 1 - _WINDOW) : line_no + _WINDOW])
        if not any(t in window for t in tokens):
            return f"{m.group(1)}:{line_no} does not mention the field {identity[2]!r}"
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
        bad = reference_problem(entry.get("reference", ""), repo_root, ids, gone)
        if bad or not entry.get("reason", "").strip():
            problems.append(f"invalid disposition for {gone}: {bad or 'empty reason'}")
    return problems
