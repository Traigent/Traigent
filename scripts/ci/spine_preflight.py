#!/usr/bin/env python3
"""Local mirror of the validation-spine PR gates, runnable before a push.

CI enforces a spine rule on Traigent (Python SDK) PRs that an agent or human
only discovers *after* pushing:

* ``spine-trail present`` (``.github/workflows/spine-trail-gate.yml``) — every
  product PR must carry a validation-spine mark in its body: a
  ``Spine-Trail: st_<id>`` line (a Tier-0 WorkIntent), a ``Spine: cs_<id>``
  line (a promoted ChangeSession), or an explicit ``Spine: none (reason: …)``
  waiver.

Unlike TraigentBackend, this repo currently has **no** ``require-spine-session``
gate and **no** ``.github/spine-policy-surface-globs.txt`` policy-surface file
(its only PR-time spine workflow, ``validation-spine-pr.yml``, is advisory /
``continue-on-error``). So this preflight:

* checks for the spine-trail breadcrumb (WARNING only — the authoritative check
  reads the PR *body*, which does not exist pre-push, and a standalone clone
  cannot read the workspace ledger), and
* if a policy-surface globs file is later added to this repo, automatically
  picks it up and enforces a ``Spine-Session:`` on policy-surface diffs (hard
  block), mirroring the TraigentBackend gate — so this stays correct if the
  repo grows that gate without needing a script change.

It is intentionally dependency-free (stdlib + git) so ``scripts/local_gate.sh``
and the pre-push hook can block the push with the same verdict CI would reach.

Exit codes: 0 = pass/clean, 1 = a blocking spine requirement is unmet.
"""

from __future__ import annotations

import argparse
import fnmatch
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
REPO_NAME = "Traigent"
# Optional: if this repo ever adds a policy-surface globs file (as
# TraigentBackend has), this preflight will start enforcing require-spine-session
# against it automatically. Absent today.
GLOBS_FILE = REPO_ROOT / ".github" / "spine-policy-surface-globs.txt"
SPINE_TRAIL = (
    REPO_ROOT.parent / "tools" / "spine-trail" / "spine_trail.py"
)  # workspace-level helper; absent in a bare/standalone checkout
SESSION_RE = re.compile(r"^\s*Spine(?:-Session)?:\s*(cs_[0-9a-z]{6,})\b", re.MULTILINE)
TRAIL_RE = re.compile(r"^\s*Spine-Trail:\s*(st_[0-9a-f]{6,})\b", re.MULTILINE)


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True
    ).stdout.strip()


def _load_globs() -> list[str]:
    if not GLOBS_FILE.exists():
        return []
    return [
        ln.strip()
        for ln in GLOBS_FILE.read_text().splitlines()
        if ln.strip() and not ln.strip().startswith("#")
    ]


def _resolve_base(explicit: str | None, branch: str) -> str:
    if explicit:
        return explicit
    # release/hotfix branches target main; everything else targets develop.
    target = "main" if re.match(r"^(release|hotfix)/", branch) else "develop"
    for ref in (f"origin/{target}", target):
        if _git("rev-parse", "--verify", "--quiet", ref):
            return ref
    return "origin/develop"


def _changed_files(base: str) -> list[str]:
    merge_base = _git("merge-base", base, "HEAD") or base
    committed = _git("diff", "--name-only", f"{merge_base}...HEAD")
    # include staged + unstaged + untracked so an about-to-be-pushed change is
    # seen even mid-edit or before the first commit (new files in a policy dir).
    dirty = _git("diff", "--name-only", "HEAD")
    untracked = _git("ls-files", "--others", "--exclude-standard")
    files = {f for f in (committed + "\n" + dirty + "\n" + untracked).splitlines() if f}
    return sorted(files)


def _matches_policy(files: list[str], globs: list[str]) -> list[str]:
    hits = []
    for f in files:
        if any(fnmatch.fnmatch(f, g) for g in globs):
            hits.append(f)
    return hits


def _branch_messages(base: str) -> str:
    merge_base = _git("merge-base", base, "HEAD") or base
    return _git("log", f"{merge_base}..HEAD", "--format=%B")


def _trail_present(branch: str) -> str | None:
    """Return the trail id if the workspace ledger has one for this branch."""
    if not SPINE_TRAIL.exists():
        return None
    out = subprocess.run(
        [
            sys.executable,
            str(SPINE_TRAIL),
            "check",
            "--repo",
            REPO_NAME,
            "--branch",
            branch,
        ],
        capture_output=True,
        text=True,
    )
    m = TRAIL_RE.search(out.stdout) or re.search(r"(st_[0-9a-f]{6,})", out.stdout)
    return m.group(1) if m else None


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base", help="base ref (default: auto from branch name)")
    args = p.parse_args(argv)

    branch = _git("rev-parse", "--abbrev-ref", "HEAD")
    base = _resolve_base(args.base, branch)
    globs = _load_globs()
    files = _changed_files(base)
    messages = _branch_messages(base)

    ok = True
    print(
        f"🧭 spine preflight — branch={branch} base={base} changed={len(files)} files"
    )

    # 1) policy surface -> require a Spine-Session signal.
    # Only active if this repo gains a policy-surface globs file; absent today.
    if globs:
        policy_hits = _matches_policy(files, globs)
        if policy_hits:
            session = SESSION_RE.search(messages)
            if session:
                print(
                    "   ✅ policy surface touched; Spine-Session "
                    f"{session.group(1)} found in branch commits"
                )
            else:
                ok = False
                shown = "\n".join(f"        - {f}" for f in policy_hits[:8])
                more = (
                    f"\n        … +{len(policy_hits) - 8} more"
                    if len(policy_hits) > 8
                    else ""
                )
                print(
                    "   ❌ POLICY SURFACE touched but no Spine-Session found.\n"
                    f"      Files:\n{shown}{more}\n"
                    "      Fix (work the spine way BEFORE pushing):\n"
                    "        1. Open/relate a ChangeSession (/spine:change).\n"
                    "        2. Add a trailer to a commit on this branch AND the"
                    " PR body:\n"
                    "             Spine: cs_xxxxxxxx\n"
                    "      (CI reads the PR BODY — keep both in sync.)"
                )
        else:
            print("   ✅ no policy-surface files in diff (no Spine-Session required)")
    else:
        print(
            "   ℹ️  no policy-surface globs file in this repo; "
            "require-spine-session not enforced (CI has no such gate here)"
        )

    # 2) spine-trail present (core repo breadcrumb) — WARNING only.
    # The authoritative check is the PR body (which doesn't exist pre-push), and
    # the workspace commit hook auto-stamps a trail; a standalone clone also
    # can't read the workspace ledger. So surface it as a reminder, don't block.
    trail_in_msg = TRAIL_RE.search(messages) or SESSION_RE.search(messages)
    trail_in_ledger = _trail_present(branch)
    if trail_in_msg or trail_in_ledger:
        tid = trail_in_msg.group(1) if trail_in_msg else trail_in_ledger
        print(f"   ✅ spine-trail present ({tid})")
    else:
        print(
            "   ⚠️  no Spine-Trail detected for this branch (CI's "
            "'spine-trail present' requires one in the PR body).\n"
            "      Create:  python3 tools/spine-trail/spine_trail.py "
            "get-or-create \\\n"
            f"                 --repo {REPO_NAME} --branch {branch} "
            f"--base-branch {base.split('/')[-1]} \\\n"
            "                 --kind <bug_fix|feature|change> "
            "--source-type <type> \\\n"
            '                 --source-ref "<ref>" --changed-paths "<files>"\n'
            "      then add its  Spine-Trail: st_xxxx  line to the PR body\n"
            "      (or  Spine: cs_xxxx , or  Spine: none (reason: …) )."
        )

    print("✅ spine preflight passed" if ok else "❌ spine preflight FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
