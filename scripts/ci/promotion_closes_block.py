#!/usr/bin/env python3
"""Build the "Closes #N" block for a develop -> main promotion PR.

GitHub only acts on closing keywords when a PR merges into the default
branch, so issues fixed by PRs merged into ``develop`` stay open after the
promotion. This helper collects the closing keywords from the commits (and,
optionally, the PR bodies) being promoted and prints one de-duplicated block
to paste into the promotion PR body.

Reads commit messages and the bodies of the PRs named in commit subjects
(needs an authenticated ``gh``; any failed lookup exits non-zero)::

    python scripts/ci/promotion_closes_block.py origin/main..origin/develop

Git-only mode misses keywords that live only in PR bodies::

    python scripts/ci/promotion_closes_block.py origin/main..origin/develop --offline

Add ``--drop-closed`` to skip issues that are already closed.

The block is printed to stdout; the check line goes to stderr. After opening
the promotion PR, the number of lines in the block should equal its
``closingIssuesReferences`` count.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from collections.abc import Callable, Iterable

_KEYWORDS = r"(?:close[sd]?|fix(?:e[sd])?|resolve[sd]?)"
# keyword [:] <#N | owner/repo#N | https://github.com/owner/repo/issues/N>
_CLOSING_RE = re.compile(
    rf"(?<![\w-]){_KEYWORDS}:?[ \t]+"
    r"(?:(?P<repo>[\w.-]+/[\w.-]+)#|https?://github\.com/(?P<urlrepo>[\w.-]+/[\w.-]+)/issues/|#)"
    r"(?P<num>\d+)(?!\w)",
    re.IGNORECASE,
)
_FENCE_RE = re.compile(r"(```|~~~).*?(\1|\Z)", re.DOTALL)
_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
_PR_REF_RE = re.compile(r"(?:\(#(\d+)\)|Merge pull request #(\d+))")

IssueRef = tuple[str, int]  # ("" for the current repo, else "owner/repo"), number


def extract_closing_refs(text: str) -> list[IssueRef]:
    """Return closing-keyword issue refs in ``text`` in order of appearance.

    HTML comments are stripped first (so an unmatched fence inside a comment
    cannot swallow later text), then fenced code blocks; text in either is ignored, as GitHub
    does not treat it as a closing reference. One keyword closes one issue, so
    ``Closes #1, #2`` yields only ``#1`` (matching GitHub behaviour).
    """
    cleaned = _FENCE_RE.sub("", _COMMENT_RE.sub("", text or ""))
    refs: list[IssueRef] = []
    for m in _CLOSING_RE.finditer(cleaned):
        repo = (m.group("repo") or m.group("urlrepo") or "").lower()
        refs.append((repo, int(m.group("num"))))
    return refs


def build_closes_block(
    texts: Iterable[str],
    *,
    exclude: Iterable[int] = (),
    repo: str = "",
) -> list[str]:
    """Return de-duplicated ``Closes`` lines: same-repo by number, then others.

    ``repo`` ("owner/name", case-insensitive) lets refs that spell out the
    current repo collapse onto the short ``#N`` form. ``exclude`` drops
    same-repo issue numbers (e.g. already closed).
    """
    here = repo.lower()
    skip = set(exclude)
    seen: set[IssueRef] = set()
    for text in texts:
        for r, n in extract_closing_refs(text):
            if r == here:
                r = ""
            if r == "" and n in skip:
                continue
            seen.add((r, n))
    ordered = sorted(seen, key=lambda ref: (ref[0] != "", ref[0], ref[1]))
    return [f"Closes {r}#{n}" if r else f"Closes #{n}" for r, n in ordered]


def pr_numbers(texts: Iterable[str]) -> list[int]:
    """PR numbers referenced by squash subjects or merge-commit subjects."""
    nums: set[int] = set()
    for t in texts:
        first = (t or "").splitlines()[0] if t else ""
        for m in _PR_REF_RE.finditer(first):
            nums.add(int(m.group(1) or m.group(2)))
    return sorted(nums)


def _git_messages(rev_range: str) -> list[str]:
    out = subprocess.run(
        ["git", "log", "--format=%B%x1e", rev_range],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return [m.strip() for m in out.split("\x1e") if m.strip()]


class GhLookupError(RuntimeError):
    """A gh lookup failed; the block would be incomplete."""


Runner = Callable[[list[str]], str]


def _run_gh(args: list[str]) -> str:
    return subprocess.run(
        ["gh", *args], check=True, capture_output=True, text=True
    ).stdout


def _gh_json(args: list[str], runner: Runner) -> dict[str, object]:
    try:
        data = json.loads(runner(args))
    except (subprocess.CalledProcessError, OSError, json.JSONDecodeError) as exc:
        raise GhLookupError(f"gh {' '.join(args[:3])} failed: {exc}") from exc
    if not isinstance(data, dict):
        raise GhLookupError(f"gh {' '.join(args[:3])} returned unexpected JSON")
    return data


def _pr_bodies(numbers: list[int], runner: Runner) -> list[str]:
    return [
        str(
            _gh_json(["pr", "view", str(n), "--json", "body"], runner).get("body") or ""
        )
        for n in numbers
    ]


def _closed_issue_numbers(numbers: list[int], runner: Runner) -> set[int]:
    closed: set[int] = set()
    for n in numbers:
        data = _gh_json(["issue", "view", str(n), "--json", "state"], runner)
        if data.get("state") == "CLOSED":
            closed.add(n)
    return closed


def main(argv: list[str] | None = None, runner: Runner | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("range", help="git revision range, e.g. origin/main..origin/develop")
    p.add_argument(
        "--offline",
        action="store_true",
        help="git history only; misses keywords that live only in PR bodies",
    )
    p.add_argument(
        "--drop-closed",
        action="store_true",
        help="drop issues gh reports as already CLOSED",
    )
    p.add_argument("--repo", default="", help="owner/name, to shorten same-repo refs")
    args = p.parse_args(argv)

    if args.offline and args.drop_closed:
        print(
            "error: --drop-closed needs gh; incompatible with --offline",
            file=sys.stderr,
        )
        return 2
    if runner is None:
        if not args.offline and not shutil.which("gh"):
            print(
                "error: gh is not available; install it or pass --offline",
                file=sys.stderr,
            )
            return 2
        runner = _run_gh
    try:
        messages = _git_messages(args.range)
    except subprocess.CalledProcessError as exc:
        print(f"error: git log failed: {exc.stderr.strip()}", file=sys.stderr)
        return 2

    texts = list(messages)
    exclude: set[int] = set()
    try:
        if args.offline:
            print(
                "warning: --offline reads commit messages only and misses closing "
                "keywords that appear only in PR bodies",
                file=sys.stderr,
            )
        else:
            texts += _pr_bodies(pr_numbers(messages), runner)
        if args.drop_closed:
            here = args.repo.lower()
            candidates = {
                n for t in texts for r, n in extract_closing_refs(t) if r in ("", here)
            }
            exclude = _closed_issue_numbers(sorted(candidates), runner)
    except GhLookupError as exc:
        print(f"error: {exc}; refusing to print a partial block", file=sys.stderr)
        return 1

    block = build_closes_block(texts, exclude=exclude, repo=args.repo)
    print("\n".join(block))
    print(
        f"\n# {len(block)} issue(s). After opening the promotion PR, check that its "
        f"closingIssuesReferences count equals {len(block)}.",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
