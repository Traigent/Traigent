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
_MERGE_PR_RE = re.compile(r"Merge pull request #(\d+)")
_SUFFIX_PR_RE = re.compile(r"\(#(\d+)\)\s*$")
_REMOTE_RE = re.compile(
    r"github\.com[:/](?P<repo>[\w.-]+/[\w.-]+?)(?:\.git)?/?$", re.IGNORECASE
)

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


def pr_candidates(texts: Iterable[str]) -> tuple[list[int], list[int]]:
    """Split subject refs into (certain PR numbers, maybe-PR numbers).

    ``Merge pull request #N`` subjects are certain PRs. A ``(#N)`` subject
    suffix may be a PR (squash merge) or an issue number, so it must be
    looked up before use.
    """
    certain: set[int] = set()
    maybe: set[int] = set()
    for t in texts:
        first = (t or "").splitlines()[0] if t else ""
        m = _MERGE_PR_RE.search(first)
        if m:
            certain.add(int(m.group(1)))
            continue
        m = _SUFFIX_PR_RE.search(first)
        if m:
            maybe.add(int(m.group(1)))
    return sorted(certain), sorted(maybe - certain)


def parse_remote_url(url: str) -> str:
    """Return ``owner/name`` for a GitHub remote URL, or ``""``."""
    m = _REMOTE_RE.search((url or "").strip())
    return m.group("repo") if m else ""


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


def _is_not_found(exc: subprocess.CalledProcessError) -> bool:
    text = f"{exc.stderr or ''} {exc.output or ''}"
    return "404" in text or "not found" in text.lower()


def _pr_bodies(
    certain: list[int], maybe: list[int], repo: str, runner: Runner
) -> list[str]:
    bodies = [
        str(
            _gh_json(["pr", "view", str(n), "--json", "body"], runner).get("body") or ""
        )
        for n in certain
    ]
    for n in maybe:
        if not repo:
            raise GhLookupError("cannot classify (#N) subjects without a repository")
        args = ["api", f"repos/{repo}/pulls/{n}"]
        try:
            data = _gh_json(args, runner)
        except GhLookupError as exc:
            cause = exc.__cause__
            if isinstance(cause, subprocess.CalledProcessError) and _is_not_found(
                cause
            ):
                continue  # an issue number, not a PR
            raise
        bodies.append(str(data.get("body") or ""))
    return bodies


def _git_remote_url() -> str:
    try:
        return subprocess.run(
            ["git", "remote", "get-url", "origin"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, OSError):
        return ""


def infer_repo(
    remote_url: Callable[[], str] = _git_remote_url,
    runner: Runner | None = None,
) -> str:
    """Infer ``owner/name`` from the origin remote, else ``gh repo view``."""
    repo = parse_remote_url(remote_url())
    if repo or runner is None:
        return repo
    try:
        data = json.loads(runner(["repo", "view", "--json", "nameWithOwner"]))
    except (subprocess.CalledProcessError, OSError, json.JSONDecodeError):
        return ""
    name = data.get("nameWithOwner") if isinstance(data, dict) else ""
    return name if isinstance(name, str) else ""


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
    remote_url = _git_remote_url
    try:
        messages = _git_messages(args.range)
    except subprocess.CalledProcessError as exc:
        print(f"error: git log failed: {exc.stderr.strip()}", file=sys.stderr)
        return 2

    texts = list(messages)
    exclude: set[int] = set()
    if args.offline and not args.repo:
        args.repo = infer_repo(remote_url)
    try:
        if args.offline:
            print(
                "warning: --offline reads commit messages only and misses closing "
                "keywords that appear only in PR bodies",
                file=sys.stderr,
            )
        else:
            certain, maybe = pr_candidates(messages)
            if maybe and not args.repo:
                args.repo = infer_repo(remote_url, runner)
            texts += _pr_bodies(certain, maybe, args.repo, runner)
        if args.drop_closed:
            if not args.repo:
                args.repo = infer_repo(remote_url, runner)
            if not args.repo:
                raise GhLookupError("cannot determine the repository for --drop-closed")
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
