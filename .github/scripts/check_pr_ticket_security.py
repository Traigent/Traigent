#!/usr/bin/env python3
"""Check that a PR body carries `Ticket:` and `Security note:` lines.

Owner decision 4A (2026-11-17), SOC 2 CMT.02 / CMT.04 / CMT.12.

Reads the body from the PR_BODY env var (never from argv or a shell
interpolation) and the author from PR_AUTHOR. Exit 0 = pass, 1 = fail.

Ticket accepts: a project key (TRAIG-123, [A-Z]{2,10}-N), a GitHub issue
reference (#123, Fixes #123, Closes/Resolves too, or an issues URL), or
`none (<reason>)`. Security note accepts one non-empty line, or
`none (<reason>)`. Unfilled template placeholders (`<...>`) fail.
"""

import os
import re
import sys

EXEMPT_AUTHORS = {"dependabot[bot]", "renovate[bot]", "github-actions[bot]"}

COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)
TICKET_LINE = re.compile(
    r"^[ \t]*Ticket:[ \t]*(.*?)[ \t]*$", re.IGNORECASE | re.MULTILINE
)
SECURITY_LINE = re.compile(
    r"^[ \t]*Security note:[ \t]*(.*?)[ \t]*$", re.IGNORECASE | re.MULTILINE
)

NONE_WITH_REASON = re.compile(r"^none[ \t]*\([ \t]*[^ \t)][^)]*\)$", re.IGNORECASE)
PLANE_KEY = re.compile(r"^[A-Z]{2,10}-[0-9]+$")
ISSUE_REF = re.compile(
    r"^((fixes|closes|resolves)[ \t]+)?([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)?#[0-9]+$",
    re.IGNORECASE,
)
ISSUE_URL = re.compile(
    r"^((fixes|closes|resolves)[ \t]+)?https://github\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+/issues/[0-9]+/?$",
    re.IGNORECASE,
)


def _values(pattern, body):
    return [m.group(1) for m in pattern.finditer(body)]


def ticket_ok(value):
    if NONE_WITH_REASON.match(value):
        return True
    return bool(
        PLANE_KEY.match(value) or ISSUE_REF.match(value) or ISSUE_URL.match(value)
    )


def security_ok(value):
    if not value or value.startswith("<"):
        return False
    if value.lower().startswith("none"):
        return bool(NONE_WITH_REASON.match(value))
    return True


def check(body, author=""):
    """Return a list of problems (empty list = pass)."""
    if author in EXEMPT_AUTHORS:
        return []
    body = (body or "").replace("\r\n", "\n").replace("\r", "\n")
    body = COMMENT.sub("", body)
    problems = []
    if not any(ticket_ok(v) for v in _values(TICKET_LINE, body)):
        problems.append(
            "Missing or invalid `Ticket:` line. Use a Plane key (TRAIG-123), a GitHub "
            "issue (#123, Fixes #123, or an issue URL), or `Ticket: none (<reason>)`."
        )
    if not any(security_ok(v) for v in _values(SECURITY_LINE, body)):
        problems.append(
            "Missing or empty `Security note:` line. Give one line on security impact, "
            "or `Security note: none (<reason>)`."
        )
    return problems


def main():
    problems = check(os.environ.get("PR_BODY", ""), os.environ.get("PR_AUTHOR", ""))
    if not problems:
        print("OK: Ticket and Security note present.")
        return 0
    for p in problems:
        print("::error title=PR body incomplete::" + p)
    return 1


if __name__ == "__main__":
    sys.exit(main())
