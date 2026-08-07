#!/usr/bin/env python3
"""Verify every dependency of ``required-pr-gate`` actually finished green.

This is the LAST line of defense before a PR is allowed to merge: the
``required-pr-gate`` job fans in from every real test/check job via
``needs:``, and this script decides whether that fan-in counts as passing.

Previously this logic was an inline ``case`` statement in the workflow YAML
that accepted ``skipped`` as a pass for EVERY dependency, unconditionally:

    case "$result" in
      success|skipped) ;;
      *) echo "::error::Required dependency finished with ${result}"; exit 1 ;;
    esac

That is a hole shared with TraigentBackend's pr-gate.yml. The ``changes`` job
classifies which files changed and sets outputs (``code_changed``) that gate
whether ``unit`` / ``collection`` / ``mcp-contract`` run at all. A comment on
TraigentBackend's own classifier already records a real incident there:
omitting deletions from the diff filter "classified that entire change class
as inert and let the aggregate gate pass with every substantive test job
skipped." The aggregator's blanket ``skipped -> pass`` is the same shape of
hole here: the next classifier bug -- in EITHER repo -- has the identical
consequence, a real change silently merging with zero tests executed, because
the aggregator cannot tell "this job skipped because the classifier correctly
said it was irrelevant" apart from "this job skipped because the classifier
is wrong."

The fix is SKIP_OK below: an explicit, small allowlist of jobs whose skip can
be proven safe WITHOUT trusting the content classifier at all -- their `if:`
conditions depend only on unforgeable GitHub event facts (draft-PR status,
event name, fork identity), never on `needs.changes.outputs.*`. Every job
whose skip DOES depend on a classifier output (`unit`, `collection`,
`mcp-contract`) is deliberately left OUT: if any of those reports `skipped`,
the gate now fails, even though most of the time that skip is legitimate
(nothing relevant changed). A false red there is cheap -- push an empty
commit, or fix the classifier and re-run. A false green is the exact failure
mode this script exists to close.

Extracted out of the workflow YAML (instead of an inline heredoc) specifically
so it has unit tests -- see tests/ci/test_check_required_gate.py.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field

# Jobs allowed to report "skipped" without failing the gate, mapped to WHY
# each one is safe. The bar for an entry here is: this job's skip can NEVER
# be caused by a bug in the `changes` (or any other) content classifier --
# only by a GitHub-supplied event fact that our own code does not compute
# and cannot get wrong the way a diff-pattern classifier can.
SKIP_OK: dict[str, str] = {
    "changes": (
        "The classifier job itself. Its own `if:` skips it only on a draft "
        "PR (`github.event.pull_request.draft == false` is false) or a "
        "malformed merge_group event -- both unforgeable GitHub event facts, "
        "not a classifier output. A draft PR cannot be merged, so a skip "
        "here is harmless (see the draft-PR note in pr-gate.yml)."
    ),
    "preflight": (
        "Carries no `if:` of its own; it depends only on `changes` and "
        "skips (via GitHub's default `if: success()`) exactly when `changes` "
        "itself skipped or failed. A `changes` failure is independently "
        "caught because `changes` is itself a required dependency of THIS "
        "gate, so allowlisting preflight's cascade opens no new hole."
    ),
    "schema-types": (
        "Its `if:` is `event_name == 'merge_group' || head.repo.full_name == "
        "github.repository` -- it has NO `needs:` on `changes` at all (see "
        "its own workflow comment: 'deliberately not gated on changes "
        "output'), so it cannot skip because the content classifier "
        "under-reported anything. Its ONLY skip vector is a fork PR (no "
        "access to the SCHEMA_TOKEN secret), which is an unforgeable GitHub "
        "event fact, structurally immune to a classifier bug -- the exact "
        "failure mode this gate defends against."
    ),
}


@dataclass
class GateResult:
    ok: bool
    problems: list[str] = field(default_factory=list)


def evaluate(needs: dict[str, dict], skip_ok: dict[str, str]) -> GateResult:
    """Decide whether every dependency in ``needs`` is green enough to pass.

    ``needs`` is the raw ``needs`` context GitHub Actions hands the job: one
    entry per job actually listed in THIS job's own ``needs:``, each at least
    carrying a ``result`` key (``success`` / ``failure`` / ``cancelled`` /
    ``skipped``). ``skip_ok`` maps job name -> justification for jobs allowed
    to report ``skipped``.

    Fails closed: an empty ``needs`` map, a ``skip_ok`` entry naming a job
    that is not present in ``needs`` (a stale allowlist -- the job was
    probably renamed and the real one is now unguarded), a ``cancelled``
    result, and an empty/absent ``result`` all count as failures. Nothing
    ever passes by falling through an unrecognised case.
    """
    problems: list[str] = []

    if not needs:
        problems.append(
            "needs map is empty -- this job has no declared dependencies to "
            "verify, which is never correct for a required gate. Check the "
            "`needs:` list on this job in the workflow."
        )
        return GateResult(ok=False, problems=problems)

    for allowlisted_job in skip_ok:
        if allowlisted_job not in needs:
            problems.append(
                f"SKIP_OK names '{allowlisted_job}', which is not a declared "
                "dependency of this job (absent from `needs`). The allowlist "
                "is stale -- most likely the job was renamed -- and the real "
                "job it used to cover is now unguarded."
            )

    for job, info in needs.items():
        result = info.get("result") if isinstance(info, dict) else None
        if result == "success":
            continue
        if result == "skipped" and job in skip_ok:
            continue
        problems.append(f"{job}={result!r}")

    return GateResult(ok=not problems, problems=problems)


def main(argv: list[str] | None = None) -> int:
    del argv  # No CLI args; input comes from the NEEDS_JSON env var (see below).

    raw = os.environ.get("NEEDS_JSON", "")
    if not raw.strip():
        print("::error::NEEDS_JSON is empty or unset; cannot verify the gate.")
        return 1
    try:
        needs = json.loads(raw)
    except json.JSONDecodeError as exc:
        print(f"::error::NEEDS_JSON is not valid JSON: {exc}")
        return 1

    # Print the full result map before deciding, so a future reader debugging
    # a red gate can see exactly what every dependency reported.
    print("Dependency result map:")
    print(json.dumps(needs, indent=2, sort_keys=True))

    result = evaluate(needs, SKIP_OK)
    if not result.ok:
        print("::error::required-pr-gate FAILED:")
        for problem in result.problems:
            print(f"::error::  {problem}")
        return 1

    print(
        "required-pr-gate passed: every dependency succeeded, or skipped for an allowlisted reason."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
