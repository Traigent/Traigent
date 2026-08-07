#!/usr/bin/env python3
"""Verify every dependency of ``required-pr-gate`` actually finished green.

This is the LAST line of defense before a PR is allowed to merge: the
``required-pr-gate`` job fans in from every real test/check job via
``needs:``, and this script decides whether that fan-in counts as passing.

Previously this logic was an inline ``case`` statement in the workflow YAML
that accepted ``skipped`` as a pass for EVERY dependency, unconditionally --
the same shape of hole TraigentBackend's own pr-gate.yml already hit once
(see that repo's classifier comment: omitting deletions from the diff filter
"classified that entire change class as inert and let the aggregate gate
pass with every substantive test job skipped").

A first pass at fixing this simply banned classifier-gated jobs from ever
skipping (an unconditional allowlist covering only `changes`/`preflight`/
`schema-types`). That is wrong in the other direction: a docs-only,
Dependabot, or workflow-only PR touches nothing the classifier matches, so
`unit`/`collection`/`mcp-contract` legitimately skip -- and banning the skip
turns a correctly-working feature into a gate that is red on arrival for a
whole class of legitimate PRs. A gate that goes red for the wrong reason gets
disabled, not fixed -- which would recreate exactly the inert mechanism this
script exists to close.

The actual defect is narrower than "jobs skip on classifier output": it is
that a BUG in the classifier and a CORRECT classification both produce the
exact same observable, `skipped`. The fix is to make them distinguishable
instead of banning the skip. For each classifier-gated job (CLASSIFIER_GATED
below), a `skipped` result is accepted ONLY when all three hold:

  1. `changes` itself concluded `success` (not `skipped`, not `failed`) --
     an unsuccessful classifier run cannot be trusted to have set any output
     correctly.
  2. The specific `changes` output that gates this job (`code_changed`) is
     the LITERAL string `'false'` -- not empty, not missing, not any other
     value. An empty or missing output means the classifier step never set
     it (a shell bug, a renamed output, a step that exited early), which is
     silently indistinguishable from "false" unless checked explicitly.
  3. `changes` reports a NON-ZERO total changed-file count
     (`changed_file_count`), independent of any path pattern. A diff range
     that produced zero files is a broken/empty range, not a "nothing
     relevant changed" verdict.

Any of those failing -> the gate fails, naming which condition broke. All
three holding -> the gate passes, and the log states the output value that
justified it. This is strictly stronger than the original permissive `case`
(it still catches a classifier crash, an empty range, a renamed output, an
`if:` bug) while not breaking legitimate skips.

UNCONDITIONAL_SKIP_OK (below) is unchanged from the first pass: `changes`/
`preflight`/`schema-types` may skip with no further verification. `changes`
and `preflight` because their skip conditions depend solely on unforgeable
GitHub event facts (draft-PR status), never on classifier output.
`schema-types` because it carries NO `needs:` on `changes` at all -- its only
skip vector is a fork PR (no `SCHEMA_TOKEN` secret access), an unforgeable
event fact structurally immune to a classifier bug, unlike `unit`/
`collection`/`mcp-contract` which are also gated on `code_changed`.

Extracted out of the workflow YAML (instead of an inline heredoc) specifically
so it has unit tests -- see tests/ci/test_check_required_gate.py.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field

# Jobs allowed to report "skipped" with NO further verification, mapped to WHY
# each one is safe. The bar for an entry here is: this job's skip can NEVER be
# caused by a bug in the `changes` (or any other) content classifier -- only
# by a GitHub-supplied event fact that our own code does not compute and
# cannot get wrong the way a diff-pattern classifier can.
UNCONDITIONAL_SKIP_OK: dict[str, str] = {
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
        "Its `if:` is `event_name == 'merge_group' || head.repo.full_name "
        "== github.repository` -- it has NO `needs:` on `changes` at all "
        "(see its own workflow comment: 'deliberately not gated on changes "
        "output'), so it cannot skip because the content classifier "
        "under-reported anything. Its ONLY skip vector is a fork PR (no "
        "access to the SCHEMA_TOKEN secret), an unforgeable GitHub event "
        "fact, structurally immune to a classifier bug -- unlike `unit`/"
        "`collection`/`mcp-contract`, which are ALSO gated on `code_changed`."
    ),
}

# Jobs whose skip is gated on a `changes` classifier output. A `skipped`
# result here is accepted ONLY when verify_classifier_gated_skip() proves it
# safe (see the module docstring's three conditions). Maps job name -> the
# `changes` output name that gates it.
CLASSIFIER_GATED: dict[str, str] = {
    "unit": "code_changed",
    "collection": "code_changed",
    "mcp-contract": "code_changed",
}

CHANGES_JOB = "changes"
# `changes` output carrying the raw file count for the diff range, BEFORE any
# path-pattern filtering. See the module docstring, condition 3.
FILE_COUNT_OUTPUT = "changed_file_count"


@dataclass
class GateResult:
    ok: bool
    problems: list[str] = field(default_factory=list)
    justifications: list[str] = field(default_factory=list)


def verify_classifier_gated_skip(
    job: str, gating_output: str, needs: dict
) -> tuple[bool, str]:
    """Decide whether `job`'s `skipped` result is provably safe.

    Returns (True, justification) if all three conditions in the module
    docstring hold, else (False, problem). Never guesses: a missing
    `changes` entry, a non-dict `outputs`, or a non-integer file count are
    all treated as failures, not as "probably fine."
    """
    changes_entry = needs.get(CHANGES_JOB)
    changes_result = (
        changes_entry.get("result") if isinstance(changes_entry, dict) else None
    )
    if changes_result != "success":
        return False, (
            f"{job}=skipped, but '{CHANGES_JOB}' did not succeed "
            f"(result={changes_result!r}) -- its outputs cannot be trusted to "
            "justify this skip"
        )

    outputs = changes_entry.get("outputs") if isinstance(changes_entry, dict) else None
    outputs = outputs if isinstance(outputs, dict) else {}

    gating_value = outputs.get(gating_output)
    if gating_value is None:
        return False, (
            f"{job}=skipped, but '{CHANGES_JOB}.outputs.{gating_output}' is missing entirely"
        )
    if gating_value != "false":
        return False, (
            f"{job}=skipped, but '{CHANGES_JOB}.outputs.{gating_output}'="
            f"{gating_value!r}, expected the literal string 'false'"
        )

    count_raw = outputs.get(FILE_COUNT_OUTPUT)
    if count_raw is None:
        return False, (
            f"{job}=skipped, but '{CHANGES_JOB}.outputs.{FILE_COUNT_OUTPUT}' is missing entirely"
        )
    try:
        count = int(count_raw)
    except (TypeError, ValueError):
        return False, (
            f"{job}=skipped, but '{CHANGES_JOB}.outputs.{FILE_COUNT_OUTPUT}'="
            f"{count_raw!r} is not a valid integer"
        )
    if count <= 0:
        return False, (
            f"{job}=skipped, and '{CHANGES_JOB}.outputs.{FILE_COUNT_OUTPUT}'="
            f"{count} -- the classifier saw zero changed files, which is a "
            "broken/empty diff range, not a legitimate 'nothing relevant "
            "changed' verdict"
        )

    return True, (
        f"{job}: skip verified safe ({CHANGES_JOB}.outputs.{gating_output}="
        f"'false', {CHANGES_JOB}.outputs.{FILE_COUNT_OUTPUT}={count})"
    )


def evaluate(
    needs: dict[str, dict],
    unconditional_skip_ok: dict[str, str] | None = None,
    classifier_gated: dict[str, str] | None = None,
) -> GateResult:
    """Decide whether every dependency in ``needs`` is green enough to pass.

    ``needs`` is the raw ``needs`` context GitHub Actions hands the job: one
    entry per job actually listed in THIS job's own ``needs:``, each at least
    carrying a ``result`` key (``success`` / ``failure`` / ``cancelled`` /
    ``skipped``), and -- for `changes` -- an ``outputs`` map.

    Fails closed: an empty ``needs`` map, an allowlist entry naming a job
    that is not present in ``needs`` (stale -- the job was probably renamed
    and the real one is now unguarded), a ``cancelled`` result, and an
    empty/absent ``result`` all count as failures. Nothing ever passes by
    falling through an unrecognised case.
    """
    unconditional_skip_ok = (
        UNCONDITIONAL_SKIP_OK
        if unconditional_skip_ok is None
        else unconditional_skip_ok
    )
    classifier_gated = (
        CLASSIFIER_GATED if classifier_gated is None else classifier_gated
    )

    problems: list[str] = []
    justifications: list[str] = []

    if not needs:
        problems.append(
            "needs map is empty -- this job has no declared dependencies to "
            "verify, which is never correct for a required gate. Check the "
            "`needs:` list on this job in the workflow."
        )
        return GateResult(ok=False, problems=problems)

    for allowlisted_job in unconditional_skip_ok:
        if allowlisted_job not in needs:
            problems.append(
                f"UNCONDITIONAL_SKIP_OK names '{allowlisted_job}', which is "
                "not a declared dependency of this job (absent from "
                "`needs`). The allowlist is stale -- most likely the job "
                "was renamed -- and the real job it used to cover is now "
                "unguarded."
            )
    for gated_job in classifier_gated:
        if gated_job not in needs:
            problems.append(
                f"CLASSIFIER_GATED names '{gated_job}', which is not a "
                "declared dependency of this job (absent from `needs`). "
                "The mapping is stale -- most likely the job was renamed -- "
                "and the real job it used to cover is now unguarded."
            )

    for job, info in needs.items():
        result = info.get("result") if isinstance(info, dict) else None
        if result == "success":
            continue
        if result == "skipped" and job in unconditional_skip_ok:
            continue
        if result == "skipped" and job in classifier_gated:
            safe, message = verify_classifier_gated_skip(
                job, classifier_gated[job], needs
            )
            if safe:
                justifications.append(message)
                continue
            problems.append(message)
            continue
        problems.append(f"{job}={result!r}")

    return GateResult(ok=not problems, problems=problems, justifications=justifications)


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
    # a red (or unexpectedly green) gate can see exactly what every
    # dependency reported, including `changes`'s outputs.
    print("Dependency result map:")
    print(json.dumps(needs, indent=2, sort_keys=True))

    result = evaluate(needs, UNCONDITIONAL_SKIP_OK, CLASSIFIER_GATED)
    for justification in result.justifications:
        print(f"  {justification}")

    if not result.ok:
        print("::error::required-pr-gate FAILED:")
        for problem in result.problems:
            print(f"::error::  {problem}")
        return 1

    print(
        "required-pr-gate passed: every dependency succeeded, or skipped for "
        "a verified-safe reason."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
