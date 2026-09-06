"""Behavioural tests for the ``principal-target-authorization`` CI gate.

The gate decides who may open a pull request against a protected branch. It is
a required status check, so a regression here either locks every author out or
— worse — lets an unintended principal through silently. The logic lives in a
shell block inside a workflow file, which nothing else executes, so these tests
extract that exact block and run it against synthesised GitHub event contexts.

The gate is fail-closed by design: anything it cannot positively verify must
deny. Most cases below therefore assert denial.
"""

from __future__ import annotations

import subprocess
import textwrap
from pathlib import Path

import pytest
import yaml

WORKFLOW = (
    Path(__file__).resolve().parents[2]
    / ".github"
    / "workflows"
    / "protected-target-authorization.yml"
)

REPO = "Traigent/Traigent"
DEPENDABOT = "dependabot[bot]"
PRINCIPAL = "nimrodbusany"

# A real `github-actions` ecosystem bump: Dependabot authored it, and every path
# is a CI workflow. The path alone does not authorise it — the LINES do. Each of
# these files changes exactly one `uses:` reference and nothing else, which is
# the shape the workflow lane admits.
GITHUB_ACTIONS_BUMP = [
    ".github/workflows/architecture-analysis.yml",
    ".github/workflows/cost-coverage.yml",
    ".github/workflows/docs-links.yml",
    ".github/workflows/examples-smoke.yml",
    ".github/workflows/js-public-parity.yml",
]

USES_ONLY_HUNK = """@@ -21,7 +21,7 @@ jobs:
       runs-on: ubuntu-latest
       steps:
-        uses: actions/checkout@11bd71901bbe5b1630ceea73d27597364c9af683 # v4.2.2
+        uses: actions/checkout@08c6903cd8c0fde910a37f88322edcfb5dd907a8 # v5.0.0
         with:
           fetch-depth: 0"""

ACTION_BUMP_PATCHES = dict.fromkeys(GITHUB_ACTIONS_BUMP, USES_ONLY_HUNK)

PIP_BUMP = ["pyproject.toml", "uv.lock", "requirements/base.txt"]


def _gate_script() -> str:
    workflow = yaml.safe_load(WORKFLOW.read_text())
    steps = workflow["jobs"]["principal-target-authorization"]["steps"]
    return steps[0]["run"]


def _run(
    tmp_path: Path,
    files: list[str] | None = None,
    gh_fails: bool = False,
    patches: dict[str, str] | None = None,
    statuses: dict[str, str] | None = None,
    hunk_files: list[str] | None = None,
    hunks_fail: bool = False,
    **env,
):
    """Execute the gate with a stubbed ``gh`` so no network call is made.

    The gate makes up to two calls to the files API and the stub answers both:

    * the name enumeration (``--jq '.[].filename'``) -> one path per line;
    * the hunk enumeration (``--jq ... | @tsv``, made only when a workflow
      path is present) -> ``filename<TAB>status<TAB>patch``, with the patch
      escaped the way ``@tsv`` escapes it, one record per line.

    ``hunk_files`` defaults to ``files`` and exists so a test can withhold a
    record and prove the gate denies a path it could not line-verify.
    """
    bindir = tmp_path / "bin"
    bindir.mkdir(exist_ok=True)

    names = tmp_path / "names.txt"
    names.write_text("".join(f"{f}\n" for f in (files or [])))

    def _tsv(text: str) -> str:
        return text.replace("\\", "\\\\").replace("\t", "\\t").replace("\n", "\\n")

    records = []
    for f in (files or []) if hunk_files is None else hunk_files:
        status = (statuses or {}).get(f, "modified")
        records.append(f"{f}\t{status}\t{_tsv((patches or {}).get(f, ''))}")
    tsv = tmp_path / "hunks.tsv"
    tsv.write_text("".join(r + "\n" for r in records))

    stub = bindir / "gh"
    stub.write_text(
        textwrap.dedent(
            f"""\
            #!/bin/bash
            if printf '%s' "$*" | grep -Fq '@tsv'; then
              {"exit 1" if hunks_fail else ""}
              cat {tsv}
              exit 0
            fi
            {"exit 1" if gh_fails else ""}
            cat {names}
            """
        )
    )
    stub.chmod(0o755)

    script = tmp_path / "gate.sh"
    script.write_text(_gate_script())

    context = {
        "PATH": f"{bindir}:/usr/bin:/bin",
        "EVENT_NAME": "pull_request_target",
        "PR_NUMBER": "1",
        "BASE_REPO": REPO,
        "HEAD_REPO": REPO,
        "PRINCIPAL_ENGINEER": PRINCIPAL,
        "DEPENDABOT": DEPENDABOT,
        "GH_TOKEN": "stub",
        "PR_CHANGED_FILES": str(len(files or [])),
    }
    context.update({k: str(v) for k, v in env.items()})
    return subprocess.run(
        ["bash", str(script)], env=context, capture_output=True, text=True
    )


def _allows(*args, **kwargs) -> bool:
    return _run(*args, **kwargs).returncode == 0


# --------------------------------------------------------------------------
# Authorised paths
# --------------------------------------------------------------------------


def test_principal_may_change_anything(tmp_path):
    """The Principal Engineer is unconditionally authorised."""
    assert _allows(
        tmp_path,
        files=["traigent/api/decorators.py"],
        PR_AUTHOR=PRINCIPAL,
        PR_AUTHOR_TYPE="User",
        TRIGGERING_ACTOR=PRINCIPAL,
    )


def test_merge_group_reproduces_the_context(tmp_path):
    """A merge-group run has no PR author and must still report the context."""
    assert _allows(
        tmp_path,
        EVENT_NAME="merge_group",
        PR_AUTHOR="",
        PR_AUTHOR_TYPE="",
        TRIGGERING_ACTOR="",
        PR_CHANGED_FILES="",
    )


def test_dependabot_manifest_only_is_authorised(tmp_path):
    """The one case the Dependabot lane exists to permit."""
    assert _allows(
        tmp_path,
        files=PIP_BUMP,
        PR_AUTHOR=DEPENDABOT,
        PR_AUTHOR_TYPE="Bot",
        TRIGGERING_ACTOR=DEPENDABOT,
    )


def test_principal_may_advance_a_dependabot_branch(tmp_path):
    """Rebasing a Dependabot PR as the principal must not deny it."""
    assert _allows(
        tmp_path,
        files=PIP_BUMP,
        PR_AUTHOR=DEPENDABOT,
        PR_AUTHOR_TYPE="Bot",
        TRIGGERING_ACTOR=PRINCIPAL,
    )


# --------------------------------------------------------------------------
# Denials: path allowlist
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "paths,reason",
    [
        (["Dockerfile"], "docker ecosystem edits image build inputs"),
        (["setup.py"], "install scripts execute at install time"),
        (["Makefile"], "build entrypoints execute at build time"),
        (["scripts/ci/install_sdk_requirements.sh"], "shell runs in CI"),
        (["Dockerfile.dev"], "any Dockerfile variant, not just the default name"),
    ],
)
def test_dependabot_denied_outside_the_manifest_allowlist(tmp_path, paths, reason):
    """Bot identity does not authorise code-execution surfaces."""
    assert not _allows(
        tmp_path,
        files=paths,
        PR_AUTHOR=DEPENDABOT,
        PR_AUTHOR_TYPE="Bot",
        TRIGGERING_ACTOR=DEPENDABOT,
    ), reason


# --------------------------------------------------------------------------
# Denials: identity
# --------------------------------------------------------------------------


def test_login_alone_is_not_identity(tmp_path):
    """An account whose login mimics Dependabot but is not a Bot is denied."""
    assert not _allows(
        tmp_path,
        files=PIP_BUMP,
        PR_AUTHOR=DEPENDABOT,
        PR_AUTHOR_TYPE="User",
        TRIGGERING_ACTOR=DEPENDABOT,
    )


def test_fork_head_is_denied(tmp_path):
    """A head branch outside the base repository cannot ride the bot lane."""
    assert not _allows(
        tmp_path,
        files=PIP_BUMP,
        PR_AUTHOR=DEPENDABOT,
        PR_AUTHOR_TYPE="Bot",
        TRIGGERING_ACTOR=DEPENDABOT,
        HEAD_REPO="attacker/Traigent",
    )


def test_other_writer_advancing_the_branch_is_denied(tmp_path):
    """A repo writer may push to a Dependabot branch; the author stays the bot.

    Without an acting-identity check that writer's commits would inherit the
    exemption, so the gate must deny.
    """
    assert not _allows(
        tmp_path,
        files=PIP_BUMP,
        PR_AUTHOR=DEPENDABOT,
        PR_AUTHOR_TYPE="Bot",
        TRIGGERING_ACTOR="some-other-writer",
    )


def test_unrelated_contributor_is_denied(tmp_path):
    assert not _allows(
        tmp_path,
        files=PIP_BUMP,
        PR_AUTHOR="outside-contributor",
        PR_AUTHOR_TYPE="User",
        TRIGGERING_ACTOR="outside-contributor",
    )


# --------------------------------------------------------------------------
# Denials: the diff cannot be proven
# --------------------------------------------------------------------------


def test_unreadable_diff_denies(tmp_path):
    """If the files API fails, the gate has no basis to authorise."""
    assert not _allows(
        tmp_path,
        files=PIP_BUMP,
        gh_fails=True,
        PR_AUTHOR=DEPENDABOT,
        PR_AUTHOR_TYPE="Bot",
        TRIGGERING_ACTOR=DEPENDABOT,
    )


def test_truncated_diff_denies(tmp_path):
    """Fewer files enumerated than the PR reports means an incomplete view."""
    assert not _allows(
        tmp_path,
        files=PIP_BUMP,
        PR_AUTHOR=DEPENDABOT,
        PR_AUTHOR_TYPE="Bot",
        TRIGGERING_ACTOR=DEPENDABOT,
        PR_CHANGED_FILES="42",
    )


def test_diff_beyond_api_ceiling_denies(tmp_path):
    """The files API caps at 3000; a larger diff cannot be enumerated."""
    assert not _allows(
        tmp_path,
        files=PIP_BUMP,
        PR_AUTHOR=DEPENDABOT,
        PR_AUTHOR_TYPE="Bot",
        TRIGGERING_ACTOR=DEPENDABOT,
        PR_CHANGED_FILES="3001",
    )


# --------------------------------------------------------------------------
# The workflow lane: the LINE is the unit of authorization, not the path
# --------------------------------------------------------------------------
#
# A `github-actions` ecosystem bump edits `uses:` references and nothing else.
# The lane admits exactly that shape. Every test below that changes one other
# kind of line in the SAME diff must deny, because a workflow change beyond an
# action pin alters what CI executes or with what privileges.


def _hunk(*changed: str) -> str:
    """A patch whose only +/- lines are the ones given."""
    return "\n".join(
        ["@@ -10,7 +10,7 @@ jobs:", "       steps:", *changed, "         with:"]
    )


def test_dependabot_action_bump_is_authorised(tmp_path):
    """The case this lane exists to permit: five files, every changed line a `uses:`."""
    assert _allows(
        tmp_path,
        files=GITHUB_ACTIONS_BUMP,
        patches=ACTION_BUMP_PATCHES,
        PR_AUTHOR=DEPENDABOT,
        PR_AUTHOR_TYPE="Bot",
        TRIGGERING_ACTOR=DEPENDABOT,
    )


def test_action_bump_alongside_manifests_is_authorised(tmp_path):
    """A mixed diff is fine as long as every path clears its own rule."""
    wf = ".github/workflows/ci.yml"
    assert _allows(
        tmp_path,
        files=PIP_BUMP + [wf],
        patches={wf: USES_ONLY_HUNK},
        PR_AUTHOR=DEPENDABOT,
        PR_AUTHOR_TYPE="Bot",
        TRIGGERING_ACTOR=DEPENDABOT,
    )


@pytest.mark.parametrize(
    "changed,reason",
    [
        (
            ["-        run: pytest -q", "+        run: pytest -q --exitfirst"],
            "a `run:` body is arbitrary code execution",
        ),
        (
            ["-      contents: read", "+      contents: write"],
            "a `permissions:` grant widens the token",
        ),
        (
            ["-        if: github.actor == 'nimrodbusany'", "+        if: always()"],
            "an `if:` condition decides whether a guard runs at all",
        ),
        (
            [
                "-          TOKEN: ${{ secrets.LOW }}",
                "+          TOKEN: ${{ secrets.HIGH }}",
            ],
            "an `env:` value can swap which secret reaches the step",
        ),
        (
            ["-  pull_request:", "+  pull_request_target:"],
            "an `on:` trigger changes the trust context of the whole workflow",
        ),
        (
            ["+        uses: ./.github/actions/local-thing"],
            "an unpinned reference names no version and cannot be an action bump",
        ),
    ],
)
def test_workflow_change_beyond_a_uses_line_is_denied(tmp_path, changed, reason):
    """One non-`uses:` changed line denies the pull request, bump or not."""
    wf = ".github/workflows/ci.yml"
    bump = "-        uses: actions/checkout@v4\n+        uses: actions/checkout@v5"
    assert not _allows(
        tmp_path,
        files=[wf],
        patches={wf: _hunk(*bump.split("\n"), *changed)},
        PR_AUTHOR=DEPENDABOT,
        PR_AUTHOR_TYPE="Bot",
        TRIGGERING_ACTOR=DEPENDABOT,
    ), reason


@pytest.mark.parametrize("status", ["added", "removed", "renamed", "copied", ""])
def test_workflow_file_not_modified_in_place_is_denied(tmp_path, status):
    """Only an in-place modification can be line-verified.

    A new workflow file's every line is an addition, so a file consisting of
    nothing but `uses:` lines would otherwise pass while introducing a workflow
    the Principal Engineer never saw.
    """
    wf = ".github/workflows/new.yml"
    assert not _allows(
        tmp_path,
        files=[wf],
        patches={wf: USES_ONLY_HUNK},
        statuses={wf: status},
        PR_AUTHOR=DEPENDABOT,
        PR_AUTHOR_TYPE="Bot",
        TRIGGERING_ACTOR=DEPENDABOT,
    )


def test_workflow_without_a_hunk_is_denied(tmp_path):
    """No patch means nothing to verify, so there is no basis to authorise."""
    wf = ".github/workflows/ci.yml"
    assert not _allows(
        tmp_path,
        files=[wf],
        patches={wf: ""},
        PR_AUTHOR=DEPENDABOT,
        PR_AUTHOR_TYPE="Bot",
        TRIGGERING_ACTOR=DEPENDABOT,
    )


def test_workflow_missing_from_the_hunk_stream_is_denied(tmp_path):
    """A path the second call did not return never reaches verified_workflows.

    This is the fail-closed seam between the two API calls: the allowlist admits
    a workflow path only by name-match against what the lane actually verified.
    """
    verified = ".github/workflows/ci.yml"
    withheld = ".github/workflows/release.yml"
    assert not _allows(
        tmp_path,
        files=[verified, withheld],
        patches={verified: USES_ONLY_HUNK},
        hunk_files=[verified],
        PR_AUTHOR=DEPENDABOT,
        PR_AUTHOR_TYPE="Bot",
        TRIGGERING_ACTOR=DEPENDABOT,
    )


def test_unreadable_hunks_deny(tmp_path):
    """If the hunk call fails, a workflow change cannot be proven safe."""
    wf = ".github/workflows/ci.yml"
    assert not _allows(
        tmp_path,
        files=[wf],
        patches={wf: USES_ONLY_HUNK},
        hunks_fail=True,
        PR_AUTHOR=DEPENDABOT,
        PR_AUTHOR_TYPE="Bot",
        TRIGGERING_ACTOR=DEPENDABOT,
    )


def test_composite_action_files_use_the_same_lane(tmp_path):
    """`.github/actions/**` is a workflow surface too, and is line-verified."""
    action = ".github/actions/setup/action.yml"
    assert _allows(
        tmp_path,
        files=[action],
        patches={action: USES_ONLY_HUNK},
        PR_AUTHOR=DEPENDABOT,
        PR_AUTHOR_TYPE="Bot",
        TRIGGERING_ACTOR=DEPENDABOT,
    )
    assert not _allows(
        tmp_path,
        files=[action],
        patches={action: _hunk("-      shell: bash", "+      shell: pwsh")},
        PR_AUTHOR=DEPENDABOT,
        PR_AUTHOR_TYPE="Bot",
        TRIGGERING_ACTOR=DEPENDABOT,
    )


def test_the_workflow_lane_does_not_widen_the_identity_checks(tmp_path):
    """A perfect action bump still needs the bot identity and a same-repo head."""
    wf = ".github/workflows/ci.yml"
    common = {"files": [wf], "patches": {wf: USES_ONLY_HUNK}, "PR_AUTHOR": DEPENDABOT}
    assert not _allows(
        tmp_path, **common, PR_AUTHOR_TYPE="User", TRIGGERING_ACTOR=DEPENDABOT
    )
    assert not _allows(
        tmp_path, **common, PR_AUTHOR_TYPE="Bot", TRIGGERING_ACTOR="some-other-writer"
    )
    assert not _allows(
        tmp_path,
        **common,
        PR_AUTHOR_TYPE="Bot",
        TRIGGERING_ACTOR=DEPENDABOT,
        HEAD_REPO="attacker/Traigent",
    )
