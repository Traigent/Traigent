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

import shutil
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
# is a CI workflow. This is the case that motivated the allowlist — bot identity
# alone must not authorise it.
GITHUB_ACTIONS_BUMP = [
    ".github/workflows/architecture-analysis.yml",
    ".github/workflows/cost-coverage.yml",
    ".github/workflows/docs-links.yml",
    ".github/workflows/examples-smoke.yml",
    ".github/workflows/js-public-parity.yml",
]

PIP_BUMP = ["pyproject.toml", "uv.lock", "requirements/base.txt"]


def _gate_script() -> str:
    workflow = yaml.safe_load(WORKFLOW.read_text())
    steps = workflow["jobs"]["principal-target-authorization"]["steps"]
    return steps[0]["run"]


def _run(tmp_path: Path, files: list[str] | None = None, gh_fails: bool = False, **env):
    """Execute the gate with a stubbed ``gh`` so no network call is made."""
    bindir = tmp_path / "bin"
    bindir.mkdir(exist_ok=True)
    stub = bindir / "gh"
    stub.write_text(
        textwrap.dedent(
            f"""\
            #!/bin/bash
            {"exit 1" if gh_fails else ""}
            printf '%s\\n' {" ".join(f"'{f}'" for f in (files or []))}
            """
        )
    )
    stub.chmod(0o755)

    script = tmp_path / "gate.sh"
    script.write_text(_gate_script())

    context = {
        "PATH": f"{bindir}:{shutil.which('bash') and '/usr/bin:/bin'}",
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
        (GITHUB_ACTIONS_BUMP, "github-actions ecosystem edits CI definitions"),
        (["Dockerfile"], "docker ecosystem edits image build inputs"),
        (["setup.py"], "install scripts execute at install time"),
        (["Makefile"], "build entrypoints execute at build time"),
        (["scripts/ci/install_sdk_requirements.sh"], "shell runs in CI"),
        (PIP_BUMP + [".github/workflows/ci.yml"], "one bad path taints the diff"),
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
