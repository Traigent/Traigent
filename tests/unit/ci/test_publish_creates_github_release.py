"""Contract checks for the publish workflow's GitHub Release step.

GitHub's "Latest" release drifted from PyPI (stuck at v0.12.0 while PyPI
published up to v0.27.0, issue #2267) because `.github/workflows/publish.yml`
verified PyPI installs but never created a GitHub Release. This test pins the
structure of the fix so the drift cannot recur silently: a dedicated job that
runs only after a real, verified tag-triggered publish, creates or (on a
rerun) idempotently updates the Release for the exact published tag, requests
no more than the `contents: write` permission it needs, and asserts the tag,
PyPI version, and GitHub Latest all agree before the run is considered green.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import stat
import subprocess
from pathlib import Path
from typing import Any

import pytest
import yaml

_REPO_ROOT = Path(__file__).resolve().parents[3]
_WORKFLOW_PATH = _REPO_ROOT / ".github" / "workflows" / "publish.yml"


def _workflow() -> dict[str, Any]:
    workflow = yaml.safe_load(_WORKFLOW_PATH.read_text(encoding="utf-8"))
    assert isinstance(workflow, dict)
    return workflow


def _release_job() -> dict[str, Any]:
    return _workflow()["jobs"]["github-release"]


def test_workflow_top_level_permission_stays_read_only() -> None:
    workflow = _workflow()
    assert workflow["permissions"] == {"contents": "read"}


def test_release_job_only_runs_after_a_verified_tag_publish() -> None:
    job = _release_job()
    assert set(job["needs"]) == {"publish", "verify-publication"}
    condition = job["if"]
    assert "needs.publish.result == 'success'" in condition
    assert "needs.verify-publication.result == 'success'" in condition
    assert "startsWith(github.ref, 'refs/tags/v')" in condition


def test_release_job_also_runs_for_a_workflow_dispatch_pypi_publish() -> None:
    # Regression for #2267 recurring on the workflow_dispatch->PyPI path:
    # in this repo's history, 12 of 27 successful publishes were manual
    # dispatches straight to production PyPI (never a tag push), and the
    # `publish` job's own "Publish to PyPI" step already treats that path as
    # a real production release (same condition, `publish.yml` line ~199).
    # The release job must mirror it, not just the tag-push half.
    job = _release_job()
    condition = job["if"]
    assert "github.event_name == 'workflow_dispatch'" in condition
    assert "inputs.environment == 'pypi'" in condition

    publish_job = _workflow()["jobs"]["publish"]
    publish_step = next(
        step for step in publish_job["steps"] if step.get("name") == "Publish to PyPI"
    )
    # Same production-publish test the upstream step already uses, so the
    # release job can never drift narrower (or wider) than it again.
    assert publish_step["if"] == (
        "startsWith(github.ref, 'refs/tags/v') || "
        "(github.event_name == 'workflow_dispatch' && inputs.environment == 'pypi')"
    )
    assert "startsWith(github.ref, 'refs/tags/v')" in publish_step["if"]
    assert "github.event_name == 'workflow_dispatch'" in publish_step["if"]
    assert "inputs.environment == 'pypi'" in publish_step["if"]


def test_release_tag_is_derived_from_the_published_version_not_the_ref() -> None:
    # A workflow_dispatch production publish never sets github.ref to a tag
    # (github.ref_name would be a branch name), so the tag used to create
    # the Release must come from the same version source verify-publication
    # already trusts: needs.publish.outputs.package_version.
    job = _release_job()
    assert job["env"]["TAG"] == "v${{ needs.publish.outputs.package_version }}"


def test_release_job_requests_only_contents_write() -> None:
    job = _release_job()
    # Minimum additional permission beyond the workflow-level `contents: read`
    # (see test above): this job is the only writer, and it only writes.
    assert job["permissions"] == {"contents": "write"}


def test_release_job_is_idempotent_create_or_update() -> None:
    job = _release_job()
    steps_text = "\n".join(step.get("run", "") for step in job["steps"])
    assert "gh release view" in steps_text
    assert "gh release edit" in steps_text
    assert "gh release create" in steps_text
    # Not --verify-tag: on the workflow_dispatch->pypi path the derived tag
    # (see test above) has no matching git tag yet, and --verify-tag would
    # abort the release rather than mint one. --target lets `gh release
    # create` mint the tag from this run's commit; for a real tag push the
    # tag already exists there, so --target is a no-op.
    assert "--target" in steps_text
    assert "--verify-tag" not in steps_text


def test_release_job_asserts_tag_pypi_and_latest_agree() -> None:
    job = _release_job()
    steps_text = "\n".join(step.get("run", "") for step in job["steps"])
    assert "EXPECTED_VERSION" in steps_text
    assert "gh release view --repo" in steps_text
    assert "json tagName" in steps_text
    # The published package version flows from the `publish` job's output,
    # which is where PyPI's actual version is read from pyproject.toml.
    assert (
        job["env"]["EXPECTED_VERSION"] == "${{ needs.publish.outputs.package_version }}"
    )


def test_release_job_authenticates_by_env_not_by_writing_a_credential_to_disk() -> None:
    """``gh`` gets its token from the job env, never from ``gh auth login``.

    ``gh auth login --with-token`` persists the token to
    ``~/.config/gh/hosts.yml`` on the runner -- a credential at rest for the
    life of the job -- whereas ``GH_TOKEN`` in the job env is read directly by
    ``gh`` and never touches the filesystem. Both are masked in logs, so the
    difference is invisible in a run and easy to reintroduce.
    """
    job = _release_job()

    assert job["env"]["GH_TOKEN"] == "${{ github.token }}"

    steps_text = "\n".join(step.get("run", "") for step in job["steps"])
    assert "gh auth login" not in steps_text, (
        "use the GH_TOKEN job env instead of gh auth login, which writes the "
        "credential to the runner's gh config"
    )


def _normalized_release_condition() -> str:
    """The release job's `if`, whitespace-normalized for exact comparison.

    Collapsing runs of whitespace is not enough on its own: a space directly
    inside a parenthesis (``( startsWith(...``) is a harmless reformat that
    survives the collapse and would fail the exact match below, making CI red
    for a change with no semantic content. Spaces adjacent to a parenthesis are
    therefore dropped too. Everything that carries meaning -- operand order and
    the ``&&``/``||`` operators between them -- is still compared exactly.
    """
    condition = " ".join(_release_job()["if"].split())
    condition = re.sub(r"\(\s+", "(", condition)
    return re.sub(r"\s+\)", ")", condition)


def test_release_condition_is_exactly_the_intended_expression() -> None:
    """Pin the whole expression, not substrings of it.

    Substring assertions cannot see the operators between the parts they match,
    so two mutations survived them: flipping the `&&` joining the two
    `needs.*.result == 'success'` checks to `||` (the job then releases when
    `publish` FAILED), and widening the environment check so a TestPyPI dispatch
    also cuts a production release. Both leave every cited substring present.
    """
    assert _normalized_release_condition() == (
        "always() && needs.publish.result == 'success' && "
        "needs.verify-publication.result == 'success' && "
        "(startsWith(github.ref, 'refs/tags/v') || "
        "(github.event_name == 'workflow_dispatch' && inputs.environment == 'pypi'))"
    )


def test_release_condition_requires_both_dependencies_to_have_succeeded() -> None:
    """The two success checks are ANDed, never ORed."""
    condition = _normalized_release_condition()
    joined = (
        "needs.publish.result == 'success' && "
        "needs.verify-publication.result == 'success'"
    )
    assert joined in condition
    assert "needs.publish.result == 'success' ||" not in condition


def test_release_is_gated_to_the_production_environment_only() -> None:
    """A TestPyPI dispatch must not cut a production GitHub Release."""
    condition = _normalized_release_condition()
    assert "inputs.environment == 'pypi'" in condition
    assert "testpypi" not in condition
    assert "inputs.environment != " not in condition


def test_release_job_verifies_the_commit_is_on_main() -> None:
    """The publish job's on-main check is tag-only, so the release job needs its own.

    On the workflow_dispatch->pypi path `github.ref` is a branch, so
    `startsWith(github.ref, 'refs/tags/v')` is false and the publish job's
    "Verify tag is on main" step is skipped. Creating the release with
    `--target $GITHUB_SHA` would then mint a v<version> tag on a non-main commit
    and mark it Latest.
    """
    steps = _release_job()["steps"]
    guard = next((s for s in steps if "on main" in s.get("name", "")), None)
    assert guard is not None, "the release job must verify its commit is on main"

    assert "compare/main..." in guard["run"], (
        "the check must compare against main, not some other ref"
    )
    assert "set -euo pipefail" in guard["run"], (
        "without it, a failing command in the middle of the guard does not "
        "abort the step and the release is cut anyway"
    )

    names = [s.get("name", "") for s in steps]
    create_index = next(
        i for i, n in enumerate(names) if n.startswith("Create or update")
    )
    assert names.index(guard["name"]) < create_index, (
        "the on-main check must run BEFORE the release is created"
    )


# ---------------------------------------------------------------------------
# Behavioural checks on the on-main guard.
#
# Asserting that the step's text contains "identical|behind" cannot see what
# the case arm DOES with it: widening acceptance to "identical|behind|ahead"
# still contains that substring, so the weaker assertion stayed green while
# the guard stopped guarding. These run the step's own script instead, with
# `gh` stubbed to report each compare status the API can return.
# ---------------------------------------------------------------------------


def _on_main_guard_script() -> str:
    steps = _release_job()["steps"]
    guard = next(s for s in steps if "on main" in s.get("name", ""))
    return guard["run"]


GUARD_SHA = "0123456789abcdef0123456789abcdef01234567"


def _run_on_main_guard(tmp_path: Path, gh_stdout: str, gh_exit: int = 0):
    """Execute the guard step with `gh` stubbed, and return the CompletedProcess.

    The stub RECORDS its arguments to ``tmp_path/gh-args.log``. An earlier
    version just printed a canned status and ignored ``$@``, which made the
    behavioural tests blind to WHICH ref the guard compares: mutating
    ``compare/main...$GITHUB_SHA`` to ``compare/main...main`` neutered the
    guard entirely and the whole suite stayed green, because the only
    assertion about the ref was a substring the mutant still contained.
    ``.args`` on the result carries the recorded invocations.
    """
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    args_log = tmp_path / "gh-args.log"
    fake_gh = bin_dir / "gh"
    fake_gh.write_text(
        "#!/usr/bin/env bash\n"
        f'printf "%s\\n" "$*" >> {args_log!s}\n'
        f"printf '%s\\n' {gh_stdout!r}\n"
        f"exit {gh_exit}\n",
        encoding="utf-8",
    )
    fake_gh.chmod(fake_gh.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)

    script = tmp_path / "guard.sh"
    script.write_text(_on_main_guard_script(), encoding="utf-8")

    env = dict(os.environ)
    env["PATH"] = f"{bin_dir}{os.pathsep}{env['PATH']}"
    env["REPO"] = "Traigent/Traigent"
    env["GITHUB_SHA"] = GUARD_SHA
    env["TAG"] = "v0.11.4"
    result = subprocess.run(
        ["bash", str(script)],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    result.gh_calls = (  # type: ignore[attr-defined]
        args_log.read_text(encoding="utf-8").splitlines() if args_log.exists() else []
    )
    return result


@pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")
@pytest.mark.parametrize("status", ["identical", "behind"])
def test_on_main_guard_accepts_a_commit_that_is_on_main(tmp_path, status) -> None:
    """`identical` is main's tip; `behind` is an ancestor of it. Both are on main."""
    result = _run_on_main_guard(tmp_path, status)
    assert result.returncode == 0, (
        f"compare status {status!r} is on main and must be accepted; "
        f"stdout={result.stdout!r} stderr={result.stderr!r}"
    )


@pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")
@pytest.mark.parametrize("status", ["ahead", "diverged"])
def test_on_main_guard_rejects_a_commit_that_is_not_on_main(tmp_path, status) -> None:
    """The mutation the old substring assertion could not catch.

    `ahead` means the commit carries work main does not have, and `diverged`
    means the two have split -- in both cases the release commit is NOT on
    main, which is the whole thing this step exists to refuse. Widening the
    case arm to `identical|behind|ahead` makes this test red; it left the
    previous `"identical|behind" in run` assertion green.
    """
    result = _run_on_main_guard(tmp_path, status)
    assert result.returncode != 0, (
        f"compare status {status!r} is NOT on main and must abort the release; "
        f"stdout={result.stdout!r}"
    )
    assert "::error::" in (result.stdout + result.stderr), (
        "the refusal must surface as a GitHub Actions error annotation"
    )


@pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")
def test_on_main_guard_fails_closed_on_an_unexpected_status(tmp_path) -> None:
    """An unrecognized status is not a licence to release."""
    result = _run_on_main_guard(tmp_path, "something_new_from_the_api")
    assert result.returncode != 0


@pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")
def test_on_main_guard_fails_closed_when_the_compare_api_errors(tmp_path) -> None:
    """A failing `gh api` call must abort, never fall through to the release.

    This is the fail-closed property for a policy surface: if we cannot
    establish that the commit is on main, we do not publish a Release for it.
    """
    result = _run_on_main_guard(tmp_path, "", gh_exit=1)
    assert result.returncode != 0, (
        "a compare API failure left the guard passing, so an unverified commit "
        "would get a GitHub Release"
    )


@pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")
def test_on_main_guard_fails_closed_on_an_empty_status(tmp_path) -> None:
    """`gh ... --jq .status` printing nothing must not be read as acceptance."""
    result = _run_on_main_guard(tmp_path, "")
    assert result.returncode != 0


# ---------------------------------------------------------------------------
# Tag identity.
#
# Review found the on-main guard validates $GITHUB_SHA and says nothing about
# where $TAG points, while the CLI reuses an existing tag rather than
# repointing it. Name parity is not commit identity.
# ---------------------------------------------------------------------------

PUBLISHED_SHA = "0123456789abcdef0123456789abcdef01234567"
OTHER_SHA = "fedcba9876543210fedcba9876543210fedcba98"


def _tag_identity_script() -> str:
    steps = _release_job()["steps"]
    guard = next(s for s in steps if "existing tag" in s.get("name", ""))
    return guard["run"]


def _run_tag_identity_guard(tmp_path: Path, responses: dict[str, str]):
    """Execute the guard with `gh` stubbed to answer per API path."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)

    cases = "\n".join(
        f'  *"{path}"*) printf %s {payload!r}; exit 0 ;;'
        for path, payload in responses.items()
    )
    fake_gh = bin_dir / "gh"
    fake_gh.write_text(
        "#!/usr/bin/env bash\n"
        'args="$*"\n'
        'case "$args" in\n'
        f"{cases}\n"
        "  *) exit 1 ;;\n"
        "esac\n",
        encoding="utf-8",
    )
    fake_gh.chmod(fake_gh.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)

    script = tmp_path / "tag_guard.sh"
    script.write_text(_tag_identity_script(), encoding="utf-8")

    env = dict(os.environ)
    env["PATH"] = f"{bin_dir}{os.pathsep}{env['PATH']}"
    env["REPO"] = "Traigent/Traigent"
    env["GITHUB_SHA"] = PUBLISHED_SHA
    env["TAG"] = "v0.11.4"
    return subprocess.run(
        ["bash", str(script)], env=env, capture_output=True, text=True, timeout=60
    )


@pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")
@pytest.mark.skipif(shutil.which("jq") is None, reason="needs jq")
def test_tag_guard_accepts_a_tag_already_on_the_published_commit(tmp_path) -> None:
    result = _run_tag_identity_guard(
        tmp_path,
        {
            "git/ref/tags": json.dumps(
                {"object": {"sha": PUBLISHED_SHA, "type": "commit"}}
            )
        },
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")
@pytest.mark.skipif(shutil.which("jq") is None, reason="needs jq")
def test_tag_guard_refuses_a_tag_pointing_somewhere_else(tmp_path) -> None:
    """The defect: version X published from B while vX still points at A."""
    result = _run_tag_identity_guard(
        tmp_path,
        {"git/ref/tags": json.dumps({"object": {"sha": OTHER_SHA, "type": "commit"}})},
    )
    assert result.returncode != 0
    assert "::error::" in (result.stdout + result.stderr)


@pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")
@pytest.mark.skipif(shutil.which("jq") is None, reason="needs jq")
def test_tag_guard_peels_an_annotated_tag_before_comparing(tmp_path) -> None:
    """An annotated tag's ref points at a tag object, not the commit.

    Without peeling, every annotated tag would compare unequal and block a
    legitimate publish -- a failure mode worse than the one being prevented.
    """
    result = _run_tag_identity_guard(
        tmp_path,
        {
            "git/ref/tags": json.dumps(
                {
                    "object": {
                        "sha": "aaaabbbbccccddddeeeeffff0000111122223333",
                        "type": "tag",
                    }
                }
            ),
            # The peel call is `gh api ... --jq '.object.sha'`, so gh itself
            # does the filtering and the stub must return the bare sha here,
            # not the envelope it returns for the ref lookup above.
            "git/tags": PUBLISHED_SHA,
        },
    )
    assert result.returncode == 0, (
        "an annotated tag pointing at the published commit was rejected: "
        f"{result.stdout!r} {result.stderr!r}"
    )


@pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")
@pytest.mark.skipif(shutil.which("jq") is None, reason="needs jq")
def test_tag_guard_allows_a_tag_that_does_not_exist_yet(tmp_path) -> None:
    """The workflow_dispatch path mints the tag; absence is not a failure."""
    result = _run_tag_identity_guard(tmp_path, {})
    assert result.returncode == 0, result.stderr


# ---------------------------------------------------------------------------
# What the guard ASKS, not just what it does with the answer.
#
# Mutation testing found three survivors, all from the same blind spot: the
# `gh` stub ignored its arguments, so no behavioural test could see which ref
# the guard actually compares. Mutating `compare/main...$GITHUB_SHA` to
# `compare/main...main` turns the guard into an always-pass (a branch is always
# identical to itself) and every test stayed green.
# ---------------------------------------------------------------------------


@pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")
def test_on_main_guard_compares_main_against_the_published_commit(tmp_path) -> None:
    """The comparison must name $GITHUB_SHA, not a branch.

    `compare/main...main` is `identical` by definition, so a guard that asks
    that question can never refuse anything -- it looks like a check and is not
    one. Asserted on the recorded `gh` invocation rather than on the script
    text, because a substring assertion cannot tell the two apart.
    """
    result = _run_on_main_guard(tmp_path, "identical")

    compare_calls = [c for c in result.gh_calls if "compare/" in c]
    assert compare_calls, (
        f"the guard never called `gh api ...compare/...`: {result.gh_calls}"
    )

    call = compare_calls[0]
    assert f"compare/main...{GUARD_SHA}" in call, (
        "the guard must compare main against the commit being published; "
        f"it asked for: {call}"
    )
    assert "compare/main...main" not in call, (
        "comparing main to itself is always `identical`, so the guard would "
        "never refuse anything"
    )


@pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")
def test_on_main_guard_extracts_only_the_status_field(tmp_path) -> None:
    """`--jq .status` is load-bearing.

    Without it `gh api` returns the whole comparison document, and the `case`
    match against `identical|behind` silently stops matching -- the guard then
    refuses every legitimate release, or, depending on the shape, stops
    discriminating. Either way the arm no longer means what it reads as.
    """
    result = _run_on_main_guard(tmp_path, "identical")
    compare_calls = [c for c in result.gh_calls if "compare/" in c]
    assert compare_calls
    assert "--jq" in compare_calls[0] and ".status" in compare_calls[0], (
        f"the guard must ask for just the status field; it asked: {compare_calls[0]}"
    )
