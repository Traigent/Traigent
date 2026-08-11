"""Negative-control tests for scripts/ci/check_required_gate.py.

`required-pr-gate` is a required status check: this script decides whether a
PR is allowed to merge based on what its dependency jobs reported. A
regression here either wrongly blocks every legitimate PR (a docs-only,
Dependabot, or workflow-only change that touches nothing the `changes`
classifier matches), or -- much worse -- silently turns the gate green when a
real test job never ran because the classifier itself is broken. These tests
exercise the decision function directly (fast, precise) plus the actual CLI
entry point with the repo's real config (end-to-end, proves the wiring
works).

Design under test: `UNCONDITIONAL_SKIP_OK` jobs (`changes`, `preflight`,
`schema-types`) may skip with no further checks -- their skip depends only on
unforgeable GitHub event facts. `CLASSIFIER_GATED` jobs (`unit`, `collection`,
`mcp-contract`) may skip ONLY when `verify_classifier_gated_skip` proves it
safe: `changes` succeeded, `code_changed` is literally `'false'`, and the
classifier saw a non-zero changed-file count. The sole zero-file exception is
an explicitly verified ancestry-only topology, reported as the literal
`ancestry_only='true'`; anything else -- `cancelled`, an empty/missing output,
a zero file count without that proof, an unsuccessful `changes` run -- fails
the gate.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "ci" / "check_required_gate.py"

sys.path.insert(0, str(SCRIPT.parent))
import check_required_gate as gate  # noqa: E402  (path insert must precede this)


def _needs(**results: str) -> dict[str, dict[str, str]]:
    """Build a `needs` context map from job=result kwargs (no outputs)."""
    return {name: {"result": result} for name, result in results.items()}


def _run_cli(needs: dict) -> subprocess.CompletedProcess[str]:
    """Invoke the real script exactly as the workflow does: via NEEDS_JSON."""
    env = {**os.environ, "NEEDS_JSON": json.dumps(needs)}
    return subprocess.run(
        [sys.executable, str(SCRIPT)],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )


class TestEvaluateBasics:
    """Generic cases independent of classifier-gated verification."""

    def test_all_success_passes(self) -> None:
        needs = _needs(a="success", b="success", c="success")
        result = gate.evaluate(needs, unconditional_skip_ok={}, classifier_gated={})
        assert result.ok, result.problems
        assert result.problems == []

    def test_non_allowlisted_non_gated_skip_fails_and_names_it(self) -> None:
        needs = _needs(a="success", b="skipped")
        result = gate.evaluate(needs, unconditional_skip_ok={}, classifier_gated={})
        assert not result.ok
        assert any(
            "b" in problem and "skipped" in problem for problem in result.problems
        )

    def test_unconditional_allowlisted_skip_passes(self) -> None:
        needs = _needs(a="success", b="skipped")
        result = gate.evaluate(
            needs, unconditional_skip_ok={"b": "test double"}, classifier_gated={}
        )
        assert result.ok, result.problems

    def test_cancelled_job_fails(self) -> None:
        needs = _needs(a="success", b="cancelled")
        result = gate.evaluate(needs, unconditional_skip_ok={}, classifier_gated={})
        assert not result.ok
        assert any("cancelled" in problem for problem in result.problems)

    def test_cancelled_job_fails_even_if_unconditionally_allowlisted(self) -> None:
        needs = _needs(a="success", b="cancelled")
        result = gate.evaluate(
            needs, unconditional_skip_ok={"b": "test double"}, classifier_gated={}
        )
        assert not result.ok

    def test_absent_result_key_fails(self) -> None:
        needs = {"a": {"result": "success"}, "b": {}}
        result = gate.evaluate(needs, unconditional_skip_ok={}, classifier_gated={})
        assert not result.ok
        assert any(problem.startswith("b=") for problem in result.problems)

    def test_empty_string_result_fails(self) -> None:
        needs = {"a": {"result": "success"}, "b": {"result": ""}}
        result = gate.evaluate(needs, unconditional_skip_ok={}, classifier_gated={})
        assert not result.ok

    def test_stale_unconditional_allowlist_entry_fails(self) -> None:
        needs = _needs(a="success")
        result = gate.evaluate(
            needs,
            unconditional_skip_ok={"renamed-away-job": "stale"},
            classifier_gated={},
        )
        assert not result.ok
        assert any("renamed-away-job" in problem for problem in result.problems)

    def test_stale_classifier_gated_entry_fails(self) -> None:
        needs = _needs(a="success")
        result = gate.evaluate(
            needs,
            unconditional_skip_ok={},
            classifier_gated={"renamed-away-job": "some_output"},
        )
        assert not result.ok
        assert any("renamed-away-job" in problem for problem in result.problems)

    def test_empty_needs_fails(self) -> None:
        result = gate.evaluate({}, unconditional_skip_ok={}, classifier_gated={})
        assert not result.ok
        assert result.problems


class TestClassifierGatedSkipVerification:
    """The seven required negative controls, against the decision function."""

    GATED = {"unit": "code_changed"}

    def _needs(
        self,
        changes_result: str = "success",
        changes_outputs: dict | None = None,
        unit_result: str = "skipped",
    ) -> dict:
        outputs = {} if changes_outputs is None else changes_outputs
        return {
            "changes": {"result": changes_result, "outputs": outputs},
            "unit": {"result": unit_result},
        }

    def test_verified_safe_skip_passes(self) -> None:
        # classifier success + output 'false' + non-zero file count + job skipped -> exit 0 (pass)
        needs = self._needs(
            changes_outputs={"code_changed": "false", "changed_file_count": "3"}
        )
        result = gate.evaluate(
            needs, unconditional_skip_ok={}, classifier_gated=self.GATED
        )
        assert result.ok, result.problems
        assert any("unit" in j and "verified safe" in j for j in result.justifications)

    def test_empty_string_output_fails_naming_output(self) -> None:
        # output empty string, job skipped -> exit 1, naming the output
        needs = self._needs(
            changes_outputs={"code_changed": "", "changed_file_count": "3"}
        )
        result = gate.evaluate(
            needs, unconditional_skip_ok={}, classifier_gated=self.GATED
        )
        assert not result.ok
        assert any("code_changed" in p for p in result.problems)

    def test_missing_output_fails(self) -> None:
        # output missing entirely -> exit 1
        needs = self._needs(changes_outputs={"changed_file_count": "3"})
        result = gate.evaluate(
            needs, unconditional_skip_ok={}, classifier_gated=self.GATED
        )
        assert not result.ok
        assert any(
            "code_changed" in p and "missing entirely" in p for p in result.problems
        )

    def test_non_false_output_value_fails(self) -> None:
        needs = self._needs(
            changes_outputs={"code_changed": "true", "changed_file_count": "3"}
        )
        result = gate.evaluate(
            needs, unconditional_skip_ok={}, classifier_gated=self.GATED
        )
        assert not result.ok

    def test_changes_skipped_fails(self) -> None:
        # `changes` itself skipped, downstream skipped -> exit 1
        needs = self._needs(
            changes_result="skipped",
            changes_outputs={"code_changed": "false", "changed_file_count": "3"},
        )
        result = gate.evaluate(
            needs, unconditional_skip_ok={}, classifier_gated=self.GATED
        )
        assert not result.ok
        assert any("did not succeed" in p for p in result.problems)

    def test_changes_failed_fails(self) -> None:
        # `changes` itself failed, downstream skipped -> exit 1
        needs = self._needs(
            changes_result="failure",
            changes_outputs={"code_changed": "false", "changed_file_count": "3"},
        )
        result = gate.evaluate(
            needs, unconditional_skip_ok={}, classifier_gated=self.GATED
        )
        assert not result.ok
        assert any("did not succeed" in p for p in result.problems)

    def test_zero_file_count_fails(self) -> None:
        # classified file count 0 -> exit 1
        needs = self._needs(
            changes_outputs={"code_changed": "false", "changed_file_count": "0"}
        )
        result = gate.evaluate(
            needs, unconditional_skip_ok={}, classifier_gated=self.GATED
        )
        assert not result.ok
        assert any("zero changed files" in p for p in result.problems)

    def test_verified_ancestry_only_zero_file_count_passes(self) -> None:
        needs = self._needs(
            changes_outputs={
                "py_changed": "false",
                "code_changed": "false",
                "changed_file_count": "0",
                "ancestry_only": "true",
            }
        )
        result = gate.evaluate(
            needs, unconditional_skip_ok={}, classifier_gated=self.GATED
        )
        assert result.ok, result.problems
        assert any("ancestry-only" in j for j in result.justifications)

    def test_false_ancestry_only_does_not_weaken_zero_file_rejection(self) -> None:
        needs = self._needs(
            changes_outputs={
                "py_changed": "false",
                "code_changed": "false",
                "changed_file_count": "0",
                "ancestry_only": "false",
            }
        )
        result = gate.evaluate(
            needs, unconditional_skip_ok={}, classifier_gated=self.GATED
        )
        assert not result.ok
        assert any("zero changed files" in p for p in result.problems)

    def test_ancestry_only_zero_file_exception_requires_py_changed_false(self) -> None:
        needs = self._needs(
            changes_outputs={
                "py_changed": "true",
                "code_changed": "false",
                "changed_file_count": "0",
                "ancestry_only": "true",
            }
        )
        result = gate.evaluate(
            needs, unconditional_skip_ok={}, classifier_gated=self.GATED
        )
        assert not result.ok
        assert any("py_changed" in p for p in result.problems)

    def test_negative_file_count_fails(self) -> None:
        needs = self._needs(
            changes_outputs={"code_changed": "false", "changed_file_count": "-1"}
        )
        result = gate.evaluate(
            needs, unconditional_skip_ok={}, classifier_gated=self.GATED
        )
        assert not result.ok

    def test_missing_file_count_fails(self) -> None:
        needs = self._needs(changes_outputs={"code_changed": "false"})
        result = gate.evaluate(
            needs, unconditional_skip_ok={}, classifier_gated=self.GATED
        )
        assert not result.ok
        assert any(
            "changed_file_count" in p and "missing entirely" in p
            for p in result.problems
        )

    def test_non_integer_file_count_fails(self) -> None:
        needs = self._needs(
            changes_outputs={
                "code_changed": "false",
                "changed_file_count": "not-a-number",
            }
        )
        result = gate.evaluate(
            needs, unconditional_skip_ok={}, classifier_gated=self.GATED
        )
        assert not result.ok
        assert any("not a valid integer" in p for p in result.problems)

    def test_cancelled_job_fails(self) -> None:
        # job cancelled -> exit 1 (classifier state is irrelevant; cancelled is never a skip)
        needs = self._needs(
            unit_result="cancelled",
            changes_outputs={"code_changed": "false", "changed_file_count": "3"},
        )
        result = gate.evaluate(
            needs, unconditional_skip_ok={}, classifier_gated=self.GATED
        )
        assert not result.ok
        assert any("cancelled" in p for p in result.problems)

    def test_outputs_not_a_dict_fails_closed(self) -> None:
        needs = {
            "changes": {"result": "success", "outputs": "not-a-dict"},
            "unit": {"result": "skipped"},
        }
        result = gate.evaluate(
            needs, unconditional_skip_ok={}, classifier_gated=self.GATED
        )
        assert not result.ok

    def test_changes_entry_absent_fails(self) -> None:
        needs = {"unit": {"result": "skipped"}}
        result = gate.evaluate(
            needs, unconditional_skip_ok={}, classifier_gated=self.GATED
        )
        assert not result.ok


class TestRealConfig:
    """The real UNCONDITIONAL_SKIP_OK / CLASSIFIER_GATED must match what the
    workflow declares."""

    def test_unconditional_skip_ok_covers_changes_preflight_and_schema_types_only(
        self,
    ) -> None:
        assert set(gate.UNCONDITIONAL_SKIP_OK) == {
            "changes",
            "preflight",
            "schema-types",
        }

    def test_classifier_gated_covers_the_three_test_tier_jobs(self) -> None:
        assert set(gate.CLASSIFIER_GATED) == {"unit", "collection", "mcp-contract"}

    def test_classifier_gated_points_at_real_changes_outputs(self) -> None:
        # Every gating output name must be one `changes` actually declares
        # (pr-gate.yml's `changes` job outputs block) -- a typo here would
        # make verify_classifier_gated_skip() always fail closed, silently
        # turning every relevant PR red.
        real_changes_outputs = {"py_changed", "code_changed", "changed_file_count"}
        assert set(gate.CLASSIFIER_GATED.values()) <= real_changes_outputs

    def test_workflow_declares_a_fail_closed_ancestry_only_verifier(self) -> None:
        workflow = (REPO_ROOT / ".github" / "workflows" / "pr-gate.yml").read_text()

        required_fragments = (
            "ancestry_only: ${{ steps.changed.outputs.ancestry_only }}",
            'echo "ancestry_only=false" >> "$GITHUB_OUTPUT"',
            'if [ "$EVENT_NAME" = "pull_request" ]; then',
            'base_sha="$PULL_REQUEST_BASE_SHA"',
            'head_sha="$PULL_REQUEST_HEAD_SHA"',
            'if [ "$head_repo" != "$REPOSITORY" ]; then',
            "git fetch --no-tags origin main",
            'git merge-base --is-ancestor "$base_sha" "$declared_head"',
            "Traigent-Ancestry-Only: true",
            'if [ "${#head_parents[@]}" -ne 2 ]',
            'if [ "${head_parents[0]}" != "$base_sha" ]',
            'if [ "${head_parents[1]}" != "$expected_main_sha" ]',
            'if [ "$declared_tree" != "$base_tree" ]',
            'if [ "$EVENT_NAME" = "merge_group" ]; then',
            'if [ "${#candidate_parents[@]}" -ne 2 ]',
            'if [ "${candidate_parents[0]}" != "$base_sha" ]',
            'if [ "$candidate_tree" != "$base_tree" ]',
            'if [ "$pr_head_tree" != "$base_tree" ]',
        )
        for fragment in required_fragments:
            assert fragment in workflow, fragment

    def test_ancestry_only_requires_main_tree_to_match_base(self) -> None:
        workflow = (REPO_ROOT / ".github" / "workflows" / "pr-gate.yml").read_text()
        assert (
            'expected_main_tree="$(git rev-parse "${expected_main_sha}^{tree}")"'
            in workflow
        )
        assert 'if [ "$expected_main_tree" != "$base_tree" ]; then' in workflow

    def test_merge_group_requires_a_same_repo_pr_attribution(self) -> None:
        workflow = (REPO_ROOT / ".github" / "workflows" / "pr-gate.yml").read_text()
        assert "pull-requests: read" in workflow
        assert (
            "MERGE_GROUP_HEAD_REF: ${{ github.event.merge_group.head_ref }}" in workflow
        )
        assert 'merge_group_ref="$MERGE_GROUP_HEAD_REF"' in workflow
        assert (
            'merge_group_ref="${{ github.event.merge_group.head_ref }}"' not in workflow
        )
        assert "gh-readonly-queue/develop/pr-([1-9][0-9]*)-" in workflow
        assert (
            'gh api --method GET "repos/${REPOSITORY}/pulls/${pr_number}"' in workflow
        )
        assert '"$pr_head_repo" != "$REPOSITORY"' in workflow

    def test_merge_group_rejects_mismatched_or_ambiguous_pr_metadata(self) -> None:
        workflow = (REPO_ROOT / ".github" / "workflows" / "pr-gate.yml").read_text()
        assert '"$pr_base_ref" != "develop"' in workflow
        assert '"$pr_base_sha" != "$base_sha"' in workflow
        assert '"$pr_head_sha_api" != "$pr_head_sha"' in workflow
        assert "merge-queue head_ref is not one exact develop PR reference" in workflow

    def test_merge_group_fails_closed_when_pr_lookup_fails(self) -> None:
        workflow = (REPO_ROOT / ".github" / "workflows" / "pr-gate.yml").read_text()
        assert "could not fetch merge-queue PR metadata" in workflow

    def test_run_blocks_do_not_interpolate_event_data_as_shell_source(self) -> None:
        workflow = (REPO_ROOT / ".github" / "workflows" / "pr-gate.yml").read_text()
        run_lines: list[str] = []
        in_run_block = False
        run_indent = 0
        for line in workflow.splitlines():
            stripped = line.lstrip()
            indent = len(line) - len(stripped)
            if stripped == "run: |":
                in_run_block = True
                run_indent = indent
                continue
            if in_run_block and stripped and indent <= run_indent:
                in_run_block = False
            if in_run_block:
                run_lines.append(line)

        assert run_lines
        assert all("${{ github.event" not in line for line in run_lines)


class TestMergeQueueTopologyReplay:
    """Execute the workflow classifier against a synthetic merge-queue graph."""

    @staticmethod
    def _git(repo: Path, *args: str, input_text: str | None = None) -> str:
        completed = subprocess.run(
            ["git", *args],
            cwd=repo,
            input=input_text,
            capture_output=True,
            text=True,
            check=False,
        )
        assert completed.returncode == 0, completed.stderr
        return completed.stdout.strip()

    def _make_graph(self, tmp_path: Path) -> dict[str, str | Path]:
        remote = tmp_path / "remote.git"
        repo = tmp_path / "repo"
        remote.mkdir()
        repo.mkdir()
        self._git(remote, "init", "--bare")
        self._git(repo, "init")
        self._git(repo, "config", "user.email", "ci@example.test")
        self._git(repo, "config", "user.name", "CI")
        self._git(repo, "remote", "add", "origin", str(remote))

        (repo / "payload.txt").write_text("stable\n")
        self._git(repo, "add", "payload.txt")
        self._git(repo, "commit", "-m", "base")
        self._git(repo, "branch", "-M", "develop")
        base_sha = self._git(repo, "rev-parse", "HEAD")

        self._git(repo, "branch", "main")
        self._git(repo, "checkout", "main")
        self._git(repo, "commit", "--allow-empty", "-m", "main")
        main_sha = self._git(repo, "rev-parse", "HEAD")
        self._git(repo, "push", "origin", "develop", "main")

        self._git(repo, "checkout", "develop")
        self._git(
            repo,
            "merge",
            "--no-ff",
            "main",
            "-m",
            "ancestry sync\n\nTraigent-Ancestry-Only: true",
        )
        declared_head_sha = self._git(repo, "rev-parse", "HEAD")

        self._git(repo, "checkout", "--detach", base_sha)
        self._git(repo, "merge", "--no-ff", declared_head_sha, "-m", "candidate")
        candidate_sha = self._git(repo, "rev-parse", "HEAD")
        return {
            "repo": repo,
            "base_sha": base_sha,
            "main_sha": main_sha,
            "declared_head_sha": declared_head_sha,
            "candidate_sha": candidate_sha,
        }

    @staticmethod
    def _render_changes_script(destination: Path, event_values: dict[str, str]) -> None:
        workflow = (REPO_ROOT / ".github" / "workflows" / "pr-gate.yml").read_text()
        start = workflow.index("        run: |\n", workflow.index("id: changed"))
        start += len("        run: |\n")
        end = workflow.index("\n\n  schema-types:", start)
        script = textwrap.dedent(workflow[start:end]) + "\n"
        # Compatibility with the pre-fix workflow is intentional: it turns
        # the injection-shaped test case into a born-red regression test.
        for expression, value in event_values.items():
            script = script.replace(expression, value)
        destination.write_text(script)

    def _run_changes(
        self,
        tmp_path: Path,
        graph: dict[str, str | Path],
        *,
        api_mode: str,
        api_json: dict[str, object] | None,
        head_ref: str,
    ) -> tuple[subprocess.CompletedProcess[str], dict[str, str], Path]:
        repo = graph["repo"]
        assert isinstance(repo, Path)
        base_sha = str(graph["base_sha"])
        candidate_sha = str(graph["candidate_sha"])

        bin_dir = tmp_path / "bin"
        bin_dir.mkdir()
        gh = bin_dir / "gh"
        gh.write_text(
            "#!/usr/bin/env bash\n"
            'if [ "$GH_STUB_MODE" = failure ]; then exit 1; fi\n'
            "printf '%s\\n' \"$GH_STUB_JSON\"\n"
        )
        gh.chmod(0o755)

        script = tmp_path / "changes.sh"
        event_values = {
            "${{ github.event_name }}": "merge_group",
            "${{ github.event.merge_group.base_sha }}": base_sha,
            "${{ github.event.merge_group.head_sha }}": candidate_sha,
            "${{ github.event.merge_group.head_ref }}": head_ref,
            "${{ github.event.pull_request.base.sha }}": "",
            "${{ github.event.pull_request.head.sha }}": "",
            "${{ github.event.pull_request.head.repo.full_name }}": "",
            "${{ github.repository }}": "Traigent/Traigent",
        }
        self._render_changes_script(script, event_values)

        output = tmp_path / "github-output"
        env = {
            **os.environ,
            "EVENT_NAME": "merge_group",
            "MERGE_GROUP_BASE_SHA": base_sha,
            "MERGE_GROUP_HEAD_SHA": candidate_sha,
            "MERGE_GROUP_HEAD_REF": head_ref,
            "PULL_REQUEST_BASE_SHA": "",
            "PULL_REQUEST_HEAD_SHA": "",
            "PULL_REQUEST_HEAD_REPO": "",
            "REPOSITORY": "Traigent/Traigent",
            "GH_STUB_MODE": api_mode,
            "GH_STUB_JSON": json.dumps(api_json or {}),
            "GITHUB_OUTPUT": str(output),
            "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
        }
        completed = subprocess.run(
            ["bash", str(script)],
            cwd=repo,
            capture_output=True,
            text=True,
            env=env,
            check=False,
        )
        outputs = dict(
            line.split("=", 1)
            for line in output.read_text().splitlines()
            if "=" in line
        )
        return completed, outputs, tmp_path / "injected"

    def test_merge_group_provenance_replay_is_fail_closed(self, tmp_path: Path) -> None:
        graph = self._make_graph(tmp_path)
        base_sha = str(graph["base_sha"])
        declared_head_sha = str(graph["declared_head_sha"])
        valid_ref = f"gh-readonly-queue/develop/pr-2163-{'a' * 40}"

        valid_api = {
            "base": {"ref": "develop", "sha": base_sha},
            "head": {
                "sha": declared_head_sha,
                "repo": {"full_name": "Traigent/Traigent"},
            },
        }
        cases = (
            ("valid", "success", valid_api, valid_ref, True),
            (
                "fork",
                "success",
                {
                    **valid_api,
                    "head": {
                        "sha": declared_head_sha,
                        "repo": {"full_name": "attacker/Traigent"},
                    },
                },
                valid_ref,
                False,
            ),
            (
                "mismatched-head",
                "success",
                {
                    **valid_api,
                    "head": {
                        "sha": "0" * 40,
                        "repo": {"full_name": "Traigent/Traigent"},
                    },
                },
                valid_ref,
                False,
            ),
            ("api-failure", "failure", None, valid_ref, False),
            ("missing-fields", "success", {}, valid_ref, False),
            (
                "stale-base",
                "success",
                {**valid_api, "base": {"ref": "develop", "sha": "0" * 40}},
                valid_ref,
                False,
            ),
            (
                "injection-shaped-ref",
                "success",
                valid_api,
                f'{valid_ref}"; touch {tmp_path / "injected"}; #',
                False,
            ),
        )

        for name, api_mode, api_json, head_ref, expected in cases:
            case_dir = tmp_path / name
            case_dir.mkdir()
            completed, outputs, injected = self._run_changes(
                case_dir,
                graph,
                api_mode=api_mode,
                api_json=api_json,
                head_ref=head_ref,
            )
            assert completed.returncode == 0, completed.stderr
            assert outputs["ancestry_only"] == str(expected).lower()
            assert not injected.exists(), name


class TestCliEndToEnd:
    """Runs the actual script subprocess, mirroring the workflow's invocation."""

    def _full_needs(
        self,
        unit_result: str = "success",
        schema_types_result: str = "success",
        **changes_outputs: str,
    ) -> dict:
        base_outputs = {
            "py_changed": "false",
            "code_changed": "false",
            "changed_file_count": "1",
            "ancestry_only": "false",
        }
        base_outputs.update(changes_outputs)
        return {
            "changes": {"result": "success", "outputs": base_outputs},
            "schema-types": {"result": schema_types_result},
            "preflight": {"result": "success"},
            "unit": {"result": unit_result},
            "collection": {"result": unit_result},
            "mcp-contract": {"result": unit_result},
        }

    def test_all_success_passes(self) -> None:
        needs = self._full_needs(unit_result="success")
        proc = _run_cli(needs)
        assert proc.returncode == 0, proc.stdout

    def test_docs_only_pr_stays_green(self) -> None:
        # The exact scenario the coordinator called out: nothing
        # classifier-relevant changed, every gated job legitimately skips,
        # and changed_file_count is non-zero -- gate must be GREEN.
        needs = self._full_needs(unit_result="skipped")
        proc = _run_cli(needs)
        assert proc.returncode == 0, proc.stdout
        assert "verified safe" in proc.stdout

    def test_broken_classifier_output_true_but_job_skipped_fails(self) -> None:
        # Simulates an `if:` bug: the gating output says 'true' (relevant
        # change detected) yet the job still reports skipped. Must be RED.
        needs = self._full_needs(unit_result="skipped", code_changed="true")
        proc = _run_cli(needs)
        assert proc.returncode == 1, proc.stdout
        assert "unit" in proc.stdout

    def test_zero_file_count_with_skips_fails(self) -> None:
        needs = self._full_needs(unit_result="skipped", changed_file_count="0")
        proc = _run_cli(needs)
        assert proc.returncode == 1, proc.stdout

    def test_declared_ancestry_only_zero_file_count_with_skips_passes(self) -> None:
        needs = self._full_needs(
            unit_result="skipped",
            changed_file_count="0",
            ancestry_only="true",
        )
        proc = _run_cli(needs)
        assert proc.returncode == 0, proc.stdout
        assert "ancestry-only" in proc.stdout

    def test_changes_failed_with_downstream_skips_fails(self) -> None:
        needs = self._full_needs(unit_result="skipped")
        needs["changes"]["result"] = "failure"
        proc = _run_cli(needs)
        assert proc.returncode == 1, proc.stdout

    def test_real_gate_rejects_unit_cancelled(self) -> None:
        needs = self._full_needs(unit_result="success")
        needs["unit"] = {"result": "cancelled"}
        proc = _run_cli(needs)
        assert proc.returncode == 1, proc.stdout
        assert "unit" in proc.stdout

    def test_schema_types_skipped_fork_pr_passes_unconditionally(self) -> None:
        # schema-types is the one job whose skip is allowlisted despite
        # gating real behaviour, because its only skip vector (a fork PR) can
        # never be produced by a classifier bug. Forks are excluded by fork-CI
        # policy, not by a missing secret -- TraigentSchema is public and that
        # job's checkout is credential-free.
        needs = self._full_needs(unit_result="success", schema_types_result="skipped")
        proc = _run_cli(needs)
        assert proc.returncode == 0, proc.stdout

    def test_real_gate_accepts_draft_pr_cascade(self) -> None:
        # On a draft PR, `changes` and `preflight` both skip (unconditionally
        # allowlisted); `unit`/`collection`/`mcp-contract` also skip, but
        # `changes` itself did NOT succeed, so verify_classifier_gated_skip()
        # correctly refuses to trust it -- the gate is red for the whole
        # draft period. `schema-types` does not depend on `changes` at all,
        # so it keeps running even on a draft (pre-existing behaviour,
        # modelled here as success to isolate the cascade being tested).
        needs = _needs(
            changes="skipped",
            **{"schema-types": "success"},
            preflight="skipped",
            collection="skipped",
            unit="skipped",
            **{"mcp-contract": "skipped"},
        )
        proc = _run_cli(needs)
        assert proc.returncode == 1, proc.stdout
