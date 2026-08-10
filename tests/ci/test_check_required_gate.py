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
classifier saw a non-zero changed-file count. Anything else -- `cancelled`,
an empty/missing output, a zero file count, an unsuccessful `changes` run --
fails the gate.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
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
