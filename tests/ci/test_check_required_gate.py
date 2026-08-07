"""Negative-control tests for scripts/ci/check_required_gate.py.

`required-pr-gate` is a required status check: this script decides whether a
PR is allowed to merge based on what its dependency jobs reported. A
regression here either wrongly blocks every PR, or -- much worse -- silently
turns the gate green when a real test job never ran. These tests exercise the
decision function directly (fast, precise) plus the actual CLI entry point
with the repo's real SKIP_OK allowlist (end-to-end, proves the wiring works).

The gate is fail-closed by design: anything not proven safe must fail. Most
cases below therefore assert failure.
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
    """Build a `needs` context map from job=result kwargs."""
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


class TestEvaluateDecisionLogic:
    """Six required negative-control cases against the pure decision function."""

    def test_all_success_passes(self) -> None:
        needs = _needs(a="success", b="success", c="success")
        result = gate.evaluate(needs, skip_ok={})
        assert result.ok, result.problems
        assert result.problems == []

    def test_non_allowlisted_job_skipped_fails_and_names_it(self) -> None:
        needs = _needs(a="success", b="skipped")
        result = gate.evaluate(needs, skip_ok={})
        assert not result.ok
        assert any(
            "b" in problem and "skipped" in problem for problem in result.problems
        )

    def test_allowlisted_job_skipped_passes(self) -> None:
        needs = _needs(a="success", b="skipped")
        result = gate.evaluate(needs, skip_ok={"b": "test double: allowlisted"})
        assert result.ok, result.problems

    def test_cancelled_job_fails(self) -> None:
        needs = _needs(a="success", b="cancelled")
        result = gate.evaluate(needs, skip_ok={})
        assert not result.ok
        assert any("cancelled" in problem for problem in result.problems)

    def test_cancelled_job_fails_even_if_allowlisted(self) -> None:
        # Allowlisting only excuses `skipped`, never `cancelled`.
        needs = _needs(a="success", b="cancelled")
        result = gate.evaluate(needs, skip_ok={"b": "test double: allowlisted"})
        assert not result.ok

    def test_absent_result_key_fails(self) -> None:
        needs = {"a": {"result": "success"}, "b": {}}
        result = gate.evaluate(needs, skip_ok={})
        assert not result.ok
        assert any(problem.startswith("b=") for problem in result.problems)

    def test_empty_string_result_fails(self) -> None:
        needs = {"a": {"result": "success"}, "b": {"result": ""}}
        result = gate.evaluate(needs, skip_ok={})
        assert not result.ok

    def test_stale_allowlist_entry_fails(self) -> None:
        # skip_ok names a job that isn't a declared dependency at all -- the
        # job was probably renamed, so the real one is now unguarded.
        needs = _needs(a="success")
        result = gate.evaluate(needs, skip_ok={"renamed-away-job": "stale"})
        assert not result.ok
        assert any("renamed-away-job" in problem for problem in result.problems)

    def test_empty_needs_fails(self) -> None:
        result = gate.evaluate({}, skip_ok={})
        assert not result.ok
        assert result.problems


class TestRealSkipOkAllowlist:
    """`SKIP_OK` in this file must actually match what the workflow declares."""

    def test_skip_ok_covers_changes_preflight_and_schema_types_only(self) -> None:
        # Load-bearing assertion: the real test-execution jobs (unit,
        # collection, mcp-contract) must NOT be in the allowlist, because
        # their skip conditions depend on the `changes` classifier's
        # `code_changed` output, and a classifier bug is exactly what this
        # gate defends against. `schema-types` IS included because it has no
        # dependency on `changes` at all -- see its SKIP_OK justification.
        assert set(gate.SKIP_OK) == {"changes", "preflight", "schema-types"}


class TestCliEndToEnd:
    """Runs the actual script subprocess, mirroring the workflow's invocation."""

    def test_real_gate_rejects_unit_skipped(self) -> None:
        needs = _needs(
            changes="success",
            **{"schema-types": "success"},
            preflight="success",
            collection="success",
            unit="skipped",
            **{"mcp-contract": "success"},
        )
        proc = _run_cli(needs)
        assert proc.returncode == 1, proc.stdout
        assert "unit" in proc.stdout

    def test_real_gate_accepts_all_success(self) -> None:
        needs = _needs(
            changes="success",
            **{"schema-types": "success"},
            preflight="success",
            collection="success",
            unit="success",
            **{"mcp-contract": "success"},
        )
        proc = _run_cli(needs)
        assert proc.returncode == 0, proc.stdout

    def test_real_gate_accepts_schema_types_skipped_fork_pr(self) -> None:
        # schema-types is the one job whose skip IS allowlisted despite
        # gating real behaviour, because its only skip vector (fork PR, no
        # SCHEMA_TOKEN) can never be produced by a classifier bug.
        needs = _needs(
            changes="success",
            **{"schema-types": "skipped"},
            preflight="success",
            collection="success",
            unit="success",
            **{"mcp-contract": "success"},
        )
        proc = _run_cli(needs)
        assert proc.returncode == 0, proc.stdout

    def test_real_gate_accepts_draft_pr_cascade(self) -> None:
        # On a draft PR, `changes` and `preflight` both skip (allowlisted);
        # `unit`/`collection`/`mcp-contract` also skip because their own
        # `if:` requires needs.preflight.result == 'success' (false) and are
        # NOT allowlisted, so the gate is red for the whole draft period.
        # `schema-types` does NOT depend on `changes`/`preflight` at all, so
        # it actually keeps running even while the PR is a draft (pre-
        # existing behaviour, unrelated to this change) -- modelled here as
        # success to isolate the cascade being tested.
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
