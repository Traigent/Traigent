from __future__ import annotations

import json
import sys
from types import SimpleNamespace

import pytest

from walkthrough.demo import langfuse_core_e2e_seed as seed


class _PhaseReached(Exception):
    """Raised by the run_phase recorder so main() stops at the first phase."""


def test_phase_env_sets_resolved_project(monkeypatch):
    monkeypatch.setenv("TRAIGENT_PROJECT_ID", "from-env")
    env = seed.phase_env(SimpleNamespace(project_id="  chosen  "))
    assert env["TRAIGENT_PROJECT_ID"] == "chosen"


def test_prompt_scenarios_fail_without_project(monkeypatch, tmp_path):
    monkeypatch.delenv("TRAIGENT_PROJECT_ID", raising=False)
    monkeypatch.delenv("TRAIGENT_JWT_TOKEN", raising=False)
    monkeypatch.delenv("LANGFUSE_CORE_E2E_EMAIL", raising=False)
    monkeypatch.delenv("LANGFUSE_CORE_E2E_PASSWORD", raising=False)
    monkeypatch.setenv("TRAIGENT_API_KEY", "test-key")

    calls: list[str] = []

    def recorder(args, phase, artifacts_dir):
        calls.append(phase)
        raise _PhaseReached(phase)

    monkeypatch.setattr(seed, "run_phase", recorder)

    for scenario in sorted(seed.PROMPT_SCENARIOS):
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "langfuse_core_e2e_seed.py",
                "--run-id",
                "no-project",
                "--scenario",
                scenario,
                "--output",
                str(tmp_path / scenario / "manifest.json"),
            ],
        )
        with pytest.raises(RuntimeError, match="project id is required"):
            seed.main()
        assert calls == [], f"run_phase reached for {scenario} without a project"


def test_run_phase_passes_explicit_project_id_to_subprocess(monkeypatch, tmp_path):
    monkeypatch.setenv("TRAIGENT_PROJECT_ID", "other")

    captured: dict[str, dict[str, str]] = {}

    def fake_run(cmd, **kwargs):
        captured["env"] = kwargs["env"]
        # run_phase reads the phase summary after the subprocess returns.
        summary = tmp_path / "run-explicit" / "baseline_summary.json"
        summary.parent.mkdir(parents=True, exist_ok=True)
        summary.write_text(json.dumps({}), encoding="utf-8")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(seed.subprocess, "run", fake_run)

    args = SimpleNamespace(
        run_id="run-explicit",
        mode="mock",
        scale="tiny",
        observability="backend",
        baseline_runs=1,
        post_runs=1,
        frontend_url="http://localhost:3000",
        project_id="explicit-id",
    )
    seed.run_phase(args, "baseline", tmp_path)

    assert captured["env"]["TRAIGENT_PROJECT_ID"] == "explicit-id"
