from __future__ import annotations

from types import SimpleNamespace

import pytest

from walkthrough.demo import langfuse_core_e2e_seed as seed


def test_phase_env_sets_resolved_project(monkeypatch):
    monkeypatch.setenv("TRAIGENT_PROJECT_ID", "from-env")
    env = seed.phase_env(SimpleNamespace(project_id="  chosen  "))
    assert env["TRAIGENT_PROJECT_ID"] == "chosen"


def test_prompt_scenarios_fail_without_project():
    for scenario in seed.PROMPT_SCENARIOS:
        assert scenario
    with pytest.raises(RuntimeError):
        seed.prompts_api_path(SimpleNamespace(project_id=""))
