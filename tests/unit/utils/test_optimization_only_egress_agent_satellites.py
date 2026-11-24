"""Egress allowlist pins for the project-level agent satellites.

Production breaches and next-step are GET reads on ``/api/v1/projects/{project_id}/agents/
{agent_ref}/...``; production-requirements stay on ``/api/v1beta``. The retired
``/agent-readiness/...`` routes must stay refused.
"""

from __future__ import annotations

import pytest

from traigent.utils import optimization_only_egress as oo

API = "https://api.example.test/api/v1"
BREACHES = API + "/projects/proj-1/agents/agent-1/production-breaches"
NEXT_STEP = API + "/projects/proj-1/agents/agent-1/next-step"


@pytest.fixture(autouse=True)
def _optimization_only(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(oo.ENV_VAR, "1")


@pytest.mark.parametrize("url", [BREACHES, NEXT_STEP])
def test_project_level_agent_reads_are_allowed(url: str) -> None:
    oo.enforce_outgoing("GET", url, {}, None, explicit=True)


@pytest.mark.parametrize(
    "url",
    [
        API + "/agent-readiness/agent-1/next-step",
        API + "/agent-readiness/agent-1/lifecycle-history",
    ],
)
def test_retired_agent_readiness_routes_are_refused(url: str) -> None:
    with pytest.raises(oo.OptimizationOnlyRefusal):
        oo.enforce_outgoing("GET", url, {}, None, explicit=True)


def test_agent_next_step_is_not_writable() -> None:
    with pytest.raises(oo.OptimizationOnlyRefusal):
        oo.enforce_outgoing("POST", NEXT_STEP, {}, "{}", explicit=True)
