"""Returned and stored best agree on a tie with a raising candidate (#2660)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import traigent
from traigent.api.decorators import EvaluationOptions, ExecutionOptions

ROWS = 8
KEEP = {"v1": 2, "v2": 3, "v3": 4, "v4": 5, "v5": 5}


def _run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, raising: str
) -> tuple[dict, dict | None]:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TRAIGENT_MOCK_LLM", "true")
    monkeypatch.setenv("TRAIGENT_COST_APPROVED", "true")
    monkeypatch.setenv("TRAIGENT_RESULTS_FOLDER", str(tmp_path / "res"))
    monkeypatch.delenv("TRAIGENT_API_KEY", raising=False)
    monkeypatch.delenv("TRAIGENT_BACKEND_URL", raising=False)
    ds = tmp_path / "qa.jsonl"
    ds.write_text(
        "".join(
            json.dumps({"input": {"question": f"q{i}"}, "output": f"a{i}"}) + "\n"
            for i in range(ROWS)
        )
    )
    qs = [f"q{i}" for i in range(ROWS)]

    @traigent.optimize(
        configuration_space={"variant": list(KEEP)},
        objectives=["accuracy"],
        evaluation=EvaluationOptions(eval_dataset=str(ds)),
        execution=ExecutionOptions(offline=True),
        max_trials=5,
    )
    def agent(question: str) -> str:
        v = traigent.get_config()["variant"]
        i = qs.index(question)
        if v == raising and i >= 5:
            raise RuntimeError("partial crash")
        return f"a{i}" if i < KEEP[v] else "I don't know"

    result = agent.optimize_sync(algorithm="grid")
    sessions = sorted((tmp_path / "res" / "sessions").glob("*.json"))
    assert sessions
    stored = json.loads(sessions[-1].read_text())
    return result.best_config, stored["best_config"]


@pytest.mark.parametrize("raising", ["v4", "v5"])
def test_returned_and_stored_best_agree_in_both_listing_orders(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, raising: str
) -> None:
    returned, stored = _run(tmp_path, monkeypatch, raising)
    assert stored == returned
