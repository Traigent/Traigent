"""Partial objective coverage is named, consistent and starless (#2664)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import traigent
from traigent.core.objectives import create_default_objectives
from traigent.utils.results_table import _find_best_trial_index

BAD_ROWS = {3, 4, 5}


def _run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, orientation: str, ret: object
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TRAIGENT_MOCK_LLM", "true")
    monkeypatch.setenv("TRAIGENT_COST_APPROVED", "true")
    monkeypatch.setenv("TRAIGENT_RESULTS_FOLDER", str(tmp_path / "res"))
    monkeypatch.delenv("TRAIGENT_API_KEY", raising=False)
    ds = tmp_path / "d.jsonl"
    ds.write_text(
        "".join(
            json.dumps({"input": {"q": f"q{i}"}, "output": str(i % 4)}) + "\n"
            for i in range(20)
        )
    )
    state = {"n": 0}

    def metric(output, expected, **kw):  # noqa: ANN001, ANN003, ANN202
        i = state["n"] % 20
        state["n"] += 1
        return ret if i in BAD_ROWS else 0.8

    @traigent.optimize(
        eval_dataset=str(ds),
        objectives=create_default_objectives(
            ["quality"], orientations={"quality": orientation}
        ),
        configuration_space={"t": [0.0]},
        metric_functions={"quality": metric},
        injection_mode="parameter",
        config_param="t",
        offline=True,
        algorithm="grid",
        max_trials=1,
    )
    def f(q, t=None):  # noqa: ANN001, ANN202
        return "1"

    return f.optimize_sync(max_trials=1)


@pytest.mark.parametrize("orientation", ["maximize", "minimize"])
@pytest.mark.parametrize("ret", [{"other": 1.0}, {}])
def test_partial_coverage_is_named_and_fields_agree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, orientation: str, ret: object
) -> None:
    result = _run(tmp_path, monkeypatch, orientation, ret)
    assert result.best_config is None
    assert "METRIC_PARTIAL_COVERAGE" in (result.warning_codes or [])
    assert result.ranking_eligible_trial_ids in (None, [])
    comp = result.trials[0].metadata["comparability"]
    comp = comp if isinstance(comp, dict) else comp.to_dict()
    assert comp["primary_objective"] == "quality"
    assert comp["ranking_eligible"] is False
    assert comp["coverage_ratio"] == pytest.approx(0.85)
    assert "MCI-002" in comp["warning_codes"]
    assert (
        _find_best_trial_index(result, result.trials, [("quality", orientation)], None)
        is None
    )


def test_stored_session_best_is_cleared_when_selection_has_no_winner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result = _run(tmp_path, monkeypatch, "maximize", {"other": 1.0})
    assert result.best_config is None
    sessions = sorted((tmp_path / "res" / "sessions").glob("*.json"))
    assert sessions
    stored = json.loads(sessions[-1].read_text())
    assert stored["status"] == "completed"
    assert stored["best_config"] is None
    assert stored["best_score"] is None


def test_warning_text_matches_selection_per_mode() -> None:
    from traigent.core.orchestrator import _format_partial_coverage_warning as fmt

    partial = [("t1", 17, 20)]
    legacy = fmt(partial, "quality", None, {"t": 0.0}, ["t1"])
    assert "still eligible" in legacy and "excluded" not in legacy
    none_ranked = fmt(partial, "quality", "NO_RANKING_ELIGIBLE_TRIALS", None, [])
    assert "No winner is declared" in none_ranked
    uncertified = fmt(partial, "quality", "NO_CERTIFIED_SELECTION", None, [])
    assert "winner was chosen" not in uncertified
    assert "NO_CERTIFIED_SELECTION" in uncertified
    strict_empty = fmt(partial, "quality", None, {}, [])
    assert "winner was chosen" not in strict_empty
    chosen = fmt(partial, "quality", None, {"t": 1}, ["t2"])
    assert "winner was chosen among the remaining" in chosen
