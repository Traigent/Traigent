"""Consensus detector + ranked worklist through the NORMAL scoring_function path.

Two families, two configs, a small fake table with 3 planted wrong labels (both
models agree on the same non-gold answer) and 1 genuinely hard row (both wrong,
DIFFERENT answers). No custom_evaluator, no LLM calls.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

import traigent

ROWS = [f"q{i:02d}" for i in range(12)]
GOLD = {r: ("yes" if i % 2 == 0 else "no") for i, r in enumerate(ROWS)}
PLANTED = {"q03": "yes", "q07": "yes", "q09": "yes"}  # both agree; gold is "no"
HARD = "q05"  # both wrong, different answers
ANS = {"a": dict(GOLD) | PLANTED, "b": dict(GOLD) | PLANTED}
ANS["a"][HARD] = "maybe"
ANS["b"][HARD] = "other"
MODELS = ["openai/gpt-sim", "anthropic/claude-sim"]


def _exact(output, expected_output=None, **_kw):
    return float(output == expected_output)


@pytest.fixture
def audit(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("TRAIGENT_OFFLINE_MODE", "true")
    monkeypatch.setenv("TRAIGENT_RUN_APPROVED", "1")
    monkeypatch.setenv("TRAIGENT_DISABLE_TELEMETRY", "1")
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    pricing = tmp_path / "pricing.json"
    pricing.write_text(
        json.dumps(
            {
                m: {"input_cost_per_token": 1e-7, "output_cost_per_token": 4e-7}
                for m in MODELS
            }
        )
    )
    monkeypatch.setenv("TRAIGENT_CUSTOM_MODEL_PRICING_FILE", str(pricing))
    monkeypatch.setenv("TRAIGENT_DATASET_ROOT", str(tmp_path))
    data = tmp_path / "eval.jsonl"
    data.write_text(
        "\n".join(json.dumps({"input": {"qid": r}, "output": GOLD[r]}) for r in ROWS)
        + "\n"
    )

    @traigent.optimize(
        configuration_space={"model": MODELS},
        objectives=["accuracy"],
        eval_dataset=str(data),
        scoring_function=_exact,
        offline=True,
        algorithm="grid",
        max_trials=4,
    )
    def answer(qid: str):
        cfg = traigent.get_config() or {"model": MODELS[0]}
        return ANS["a" if cfg["model"] == MODELS[0] else "b"][qid]

    result = asyncio.run(answer.optimize())
    return result.eval_audit


def test_consensus_flags_exactly_planted_and_ranks_them_top(audit) -> None:
    assert audit is not None
    assert audit.summary.family_count == 2
    d7 = {
        f.example_id
        for f in audit.flagged
        if "cross-family-consensus-on-wrong" in f.detectors
    }
    planted_ids = {f"example_{ROWS.index(r)}" for r in PLANTED}
    hard_id = f"example_{ROWS.index(HARD)}"
    assert d7 == planted_ids
    top3 = {s.example_id for s in audit.scored[:3]}
    assert top3 == planted_ids
    by_id = {s.example_id: s for s in audit.scored}
    assert all(by_id[i].primary_reason == "consensus_wrong" for i in planted_ids)
    assert by_id[hard_id].primary_reason == "mean_wrong"
    assert [s.example_id for s in audit.scored].index(hard_id) == 3
