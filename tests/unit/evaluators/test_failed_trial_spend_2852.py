"""Traigent#2852: a scorer exception is the trial's failure, not an "unpriced call".

The model call behind the failed row was priced; its spend is kept on the
failed trial instead of being dropped, and a cost objective no longer turns the
scorer error into a pricing error. Keyless: LiteLLM ``mock_response``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

litellm = pytest.importorskip("litellm")

import traigent  # noqa: E402
from traigent import Choices  # noqa: E402

MODEL = "openrouter/openai/gpt-4o-mini"


@pytest.fixture(autouse=True)
def _env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    monkeypatch.setenv("TRAIGENT_OFFLINE_MODE", "true")
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("TRAIGENT_MOCK_LLM", raising=False)
    monkeypatch.delenv("TRAIGENT_STRICT_COST_ACCOUNTING", raising=False)
    monkeypatch.chdir(tmp_path)


def _run(rows: int, scorer):
    Path("ds.jsonl").write_text(
        "".join(
            json.dumps({"input": {"q": f"q{i}"}, "output": "x"}) + "\n"
            for i in range(rows)
        )
    )

    @traigent.optimize(
        configuration_space={"model": Choices([MODEL])},
        eval_dataset="ds.jsonl",
        objectives=["accuracy", "cost"],
        scoring_function=scorer,
        offline=True,
    )
    def ask(q: str) -> str:
        r = litellm.completion(
            model=MODEL,
            messages=[{"role": "user", "content": q}],
            mock_response="paris",
        )
        return r.choices[0].message.content

    return ask.optimize_sync(max_trials=1, algorithm="grid")


def _boom(output, expected, *args, **kwargs):
    raise ValueError("model output is not JSON")


@pytest.mark.parametrize("rows", [1, 3])
def test_scorer_error_is_a_failed_trial_with_its_spend(rows: int) -> None:
    result = _run(rows, _boom)

    assert [str(getattr(t.status, "value", t.status)) for t in result.trials] == [
        "failed"
    ]
    trial = result.trials[0]
    assert "model output is not JSON" in (trial.error_message or "")
    one_call = litellm.completion_cost(
        completion_response=litellm.completion(
            model=MODEL,
            messages=[{"role": "user", "content": "q"}],
            mock_response="paris",
        )
    )
    assert trial.metrics["cost"] == pytest.approx(rows * one_call)
    assert not trial.metrics.get("cost_unmeasured")
