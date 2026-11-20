"""A model released after the price table shipped must not abort the run it is admitted to.

Only the admitted value may run with its cost unknown; every other unpriced model
keeps failing the cost preflight, and an unknown cost is never scored as free.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

import traigent
from traigent.core.optimized_function import UnknownModelError
from traigent.tuning.controller import traigent_optimize_fn
from traigent.tuning.evidence import (
    COST_UNKNOWN,
    EvidenceUnavailable,
    build_promotion_evidence,
)

PRICED = "anthropic/claude-haiku-4-5"
ADMITTED = "anthropic/claude-haiku-9-9"  # released after the price table
OTHER_UNPRICED = "newvendor/model-x"


@pytest.fixture(autouse=True)
def _mock_non_interactive(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TRAIGENT_MOCK_LLM", "true")
    for name in (
        "TRAIGENT_COST_APPROVED",
        "TRAIGENT_COST_PREFLIGHT_MOCK_EXEMPT",
        "TRAIGENT_STRICT_COST_ACCOUNTING",
    ):
        monkeypatch.delenv(name, raising=False)


def _live() -> Any:
    @traigent.optimize(
        configuration_space={"model": [PRICED]},
        objectives=["accuracy"],
        eval_dataset=[{"input": {"q": "a"}, "output": "a"}],
        offline=True,
        max_trials=2,
        default_config={"model": PRICED},
    )
    def live(q: str) -> str:
        return str(traigent.get_config().get("model") and "a")

    return live


def test_an_unpriced_model_still_fails_the_preflight_by_default() -> None:
    live = _live()
    with pytest.raises(UnknownModelError):
        live._preflight_model_cost_coverage({"model": [PRICED, ADMITTED]})


def test_only_the_admitted_model_may_run_unpriced() -> None:
    live = _live()
    live._unpriced_models_allowed = frozenset({ADMITTED})
    live._preflight_model_cost_coverage({"model": [PRICED, ADMITTED]})  # no abort
    with pytest.raises(UnknownModelError) as err:  # another unpriced model: guard holds
        live._preflight_model_cost_coverage(
            {"model": [PRICED, ADMITTED, OTHER_UNPRICED]}
        )
    assert OTHER_UNPRICED in str(err.value) and ADMITTED not in str(err.value)


def test_the_adapter_runs_an_admitted_unpriced_model_instead_of_aborting() -> None:
    live = _live()
    run = traigent_optimize_fn(live, algorithm="grid", max_trials=2)
    space = {"model": [PRICED, ADMITTED]}
    with pytest.raises(UnknownModelError):
        run(space)  # without the admission the run aborts, as before
    result = run(space, admitted_models=[ADMITTED])
    assert result.trials
    with pytest.raises(UnknownModelError):  # the exemption is scoped to one run
        run({"model": [PRICED, OTHER_UNPRICED]}, admitted_models=[ADMITTED])
    assert not hasattr(
        live, "_unpriced_models_allowed"
    )  # the live function is untouched


MODULE: dict[str, Any] = {
    "objectives": [{"name": "accuracy", "direction": "maximize"}],
    "promotion_policy": {"alpha": 0.05, "min_effect": {"accuracy": 0.02}},
    "requirements": [
        {
            "name": "failure_rate",
            "kind": "rate_bound",
            "unit": "invocation",
            "max_rate": 0.08,
            "violation": {"checker": "customer:c", "label_in": ["fail"]},
        }
    ],
    "monitors": [
        {
            "name": "gate",
            "requirement": "failure_rate",
            "phase": "promotion",
            "evidence": {"source": "evaluation_set"},
            "decision": {
                "method": "clopper_pearson_rational_v2",
                "confidence": 0.95,
                "false_alarm_budget": 0.05,
            },
        }
    ],
}


def _agent(x: int) -> int:
    from traigent.config.context import config_context

    cfg = config_context.get() or {}
    right = x % 2
    return right if cfg.get("good") or x % 4 < 2 else 1 - right


def _evidence(tmp_path: Path, module: dict[str, Any]) -> Any:
    path = tmp_path / "h.jsonl"
    rows = [{"input": {"x": i}, "output": i % 2} for i in range(200)]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return build_promotion_evidence(
        module,
        fn=_agent,
        checkers={"customer:c": lambda ex, out: "pass"},
        checker_labels={"customer:c": ["pass", "fail"]},
        holdout=path,
        candidate_config={"good": True, "model": ADMITTED},
        incumbent_config={"good": False, "model": PRICED},
        unpriced_models=[ADMITTED],
    )


def test_unknown_cost_is_never_scored_as_free(tmp_path: Path) -> None:
    module = json.loads(json.dumps(MODULE))
    module["objectives"].append({"name": "cost", "direction": "minimize"})
    with pytest.raises(EvidenceUnavailable) as err:
        _evidence(tmp_path, module)
    assert err.value.reason == "cost_unknown_unpriced_model"  # the gate says why


def test_without_a_cost_objective_the_gate_still_says_the_cost_is_unknown(
    tmp_path: Path,
) -> None:
    evidence = _evidence(tmp_path, json.loads(json.dumps(MODULE)))
    assert evidence.legacy_decision == "Promote"
    assert COST_UNKNOWN in evidence.errors
