"""A decorator-time ``tvl_spec`` must apply its exploration budget and strategy (#3054)."""

from __future__ import annotations

from traigent import optimize

SPEC = """tvl:
  module: test.budget
tvl_version: "0.9"
tvars:
  - name: k
    type: int
    domain:
      range: [1, 6]
objectives:
  - name: accuracy
    direction: maximize
exploration:
  strategy: random
  budget:
    max_trials: 3
"""

DATA = [{"input": {"q": "a"}, "output": "a"}]


def _decorate(tmp_path, **kwargs):
    spec = tmp_path / "budget.tvl.yml"
    spec.write_text(SPEC)

    @optimize(tvl_spec=spec, eval_dataset=DATA, **kwargs)
    def agent(q: str) -> str:
        return q

    return agent


def test_decorator_tvl_spec_applies_max_trials_and_algorithm(tmp_path) -> None:
    agent = _decorate(tmp_path)

    assert agent.max_trials == 3
    assert agent.algorithm == "random"


def test_explicit_decorator_values_win_over_tvl_spec(tmp_path) -> None:
    agent = _decorate(tmp_path, max_trials=7, algorithm="grid")

    assert agent.max_trials == 7
    assert agent.algorithm == "grid"
