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


def test_explicit_auto_algorithm_is_not_overridden_by_spec(
    tmp_path, connected_policy_env
) -> None:
    # Offline (the default unit lane) resolves "auto" to the local runtime
    # algorithm; a connected policy keeps "auto" as chosen.
    agent = _decorate(tmp_path, algorithm="auto")

    assert agent.algorithm == "auto"
    assert agent.max_trials == 3


def test_explicit_max_trials_equal_to_default_wins_over_spec(tmp_path) -> None:
    from traigent.api.decorators import DEFAULT_MAX_TRIALS

    agent = _decorate(tmp_path, max_trials=DEFAULT_MAX_TRIALS)

    assert agent.max_trials == DEFAULT_MAX_TRIALS


def test_explicit_auto_conflicts_with_different_legacy_algorithm(tmp_path) -> None:
    import pytest

    from traigent.api.decorators import LegacyOptimizeArgs

    spec = tmp_path / "budget.tvl.yml"
    spec.write_text(SPEC)

    with pytest.raises(TypeError, match="algorithm"):

        @optimize(
            tvl_spec=spec,
            eval_dataset=DATA,
            algorithm="auto",
            legacy=LegacyOptimizeArgs(algorithm="grid"),
        )
        def agent(q: str) -> str:
            return q
