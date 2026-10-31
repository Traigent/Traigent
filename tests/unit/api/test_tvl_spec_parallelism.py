"""A TVL spec's ``max_parallel_trials`` must not break decoration.

Found while verifying the ``tvl_spec=`` examples for #2836: the spec's
parallelism reached ``OptimizedFunction`` twice (explicitly and through the
runtime overrides) and decoration raised
``TypeError: got multiple values for keyword argument 'parallel_config'``.
"""

from __future__ import annotations

from traigent import optimize
from traigent.core.optimized_function import OptimizedFunction

SPEC = """tvl:
  module: test.parallelism
tvl_version: "0.9"
tvars:
  - name: model
    type: enum[str]
    domain: ["gpt-4o-mini", "gpt-4o"]
objectives:
  - name: accuracy
    direction: maximize
exploration:
  budgets:
    max_trials: 4
  parallelism:
    max_parallel_trials: 3
"""


def test_tvl_spec_with_parallelism_decorates(tmp_path) -> None:
    spec = tmp_path / "parallel.tvl.yml"
    spec.write_text(SPEC)

    @optimize(tvl_spec=spec, eval_dataset=[{"input": {"q": "a"}, "output": "a"}])
    def agent(q: str) -> str:
        return q

    assert isinstance(agent, OptimizedFunction)
    assert agent.parallel_config.trial_concurrency == 3


def test_decorator_parallel_config_wins_over_tvl_spec(tmp_path) -> None:
    spec = tmp_path / "parallel.tvl.yml"
    spec.write_text(SPEC)

    @optimize(
        tvl_spec=spec,
        eval_dataset=[{"input": {"q": "a"}, "output": "a"}],
        parallel_config={"trial_concurrency": 1},
    )
    def agent(q: str) -> str:
        return q

    assert agent.parallel_config.trial_concurrency == 1


def test_reused_decorator_keeps_tvl_parallelism(tmp_path) -> None:
    spec = tmp_path / "parallel.tvl.yml"
    spec.write_text(SPEC)
    decorate = optimize(
        tvl_spec=spec, eval_dataset=[{"input": {"q": "a"}, "output": "a"}]
    )

    def first(q: str) -> str:
        return q

    def second(q: str) -> str:
        return q

    assert decorate(first).parallel_config.trial_concurrency == 3
    assert decorate(second).parallel_config.trial_concurrency == 3
