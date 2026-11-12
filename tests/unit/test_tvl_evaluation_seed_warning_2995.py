"""evaluation_set.seed is not used by the SDK; setting it warns once per spec."""

from __future__ import annotations

import warnings
from types import SimpleNamespace

import pytest

from traigent.api import decorators


def _artifact(path: str, seed):
    return SimpleNamespace(
        path=path, evaluation_set=SimpleNamespace(dataset="d.jsonl", seed=seed)
    )


@pytest.fixture(autouse=True)
def _reset():
    decorators._SEED_WARNED_SPECS.clear()
    yield
    decorators._SEED_WARNED_SPECS.clear()


def test_seed_set_warns_once_per_spec_without_value_2995():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        decorators._warn_evaluation_seed_unused(_artifact("a.tvl.yml", 424242))
        decorators._warn_evaluation_seed_unused(_artifact("a.tvl.yml", 424242))
    assert len(caught) == 1
    text = str(caught[0].message)
    assert "evaluation_set.seed" in text
    assert "424242" not in text


def test_each_spec_warns_and_unset_seed_is_silent_2995():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        decorators._warn_evaluation_seed_unused(_artifact("a.tvl.yml", 1))
        decorators._warn_evaluation_seed_unused(_artifact("b.tvl.yml", 1))
        decorators._warn_evaluation_seed_unused(_artifact("c.tvl.yml", None))
        decorators._warn_evaluation_seed_unused(
            SimpleNamespace(path="d", evaluation_set=None)
        )
    assert len(caught) == 2


def test_decorating_with_seeded_tvl_spec_warns_exactly_once_2995(tmp_path):
    from traigent import optimize

    spec_path = tmp_path / "seeded.tvl.yml"
    spec_path.write_text("""tvl:
  module: test.seeded
tvl_version: "0.9"
evaluation_set:
  dataset: test.jsonl
  seed: 424242
tvars:
  - name: model
    type: enum[str]
    domain: ["gpt-4"]
objectives:
  - name: accuracy
    direction: maximize
""")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")

        @optimize(tvl_spec=spec_path)
        def first(question: str) -> str:
            return question

        @optimize(tvl_spec=spec_path)
        def second(question: str) -> str:
            return question

    seed_warnings = [w for w in caught if "evaluation_set.seed" in str(w.message)]
    assert len(seed_warnings) == 1
    assert "424242" not in str(seed_warnings[0].message)
