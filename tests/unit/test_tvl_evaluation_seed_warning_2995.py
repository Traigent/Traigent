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
