"""Wrong-typed or non-positive exploration budget values must be rejected (#3055)."""

from __future__ import annotations

import pytest

from traigent.tvl.spec_loader import load_tvl_spec
from traigent.utils.exceptions import TVLValidationError

HEADER = """tvl:
  module: test.budgetvalidation
tvl_version: "0.9"
tvars:
  - name: model
    type: enum[str]
    domain: ["a", "b"]
objectives:
  - name: accuracy
    direction: maximize
"""


def _load(tmp_path, exploration: str):
    spec = tmp_path / "s.tvl.yml"
    spec.write_text(HEADER + exploration)
    return load_tvl_spec(spec_path=spec)


def test_valid_budget_still_loads(tmp_path) -> None:
    artifact = _load(
        tmp_path,
        "exploration:\n  budget:\n    max_trials: 20\n    timeout_seconds: 60\n"
        "  parallelism:\n    max_parallel_trials: 2\n",
    )

    assert artifact.runtime_overrides()["max_trials"] == 20


@pytest.mark.parametrize(
    ("exploration", "field"),
    [
        ("budget:\n    timeout_seconds: '60'", "timeout_seconds"),
        ("budget:\n    timeout_seconds: 0", "timeout_seconds"),
        ("budget:\n    max_trials: 0", "max_trials"),
        ("budget:\n    max_trials: -5", "max_trials"),
        ("budget:\n    max_trials: true", "max_trials"),
        ("budget:\n    parallel_trials: '3'", "parallel_trials"),
        ("budget:\n    parallel_trials: 0", "parallel_trials"),
        ("budget:\n    parallel_trials: 2.5", "parallel_trials"),
        ("budget:\n    samples_include_pruned: 'false'", "samples_include_pruned"),
        ("budget: 20", "budget"),
        ("parallelism: 4", "parallelism"),
        ("parallelism:\n    max_parallel_trials: '4'", "max_parallel_trials"),
        ("parallelism:\n    max_parallel_trials: 0", "max_parallel_trials"),
    ],
)
def test_bad_budget_value_raises_naming_field(tmp_path, exploration, field) -> None:
    with pytest.raises(TVLValidationError, match=field):
        _load(tmp_path, "exploration:\n  " + exploration + "\n")
