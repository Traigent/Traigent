"""Traigent#3137: coercion helpers keep their documented contract on huge/inf numbers."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from traigent.cloud.client import _best_metrics_from_score
from traigent.cloud.dtos import _int_or_default
from traigent.core.result_selection import _coerce_float, _coerce_int
from traigent.hybrid.protocol import EstimatedTokensPerExample
from traigent.security.input_validation import InputValidator, ValidationError
from traigent.storage.local_storage import _coerce_optional_cost
from traigent.tvl.spec_loader import load_tvl_spec
from traigent.utils.discrete_domains import _float_range_args, _int_range_args
from traigent.utils.exceptions import MetricExtractionError, TVLValidationError
from traigent.utils.objectives import coerce_finite_objective_score
from traigent.utils.trial_costs import coerce_finite_cost
from traigent.utils.validation import validate_numeric_metric

HUGE = 10**400
INF = float("inf")


def test_none_returning_helpers_on_huge_int() -> None:
    assert coerce_finite_cost(HUGE) is None
    assert coerce_finite_objective_score(HUGE) is None
    assert _coerce_optional_cost(HUGE) is None
    assert _coerce_float(HUGE) is None
    assert _best_metrics_from_score("accuracy", HUGE) == {}


def test_int_helpers_on_inf() -> None:
    assert _coerce_int(INF) is None
    assert _int_or_default(INF, 1) == 1
    assert _int_range_args({"step": 1}, INF, 5) is None


def test_float_range_args_on_huge_int() -> None:
    assert _float_range_args({"step": 1}, HUGE, 5) is None


def test_estimated_tokens_from_json_infinity() -> None:
    est = EstimatedTokensPerExample.from_dict(
        json.loads('{"input_tokens":1e999,"output_tokens":1}')
    )
    assert (est.input_tokens, est.output_tokens) == (0, 1)


def test_input_validator_raises_validation_error() -> None:
    with pytest.raises(ValidationError):
        InputValidator.validate_numeric(HUGE)
    with pytest.raises(ValidationError):
        InputValidator.validate_integer(INF)


def test_validate_numeric_metric_raises_metric_error() -> None:
    with pytest.raises(MetricExtractionError):
        validate_numeric_metric(HUGE, "accuracy")
    with pytest.raises(MetricExtractionError):
        validate_numeric_metric(INF, "accuracy")


_BASE = """\
tvl:
  module: corp.validation.overflow_3137
environment:
  snapshot_id: "2025-01-01T00:00:00Z"
evaluation_set:
  dataset: s3://datasets/validation/dev.parquet
tvars:
  - name: model
    type: enum[str]
    domain: ["gpt-4", "claude-3"]
constraints:
  structural: []
objectives:
  - name: quality
    metric_ref: metrics.quality.v1
    direction: maximize
"""


@pytest.mark.parametrize("digits", [400, 5000])
def test_load_tvl_spec_huge_integer_is_validation_error(
    tmp_path: Path, digits: int
) -> None:
    path = tmp_path / "spec.tvl.yml"
    path.write_text(
        _BASE
        + "exploration:\n  strategy:\n    type: random\n  budgets:\n"
        + f"    max_spend_usd: {'9' * digits}\n",
        encoding="utf-8",
    )
    with pytest.raises(TVLValidationError):
        load_tvl_spec(spec_path=path)
