"""Example pytest module that exercises the Chapter 3 CEL constraints."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict
import re

import yaml


SPEC_PATH = Path(__file__).with_name("ch3_constraints_units.tvl.yml")
SPEC = yaml.safe_load(SPEC_PATH.read_text(encoding="utf-8"))


def evaluate_constraint(expr: str, params: Dict[str, Any]) -> bool:
    """Tiny CEL-like evaluator for demonstration purposes."""
    python_expr = expr.replace("&&", " and ").replace("||", " or ")
    python_expr = re.sub(r"!(?!=)", " not ", python_expr)
    activation = {"params": SimpleNamespace(**params)}
    return bool(eval(python_expr, {}, activation))


def test_temperature_token_budget_passes() -> None:
    params = {
        "temperature": 0.5,
        "max_tokens": 512,
        "cache_ttl_hours": 2.0,
        "rerank_weight": 0.5,
    }
    expr = next(c["rule"] for c in SPEC["constraints"] if c["id"] == "token-temperature-budget")
    assert evaluate_constraint(expr, params)


def test_temperature_token_budget_fails() -> None:
    params = {
        "temperature": 0.95,
        "max_tokens": 768,
        "cache_ttl_hours": 3.0,
        "rerank_weight": 0.6,
    }
    expr = next(c["rule"] for c in SPEC["constraints"] if c["id"] == "token-temperature-budget")
    assert not evaluate_constraint(expr, params)


def test_rerank_requirement_triggers() -> None:
    constraint = next(c for c in SPEC["constraints"] if c["id"] == "rerank-mandate")
    params = {
        "temperature": 0.3,
        "max_tokens": 640,
        "cache_ttl_hours": 1.5,
        "rerank_weight": 0.2,
    }
    when_ok = evaluate_constraint(constraint["when"], params)
    then_ok = evaluate_constraint(constraint["then"], params)
    assert when_ok and not then_ok
