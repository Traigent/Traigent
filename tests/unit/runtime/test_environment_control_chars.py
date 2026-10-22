"""Environment names containing control characters are rejected at input."""

from __future__ import annotations

import pytest
import yaml

import traigent
from traigent.runtime.model import E_INVALID, RuntimeRequirementError
from traigent.runtime.tvl_fragment import parse_fragment

from .conftest import FIXTURES

# Embedded (not leading/trailing) so str.strip() cannot remove them first.
BAD = [
    "prod\x1bx",
    "prod\x00x",
    "prod\x7fx",
    "prod\u0085x",
    "prod x",
    "prod x",
    "prod\nx",
    "prod\x9bx",
]
GOOD = ["production", "eu-west-1", "staging_2", "  production  "]


def _declare(environment: str):
    @traigent.requirement(
        latency_ms=2000,
        max_violation_rate=0.05,
        on_breach="log",
        environment=environment,
    )
    def fn() -> None: ...

    return fn.requirements[0]


@pytest.mark.parametrize("env", BAD)
def test_decorator_rejects_control_characters(env: str) -> None:
    with pytest.raises(RuntimeRequirementError) as err:
        _declare(env)
    assert err.value.code == E_INVALID
    assert err.value.path == "environment"
    assert "prod" not in str(err.value) and "prod" not in repr(err.value.path)


@pytest.mark.parametrize("env", BAD)
def test_tvl_fragment_rejects_control_characters(env: str) -> None:
    doc = yaml.safe_load((FIXTURES / "latency.tvl.yml").read_text())
    doc["monitors"][0]["evidence"]["deployment_environment"] = env
    with pytest.raises(RuntimeRequirementError) as err:
        parse_fragment(doc)
    assert err.value.code == E_INVALID
    assert "prod" not in str(err.value) and "\x1b" not in str(err.value)


@pytest.mark.parametrize("env", GOOD)
def test_ordinary_environment_names_still_accepted(env: str) -> None:
    assert _declare(env).environment == env
