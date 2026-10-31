"""Environment names containing control characters are rejected at input."""

from __future__ import annotations

import pytest
import yaml

import traigent
from traigent.runtime.model import E_INVALID, RuntimeRequirementError
from traigent.runtime.tvl_fragment import parse_fragment

from .conftest import FIXTURES

# S2 shared fixture (identical in the JS SDK). The check runs on the ORIGINAL input,
# before trim, so unsafe characters at the ends are rejected too.
_UNSAFE = [
    "\x1b",
    "\x00",
    "\x7f",
    "\u0085",
    "\u2028",
    "\u202e",
    "\ufeff",
]
BAD = (
    [f"prod{c}x" for c in _UNSAFE]
    + [f"{c}production" for c in _UNSAFE]
    + [f"production{c}" for c in _UNSAFE]
    + ["prod\nx", "prod\x9bx", "prod\u200bx", "prod\u061cx", "a" * 65]
    + ["\u0130" * 64]  # lowercases to 128 code points
)
GOOD = [
    "production",
    "eu-west-1",
    "staging_2",
    "  production  ",
    "z\u200dw",  # ZWJ is a legitimate joiner
    "\U0001f600" * 40,  # 40 code points, 80 UTF-16 units
]


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
    assert str(err.value) == "invalid_requirement at environment"


@pytest.mark.parametrize("env", BAD)
def test_tvl_fragment_rejects_control_characters(env: str) -> None:
    doc = yaml.safe_load((FIXTURES / "latency.tvl.yml").read_text())
    doc["monitors"][0]["evidence"]["deployment_environment"] = env
    with pytest.raises(RuntimeRequirementError) as err:
        parse_fragment(doc)
    assert err.value.code == E_INVALID
    assert str(err.value) == "invalid_requirement at environment"


@pytest.mark.parametrize("env", GOOD)
def test_ordinary_environment_names_still_accepted(env: str) -> None:
    assert _declare(env).environment == env


def test_exact_boundary_lengths() -> None:
    assert _declare("a" * 64)
    assert _declare("\u0130" * 32)  # 32 -> 64 code points once lowercased
