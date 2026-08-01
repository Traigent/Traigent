"""Tests for deterministic cold-start scenario generators."""

from __future__ import annotations

import pytest

from traigent.generation.coldstart.contracts import (
    ColdStartConfigurationError,
    ParameterSpec,
    SystemSpec,
)
from traigent.generation.coldstart.generators import ContractGroundedGenerator


def _system(*parameters: ParameterSpec) -> SystemSpec:
    return SystemSpec("answer", "example", parameters, "str", (), "a" * 64)


def test_contract_grounded_generator_is_deterministic_and_input_only() -> None:
    generator = ContractGroundedGenerator()
    system = _system(
        ParameterSpec("question", "str", True), ParameterSpec("retry", "int", False)
    )

    candidates = generator.propose(system, count=2, seed=7)

    assert [candidate.inputs for candidate in candidates] == [
        {"question": "coldstart-question-7-0", "retry": 7},
        {"question": "coldstart-question-7-1", "retry": 8},
    ]
    assert all(candidate.ground_truth is None for candidate in candidates)
    assert generator.propose(system, count=2, seed=7) == candidates


def test_contract_grounded_generator_supports_pep604_optional_annotations() -> None:
    generator = ContractGroundedGenerator()
    system = _system(
        ParameterSpec("question", "str | None", False),
        ParameterSpec("retry", "None | int", False),
        ParameterSpec("ratio", "float | None", False),
        ParameterSpec("enabled", "None | bool", False),
    )

    candidates = generator.propose(system, count=1, seed=7)

    assert [dict(candidate.inputs) for candidate in candidates] == [
        {
            "question": "coldstart-question-7-0",
            "retry": 7,
            "ratio": 7.0,
            "enabled": True,
        }
    ]


def test_contract_grounded_generator_fails_closed_for_unsupported_or_invalid_contracts() -> (
    None
):
    generator = ContractGroundedGenerator()
    unsupported = _system(ParameterSpec("document", "dict[str, str]", True))

    assert generator.propose(unsupported, count=1, seed=0) == []
    with pytest.raises(ColdStartConfigurationError, match="positive"):
        generator.propose(unsupported, count=0, seed=0)
