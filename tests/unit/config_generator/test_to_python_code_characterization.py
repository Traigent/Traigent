"""Golden characterization for AutoConfigResult.to_python_code."""

from __future__ import annotations

import pytest

from traigent.config_generator.types import (
    AutoConfigResult,
    ObjectiveSpec,
    SafetySpec,
    StructuralConstraintSpec,
    TVarRecommendation,
    TVarSpec,
)

TVAR = TVarSpec("temperature", "Range", {"low": 0.0, "high": 2.0})
TVAR2 = TVarSpec("model", "Choices", {"values": ["a", "b"]})
OBJ = ObjectiveSpec("accuracy", "maximize", 0.5)
REC = TVarRecommendation("top_k", "IntRange", {"low": 1, "high": 9})


def test_empty_result():
    assert AutoConfigResult().to_python_code() == "@traigent.optimize(\n)"


def test_all_sections_in_order():
    result = AutoConfigResult(
        tvars=(TVAR, TVAR2),
        objectives=(OBJ, ObjectiveSpec("cost", "minimize")),
        safety_constraints=(
            SafetySpec("hallucination_rate", "<=", 0.1),
            SafetySpec("custom_metric", ">=", 0.9),
            SafetySpec("custom_metric", "<=", 3),
        ),
        recommendations=(REC,),
    )
    assert result.to_python_code().split("\n") == [
        "@traigent.optimize(",
        "    configuration_space={",
        "        'temperature': Range(low=0.0, high=2.0),",
        "        'model': Choices(values=['a', 'b']),",
        "    },",
        "    objectives=ObjectiveSchema.from_objectives([",
        "        ObjectiveDefinition(name='accuracy', orientation='maximize', weight=0.5),",
        "        ObjectiveDefinition(name='cost', orientation='minimize', weight=1.0),",
        "    ]),",
        "    safety_constraints=[",
        "        hallucination_rate().below(0.1),",
        "        custom_metric.above(0.9),",
        "        custom_metric.below(3),",
        "    ],",
        "    # Recommended additional TVars:",
        "    #   'top_k': IntRange(low=1, high=9),",
        ")",
    ]


def test_recommendations_only_has_comment_lines():
    assert AutoConfigResult(recommendations=(REC,)).to_python_code() == (
        "@traigent.optimize(\n"
        "    # Recommended additional TVars:\n"
        "    #   'top_k': IntRange(low=1, high=9),\n)"
    )


def test_structural_constraints_raise_before_anything():
    sc = StructuralConstraintSpec("a needs b", "x > y")
    with pytest.raises(ValueError, match="1 generated structural constraint"):
        AutoConfigResult(tvars=(TVAR,), structural_constraints=(sc,)).to_python_code()


def test_structural_error_truncates_after_three():
    scs = tuple(StructuralConstraintSpec(f"d{i}", f"c{i}") for i in range(5))
    with pytest.raises(ValueError, match=r"d0; d1; d2 \(\+2 more\)"):
        AutoConfigResult(structural_constraints=scs).to_python_code()
