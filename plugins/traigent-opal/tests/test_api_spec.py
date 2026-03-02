from __future__ import annotations

import pytest
from traigent_opal import (
    ConstraintSpec,
    DomainChoices,
    DomainRange,
    chance_constraint,
    choices,
    compile_opal_spec,
    constraint,
    frange,
    maximize,
    minimize,
    program,
    tvar,
    when,
)


def test_choices_requires_non_empty_domain() -> None:
    with pytest.raises(ValueError):
        choices()


def test_frange_rejects_non_positive_step() -> None:
    with pytest.raises(ValueError):
        frange(0.0, 1.0, 0.0)


def test_choices_rejects_set_for_determinism() -> None:
    with pytest.raises(ValueError):
        choices({1, 2, 3})


def test_program_rejects_duplicate_tvars() -> None:
    with pytest.raises(ValueError):
        program(
            tvars=[
                tvar("temperature", choices(0.1, 0.3)),
                tvar("temperature", choices(0.5, 0.7)),
            ]
        )


def test_program_rejects_assignment_tvar_collision() -> None:
    with pytest.raises(ValueError):
        program(
            tvars=[tvar("model", choices("gpt-4o-mini", "gpt-4o"))],
            assignments={"model": "gpt-4o-mini"},
        )


def test_compile_opal_spec_emits_expected_artifact() -> None:
    spec = program(
        module="corp.support.rag_bot",
        tvars=[
            tvar("model", choices("gpt-4o-mini", "gpt-4o")),
            tvar("temperature", frange(0.0, 0.6, 0.3)),
            tvar("top_k", choices(3, 5, 8, 12)),
        ],
        assignments={"max_tokens": 512},
        objectives=[maximize("quality", on="eval_dataset", weight=0.6), minimize("cost", weight=0.4)],
        constraints=[when("model", "gpt-4o", "temperature <= 0.3"), constraint("cost <= 0.02")],
        chance_constraints=[chance_constraint("latency > 1200ms", threshold=0.03, confidence=0.95)],
    )

    artifact = compile_opal_spec(spec)

    assert artifact.module_name == "corp.support.rag_bot"
    assert artifact.tuned_variables["model"] == ["gpt-4o-mini", "gpt-4o"]
    assert artifact.tuned_variables["temperature"] == [0.0, 0.3, 0.6]
    assert artifact.assignments["max_tokens"] == 512
    assert artifact.constraints[0] == "when model is 'gpt-4o': temperature <= 0.3"
    assert artifact.constraints[1] == "cost <= 0.02"
    assert artifact.chance_constraints == ["P(latency > 1200ms) <= 0.03 confidence 0.95"]


def test_compile_opal_spec_emits_objective_schema() -> None:
    spec = program(
        tvars=[tvar("model", choices("gpt-4o-mini", "gpt-4o"))],
        objectives=[maximize("quality", weight=0.7), minimize("cost", weight=0.3)],
    )

    artifact = compile_opal_spec(spec)
    kwargs = artifact.to_optimize_kwargs()

    assert artifact.objective_schema is not None
    schema = kwargs["objectives"]
    assert schema.get_orientation("quality") == "maximize"
    assert schema.get_orientation("cost") == "minimize"


def test_domain_range_rejects_invalid_bounds() -> None:
    with pytest.raises(ValueError):
        DomainRange(lo=1.0, hi=0.0, step=0.1)


def test_domain_types_are_exposed() -> None:
    assert isinstance(choices("a", "b"), DomainChoices)
    assert isinstance(frange(0.0, 1.0, 0.1), DomainRange)


def test_when_requires_non_none_is_value_and_non_empty_then_expr() -> None:
    with pytest.raises(ValueError):
        # missing is_value
        when("model", None, "temperature <= 0.3")

    with pytest.raises(ValueError):
        # empty then arm
        when("model", "gpt-4o", "")


def test_constraint_spec_rejects_mixed_expr_and_guard_form() -> None:
    with pytest.raises(ValueError):
        ConstraintSpec(
            expr="cost <= 0.02",
            when_name="model",
            is_value="gpt-4o",
            then_expr="temperature <= 0.3",
        )
