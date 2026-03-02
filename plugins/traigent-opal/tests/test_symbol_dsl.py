from __future__ import annotations

from typing import Any

import pytest
from traigent_opal import (
    choices,
    compile_opal_spec,
    frange,
    opal_program,
    program,
    tv,
    tvar,
)

from traigent import Choices, IntRange, Range


def test_scoped_symbol_dsl_builds_program_spec() -> None:
    with opal_program(module="corp.support.rag_bot") as v:
        v.model_name in tv(choices("gpt-4o-mini", "gpt-4o"))
        v.temperature in tv(frange(0.0, 0.6, 0.3))
        v.assign("max_tokens", 512)
        v.maximize("quality", on="eval_dataset")
        v.minimize("cost_per_query")
        v.when(v.model_name, "gpt-4o", "temperature <= 0.3")
        v.constraint("cost_per_query <= 0.02")
        v.chance_constraint("latency > 1200ms", threshold=0.03, confidence=0.95)
        spec = v.spec

    assert spec.module == "corp.support.rag_bot"
    assert [tv_.name for tv_ in spec.tvars] == ["model_name", "temperature"]
    assert spec.assignments["max_tokens"] == 512
    assert len(spec.objectives) == 2
    assert len(spec.constraints) == 2
    assert len(spec.chance_constraints) == 1


def test_symbol_dsl_compilation_equivalent_to_object_api() -> None:
    with opal_program(module="examples.model_selection") as v:
        v.model in tv(choices("gpt-4o-mini", "gpt-4o"))
        v.temperature in tv(frange(0.0, 1.0, 0.1))
        v.assign("max_tokens", 512)
        v.maximize("quality", on="eval_dataset")
        v.minimize("cost_per_query")
        v.constraint("cost_per_query <= 0.02")
        scoped_spec = v.build()

    object_spec = program(
        module="examples.model_selection",
        tvars=[
            tvar("model", choices("gpt-4o-mini", "gpt-4o")),
            tvar("temperature", frange(0.0, 1.0, 0.1)),
        ],
        assignments={"max_tokens": 512},
        objectives=[],
        constraints=[],
    )
    object_spec.objectives.extend(scoped_spec.objectives)
    object_spec.constraints.extend(scoped_spec.constraints)

    a = compile_opal_spec(scoped_spec)
    b = compile_opal_spec(object_spec)

    assert a.module_name == b.module_name
    assert a.tuned_variables == b.tuned_variables
    assert a.assignments == b.assignments
    assert a.constraints == b.constraints
    assert [o.metric for o in a.objectives] == [o.metric for o in b.objectives]


def test_symbol_dsl_rejects_outside_context() -> None:
    builder = opal_program(module="examples.bad")
    with pytest.raises(AttributeError):
        builder.model in tv(choices("a", "b"))


def test_symbol_dsl_rejects_assignment_tvar_collision() -> None:
    with pytest.raises(ValueError):
        with opal_program(module="examples.bad") as v:
            v.assign("model", "gpt-4o-mini")
            v.model in tv(choices("gpt-4o-mini", "gpt-4o"))


def test_tv_accepts_traigent_core_range_types() -> None:
    with opal_program(module="examples.core_bridge") as v:
        v.model in tv(Choices(["a", "b"]))
        v.temperature in tv(Range(0.0, 1.0))
        v.top_k in tv(IntRange(2, 8, step=2))
        spec = v.build()

    artifact = compile_opal_spec(spec)
    assert artifact.tuned_variables["model"] == ["a", "b"]
    assert artifact.tuned_variables["temperature"] == (0.0, 1.0)
    assert artifact.tuned_variables["top_k"] == [2.0, 4.0, 6.0, 8.0]


def test_builder_objective_defaults_to_evaluation_set() -> None:
    with opal_program(module="examples.eval", evaluation_set="eval_dataset") as v:
        v.model in tv(choices("a", "b"))
        v.maximize("quality")
        spec = v.build()

    artifact = compile_opal_spec(spec)
    assert artifact.objectives[0].raw == "objective maximize quality on eval_dataset"


def test_builder_deploy_binds_values_without_optimizer() -> None:
    with opal_program(module="examples.deploy") as v:
        v.model in tv(choices("a", "b"))
        v.maximize("quality")

    @v.deploy(config={"model": "b"})
    def run() -> str:
        return v.model

    assert run() == "b"


def test_builder_optimize_uses_traigent_decorator(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def fake_optimize(**kwargs: Any):
        captured.update(kwargs)

        def _decorator(fn):
            return fn

        return _decorator

    import traigent

    monkeypatch.setattr(traigent, "optimize", fake_optimize)
    monkeypatch.setattr(traigent, "get_config", lambda: {"model": "a"})

    with opal_program(module="examples.optimize") as v:
        v.model in tv(choices("a", "b"))
        v.maximize("quality")

    @v.optimize(max_trials=3)
    def run() -> str:
        return v.model

    assert run() == "a"
    assert captured["model"] == ["a", "b"]
    assert captured["max_trials"] == 3


def test_builder_optimize_without_parentheses(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def fake_optimize(**kwargs: Any):
        captured.update(kwargs)

        def _decorator(fn):
            return fn

        return _decorator

    import traigent

    monkeypatch.setattr(traigent, "optimize", fake_optimize)
    monkeypatch.setattr(traigent, "get_config", lambda: {"model": "b"})

    with opal_program(module="examples.optimize.noargs") as v:
        v.model in tv(choices("a", "b"))
        v.maximize("quality")

    @v.optimize
    def run() -> str:
        return v.model

    assert run() == "b"
    assert captured["model"] == ["a", "b"]
