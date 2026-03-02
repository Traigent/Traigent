from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from traigent_opal import (
    OpalCompileError,
    choices,
    compile_opal_file,
    compile_opal_source,
    compile_opal_spec,
    constraint,
    maximize,
    minimize,
    opal_optimize,
    program,
    tvar,
)


def test_compile_opal_source_supports_python_comment_directives() -> None:
    source = """
    module examples.simple

    model in {"gpt-4o-mini", "gpt-4o"}
    temperature in [0.0, 1.0] step 0.5
    max_tokens = 512

    # opal: objective maximize quality on eval_dataset
    # opal: objective minimize cost_per_query
    # opal: constraint cost_per_query <= 0.02
    # opal: chance_constraint latency_ms <= 400 confidence 0.95

    def run(query: str) -> str:
        return query
    """

    artifact = compile_opal_source(source)

    assert artifact.module_name == "examples.simple"
    assert artifact.tuned_variables["model"] == ["gpt-4o-mini", "gpt-4o"]
    assert artifact.tuned_variables["temperature"] == [0.0, 0.5, 1.0]
    assert artifact.assignments["max_tokens"] == 512

    assert [o.metric for o in artifact.objectives] == ["quality", "cost_per_query"]
    assert [o.direction for o in artifact.objectives] == ["maximize", "minimize"]
    assert artifact.objectives[0].dataset == "eval_dataset"
    assert artifact.constraints == ["cost_per_query <= 0.02"]
    assert artifact.chance_constraints == ["latency_ms <= 400 confidence 0.95"]


def test_compile_opal_source_returns_optimize_kwargs() -> None:
    source = """
    model in {"gpt-4o-mini", "gpt-4o"}
    temperature in [0.0, 1.0]
    max_tokens = 256
    # opal: objective maximize quality
    # opal: objective minimize cost
    """

    kwargs = compile_opal_source(source).to_optimize_kwargs()

    assert kwargs["model"] == ["gpt-4o-mini", "gpt-4o"]
    assert kwargs["temperature"] == (0.0, 1.0)
    assert kwargs["max_tokens"] == 256
    assert kwargs["objectives"].get_orientation("quality") == "maximize"
    assert kwargs["objectives"].get_orientation("cost") == "minimize"


def test_compile_opal_file_roundtrip(tmp_path: Path) -> None:
    opal_file = tmp_path / "agent.opal"
    opal_file.write_text(
        """
model in {"a", "b"}
# opal: objective maximize score
""".strip(),
        encoding="utf-8",
    )

    artifact = compile_opal_file(opal_file)
    assert artifact.tuned_variables["model"] == ["a", "b"]
    assert artifact.objectives[0].metric == "score"


def test_compile_rejects_assignment_tvar_collision() -> None:
    source = """
    model = "gpt-4o-mini"
    model in {"gpt-4o-mini", "gpt-4o"}
    """

    with pytest.raises(OpalCompileError) as exc:
        compile_opal_source(source)

    assert "cannot redeclare with 'in'" in str(exc.value)


def test_compile_assignment_spacing_variants_are_supported() -> None:
    source = """
    a=1
    b =1
    c= 1
    d = 1
    # opal: objective maximize quality
    """

    artifact = compile_opal_source(source)
    assert artifact.assignments == {"a": 1, "b": 1, "c": 1, "d": 1}


def test_compile_range_discretization_normalizes_decimal_artifacts() -> None:
    source = """
    temperature in [0.0, 1.0] step 0.1
    # opal: objective maximize quality
    """

    artifact = compile_opal_source(source)
    assert artifact.tuned_variables["temperature"] == [
        0.0,
        0.1,
        0.2,
        0.3,
        0.4,
        0.5,
        0.6,
        0.7,
        0.8,
        0.9,
        1.0,
    ]


def test_opal_optimize_builds_decorator_kwargs(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def fake_optimize(**kwargs: Any):
        captured.update(kwargs)

        def _decorator(fn):
            return fn

        return _decorator

    import traigent

    monkeypatch.setattr(traigent, "optimize", fake_optimize)

    source = """
    model in {"gpt-4o-mini", "gpt-4o"}
    # opal: objective maximize quality
    """

    decorator = opal_optimize(source, max_trials=5)

    @decorator
    def _agent(query: str) -> str:
        return query

    assert captured["model"] == ["gpt-4o-mini", "gpt-4o"]
    assert captured["objectives"].get_orientation("quality") == "maximize"
    assert captured["max_trials"] == 5


def test_to_optimize_kwargs_rejects_manual_artifact_collisions() -> None:
    artifact = compile_opal_source(
        """
        model in {"a", "b"}
        # opal: objective maximize quality
        """
    )
    artifact.assignments["model"] = "a"

    with pytest.raises(OpalCompileError):
        artifact.to_optimize_kwargs()

def test_compile_opal_source_supports_raw_objective_lines() -> None:
    source = """
    model in {"gpt-4o-mini", "gpt-4o"}
    objective maximize quality on eval_dataset
    constraint cost_per_query <= 0.02
    """

    artifact = compile_opal_source(source)
    assert artifact.objectives[0].metric == "quality"
    assert artifact.constraints == ["cost_per_query <= 0.02"]


def test_python_to_opal_to_python_opal_equivalence() -> None:
    raw_opal = """
    module examples.model_selection
    model in {"gpt-4o-mini", "gpt-4o"}
    temperature in [0.0, 1.0] step 0.1
    max_tokens = 512
    objective maximize quality on eval_dataset
    objective minimize cost_per_query
    constraint cost_per_query <= 0.02
    """

    python_opal = """
    # Plain behavior code can coexist with OPAL directives
    model in {"gpt-4o-mini", "gpt-4o"}
    temperature in [0.0, 1.0] step 0.1
    max_tokens = 512
    # opal: objective maximize quality on eval_dataset
    # opal: objective minimize cost_per_query
    # opal: constraint cost_per_query <= 0.02
    llm = ChatOpenAI(model=model, temperature=temperature, max_tokens=max_tokens)
    """

    raw_artifact = compile_opal_source(raw_opal)
    py_artifact = compile_opal_source(python_opal)

    assert raw_artifact.tuned_variables == py_artifact.tuned_variables
    assert raw_artifact.assignments == py_artifact.assignments
    assert [o.metric for o in raw_artifact.objectives] == [
        o.metric for o in py_artifact.objectives
    ]
    assert raw_artifact.constraints == py_artifact.constraints

def test_compile_ignores_behavior_plane_assignments() -> None:
    source = """
    model in {"gpt-4o-mini", "gpt-4o"}
    # opal: objective maximize quality
    llm = ChatOpenAI(model=model, temperature=0.2)
    """

    artifact = compile_opal_source(source)
    assert "llm" not in artifact.assignments
    assert artifact.tuned_variables["model"] == ["gpt-4o-mini", "gpt-4o"]


def test_opal_plugin_declares_tvl_feature() -> None:
    from traigent_opal import OpalPlugin

    from traigent.plugins import FEATURE_TVL

    plugin = OpalPlugin()
    assert plugin.name == "traigent-opal"
    assert FEATURE_TVL in plugin.provides_features()


def test_compile_opal_spec_and_source_equivalence_subset() -> None:
    source = """
    module examples.model_selection
    model in {"gpt-4o-mini", "gpt-4o"}
    temperature in [0.0, 1.0] step 0.1
    max_tokens = 512
    objective maximize quality on eval_dataset
    objective minimize cost_per_query
    constraint cost_per_query <= 0.02
    """

    spec = program(
        module="examples.model_selection",
        tvars=[
            tvar("model", choices("gpt-4o-mini", "gpt-4o")),
            tvar("temperature", choices(*[round(x * 0.1, 1) for x in range(11)])),
        ],
        assignments={"max_tokens": 512},
        objectives=[maximize("quality", on="eval_dataset"), minimize("cost_per_query")],
        constraints=[constraint("cost_per_query <= 0.02")],
    )

    a = compile_opal_source(source)
    b = compile_opal_spec(spec)

    assert a.module_name == b.module_name
    assert a.assignments == b.assignments
    assert [o.metric for o in a.objectives] == [o.metric for o in b.objectives]
    assert a.constraints == b.constraints


def test_opal_optimize_accepts_program_spec(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def fake_optimize(**kwargs: Any):
        captured.update(kwargs)

        def _decorator(fn):
            return fn

        return _decorator

    import traigent

    monkeypatch.setattr(traigent, "optimize", fake_optimize)

    spec = program(
        tvars=[tvar("model", choices("gpt-4o-mini", "gpt-4o"))],
        objectives=[maximize("quality"), minimize("cost")],
    )

    decorator = opal_optimize(spec, max_trials=7)

    @decorator
    def _agent(query: str) -> str:
        return query

    assert captured["model"] == ["gpt-4o-mini", "gpt-4o"]
    assert captured["max_trials"] == 7
    assert captured["objectives"].get_orientation("quality") == "maximize"
    assert captured["objectives"].get_orientation("cost") == "minimize"


def test_opal_optimize_forwards_full_traigent_options(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def fake_optimize(**kwargs: Any):
        captured.update(kwargs)

        def _decorator(fn):
            return fn

        return _decorator

    import traigent

    monkeypatch.setattr(traigent, "optimize", fake_optimize)

    source = """
    model in {"gpt-4o-mini", "gpt-4o"}
    # opal: objective maximize quality
    """
    decorator = opal_optimize(
        source,
        algorithm="bayesian",
        max_trials=25,
        injection_mode="parameter",
        config_param="cfg",
        evaluation={"eval_dataset": "eval.jsonl"},
        injection={"injection_mode": "context"},
        execution={
            "execution_mode": "edge_analytics",
            "minimal_logging": False,
            "runtime": "python",
        },
        budget_limit=2.5,
        budget_metric="cost_per_query",
        budget_include_pruned=True,
        max_total_examples=200,
        samples_include_pruned=False,
    )

    @decorator
    def _agent(query: str) -> str:
        return query

    assert captured["model"] == ["gpt-4o-mini", "gpt-4o"]
    assert captured["objectives"].get_orientation("quality") == "maximize"
    assert captured["algorithm"] == "bayesian"
    assert captured["max_trials"] == 25
    assert captured["injection_mode"] == "parameter"
    assert captured["config_param"] == "cfg"
    assert captured["evaluation"] == {"eval_dataset": "eval.jsonl"}
    assert captured["injection"] == {"injection_mode": "context"}
    assert captured["execution"]["execution_mode"] == "edge_analytics"
    assert captured["execution"]["runtime"] == "python"
    assert captured["budget_limit"] == 2.5
    assert captured["budget_metric"] == "cost_per_query"
    assert captured["budget_include_pruned"] is True
    assert captured["max_total_examples"] == 200
    assert captured["samples_include_pruned"] is False


def test_opal_optimize_missing_opal_file_raises() -> None:
    with pytest.raises(OpalCompileError):
        opal_optimize("missing_spec.opal")
