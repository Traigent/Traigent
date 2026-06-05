"""Tests for config_generator.types."""

from __future__ import annotations

import pytest

from traigent.config_generator.types import (
    AutoConfigResult,
    BenchmarkSpec,
    EvidenceRef,
    ObjectiveSpec,
    SafetySpec,
    StructuralConstraintSpec,
    TVarRecommendation,
    TVarSpec,
)


class TestTVarSpec:
    def test_frozen(self) -> None:
        spec = TVarSpec(
            name="temperature",
            range_type="Range",
            range_kwargs={"low": 0.0, "high": 1.0},
        )
        with pytest.raises(AttributeError):
            spec.name = "other"  # type: ignore[misc]

    def test_to_range_code_float(self) -> None:
        spec = TVarSpec(
            name="temperature",
            range_type="Range",
            range_kwargs={"low": 0.0, "high": 2.0},
        )
        assert spec.to_range_code() == "Range(low=0.0, high=2.0)"

    def test_to_range_code_int(self) -> None:
        spec = TVarSpec(
            name="max_tokens",
            range_type="IntRange",
            range_kwargs={"low": 256, "high": 4096},
        )
        assert spec.to_range_code() == "IntRange(low=256, high=4096)"

    def test_to_range_code_choices(self) -> None:
        spec = TVarSpec(
            name="model",
            range_type="Choices",
            range_kwargs={"values": ["gpt-4o", "gpt-4o-mini"]},
        )
        assert "Choices(values=" in spec.to_range_code()

    def test_defaults(self) -> None:
        spec = TVarSpec(name="x", range_type="Range")
        assert spec.source == "preset"
        assert spec.confidence == 1.0
        assert spec.reasoning == ""
        assert spec.range_kwargs == {}


class TestObjectiveSpec:
    def test_defaults(self) -> None:
        obj = ObjectiveSpec(name="accuracy")
        assert obj.orientation == "maximize"
        assert obj.weight == 1.0
        assert obj.source == "default"

    def test_custom_values(self) -> None:
        obj = ObjectiveSpec(
            name="cost", orientation="minimize", weight=0.3, source="heuristic"
        )
        assert obj.orientation == "minimize"
        assert obj.weight == 0.3


class TestBenchmarkSpec:
    def test_creation(self) -> None:
        bm = BenchmarkSpec(name="QA Benchmark", description="Standard QA")
        assert bm.source == "catalog"
        assert bm.sample_examples == ()

    def test_with_schema(self) -> None:
        schema = {"input": {"question": "str"}, "output": "str"}
        bm = BenchmarkSpec(name="QA", example_schema=schema)
        assert bm.example_schema == schema


class TestSafetySpec:
    def test_creation(self) -> None:
        sc = SafetySpec(metric_name="hallucination_rate", operator="<=", threshold=0.15)
        assert sc.agent_type == ""
        assert sc.source == "preset"

    def test_with_agent_type(self) -> None:
        sc = SafetySpec(
            metric_name="faithfulness",
            operator=">=",
            threshold=0.85,
            agent_type="rag",
            reasoning="RAG agents need high faithfulness",
        )
        assert sc.agent_type == "rag"


class TestStructuralConstraintSpec:
    def test_creation(self) -> None:
        sc = StructuralConstraintSpec(
            description="Conservative temperature for factual models",
            constraint_code="implies(model.equals('gpt-4o'), temperature.lte(1.0))",
            requires_tvars=("model", "temperature"),
        )
        assert "implies" in sc.constraint_code
        assert sc.source == "template"


class TestEvidenceRef:
    def test_creation_defaults(self) -> None:
        ref = EvidenceRef(
            scope="isolation",
            metric="execution_accuracy",
            n=10,
            model="bedrock/us.anthropic.claude-haiku-4-5",
            baseline="none",
            candidate="linked_top6",
        )
        assert ref.delta is None
        assert ref.limitations == ()
        # Public-safe contract: no internal artifact paths / run IDs.
        assert not hasattr(ref, "artifact_path")
        assert not hasattr(ref, "run_id")

    def test_frozen(self) -> None:
        ref = EvidenceRef(
            scope="isolation",
            metric="execution_accuracy",
            n=10,
            model="model",
            baseline="none",
            candidate="linked_top6",
        )
        with pytest.raises(AttributeError):
            ref.metric = "other"  # type: ignore[misc]


class TestTVarRecommendation:
    def test_to_range_code(self) -> None:
        rec = TVarRecommendation(
            name="prompting_strategy",
            range_type="Choices",
            range_kwargs={"values": ["direct", "chain_of_thought"]},
            category="prompting",
            reasoning="Try different prompting strategies",
            impact_estimate="high",
        )
        assert "Choices(values=" in rec.to_range_code()

    def test_defaults(self) -> None:
        rec = TVarRecommendation(name="x", range_type="Range")
        assert rec.impact_estimate == "medium"
        assert rec.category == ""
        assert rec.entry_id == ""
        assert rec.catalog_entry_id == ""
        assert rec.kind == ""
        assert rec.effectuation_status == ""
        assert rec.effectuation_strategy == ""
        assert rec.evidence_refs == ()
        assert rec.apply_guidance == ""

    def test_with_evidence_and_guidance(self) -> None:
        ref = EvidenceRef(
            scope="isolation",
            metric="execution_accuracy",
            n=10,
            model="model",
            baseline="none",
            candidate="linked_top6",
            delta=0.4,
            limitations=("single_slice", "not_sota"),
        )
        rec = TVarRecommendation(
            name="schema_context",
            range_type="Choices",
            catalog_entry_id="code_gen.schema_context.v1",
            kind="topology",
            effectuation_status="manual_guidance",
            evidence_refs=(ref,),
            apply_guidance="Manual wiring required.",
            recommended_values=("linked_top10",),
        )
        assert rec.evidence_refs == (ref,)
        assert rec.apply_guidance == "Manual wiring required."
        assert rec.catalog_entry_id == "code_gen.schema_context.v1"
        assert rec.kind == "topology"
        assert rec.effectuation_status == "manual_guidance"
        assert rec.recommended_values == ("linked_top10",)


class TestAutoConfigResult:
    def test_empty_result(self) -> None:
        result = AutoConfigResult()
        assert result.tvars == ()
        assert result.objectives == ()
        assert result.llm_calls_made == 0
        assert result.llm_cost_usd == 0.0

    def test_to_decorator_kwargs_empty(self) -> None:
        result = AutoConfigResult()
        kwargs = result.to_decorator_kwargs()
        assert kwargs == {}

    def test_to_decorator_kwargs_returns_live_objects(self) -> None:
        from traigent.api.parameter_ranges import Choices, Range

        result = AutoConfigResult(
            tvars=(
                TVarSpec(
                    name="temperature",
                    range_type="Range",
                    range_kwargs={"low": 0.0, "high": 1.0},
                ),
                TVarSpec(
                    name="model",
                    range_type="Choices",
                    range_kwargs={"values": ["gpt-4o"]},
                ),
            ),
            objectives=(
                ObjectiveSpec(name="accuracy"),
                ObjectiveSpec(name="cost", orientation="minimize"),
            ),
        )
        kwargs = result.to_decorator_kwargs()
        assert "configuration_space" in kwargs
        assert "objectives" in kwargs

        # Must return actual ParameterRange objects, not dicts
        temp = kwargs["configuration_space"]["temperature"]
        assert isinstance(temp, Range)
        assert temp.low == pytest.approx(0.0)
        assert temp.high == pytest.approx(1.0)

        model = kwargs["configuration_space"]["model"]
        assert isinstance(model, Choices)
        assert "gpt-4o" in model.values

        assert kwargs["objectives"] == ["accuracy", "cost"]

    def test_to_decorator_kwargs_with_safety(self) -> None:
        from traigent.api.safety import SafetyConstraint

        result = AutoConfigResult(
            safety_constraints=(
                SafetySpec(
                    metric_name="faithfulness",
                    operator=">=",
                    threshold=0.85,
                ),
                SafetySpec(
                    metric_name="hallucination_rate",
                    operator="<=",
                    threshold=0.15,
                ),
            ),
        )
        kwargs = result.to_decorator_kwargs()
        assert "safety_constraints" in kwargs
        constraints = kwargs["safety_constraints"]
        assert len(constraints) == 2
        for c in constraints:
            assert isinstance(c, SafetyConstraint)

    def test_to_dict_kwargs_returns_dicts(self) -> None:
        result = AutoConfigResult(
            tvars=(
                TVarSpec(
                    name="temperature",
                    range_type="Range",
                    range_kwargs={"low": 0.0, "high": 1.0},
                ),
            ),
            objectives=(ObjectiveSpec(name="accuracy"),),
        )
        kwargs = result.to_dict_kwargs()
        assert kwargs["configuration_space"]["temperature"] == {
            "type": "Range",
            "low": 0.0,
            "high": 1.0,
        }
        assert kwargs["objectives"] == ["accuracy"]

    def test_to_tvl_spec_basic(self) -> None:
        result = AutoConfigResult(
            tvars=(
                TVarSpec(
                    name="temperature",
                    range_type="Range",
                    range_kwargs={"low": 0.0, "high": 1.0},
                ),
            ),
            objectives=(
                ObjectiveSpec(name="accuracy", orientation="maximize", weight=0.6),
            ),
        )
        spec = result.to_tvl_spec(module_name="test_agent")
        assert spec["header"]["module"] == "test_agent"
        assert "tvars" in spec
        assert "objectives" in spec
        assert spec["objectives"][0]["name"] == "accuracy"
        assert spec["objectives"][0]["direction"] == "maximize"

    def test_to_tvl_spec_with_safety(self) -> None:
        result = AutoConfigResult(
            tvars=(
                TVarSpec(
                    name="temperature",
                    range_type="Range",
                    range_kwargs={"low": 0.0, "high": 1.0},
                ),
            ),
            safety_constraints=(
                SafetySpec(metric_name="faithfulness", operator=">=", threshold=0.85),
            ),
        )
        spec = result.to_tvl_spec()
        assert "safety" in spec
        assert spec["safety"][0]["metric"] == "faithfulness"
        assert spec["safety"][0]["threshold"] == pytest.approx(0.85)

    def test_to_tvl_spec_with_structural_constraints(self) -> None:
        result = AutoConfigResult(
            tvars=(
                TVarSpec(
                    name="temperature",
                    range_type="Range",
                    range_kwargs={"low": 0.0, "high": 1.0},
                ),
            ),
            structural_constraints=(
                StructuralConstraintSpec(
                    description="Low temp for factual models",
                    constraint_code="implies(model.equals('gpt-4o'), temperature.lte(1.0))",
                    requires_tvars=("model", "temperature"),
                ),
            ),
        )
        spec = result.to_tvl_spec()
        assert "constraints" in spec
        structural = spec["constraints"]["structural"]
        assert len(structural) == 1
        assert structural[0]["description"] == "Low temp for factual models"
        assert "implies" in structural[0]["code"]
        assert structural[0]["requires"] == ["model", "temperature"]

    def test_reconstruct_range_unknown_type(self) -> None:
        from traigent.config_generator.types import _reconstruct_range

        tvar = TVarSpec(name="x", range_type="UnknownRange")
        with pytest.raises(ValueError, match="Unknown range type"):
            _reconstruct_range(tvar)

    def test_reconstruct_safety_unknown_metric(self) -> None:
        from traigent.config_generator.types import _reconstruct_safety

        sc = SafetySpec(metric_name="made_up_metric", operator=">=", threshold=0.5)
        with pytest.raises(ValueError, match="Unknown safety metric"):
            _reconstruct_safety(sc)

    def test_to_python_code_basic(self) -> None:
        result = AutoConfigResult(
            tvars=(
                TVarSpec(
                    name="temperature",
                    range_type="Range",
                    range_kwargs={"low": 0.0, "high": 1.0},
                ),
            ),
            objectives=(ObjectiveSpec(name="accuracy"),),
        )
        code = result.to_python_code()
        assert "@traigent.optimize(" in code
        assert "configuration_space" in code
        assert "'temperature'" in code
        assert "Range(low=0.0, high=1.0)" in code
        assert "objectives=['accuracy']" in code

    def test_to_python_code_with_safety(self) -> None:
        result = AutoConfigResult(
            tvars=(
                TVarSpec(
                    name="temperature",
                    range_type="Range",
                    range_kwargs={"low": 0.0, "high": 1.0},
                ),
            ),
            safety_constraints=(
                SafetySpec(
                    metric_name="hallucination_rate", operator="<=", threshold=0.15
                ),
            ),
        )
        code = result.to_python_code()
        assert "safety_constraints" in code
        # hallucination_rate is a factory function — must emit "()"
        assert "hallucination_rate().below" in code

    def test_to_python_code_factory_vs_instance_metrics(self) -> None:
        result = AutoConfigResult(
            safety_constraints=(
                # Factory function metrics need "()"
                SafetySpec(
                    metric_name="hallucination_rate", operator="<=", threshold=0.15
                ),
                SafetySpec(metric_name="toxicity_score", operator="<=", threshold=0.05),
                # Module-level instance metrics don't need "()"
                SafetySpec(metric_name="faithfulness", operator=">=", threshold=0.85),
            ),
        )
        code = result.to_python_code()
        assert "hallucination_rate().below(0.15)" in code
        assert "toxicity_score().below(0.05)" in code
        assert "faithfulness.above(0.85)" in code
        # Ensure faithfulness does NOT get "()"
        assert "faithfulness().above" not in code

    def test_to_python_code_rejects_structural_constraints(self) -> None:
        """to_python_code() must refuse to silently drop generated constraints."""
        result = AutoConfigResult(
            tvars=(
                TVarSpec(
                    name="temperature",
                    range_type="Range",
                    range_kwargs={"low": 0.0, "high": 1.0},
                ),
            ),
            structural_constraints=(
                StructuralConstraintSpec(
                    description="Low temp for factual models",
                    constraint_code="implies(model.equals('gpt-4o'), temperature.lte(1.0))",
                    requires_tvars=("model", "temperature"),
                ),
            ),
        )
        with pytest.raises(ValueError) as excinfo:
            result.to_python_code()
        message = str(excinfo.value)
        # Error must direct user to the TVL export path.
        assert "tvl" in message.lower()
        assert "structural constraint" in message.lower()
        # Must report how many were dropped and include the description.
        assert "1" in message
        assert "Low temp for factual models" in message

    def test_to_python_code_rejects_multiple_structural_constraints(self) -> None:
        """Count and first descriptions appear in the rejection message."""
        result = AutoConfigResult(
            structural_constraints=tuple(
                StructuralConstraintSpec(
                    description=f"constraint-{i}",
                    constraint_code=f"code-{i}",
                )
                for i in range(5)
            ),
        )
        with pytest.raises(ValueError) as excinfo:
            result.to_python_code()
        message = str(excinfo.value)
        assert "5" in message
        # Truncates after first 3 with a "+N more" suffix.
        assert "constraint-0" in message
        assert "+2 more" in message

    def test_to_python_code_no_constraints_still_works(self) -> None:
        """Result without structural constraints emits decorator normally."""
        result = AutoConfigResult(
            tvars=(
                TVarSpec(
                    name="temperature",
                    range_type="Range",
                    range_kwargs={"low": 0.0, "high": 1.0},
                ),
            ),
        )
        code = result.to_python_code()
        assert "@traigent.optimize(" in code

    def test_to_python_code_with_recommendations(self) -> None:
        result = AutoConfigResult(
            tvars=(
                TVarSpec(
                    name="temperature",
                    range_type="Range",
                    range_kwargs={"low": 0.0, "high": 1.0},
                ),
            ),
            recommendations=(
                TVarRecommendation(
                    name="prompting_strategy",
                    range_type="Choices",
                    range_kwargs={"values": ["direct", "cot"]},
                ),
            ),
        )
        code = result.to_python_code()
        assert "# Recommended additional TVars:" in code
        assert "prompting_strategy" in code
