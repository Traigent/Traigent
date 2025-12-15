"""Tests for loading TVL specifications."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from traigent.tvl.spec_loader import load_tvl_spec
from traigent.utils.exceptions import TVLValidationError

FIXTURE_SPEC = Path(
    "docs/tvl/tvl-website/client/public/examples/ch1_motivation_experiment.tvl.yml"
)


def test_loads_configuration_space_and_objectives() -> None:
    """Ensure the sample spec normalizes spaces/objectives."""

    artifact = load_tvl_spec(spec_path=FIXTURE_SPEC)

    assert "model" in artifact.configuration_space
    assert artifact.configuration_space["temperature"] == (0.1, 0.8)
    assert artifact.objective_schema is not None
    assert [obj.name for obj in artifact.objective_schema.objectives] == [
        "answer_quality",
        "response_latency",
        "token_cost",
    ]
    assert artifact.budget.max_trials == 60
    assert artifact.metadata["spec_id"] == "rag-campus-orientation"


def test_environment_overrides_budget_and_space() -> None:
    """Environment overlays update the merged spec before normalization."""

    artifact = load_tvl_spec(spec_path=FIXTURE_SPEC, environment="finals_week")

    assert artifact.budget.max_trials == 90
    assert artifact.configuration_space["retrieval_depth"] == (3.0, 8.0)


def test_compiled_constraints_attach_metadata() -> None:
    """Constraints are converted to decorator callables with metadata."""

    artifact = load_tvl_spec(spec_path=FIXTURE_SPEC)
    assert artifact.constraints, "spec defines constraints"
    constraint = artifact.constraints[0]
    meta = getattr(constraint, "__tvl_constraint__", {})
    assert meta["id"] == "campus-hour-latency"
    assert isinstance(constraint({"response_latency_ms": 800}), bool)


class TestDerivedConstraintParsing:
    """Tests for TVL 0.9 derived constraint parsing."""

    def test_parses_derived_constraints(self) -> None:
        """Derived constraints are parsed and stored."""
        spec_content = """
tvars:
  - name: model
    type: enum[str]
    domain: ["gpt-4", "gpt-3.5"]
  - name: max_tokens
    type: int
    domain:
      range: [256, 4096]

constraints:
  structural:
    - when: "params.model == 'gpt-4'"
      then: "params.max_tokens <= 2048"
  derived:
    - require: "env.budget >= 1000"
    - require: "env.price_per_token * max_tokens <= env.quota"
      description: "Token budget constraint"

objectives:
  - name: accuracy
    direction: maximize
"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".tvl.yml", delete=False
        ) as f:
            f.write(spec_content)
            f.flush()

            artifact = load_tvl_spec(spec_path=f.name)

            # Derived constraints should be parsed
            assert artifact.derived_constraints is not None
            assert len(artifact.derived_constraints) == 2
            assert artifact.derived_constraints[0].require == "env.budget >= 1000"
            assert (
                artifact.derived_constraints[1].require
                == "env.price_per_token * max_tokens <= env.quota"
            )
            assert (
                artifact.derived_constraints[1].description == "Token budget constraint"
            )

    def test_no_derived_constraints_returns_none(self) -> None:
        """When no derived constraints exist, returns None."""
        spec_content = """
tvars:
  - name: model
    type: enum[str]
    domain: ["gpt-4", "gpt-3.5"]

constraints:
  structural:
    - expr: "params.model in ['gpt-4', 'gpt-3.5']"

objectives:
  - name: accuracy
    direction: maximize
"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".tvl.yml", delete=False
        ) as f:
            f.write(spec_content)
            f.flush()

            artifact = load_tvl_spec(spec_path=f.name)
            assert artifact.derived_constraints is None

    def test_invalid_derived_constraint_raises(self) -> None:
        """Invalid derived constraint raises TVLValidationError."""
        spec_content = """
tvars:
  - name: model
    type: enum[str]
    domain: ["gpt-4"]

constraints:
  derived:
    - invalid_key: "something"

objectives:
  - name: accuracy
    direction: maximize
"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".tvl.yml", delete=False
        ) as f:
            f.write(spec_content)
            f.flush()

            with pytest.raises(TVLValidationError, match="requires a 'require' string"):
                load_tvl_spec(spec_path=f.name)

    def test_legacy_format_has_no_derived_constraints(self) -> None:
        """Legacy format (list constraints) returns None for derived_constraints."""
        # The fixture uses legacy format
        artifact = load_tvl_spec(spec_path=FIXTURE_SPEC)
        assert artifact.derived_constraints is None


class TestTVL09FormatParsing:
    """Tests for TVL 0.9 format with tvars."""

    def test_tvars_parsed_correctly(self) -> None:
        """TVL 0.9 tvars format is parsed into TVarDecl objects."""
        spec_content = """
tvars:
  - name: model
    type: enum[str]
    domain: ["gpt-4", "gpt-3.5", "claude-3"]
    default: "gpt-4"
  - name: temperature
    type: float
    domain:
      range: [0.0, 2.0]
      resolution: 0.1
    unit: "unitless"
  - name: use_cot
    type: bool
    default: true

objectives:
  - name: accuracy
    direction: maximize
"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".tvl.yml", delete=False
        ) as f:
            f.write(spec_content)
            f.flush()

            artifact = load_tvl_spec(spec_path=f.name)

            # TVars should be parsed
            assert artifact.tvars is not None
            assert len(artifact.tvars) == 3

            # Check model TVAR
            model_tvar = artifact.tvars[0]
            assert model_tvar.name == "model"
            assert model_tvar.type == "enum"
            assert model_tvar.raw_type == "enum[str]"
            assert model_tvar.default == "gpt-4"

            # Check temperature TVAR
            temp_tvar = artifact.tvars[1]
            assert temp_tvar.name == "temperature"
            assert temp_tvar.type == "float"
            assert temp_tvar.domain.kind == "range"
            assert temp_tvar.domain.range == (0.0, 2.0)
            assert abs(temp_tvar.domain.resolution - 0.1) < 1e-10
            assert temp_tvar.unit == "unitless"

            # Check bool TVAR
            cot_tvar = artifact.tvars[2]
            assert cot_tvar.name == "use_cot"
            assert cot_tvar.type == "bool"
            assert cot_tvar.default is True

            # Configuration space should also be populated
            assert artifact.configuration_space["model"] == [
                "gpt-4",
                "gpt-3.5",
                "claude-3",
            ]
            assert artifact.configuration_space["temperature"] == (0.0, 2.0)
            assert artifact.configuration_space["use_cot"] == [True, False]

    def test_registry_domain_raises_not_implemented(self) -> None:
        """Registry domain raises NotImplementedError (fail-fast behavior)."""
        spec_content = """
tvars:
  - name: scorer
    type: callable[ScorerProto]
    domain:
      registry: scorers
      filter: "version >= 2"

objectives:
  - name: accuracy
    direction: maximize
"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".tvl.yml", delete=False
        ) as f:
            f.write(spec_content)
            f.flush()

            # Registry domains should fail-fast until a resolver is configured
            with pytest.raises(NotImplementedError) as exc_info:
                load_tvl_spec(spec_path=f.name)

            assert "scorers" in str(exc_info.value)
            assert "not yet supported" in str(exc_info.value)


class TestBandedObjectiveParsing:
    """Tests for TVL 0.9 banded objective parsing."""

    def test_parses_banded_objective_with_interval(self, tmp_path):
        """Test parsing banded objective with [L, U] interval."""
        spec_path = tmp_path / "banded.yaml"
        spec_content = """tvl:
  module: test.banded
tvl_version: "0.9"
environment:
  snapshot_id: "2025-01-01T00:00:00Z"
evaluation_set:
  dataset: s3://test/data.parquet
tvars:
  - name: temperature
    type: float
    domain:
      range: [0.0, 1.0]
objectives:
  - name: response_length
    band:
      target: [100, 200]
      test: TOST
      alpha: 0.05
promotion_policy:
  dominance: epsilon_pareto
"""
        spec_path.write_text(spec_content)
        artifact = load_tvl_spec(spec_path=spec_path)
        assert artifact.objective_schema is not None
        obj = artifact.objective_schema.objectives[0]
        assert obj.name == "response_length"
        assert obj.orientation == "band"
        assert obj.band is not None
        assert obj.band.low == 100
        assert obj.band.high == 200


class TestTVL09EnvironmentParsing:
    """Tests for TVL 0.9 environment section parsing."""

    def test_parses_environment_snapshot(self, tmp_path):
        """Test parsing environment snapshot with components."""
        spec_path = tmp_path / "env.yaml"
        spec_content = """tvl:
  module: test.env
tvl_version: "0.9"
environment:
  snapshot_id: "2025-01-15T10:30:00Z"
  components:
    retriever: bm25-v3
evaluation_set:
  dataset: s3://test/data.parquet
tvars:
  - name: model
    type: enum[str]
    domain: ["gpt-4"]
objectives:
  - name: quality
    direction: maximize
promotion_policy:
  dominance: epsilon_pareto
"""
        spec_path.write_text(spec_content)
        artifact = load_tvl_spec(spec_path=spec_path)
        assert artifact.environment_snapshot is not None
        assert artifact.environment_snapshot.snapshot_id == "2025-01-15T10:30:00Z"
        assert artifact.environment_snapshot.components["retriever"] == "bm25-v3"


class TestTVL09ExplorationParsing:
    """Tests for TVL 0.9 exploration section parsing."""

    def test_parses_convergence_criteria(self, tmp_path):
        """Test parsing convergence criteria."""
        spec_path = tmp_path / "convergence.yaml"
        spec_content = """tvl:
  module: test.conv
tvl_version: "0.9"
environment:
  snapshot_id: "2025-01-01T00:00:00Z"
evaluation_set:
  dataset: s3://test/data.parquet
tvars:
  - name: model
    type: enum[str]
    domain: ["gpt-4"]
objectives:
  - name: quality
    direction: maximize
promotion_policy:
  dominance: epsilon_pareto
exploration:
  strategy:
    type: nsga2
  convergence:
    metric: hypervolume_improvement
    window: 20
    threshold: 0.001
  budgets:
    max_trials: 200
"""
        spec_path.write_text(spec_content)
        artifact = load_tvl_spec(spec_path=spec_path)
        assert artifact.convergence is not None
        assert artifact.convergence.metric == "hypervolume_improvement"
        assert artifact.convergence.window == 20
        assert artifact.exploration_budgets is not None
        assert artifact.exploration_budgets.max_trials == 200
