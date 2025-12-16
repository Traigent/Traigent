"""Tests for loading TVL specifications."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from traigent.tvl.spec_loader import (
    _parse_exploration_parallelism,
    _resolve_algorithm,
    load_tvl_spec,
)
from traigent.utils.exceptions import TVLValidationError

FIXTURE_SPEC = Path(
    "docs/tvl/tvl-website/client/public/examples/ch1_motivation_experiment.tvl.yml"
)


class TestLoadTvlSpecErrors:
    """Tests for error handling in load_tvl_spec."""

    def test_file_not_found_raises_error(self, tmp_path: Path) -> None:
        """Missing spec file raises TVLValidationError."""
        nonexistent = tmp_path / "does_not_exist.tvl.yml"
        with pytest.raises(TVLValidationError, match="not found"):
            load_tvl_spec(spec_path=nonexistent)

    def test_non_mapping_spec_raises_error(self, tmp_path: Path) -> None:
        """Spec that is not a mapping raises TVLValidationError."""
        spec_file = tmp_path / "invalid.tvl.yml"
        spec_file.write_text("- item1\n- item2\n")  # List instead of mapping

        with pytest.raises(TVLValidationError, match="must be a mapping"):
            load_tvl_spec(spec_path=spec_file)

    def test_tvl_version_numeric_converted_to_string(self, tmp_path: Path) -> None:
        """Numeric tvl_version is converted to string."""
        spec_content = """
tvl_version: 0.9

tvars:
  - name: model
    type: enum[str]
    domain: ["gpt-4o"]

objectives:
  - name: accuracy
    direction: maximize
"""
        spec_file = tmp_path / "test.tvl.yml"
        spec_file.write_text(spec_content)

        artifact = load_tvl_spec(spec_path=spec_file)
        # tvl_version should be string
        assert artifact.tvl_version == "0.9"

    def test_spec_without_tvars_or_configuration_space(self, tmp_path: Path) -> None:
        """Spec without tvars or configuration_space gets empty config space."""
        spec_content = """
objectives:
  - name: accuracy
    direction: maximize
"""
        spec_file = tmp_path / "test.tvl.yml"
        spec_file.write_text(spec_content)

        artifact = load_tvl_spec(spec_path=spec_file)
        # Should have empty configuration space
        assert artifact.configuration_space == {}


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
    # TVL 0.9 uses exploration_budgets instead of legacy budget
    assert artifact.exploration_budgets is not None
    assert artifact.exploration_budgets.max_trials == 60
    assert artifact.metadata["spec_id"] == "rag-campus-orientation"


def test_environment_overrides_budget_and_space() -> None:
    """Environment overlays update the merged spec before normalization."""

    artifact = load_tvl_spec(spec_path=FIXTURE_SPEC, environment="finals_week")

    # TVL 0.9 uses exploration_budgets instead of legacy budget
    assert artifact.exploration_budgets is not None
    assert artifact.exploration_budgets.max_trials == 90
    # Integer tvars produce int ranges, not float
    assert artifact.configuration_space["retrieval_depth"] == (3, 8)


def test_compiled_constraints_attach_metadata() -> None:
    """Constraints are converted to decorator callables with metadata."""

    artifact = load_tvl_spec(spec_path=FIXTURE_SPEC)
    assert artifact.constraints, "spec defines constraints"
    constraint = artifact.constraints[0]
    meta = getattr(constraint, "__tvl_constraint__", {})
    # TVL 0.9 structural constraints use id from the constraint definition
    assert meta["id"] == "campus-hour-latency"
    assert isinstance(constraint({"response_latency_ms": 800}), bool)


def test_legacy_formats_emit_deprecation_warnings() -> None:
    """Legacy TVL formats emit DeprecationWarnings during spec loading."""
    # Use the environment_overlays example which is intentionally legacy format
    legacy_spec = Path("examples/tvl/environment_overlays/overlays.tvl.yml")
    with pytest.warns(DeprecationWarning) as record:
        load_tvl_spec(spec_path=legacy_spec)

    messages = [str(w.message) for w in record]
    assert any("configuration_space" in message for message in messages)
    assert any("constraints" in message for message in messages)
    assert any("optimization" in message for message in messages)


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

    def test_tvars_and_configuration_space_conflict_raises(self) -> None:
        """Using both tvars and configuration_space is an error."""
        spec_content = """
tvars:
  - name: model
    type: enum[str]
    domain: ["gpt-4"]

configuration_space:
  model:
    type: categorical
    values: ["gpt-4"]

objectives:
  - name: accuracy
    direction: maximize
"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".tvl.yml", delete=False
        ) as f:
            f.write(spec_content)
            f.flush()

            with pytest.raises(TVLValidationError, match="both 'tvars'"):
                load_tvl_spec(spec_path=f.name)

    def test_registry_domain_requires_resolver(self) -> None:
        """Registry domains fail fast unless a resolver is provided."""
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

            with pytest.raises(TVLValidationError, match="registry_resolver"):
                load_tvl_spec(spec_path=f.name)

    def test_registry_domain_resolves_with_resolver(self) -> None:
        """Registry domains are resolved into concrete configuration values."""
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

        class DummyResolver:
            def __init__(self) -> None:
                self.calls: list[tuple[str, str | None, str | None]] = []

            def resolve(
                self,
                registry_id: str,
                filter_expr: str | None = None,
                version: str | None = None,
            ) -> list[str]:
                self.calls.append((registry_id, filter_expr, version))
                return ["scorer_v2", "scorer_v3"]

        resolver = DummyResolver()
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".tvl.yml", delete=False
        ) as f:
            f.write(spec_content)
            f.flush()

            artifact = load_tvl_spec(spec_path=f.name, registry_resolver=resolver)

            assert artifact.tvars is not None
            assert len(artifact.tvars) == 1

            scorer_tvar = artifact.tvars[0]
            assert scorer_tvar.domain.kind == "registry"
            assert scorer_tvar.domain.registry == "scorers"
            assert scorer_tvar.domain.filter == "version >= 2"

            assert artifact.configuration_space["scorer"] == ["scorer_v2", "scorer_v3"]
            assert resolver.calls == [("scorers", "version >= 2", None)]


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


class TestExplorationBudgetsWiring:
    """Tests for TVL 0.9 exploration.budgets wiring to runtime_overrides."""

    def test_exploration_budgets_wired_to_runtime(self, tmp_path: Path) -> None:
        """TVL 0.9 exploration.budgets should be wired to runtime_overrides."""
        spec_content = """
tvars:
  - name: model
    type: enum[str]
    domain: ["gpt-4o"]

objectives:
  - name: accuracy
    direction: maximize

exploration:
  strategy:
    type: nsga2
  parallelism:
    max_parallel_trials: 4
  budgets:
    max_trials: 50
    max_spend_usd: 10.0
    max_wallclock_s: 1800
"""
        spec_file = tmp_path / "test.tvl.yml"
        spec_file.write_text(spec_content)

        artifact = load_tvl_spec(spec_path=spec_file)
        overrides = artifact.runtime_overrides()

        assert overrides.get("algorithm") == "nsga2"
        assert overrides.get("max_trials") == 50
        assert overrides.get("cost_limit") == 10.0
        assert overrides.get("timeout") == 1800
        assert overrides.get("parallel_trials") == 4

    def test_exploration_and_optimization_conflict_raises(self, tmp_path: Path) -> None:
        """Error should be raised if both exploration and optimization exist."""
        spec_content = """
tvars:
  - name: model
    type: enum[str]
    domain: ["gpt-4o"]

objectives:
  - name: accuracy
    direction: maximize

optimization:
  budget:
    max_trials:
      value: 30

exploration:
  budgets:
    max_trials: 50
"""
        spec_file = tmp_path / "test.tvl.yml"
        spec_file.write_text(spec_content)

        with pytest.raises(
            TVLValidationError, match="both 'exploration' and 'optimization'"
        ):
            load_tvl_spec(spec_path=spec_file)

    def test_strategy_dict_format_works(self, tmp_path: Path) -> None:
        """Strategy as dict {type: ...} should be parsed correctly."""
        spec_content = """
tvars:
  - name: model
    type: enum[str]
    domain: ["gpt-4o"]

objectives:
  - name: accuracy
    direction: maximize

exploration:
  strategy:
    type: bayesian
    initial_sampling: sobol
"""
        spec_file = tmp_path / "test.tvl.yml"
        spec_file.write_text(spec_content)

        artifact = load_tvl_spec(spec_path=spec_file)
        overrides = artifact.runtime_overrides()

        assert overrides.get("algorithm") == "bayesian"

    def test_legacy_budget_takes_precedence_over_exploration(
        self, tmp_path: Path
    ) -> None:
        """Legacy optimization.budget should take precedence over exploration.budgets."""
        # This test uses only optimization (legacy) to verify it works
        spec_content = """
tvars:
  - name: model
    type: enum[str]
    domain: ["gpt-4o"]

objectives:
  - name: accuracy
    direction: maximize

optimization:
  strategy: nsga2
  budget:
    max_trials:
      value: 100
    timeout_seconds: 3600
    parallel_trials: 8
"""
        spec_file = tmp_path / "test.tvl.yml"
        spec_file.write_text(spec_content)

        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            artifact = load_tvl_spec(spec_path=spec_file)

        overrides = artifact.runtime_overrides()

        # Legacy budget values should be used
        assert overrides.get("max_trials") == 100
        assert overrides.get("timeout") == 3600
        assert overrides.get("parallel_trials") == 8

    def test_exploration_budgets_only_when_no_legacy(self, tmp_path: Path) -> None:
        """exploration.budgets fields are used when legacy budget is not set."""
        spec_content = """
tvars:
  - name: model
    type: enum[str]
    domain: ["gpt-4o"]

objectives:
  - name: accuracy
    direction: maximize

exploration:
  budgets:
    max_trials: 75
    max_spend_usd: 25.0
    max_wallclock_s: 2400
"""
        spec_file = tmp_path / "test.tvl.yml"
        spec_file.write_text(spec_content)

        artifact = load_tvl_spec(spec_path=spec_file)
        overrides = artifact.runtime_overrides()

        assert overrides.get("max_trials") == 75
        assert overrides.get("cost_limit") == 25.0
        assert overrides.get("timeout") == 2400

    def test_exploration_parallelism_only(self, tmp_path: Path) -> None:
        """exploration.parallelism alone adds parallel_trials."""
        spec_content = """
tvars:
  - name: model
    type: enum[str]
    domain: ["gpt-4o"]

objectives:
  - name: accuracy
    direction: maximize

exploration:
  parallelism:
    max_parallel_trials: 6
"""
        spec_file = tmp_path / "test.tvl.yml"
        spec_file.write_text(spec_content)

        artifact = load_tvl_spec(spec_path=spec_file)
        overrides = artifact.runtime_overrides()

        assert overrides.get("parallel_trials") == 6
        # Should not have other budget keys
        assert "max_trials" not in overrides
        assert "timeout" not in overrides

    def test_empty_exploration_returns_minimal_overrides(self, tmp_path: Path) -> None:
        """Empty exploration section returns minimal overrides."""
        spec_content = """
tvars:
  - name: model
    type: enum[str]
    domain: ["gpt-4o"]

objectives:
  - name: accuracy
    direction: maximize

exploration: {}
"""
        spec_file = tmp_path / "test.tvl.yml"
        spec_file.write_text(spec_content)

        artifact = load_tvl_spec(spec_path=spec_file)
        overrides = artifact.runtime_overrides()

        # Should not have budget keys when exploration is empty
        assert "max_trials" not in overrides
        assert "timeout" not in overrides
        assert "parallel_trials" not in overrides
        assert "cost_limit" not in overrides

    def test_legacy_budget_max_total_examples(self, tmp_path: Path) -> None:
        """Legacy budget.max_total_examples is wired to runtime_overrides."""
        spec_content = """
tvars:
  - name: model
    type: enum[str]
    domain: ["gpt-4o"]

objectives:
  - name: accuracy
    direction: maximize

optimization:
  budget:
    max_trials:
      value: 50
    max_total_examples: 1000
"""
        spec_file = tmp_path / "test.tvl.yml"
        spec_file.write_text(spec_content)

        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            artifact = load_tvl_spec(spec_path=spec_file)

        overrides = artifact.runtime_overrides()

        assert overrides.get("max_total_examples") == 1000

    def test_legacy_budget_samples_include_pruned(self, tmp_path: Path) -> None:
        """Legacy budget.samples_include_pruned is wired to runtime_overrides."""
        spec_content = """
tvars:
  - name: model
    type: enum[str]
    domain: ["gpt-4o"]

objectives:
  - name: accuracy
    direction: maximize

optimization:
  budget:
    max_trials:
      value: 50
    samples_include_pruned: true
"""
        spec_file = tmp_path / "test.tvl.yml"
        spec_file.write_text(spec_content)

        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            artifact = load_tvl_spec(spec_path=spec_file)

        overrides = artifact.runtime_overrides()

        assert overrides.get("samples_include_pruned") is True


class TestParseExplorationParallelism:
    """Unit tests for _parse_exploration_parallelism() function."""

    def test_returns_none_for_none_input(self) -> None:
        """None input returns None."""
        assert _parse_exploration_parallelism(None) is None

    def test_returns_none_for_string_input(self) -> None:
        """String input (non-dict) returns None."""
        assert _parse_exploration_parallelism("invalid") is None

    def test_returns_none_for_list_input(self) -> None:
        """List input (non-dict) returns None."""
        assert _parse_exploration_parallelism([1, 2, 3]) is None

    def test_returns_none_for_int_input(self) -> None:
        """Integer input (non-dict) returns None."""
        assert _parse_exploration_parallelism(42) is None

    def test_returns_none_when_parallelism_key_missing(self) -> None:
        """Empty dict (no parallelism key) returns None."""
        assert _parse_exploration_parallelism({}) is None

    def test_returns_none_when_parallelism_is_string(self) -> None:
        """parallelism as string (not dict) returns None."""
        assert _parse_exploration_parallelism({"parallelism": "not_dict"}) is None

    def test_returns_none_when_parallelism_is_list(self) -> None:
        """parallelism as list (not dict) returns None."""
        assert _parse_exploration_parallelism({"parallelism": []}) is None

    def test_returns_none_when_max_parallel_trials_missing(self) -> None:
        """parallelism dict without max_parallel_trials returns None."""
        assert _parse_exploration_parallelism({"parallelism": {}}) is None

    def test_returns_none_when_max_parallel_trials_is_string(self) -> None:
        """max_parallel_trials as string (not int) returns None."""
        result = _parse_exploration_parallelism(
            {"parallelism": {"max_parallel_trials": "4"}}
        )
        assert result is None

    def test_returns_none_when_max_parallel_trials_is_float(self) -> None:
        """max_parallel_trials as float (not int) returns None."""
        result = _parse_exploration_parallelism(
            {"parallelism": {"max_parallel_trials": 4.5}}
        )
        assert result is None

    def test_extracts_max_parallel_trials_correctly(self) -> None:
        """Valid max_parallel_trials integer is extracted."""
        result = _parse_exploration_parallelism(
            {"parallelism": {"max_parallel_trials": 4}}
        )
        assert result == 4

    def test_handles_zero_parallel_trials(self) -> None:
        """Zero is a valid integer value."""
        result = _parse_exploration_parallelism(
            {"parallelism": {"max_parallel_trials": 0}}
        )
        assert result == 0

    def test_handles_large_parallel_trials(self) -> None:
        """Large integer values are handled."""
        result = _parse_exploration_parallelism(
            {"parallelism": {"max_parallel_trials": 999999}}
        )
        assert result == 999999

    def test_boolean_passes_int_check(self) -> None:
        """Python quirk: isinstance(True, int) == True.

        This documents current behavior. In Python, bool is a subclass of int,
        so True/False pass isinstance(x, int) checks.
        """
        result = _parse_exploration_parallelism(
            {"parallelism": {"max_parallel_trials": True}}
        )
        # Current behavior: True passes the int check
        assert result is True


class TestResolveAlgorithm:
    """Unit tests for _resolve_algorithm() function."""

    def test_returns_none_for_none_section(self) -> None:
        """None section returns None."""
        assert _resolve_algorithm(None) is None

    def test_returns_none_for_string_section(self) -> None:
        """String section (non-dict) returns None."""
        assert _resolve_algorithm("invalid") is None

    def test_returns_none_for_list_section(self) -> None:
        """List section (non-dict) returns None."""
        assert _resolve_algorithm([1, 2, 3]) is None

    def test_returns_none_when_strategy_missing(self) -> None:
        """Empty dict (no strategy key) returns None."""
        assert _resolve_algorithm({}) is None

    def test_returns_none_when_strategy_is_none(self) -> None:
        """strategy: None returns None."""
        assert _resolve_algorithm({"strategy": None}) is None

    def test_returns_none_when_strategy_is_int(self) -> None:
        """strategy as int (not str or dict) returns None."""
        assert _resolve_algorithm({"strategy": 123}) is None

    def test_dict_strategy_extracts_type(self) -> None:
        """Dict strategy with type key extracts the type."""
        result = _resolve_algorithm({"strategy": {"type": "bayesian"}})
        assert result == "bayesian"

    def test_dict_strategy_without_type_returns_none(self) -> None:
        """Dict strategy without type key returns None."""
        assert _resolve_algorithm({"strategy": {"no_type": "x"}}) is None

    def test_empty_dict_strategy_returns_none(self) -> None:
        """Empty dict strategy returns None."""
        assert _resolve_algorithm({"strategy": {}}) is None

    def test_string_strategy_normalizes_case(self) -> None:
        """String strategy is case-insensitive."""
        assert _resolve_algorithm({"strategy": "PARETO_OPTIMAL"}) == "nsga2"
        assert _resolve_algorithm({"strategy": "Pareto_Optimal"}) == "nsga2"

    def test_unknown_strategy_returned_lowercase(self) -> None:
        """Unknown strategy values are returned as-is (lowercased)."""
        assert _resolve_algorithm({"strategy": "Custom_Algo"}) == "custom_algo"

    def test_mapping_pareto_optimal(self) -> None:
        """pareto_optimal maps to nsga2."""
        assert _resolve_algorithm({"strategy": "pareto_optimal"}) == "nsga2"

    def test_mapping_nsga2(self) -> None:
        """nsga2 maps to nsga2."""
        assert _resolve_algorithm({"strategy": "nsga2"}) == "nsga2"

    def test_mapping_grid(self) -> None:
        """grid maps to grid."""
        assert _resolve_algorithm({"strategy": "grid"}) == "grid"

    def test_mapping_grid_search(self) -> None:
        """grid_search maps to grid."""
        assert _resolve_algorithm({"strategy": "grid_search"}) == "grid"

    def test_mapping_random(self) -> None:
        """random maps to random."""
        assert _resolve_algorithm({"strategy": "random"}) == "random"

    def test_mapping_random_search(self) -> None:
        """random_search maps to random."""
        assert _resolve_algorithm({"strategy": "random_search"}) == "random"

    def test_mapping_bayesian(self) -> None:
        """bayesian maps to bayesian."""
        assert _resolve_algorithm({"strategy": "bayesian"}) == "bayesian"

    def test_mapping_tpe(self) -> None:
        """tpe maps to optuna."""
        assert _resolve_algorithm({"strategy": "tpe"}) == "optuna"

    def test_dict_strategy_with_extra_fields(self) -> None:
        """Dict strategy with extra fields still extracts type."""
        result = _resolve_algorithm(
            {
                "strategy": {
                    "type": "bayesian",
                    "initial_sampling": "sobol",
                    "n_warmup": 10,
                }
            }
        )
        assert result == "bayesian"
