"""Unit tests for TVL 0.9 spec loader parsing functions.

This test suite covers the new parsing functions introduced in TVL 0.9:
- _parse_tvl_header
- _parse_environment_snapshot
- _parse_evaluation_set
- _parse_tvars
- _parse_convergence
- _parse_exploration_budgets
- _parse_exploration_parallelism

Tests follow patterns from tests/unit/tvl/test_spec_loader.py.

Reference: Issue #34, docs/plans/TVL_0.9_IMPLEMENTATION_PLAN.md Section 7.1
"""

from __future__ import annotations

import pytest

from traigent.tvl.models import (
    ConvergenceCriteria,
    EnvironmentSnapshot,
    EvaluationSet,
    ExplorationBudgets,
    TVLHeader,
)
from traigent.tvl.spec_loader import (
    _parse_convergence,
    _parse_environment_snapshot,
    _parse_evaluation_set,
    _parse_exploration_budgets,
    _parse_exploration_parallelism,
    _parse_tvars,
    _parse_tvl_header,
)
from traigent.utils.exceptions import TVLValidationError


class TestParseTVLHeader:
    """Tests for _parse_tvl_header function."""

    def test_valid_header_with_module(self) -> None:
        """Parse valid header with module identifier."""
        header_data = {"module": "corp.product.optimization_spec"}
        result = _parse_tvl_header(header_data)

        assert result is not None
        assert isinstance(result, TVLHeader)
        assert result.module == "corp.product.optimization_spec"
        assert result.skip_budget_checks is False
        assert result.skip_cost_estimation is False

    def test_header_with_validation_flags(self) -> None:
        """Parse header with validation configuration."""
        header_data = {
            "module": "test.module",
            "validation": {
                "skip_budget_checks": True,
                "skip_cost_estimation": True,
            },
        }
        result = _parse_tvl_header(header_data)

        assert result is not None
        assert result.skip_budget_checks is True
        assert result.skip_cost_estimation is True

    def test_header_missing_module_raises_error(self) -> None:
        """Missing module field raises TVLValidationError."""
        header_data = {"validation": {"skip_budget_checks": True}}

        with pytest.raises(TVLValidationError, match="requires a 'module' string"):
            _parse_tvl_header(header_data)

    def test_header_with_non_string_module_raises_error(self) -> None:
        """Non-string module field raises TVLValidationError."""
        header_data = {"module": 123}

        with pytest.raises(TVLValidationError, match="requires a 'module' string"):
            _parse_tvl_header(header_data)

    def test_none_header_returns_none(self) -> None:
        """None input returns None."""
        result = _parse_tvl_header(None)
        assert result is None

    def test_non_dict_header_raises_error(self) -> None:
        """Non-dict header raises TVLValidationError."""
        with pytest.raises(TVLValidationError, match="must be a mapping"):
            _parse_tvl_header("not a dict")


class TestParseEnvironmentSnapshot:
    """Tests for _parse_environment_snapshot function."""

    def test_valid_snapshot_with_id_and_components(self) -> None:
        """Parse valid environment snapshot."""
        env_data = {
            "snapshot_id": "2026-01-28T12:00:00Z",
            "components": {
                "llm_provider": "openai",
                "model_version": "gpt-4o-2024-08",
                "api_version": "2024-08-06",
            },
        }
        result = _parse_environment_snapshot(env_data)

        assert result is not None
        assert isinstance(result, EnvironmentSnapshot)
        assert result.snapshot_id == "2026-01-28T12:00:00Z"
        assert result.components["llm_provider"] == "openai"
        assert result.components["model_version"] == "gpt-4o-2024-08"

    def test_snapshot_with_only_id(self) -> None:
        """Parse snapshot with only snapshot_id (no components)."""
        env_data = {"snapshot_id": "2026-01-01T00:00:00Z"}
        result = _parse_environment_snapshot(env_data)

        assert result is not None
        assert result.snapshot_id == "2026-01-01T00:00:00Z"
        assert result.components == {}

    def test_snapshot_missing_id_raises_error(self) -> None:
        """Missing snapshot_id raises TVLValidationError."""
        env_data = {"components": {"key": "value"}}

        with pytest.raises(TVLValidationError, match="requires a 'snapshot_id' string"):
            _parse_environment_snapshot(env_data)

    def test_snapshot_non_string_id_raises_error(self) -> None:
        """Non-string snapshot_id raises TVLValidationError."""
        env_data = {"snapshot_id": 12345}

        with pytest.raises(TVLValidationError, match="requires a 'snapshot_id' string"):
            _parse_environment_snapshot(env_data)

    def test_none_env_returns_none(self) -> None:
        """None input returns None."""
        result = _parse_environment_snapshot(None)
        assert result is None

    def test_non_dict_env_raises_error(self) -> None:
        """Non-dict environment raises TVLValidationError."""
        with pytest.raises(TVLValidationError, match="must be a mapping"):
            _parse_environment_snapshot(["not", "a", "dict"])


class TestParseEvaluationSet:
    """Tests for _parse_evaluation_set function."""

    def test_valid_evaluation_set_with_dataset(self) -> None:
        """Parse valid evaluation set with dataset identifier."""
        eval_data = {"dataset": "company/eval-dataset-v1"}
        result = _parse_evaluation_set(eval_data)

        assert result is not None
        assert isinstance(result, EvaluationSet)
        assert result.dataset == "company/eval-dataset-v1"
        assert result.seed is None

    def test_evaluation_set_with_optional_seed(self) -> None:
        """Parse evaluation set with optional seed."""
        eval_data = {"dataset": "test-dataset", "seed": 42}
        result = _parse_evaluation_set(eval_data)

        assert result is not None
        assert result.dataset == "test-dataset"
        assert result.seed == 42

    def test_evaluation_set_missing_dataset_raises_error(self) -> None:
        """Missing dataset raises TVLValidationError."""
        eval_data = {"seed": 42}

        with pytest.raises(TVLValidationError, match="requires a 'dataset' string"):
            _parse_evaluation_set(eval_data)

    def test_evaluation_set_non_string_dataset_raises_error(self) -> None:
        """Non-string dataset raises TVLValidationError."""
        eval_data = {"dataset": ["list", "of", "datasets"]}

        with pytest.raises(TVLValidationError, match="requires a 'dataset' string"):
            _parse_evaluation_set(eval_data)

    def test_evaluation_set_non_int_seed_raises_error(self) -> None:
        """Non-integer seed raises TVLValidationError."""
        eval_data = {"dataset": "test-dataset", "seed": "not-an-int"}

        with pytest.raises(TVLValidationError, match="'seed' must be an integer"):
            _parse_evaluation_set(eval_data)

    def test_none_eval_returns_none(self) -> None:
        """None input returns None."""
        result = _parse_evaluation_set(None)
        assert result is None

    def test_non_dict_eval_raises_error(self) -> None:
        """Non-dict evaluation_set raises TVLValidationError."""
        with pytest.raises(TVLValidationError, match="must be a mapping"):
            _parse_evaluation_set("string-not-dict")


class TestParseTvars:
    """Tests for _parse_tvars function."""

    def test_parse_enum_str_type(self) -> None:
        """Parse enum[str] tvar type."""
        resolved = {
            "tvars": [
                {
                    "name": "model",
                    "type": "enum[str]",
                    "domain": ["gpt-4o", "gpt-4o-mini", "claude-3-5-sonnet"],
                }
            ]
        }
        tvars, config_space, defaults, units = _parse_tvars(resolved)

        assert len(tvars) == 1
        assert tvars[0].name == "model"
        assert tvars[0].raw_type == "enum[str]"
        assert "model" in config_space
        assert config_space["model"] == ["gpt-4o", "gpt-4o-mini", "claude-3-5-sonnet"]

    def test_parse_float_range_type(self) -> None:
        """Parse float tvar with range domain."""
        resolved = {
            "tvars": [
                {
                    "name": "temperature",
                    "type": "float",
                    "domain": {"range": [0.0, 1.0]},
                    "default": 0.7,
                }
            ]
        }
        tvars, config_space, defaults, units = _parse_tvars(resolved)

        assert len(tvars) == 1
        assert tvars[0].name == "temperature"
        assert defaults["temperature"] == 0.7
        assert "temperature" in config_space

    def test_parse_int_range_type(self) -> None:
        """Parse int tvar with range domain."""
        resolved = {
            "tvars": [
                {
                    "name": "max_tokens",
                    "type": "int",
                    "domain": {"range": [100, 4096]},
                }
            ]
        }
        tvars, config_space, defaults, units = _parse_tvars(resolved)

        assert len(tvars) == 1
        assert tvars[0].name == "max_tokens"
        assert "max_tokens" in config_space

    def test_parse_bool_type(self) -> None:
        """Parse bool tvar type."""
        resolved = {
            "tvars": [
                {
                    "name": "stream",
                    "type": "bool",
                    "domain": [True, False],
                    "default": False,
                }
            ]
        }
        tvars, config_space, defaults, units = _parse_tvars(resolved)

        assert len(tvars) == 1
        assert tvars[0].name == "stream"
        assert defaults["stream"] is False
        assert config_space["stream"] == [True, False]

    def test_parse_tvar_with_unit(self) -> None:
        """Parse tvar with unit specification."""
        resolved = {
            "tvars": [
                {
                    "name": "latency_threshold",
                    "type": "float",
                    "domain": {"range": [0.1, 5.0]},
                    "unit": "seconds",
                }
            ]
        }
        tvars, _, _, units = _parse_tvars(resolved)

        assert len(tvars) == 1
        assert tvars[0].unit == "seconds"
        assert units["latency_threshold"] == "seconds"

    def test_parse_tvar_with_agent(self) -> None:
        """Parse tvar with agent assignment (multi-agent support)."""
        resolved = {
            "tvars": [
                {
                    "name": "prompt_template",
                    "type": "enum[str]",
                    "domain": ["template_a", "template_b"],
                    "agent": "prompt_agent",
                }
            ]
        }
        tvars, config_space, defaults, units = _parse_tvars(resolved)

        assert len(tvars) == 1
        assert tvars[0].agent == "prompt_agent"

    def test_parse_multiple_tvars(self) -> None:
        """Parse multiple tvars in sequence."""
        resolved = {
            "tvars": [
                {"name": "model", "type": "enum[str]", "domain": ["gpt-4o"]},
                {"name": "temperature", "type": "float", "domain": [0.0, 1.0]},
                {"name": "top_p", "type": "float", "domain": [0.0, 1.0]},
            ]
        }
        tvars, config_space, defaults, units = _parse_tvars(resolved)

        assert len(tvars) == 3
        assert {t.name for t in tvars} == {"model", "temperature", "top_p"}
        assert len(config_space) == 3

    def test_empty_tvars_raises_error(self) -> None:
        """Empty tvars list raises TVLValidationError."""
        resolved = {"tvars": []}

        with pytest.raises(TVLValidationError, match="must be a non-empty array"):
            _parse_tvars(resolved)

    def test_non_list_tvars_raises_error(self) -> None:
        """Non-list tvars raises TVLValidationError."""
        resolved = {"tvars": {"model": {"type": "enum[str]"}}}

        with pytest.raises(TVLValidationError, match="must be a non-empty array"):
            _parse_tvars(resolved)

    def test_missing_name_raises_error(self) -> None:
        """TVAR without name raises TVLValidationError."""
        resolved = {"tvars": [{"type": "enum[str]", "domain": ["a", "b"]}]}

        with pytest.raises(TVLValidationError, match="requires a 'name' string"):
            _parse_tvars(resolved)

    def test_missing_type_raises_error(self) -> None:
        """TVAR without type raises TVLValidationError."""
        resolved = {"tvars": [{"name": "model", "domain": ["a", "b"]}]}

        with pytest.raises(TVLValidationError, match="requires a 'type' string"):
            _parse_tvars(resolved)

    def test_unsupported_type_raises_error(self) -> None:
        """TVAR with unsupported type raises TVLValidationError."""
        resolved = {"tvars": [{"name": "x", "type": "complex", "domain": [1, 2]}]}

        with pytest.raises(TVLValidationError, match="unsupported type"):
            _parse_tvars(resolved)

    def test_registry_domain_without_resolver_raises_error(self) -> None:
        """Registry domain without resolver raises TVLValidationError."""
        resolved = {
            "tvars": [
                {
                    "name": "retriever",
                    "type": "enum[str]",
                    "domain": {"type": "registry", "provider": "rag.retrievers"},
                }
            ]
        }

        with pytest.raises(TVLValidationError, match="[Rr]egistry"):
            _parse_tvars(resolved, registry_resolver=None)


class TestParseConvergence:
    """Tests for _parse_convergence function."""

    def test_valid_convergence_criteria(self) -> None:
        """Parse valid convergence criteria from exploration."""
        exploration_data = {
            "convergence": {
                "metric": "hypervolume_improvement",
                "window": 10,
                "threshold": 0.005,
            }
        }
        result = _parse_convergence(exploration_data)

        assert result is not None
        assert isinstance(result, ConvergenceCriteria)
        assert result.metric == "hypervolume_improvement"
        assert result.window == 10
        assert result.threshold == 0.005

    def test_convergence_with_defaults(self) -> None:
        """Parse convergence with default values."""
        exploration_data = {"convergence": {}}
        result = _parse_convergence(exploration_data)

        assert result is not None
        assert result.metric == "hypervolume_improvement"
        assert result.window == 5
        assert result.threshold == 0.01

    def test_convergence_none_metric(self) -> None:
        """Parse convergence with 'none' metric."""
        exploration_data = {"convergence": {"metric": "none"}}
        result = _parse_convergence(exploration_data)

        assert result is not None
        assert result.metric == "none"

    def test_no_convergence_section_returns_none(self) -> None:
        """Missing convergence section returns None."""
        exploration_data = {"budgets": {"max_trials": 100}}
        result = _parse_convergence(exploration_data)

        assert result is None

    def test_non_dict_exploration_returns_none(self) -> None:
        """Non-dict exploration input returns None."""
        result = _parse_convergence("not a dict")
        assert result is None

    def test_non_dict_convergence_raises_error(self) -> None:
        """Non-dict convergence raises TVLValidationError."""
        exploration_data = {"convergence": "invalid"}

        with pytest.raises(TVLValidationError, match="must be a mapping"):
            _parse_convergence(exploration_data)


class TestParseExplorationBudgets:
    """Tests for _parse_exploration_budgets function."""

    def test_valid_budgets_all_fields(self) -> None:
        """Parse valid exploration budgets with all fields."""
        exploration_data = {
            "budgets": {
                "max_trials": 100,
                "max_spend_usd": 50.0,
                "max_wallclock_s": 3600,
            }
        }
        result = _parse_exploration_budgets(exploration_data)

        assert result is not None
        assert isinstance(result, ExplorationBudgets)
        assert result.max_trials == 100
        assert result.max_spend_usd == 50.0
        assert result.max_wallclock_s == 3600

    def test_budgets_partial_fields(self) -> None:
        """Parse budgets with only some fields set."""
        exploration_data = {"budgets": {"max_trials": 50}}
        result = _parse_exploration_budgets(exploration_data)

        assert result is not None
        assert result.max_trials == 50
        assert result.max_spend_usd is None
        assert result.max_wallclock_s is None

    def test_budgets_empty_dict(self) -> None:
        """Parse empty budgets dict (all None)."""
        exploration_data = {"budgets": {}}
        result = _parse_exploration_budgets(exploration_data)

        assert result is not None
        assert result.max_trials is None
        assert result.max_spend_usd is None
        assert result.max_wallclock_s is None

    def test_no_budgets_section_returns_none(self) -> None:
        """Missing budgets section returns None."""
        exploration_data = {"convergence": {"metric": "none"}}
        result = _parse_exploration_budgets(exploration_data)

        assert result is None

    def test_non_dict_exploration_returns_none(self) -> None:
        """Non-dict exploration input returns None."""
        result = _parse_exploration_budgets([1, 2, 3])
        assert result is None

    def test_non_dict_budgets_raises_error(self) -> None:
        """Non-dict budgets raises TVLValidationError."""
        exploration_data = {"budgets": 100}

        with pytest.raises(TVLValidationError, match="must be a mapping"):
            _parse_exploration_budgets(exploration_data)


class TestParseExplorationParallelism:
    """Tests for _parse_exploration_parallelism function."""

    def test_valid_parallelism_config(self) -> None:
        """Parse valid parallelism configuration."""
        exploration_data = {"parallelism": {"max_parallel_trials": 4}}
        result = _parse_exploration_parallelism(exploration_data)

        assert result == 4

    def test_parallelism_with_other_fields(self) -> None:
        """Parse parallelism ignoring extra fields."""
        exploration_data = {
            "parallelism": {
                "max_parallel_trials": 8,
                "executor": "thread_pool",  # Extra field ignored
            }
        }
        result = _parse_exploration_parallelism(exploration_data)

        assert result == 8

    def test_no_parallelism_section_returns_none(self) -> None:
        """Missing parallelism section returns None."""
        exploration_data = {"budgets": {"max_trials": 100}}
        result = _parse_exploration_parallelism(exploration_data)

        assert result is None

    def test_parallelism_without_max_parallel_returns_none(self) -> None:
        """Parallelism dict without max_parallel_trials returns None."""
        exploration_data = {"parallelism": {"executor": "process_pool"}}
        result = _parse_exploration_parallelism(exploration_data)

        assert result is None

    def test_non_dict_exploration_returns_none(self) -> None:
        """Non-dict exploration input returns None."""
        result = _parse_exploration_parallelism(None)
        assert result is None

    def test_non_dict_parallelism_returns_none(self) -> None:
        """Non-dict parallelism returns None (no error)."""
        exploration_data = {"parallelism": "auto"}
        result = _parse_exploration_parallelism(exploration_data)

        assert result is None

    def test_non_int_max_parallel_returns_none(self) -> None:
        """Non-int max_parallel_trials returns None."""
        exploration_data = {"parallelism": {"max_parallel_trials": "auto"}}
        result = _parse_exploration_parallelism(exploration_data)

        assert result is None
