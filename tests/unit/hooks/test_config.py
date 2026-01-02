"""Unit tests for traigent.hooks.config.

Tests for configuration loading and validation for Traigent hooks,
including YAML parsing, config file discovery, and constraint definitions.
"""

# Traceability: CONC-Layer-Config CONC-Quality-Reliability CONC-Quality-Maintainability FUNC-CONFIG-LOAD REQ-CONFIG-001

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
import yaml

from traigent.hooks.config import (
    CONFIG_FILE_NAMES,
    CostConstraints,
    HooksConfig,
    HooksConstraints,
    ModelConstraints,
    PerformanceConstraints,
    create_default_config,
    find_config_file,
    load_hooks_config,
)


class TestCostConstraints:
    """Tests for CostConstraints dataclass."""

    def test_default_initialization(self) -> None:
        """Test CostConstraints initializes with default values."""
        constraints = CostConstraints()
        assert constraints.max_cost_per_query is None
        assert constraints.max_monthly_budget is None
        assert constraints.warn_threshold_pct == 0.8

    def test_initialization_with_values(self) -> None:
        """Test CostConstraints initializes with provided values."""
        constraints = CostConstraints(
            max_cost_per_query=0.05,
            max_monthly_budget=1000.0,
            warn_threshold_pct=0.9,
        )
        assert constraints.max_cost_per_query == 0.05
        assert constraints.max_monthly_budget == 1000.0
        assert constraints.warn_threshold_pct == 0.9

    def test_partial_initialization(self) -> None:
        """Test CostConstraints initializes with partial values."""
        constraints = CostConstraints(max_cost_per_query=0.10)
        assert constraints.max_cost_per_query == 0.10
        assert constraints.max_monthly_budget is None
        assert constraints.warn_threshold_pct == 0.8


class TestPerformanceConstraints:
    """Tests for PerformanceConstraints dataclass."""

    def test_default_initialization(self) -> None:
        """Test PerformanceConstraints initializes with default values."""
        constraints = PerformanceConstraints()
        assert constraints.min_accuracy is None
        assert constraints.max_latency_ms is None
        assert constraints.min_success_rate is None

    def test_initialization_with_values(self) -> None:
        """Test PerformanceConstraints initializes with provided values."""
        constraints = PerformanceConstraints(
            min_accuracy=0.85, max_latency_ms=500, min_success_rate=0.95
        )
        assert constraints.min_accuracy == 0.85
        assert constraints.max_latency_ms == 500
        assert constraints.min_success_rate == 0.95

    def test_partial_initialization(self) -> None:
        """Test PerformanceConstraints initializes with partial values."""
        constraints = PerformanceConstraints(min_accuracy=0.9)
        assert constraints.min_accuracy == 0.9
        assert constraints.max_latency_ms is None
        assert constraints.min_success_rate is None


class TestModelConstraints:
    """Tests for ModelConstraints dataclass."""

    def test_default_initialization(self) -> None:
        """Test ModelConstraints initializes with default empty lists."""
        constraints = ModelConstraints()
        assert constraints.allowed_models == []
        assert constraints.blocked_models == []
        assert constraints.blocked_reasons == {}

    def test_initialization_with_values(self) -> None:
        """Test ModelConstraints initializes with provided values."""
        constraints = ModelConstraints(
            allowed_models=["gpt-4o", "claude-3-sonnet"],
            blocked_models=["gpt-4-32k"],
            blocked_reasons={"gpt-4-32k": "Too expensive"},
        )
        assert constraints.allowed_models == ["gpt-4o", "claude-3-sonnet"]
        assert constraints.blocked_models == ["gpt-4-32k"]
        assert constraints.blocked_reasons == {"gpt-4-32k": "Too expensive"}

    def test_empty_lists_are_mutable(self) -> None:
        """Test that default factory creates new lists for each instance."""
        constraints1 = ModelConstraints()
        constraints2 = ModelConstraints()
        constraints1.allowed_models.append("test-model")
        assert len(constraints1.allowed_models) == 1
        assert len(constraints2.allowed_models) == 0


class TestHooksConstraints:
    """Tests for HooksConstraints dataclass."""

    def test_default_initialization(self) -> None:
        """Test HooksConstraints initializes with default constraint objects."""
        constraints = HooksConstraints()
        assert isinstance(constraints.cost, CostConstraints)
        assert isinstance(constraints.performance, PerformanceConstraints)
        assert isinstance(constraints.models, ModelConstraints)

    def test_initialization_with_values(self) -> None:
        """Test HooksConstraints initializes with provided constraint objects."""
        cost = CostConstraints(max_cost_per_query=0.05)
        performance = PerformanceConstraints(min_accuracy=0.85)
        models = ModelConstraints(allowed_models=["gpt-4o"])
        constraints = HooksConstraints(
            cost=cost, performance=performance, models=models
        )
        assert constraints.cost.max_cost_per_query == 0.05
        assert constraints.performance.min_accuracy == 0.85
        assert constraints.models.allowed_models == ["gpt-4o"]


class TestHooksConfig:
    """Tests for HooksConfig dataclass and methods."""

    def test_default_initialization(self) -> None:
        """Test HooksConfig initializes with default values."""
        config = HooksConfig()
        assert config.enabled is True
        assert config.fail_on_warning is False
        assert config.skip_patterns == []
        assert config.pre_push_hooks == ["traigent-validate"]
        assert config.pre_commit_hooks == []
        assert isinstance(config.constraints, HooksConstraints)
        assert config.raw_config == {}

    def test_initialization_with_values(self) -> None:
        """Test HooksConfig initializes with provided values."""
        config = HooksConfig(
            enabled=False,
            fail_on_warning=True,
            skip_patterns=["*.test.py"],
            pre_push_hooks=["custom-hook"],
            pre_commit_hooks=["pre-hook"],
        )
        assert config.enabled is False
        assert config.fail_on_warning is True
        assert config.skip_patterns == ["*.test.py"]
        assert config.pre_push_hooks == ["custom-hook"]
        assert config.pre_commit_hooks == ["pre-hook"]

    def test_from_dict_minimal(self) -> None:
        """Test from_dict with minimal empty configuration."""
        data: dict[str, Any] = {}
        config = HooksConfig.from_dict(data)
        assert config.enabled is True
        assert config.fail_on_warning is False
        assert config.pre_push_hooks == ["traigent-validate"]
        assert config.pre_commit_hooks == []

    def test_from_dict_validation_settings(self) -> None:
        """Test from_dict with validation settings."""
        data = {
            "validation": {
                "enabled": False,
                "fail_on_warning": True,
                "skip_patterns": ["test_*.py", "*.tmp"],
            }
        }
        config = HooksConfig.from_dict(data)
        assert config.enabled is False
        assert config.fail_on_warning is True
        assert config.skip_patterns == ["test_*.py", "*.tmp"]

    def test_from_dict_hooks_as_list(self) -> None:
        """Test from_dict with hooks configured as lists."""
        data = {
            "validation": {
                "hooks": {
                    "pre-push": ["hook1", "hook2"],
                    "pre-commit": ["hook3"],
                }
            }
        }
        config = HooksConfig.from_dict(data)
        assert config.pre_push_hooks == ["hook1", "hook2"]
        assert config.pre_commit_hooks == ["hook3"]

    def test_from_dict_hooks_as_string(self) -> None:
        """Test from_dict converts single hook string to list."""
        data = {
            "validation": {
                "hooks": {
                    "pre-push": "single-hook",
                    "pre-commit": "single-commit-hook",
                }
            }
        }
        config = HooksConfig.from_dict(data)
        assert config.pre_push_hooks == ["single-hook"]
        assert config.pre_commit_hooks == ["single-commit-hook"]

    def test_from_dict_cost_constraints_nested(self) -> None:
        """Test from_dict with nested cost constraints."""
        data = {
            "constraints": {
                "cost": {
                    "max_cost_per_query": 0.05,
                    "max_monthly_budget": 1000.0,
                    "warn_threshold_pct": 0.75,
                }
            }
        }
        config = HooksConfig.from_dict(data)
        assert config.constraints.cost.max_cost_per_query == 0.05
        assert config.constraints.cost.max_monthly_budget == 1000.0
        assert config.constraints.cost.warn_threshold_pct == 0.75

    def test_from_dict_cost_constraints_flat(self) -> None:
        """Test from_dict with flat cost constraints for backwards compatibility."""
        data = {
            "constraints": {
                "max_cost_per_query": 0.10,
                "max_monthly_budget": 2000.0,
            }
        }
        config = HooksConfig.from_dict(data)
        assert config.constraints.cost.max_cost_per_query == 0.10
        assert config.constraints.cost.max_monthly_budget == 2000.0

    def test_from_dict_cost_constraints_nested_overrides_flat(self) -> None:
        """Test from_dict where nested cost constraints take precedence over flat."""
        data = {
            "constraints": {
                "cost": {"max_cost_per_query": 0.05},
                "max_cost_per_query": 0.10,
            }
        }
        config = HooksConfig.from_dict(data)
        assert config.constraints.cost.max_cost_per_query == 0.05

    def test_from_dict_performance_constraints_nested(self) -> None:
        """Test from_dict with nested performance constraints."""
        data = {
            "constraints": {
                "performance": {
                    "min_accuracy": 0.85,
                    "max_latency_ms": 500,
                    "min_success_rate": 0.95,
                }
            }
        }
        config = HooksConfig.from_dict(data)
        assert config.constraints.performance.min_accuracy == 0.85
        assert config.constraints.performance.max_latency_ms == 500
        assert config.constraints.performance.min_success_rate == 0.95

    def test_from_dict_performance_constraints_flat(self) -> None:
        """Test from_dict with flat performance constraints."""
        data = {
            "constraints": {
                "min_accuracy": 0.90,
                "max_latency_ms": 1000,
            }
        }
        config = HooksConfig.from_dict(data)
        assert config.constraints.performance.min_accuracy == 0.90
        assert config.constraints.performance.max_latency_ms == 1000

    def test_from_dict_model_constraints_nested(self) -> None:
        """Test from_dict with nested model constraints."""
        data = {
            "constraints": {
                "models": {
                    "allowed": ["gpt-4o", "claude-3-sonnet"],
                    "blocked": ["gpt-4-32k"],
                    "blocked_reasons": {"gpt-4-32k": "Too expensive"},
                }
            }
        }
        config = HooksConfig.from_dict(data)
        assert config.constraints.models.allowed_models == ["gpt-4o", "claude-3-sonnet"]
        assert config.constraints.models.blocked_models == ["gpt-4-32k"]
        assert config.constraints.models.blocked_reasons == {
            "gpt-4-32k": "Too expensive"
        }

    def test_from_dict_model_constraints_flat(self) -> None:
        """Test from_dict with flat model constraints."""
        data = {
            "constraints": {
                "allowed_models": ["gpt-4o"],
                "blocked_models": ["gpt-4-32k"],
            }
        }
        config = HooksConfig.from_dict(data)
        assert config.constraints.models.allowed_models == ["gpt-4o"]
        assert config.constraints.models.blocked_models == ["gpt-4-32k"]

    def test_from_dict_stores_raw_config(self) -> None:
        """Test from_dict stores the raw configuration dictionary."""
        data = {
            "validation": {"enabled": True},
            "custom_field": "custom_value",
        }
        config = HooksConfig.from_dict(data)
        assert config.raw_config == data
        assert config.raw_config["custom_field"] == "custom_value"

    def test_to_dict_default_config(self) -> None:
        """Test to_dict with default configuration."""
        config = HooksConfig()
        result = config.to_dict()
        assert result["validation"]["enabled"] is True
        assert result["validation"]["fail_on_warning"] is False
        assert result["validation"]["skip_patterns"] == []
        assert result["validation"]["hooks"]["pre-push"] == ["traigent-validate"]
        assert result["validation"]["hooks"]["pre-commit"] == []

    def test_to_dict_with_constraints(self) -> None:
        """Test to_dict includes constraint values."""
        cost = CostConstraints(max_cost_per_query=0.05, max_monthly_budget=1000.0)
        performance = PerformanceConstraints(min_accuracy=0.85, max_latency_ms=500)
        models = ModelConstraints(
            allowed_models=["gpt-4o"], blocked_models=["gpt-4-32k"]
        )
        constraints = HooksConstraints(
            cost=cost, performance=performance, models=models
        )
        config = HooksConfig(constraints=constraints)
        result = config.to_dict()
        assert result["constraints"]["max_cost_per_query"] == 0.05
        assert result["constraints"]["max_monthly_budget"] == 1000.0
        assert result["constraints"]["min_accuracy"] == 0.85
        assert result["constraints"]["max_latency_ms"] == 500
        assert result["constraints"]["allowed_models"] == ["gpt-4o"]
        assert result["constraints"]["blocked_models"] == ["gpt-4-32k"]

    def test_to_dict_round_trip(self) -> None:
        """Test that from_dict and to_dict are compatible."""
        original_data = {
            "validation": {
                "enabled": False,
                "fail_on_warning": True,
                "skip_patterns": ["*.test"],
                "hooks": {
                    "pre-push": ["hook1"],
                    "pre-commit": ["hook2"],
                },
            },
            "constraints": {
                "max_cost_per_query": 0.05,
                "max_monthly_budget": 1000.0,
                "min_accuracy": 0.85,
                "max_latency_ms": 500,
                "allowed_models": ["gpt-4o"],
                "blocked_models": ["gpt-4-32k"],
            },
        }
        config = HooksConfig.from_dict(original_data)
        result = config.to_dict()
        # Check key fields match
        assert result["validation"]["enabled"] == original_data["validation"]["enabled"]
        assert (
            result["validation"]["fail_on_warning"]
            == original_data["validation"]["fail_on_warning"]
        )
        assert (
            result["constraints"]["max_cost_per_query"]
            == original_data["constraints"]["max_cost_per_query"]
        )


class TestFindConfigFile:
    """Tests for find_config_file function."""

    def test_find_config_from_cwd(self) -> None:
        """Test find_config_file finds config in current directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            config_path = tmp_path / "traigent.yml"
            config_path.touch()
            result = find_config_file(tmp_path)
            assert result == config_path

    def test_find_config_with_yaml_extension(self) -> None:
        """Test find_config_file finds .yaml extension."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            config_path = tmp_path / "traigent.yaml"
            config_path.touch()
            result = find_config_file(tmp_path)
            assert result == config_path

    def test_find_config_with_dot_prefix(self) -> None:
        """Test find_config_file finds hidden config files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            config_path = tmp_path / ".traigent.yml"
            config_path.touch()
            result = find_config_file(tmp_path)
            assert result == config_path

    def test_find_config_prioritizes_order(self) -> None:
        """Test find_config_file respects CONFIG_FILE_NAMES priority."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            # Create all config files
            (tmp_path / "traigent.yml").touch()
            (tmp_path / "traigent.yaml").touch()
            (tmp_path / ".traigent.yml").touch()
            result = find_config_file(tmp_path)
            # Should find the first one in CONFIG_FILE_NAMES
            assert result == tmp_path / CONFIG_FILE_NAMES[0]

    def test_find_config_in_parent_directory(self) -> None:
        """Test find_config_file searches parent directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            config_path = tmp_path / "traigent.yml"
            config_path.touch()
            subdir = tmp_path / "subdir"
            subdir.mkdir()
            result = find_config_file(subdir)
            assert result == config_path

    def test_find_config_multiple_levels_up(self) -> None:
        """Test find_config_file searches multiple levels up."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            config_path = tmp_path / "traigent.yml"
            config_path.touch()
            deep_dir = tmp_path / "a" / "b" / "c"
            deep_dir.mkdir(parents=True)
            result = find_config_file(deep_dir)
            assert result == config_path

    def test_find_config_returns_none_when_not_found(self) -> None:
        """Test find_config_file returns None when no config exists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            result = find_config_file(tmp_path)
            assert result is None

    def test_find_config_stops_at_filesystem_root(self) -> None:
        """Test find_config_file stops at filesystem root."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            result = find_config_file(tmp_path)
            assert result is None

    def test_find_config_defaults_to_cwd(self) -> None:
        """Test find_config_file defaults to current working directory."""
        with patch("traigent.hooks.config.Path.cwd") as mock_cwd:
            with tempfile.TemporaryDirectory() as tmpdir:
                tmp_path = Path(tmpdir)
                mock_cwd.return_value = tmp_path
                config_path = tmp_path / "traigent.yml"
                config_path.touch()
                result = find_config_file(None)
                assert result == config_path

    def test_find_config_ignores_directories(self) -> None:
        """Test find_config_file ignores directories with config names."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            # Create a directory with config name
            (tmp_path / "traigent.yml").mkdir()
            result = find_config_file(tmp_path)
            assert result is None


class TestLoadHooksConfig:
    """Tests for load_hooks_config function."""

    def test_load_config_with_valid_file(self) -> None:
        """Test load_hooks_config loads valid YAML configuration."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            config_path = tmp_path / "traigent.yml"
            config_data = {
                "validation": {"enabled": True, "fail_on_warning": False},
                "constraints": {"max_cost_per_query": 0.05},
            }
            with open(config_path, "w") as f:
                yaml.dump(config_data, f)
            config = load_hooks_config(config_path)
            assert config.enabled is True
            assert config.fail_on_warning is False
            assert config.constraints.cost.max_cost_per_query == 0.05

    def test_load_config_with_empty_file(self) -> None:
        """Test load_hooks_config handles empty YAML file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            config_path = tmp_path / "traigent.yml"
            config_path.touch()
            config = load_hooks_config(config_path)
            assert isinstance(config, HooksConfig)
            assert config.enabled is True

    def test_load_config_auto_detects_file(self) -> None:
        """Test load_hooks_config auto-detects config file when path is None."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            config_path = tmp_path / "traigent.yml"
            config_data = {"validation": {"enabled": False}}
            with open(config_path, "w") as f:
                yaml.dump(config_data, f)
            with patch("traigent.hooks.config.find_config_file") as mock_find:
                mock_find.return_value = config_path
                config = load_hooks_config(None)
                assert config.enabled is False
                mock_find.assert_called_once()

    def test_load_config_returns_default_when_not_found(self) -> None:
        """Test load_hooks_config returns default config when file not found."""
        with patch("traigent.hooks.config.find_config_file") as mock_find:
            mock_find.return_value = None
            config = load_hooks_config(None)
            assert isinstance(config, HooksConfig)
            assert config.enabled is True

    def test_load_config_raises_on_missing_explicit_path(self) -> None:
        """Test load_hooks_config raises FileNotFoundError for explicit missing path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            config_path = tmp_path / "nonexistent.yml"
            with pytest.raises(FileNotFoundError, match="Configuration file not found"):
                load_hooks_config(config_path)

    def test_load_config_accepts_string_path(self) -> None:
        """Test load_hooks_config accepts string paths."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "traigent.yml"
            config_data = {"validation": {"enabled": False}}
            with open(config_path, "w") as f:
                yaml.dump(config_data, f)
            config = load_hooks_config(str(config_path))
            assert config.enabled is False

    def test_load_config_handles_invalid_yaml(self) -> None:
        """Test load_hooks_config raises YAMLError for invalid YAML."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            config_path = tmp_path / "traigent.yml"
            with open(config_path, "w") as f:
                f.write("invalid: yaml: content: [")
            with pytest.raises(yaml.YAMLError):
                load_hooks_config(config_path)

    def test_load_config_complex_configuration(self) -> None:
        """Test load_hooks_config with complex nested configuration."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            config_path = tmp_path / "traigent.yml"
            config_data = {
                "validation": {
                    "enabled": True,
                    "fail_on_warning": True,
                    "skip_patterns": ["test_*.py"],
                    "hooks": {
                        "pre-push": ["hook1", "hook2"],
                        "pre-commit": ["hook3"],
                    },
                },
                "constraints": {
                    "cost": {
                        "max_cost_per_query": 0.05,
                        "max_monthly_budget": 1000.0,
                        "warn_threshold_pct": 0.9,
                    },
                    "performance": {
                        "min_accuracy": 0.85,
                        "max_latency_ms": 500,
                        "min_success_rate": 0.95,
                    },
                    "models": {
                        "allowed": ["gpt-4o", "claude-3-sonnet"],
                        "blocked": ["gpt-4-32k"],
                        "blocked_reasons": {"gpt-4-32k": "Too expensive"},
                    },
                },
            }
            with open(config_path, "w") as f:
                yaml.dump(config_data, f)
            config = load_hooks_config(config_path)
            assert config.enabled is True
            assert config.fail_on_warning is True
            assert config.skip_patterns == ["test_*.py"]
            assert config.pre_push_hooks == ["hook1", "hook2"]
            assert config.pre_commit_hooks == ["hook3"]
            assert config.constraints.cost.max_cost_per_query == 0.05
            assert config.constraints.cost.max_monthly_budget == 1000.0
            assert config.constraints.performance.min_accuracy == 0.85
            assert config.constraints.performance.max_latency_ms == 500
            assert config.constraints.models.allowed_models == [
                "gpt-4o",
                "claude-3-sonnet",
            ]


class TestCreateDefaultConfig:
    """Tests for create_default_config function."""

    def test_create_default_config_at_default_path(self) -> None:
        """Test create_default_config creates file at default location."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            with patch("traigent.hooks.config.Path.cwd") as mock_cwd:
                mock_cwd.return_value = tmp_path
                result = create_default_config(None)
                assert result == tmp_path / "traigent.yml"
                assert result.exists()

    def test_create_default_config_at_custom_path(self) -> None:
        """Test create_default_config creates file at specified path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            custom_path = tmp_path / "custom.yml"
            result = create_default_config(custom_path)
            assert result == custom_path
            assert custom_path.exists()

    def test_create_default_config_accepts_string_path(self) -> None:
        """Test create_default_config accepts string paths."""
        with tempfile.TemporaryDirectory() as tmpdir:
            custom_path = str(Path(tmpdir) / "custom.yml")
            result = create_default_config(custom_path)
            assert Path(result).exists()

    def test_create_default_config_content_is_valid_yaml(self) -> None:
        """Test create_default_config produces valid YAML content."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            config_path = tmp_path / "traigent.yml"
            create_default_config(config_path)
            with open(config_path) as f:
                data = yaml.safe_load(f)
            assert isinstance(data, dict)
            assert "validation" in data
            assert "constraints" in data

    def test_create_default_config_has_expected_structure(self) -> None:
        """Test create_default_config has expected configuration structure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            config_path = tmp_path / "traigent.yml"
            create_default_config(config_path)
            with open(config_path) as f:
                data = yaml.safe_load(f)
            assert data["validation"]["enabled"] is True
            assert isinstance(data["validation"]["hooks"], dict)
            assert "pre-push" in data["validation"]["hooks"]
            assert isinstance(data["constraints"], dict)
            assert "max_cost_per_query" in data["constraints"]

    def test_create_default_config_overwrites_existing(self) -> None:
        """Test create_default_config overwrites existing file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            config_path = tmp_path / "traigent.yml"
            # Create existing file
            with open(config_path, "w") as f:
                f.write("old content")
            create_default_config(config_path)
            with open(config_path) as f:
                content = f.read()
            assert "old content" not in content
            assert "Traigent Agent Configuration" in content

    def test_create_default_config_loadable(self) -> None:
        """Test create_default_config produces loadable configuration."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            config_path = tmp_path / "traigent.yml"
            create_default_config(config_path)
            config = load_hooks_config(config_path)
            assert isinstance(config, HooksConfig)
            assert config.enabled is True
            assert config.constraints.cost.max_cost_per_query == 0.05
            assert config.constraints.cost.max_monthly_budget == 1000
            assert config.constraints.performance.min_accuracy == 0.85
            assert config.constraints.performance.max_latency_ms == 500
