"""Comprehensive tests for traigent.core.trial_context module.

Tests cover TrialRunContext dataclass for trial execution.
"""

from __future__ import annotations

from unittest.mock import Mock

import pytest

from traigent.core.trial_context import TrialRunContext
from traigent.evaluators.base import Dataset


@pytest.fixture
def mock_function():
    """Create mock function."""

    def test_func(input_data):
        return {"output": input_data}

    return test_func


@pytest.fixture
def mock_dataset():
    """Create mock dataset."""
    dataset = Mock(spec=Dataset)
    dataset.examples = [{"input": "test1"}, {"input": "test2"}]
    return dataset


@pytest.fixture
def sample_config():
    """Create sample configuration."""
    return {"model": "gpt-4", "temperature": 0.7}


class TestTrialRunContext:
    """Test TrialRunContext dataclass."""

    def test_basic_creation(self, mock_function, mock_dataset, sample_config):
        """Test basic TrialRunContext creation."""
        context = TrialRunContext(
            func=mock_function,
            config=sample_config,
            dataset=mock_dataset,
            trial_number=1,
            session_id="session_123",
        )

        assert context.func == mock_function
        assert context.config == sample_config
        assert context.dataset == mock_dataset
        assert context.trial_number == 1
        assert context.session_id == "session_123"
        assert context.optuna_trial_id is None

    def test_with_optuna_trial_id(self, mock_function, mock_dataset, sample_config):
        """Test TrialRunContext with optuna trial ID."""
        context = TrialRunContext(
            func=mock_function,
            config=sample_config,
            dataset=mock_dataset,
            trial_number=5,
            session_id="session_456",
            optuna_trial_id=42,
        )

        assert context.optuna_trial_id == 42

    def test_no_session_id(self, mock_function, mock_dataset, sample_config):
        """Test TrialRunContext without session ID (backend disabled)."""
        context = TrialRunContext(
            func=mock_function,
            config=sample_config,
            dataset=mock_dataset,
            trial_number=3,
            session_id=None,
        )

        assert context.session_id is None

    def test_attribute_access(self, mock_function, mock_dataset, sample_config):
        """Test all attributes are accessible."""
        context = TrialRunContext(
            func=mock_function,
            config=sample_config,
            dataset=mock_dataset,
            trial_number=2,
            session_id="session_789",
            optuna_trial_id=99,
        )

        assert hasattr(context, "func")
        assert hasattr(context, "config")
        assert hasattr(context, "dataset")
        assert hasattr(context, "trial_number")
        assert hasattr(context, "session_id")
        assert hasattr(context, "optuna_trial_id")

    def test_different_trial_numbers(self, mock_function, mock_dataset, sample_config):
        """Test contexts with different trial numbers."""
        context1 = TrialRunContext(
            func=mock_function,
            config=sample_config,
            dataset=mock_dataset,
            trial_number=1,
            session_id="session_1",
        )

        context2 = TrialRunContext(
            func=mock_function,
            config=sample_config,
            dataset=mock_dataset,
            trial_number=10,
            session_id="session_1",
        )

        assert context1.trial_number == 1
        assert context2.trial_number == 10

    def test_different_configs(self, mock_function, mock_dataset):
        """Test contexts with different configurations."""
        config1 = {"model": "gpt-3.5-turbo", "temperature": 0.5}
        config2 = {"model": "gpt-4", "temperature": 0.9}

        context1 = TrialRunContext(
            func=mock_function,
            config=config1,
            dataset=mock_dataset,
            trial_number=1,
            session_id="session_1",
        )

        context2 = TrialRunContext(
            func=mock_function,
            config=config2,
            dataset=mock_dataset,
            trial_number=2,
            session_id="session_1",
        )

        assert context1.config["temperature"] == 0.5
        assert context2.config["temperature"] == 0.9

    def test_dataclass_equality(self, mock_function, mock_dataset, sample_config):
        """Test dataclass equality comparison."""
        context1 = TrialRunContext(
            func=mock_function,
            config=sample_config,
            dataset=mock_dataset,
            trial_number=1,
            session_id="session_1",
            optuna_trial_id=42,
        )

        context2 = TrialRunContext(
            func=mock_function,
            config=sample_config,
            dataset=mock_dataset,
            trial_number=1,
            session_id="session_1",
            optuna_trial_id=42,
        )

        # Dataclasses with same values should be equal
        assert context1 == context2

    def test_dataclass_inequality(self, mock_function, mock_dataset, sample_config):
        """Test dataclass inequality with different values."""
        context1 = TrialRunContext(
            func=mock_function,
            config=sample_config,
            dataset=mock_dataset,
            trial_number=1,
            session_id="session_1",
        )

        context2 = TrialRunContext(
            func=mock_function,
            config=sample_config,
            dataset=mock_dataset,
            trial_number=2,  # Different trial number
            session_id="session_1",
        )

        assert context1 != context2

    def test_function_callable(self, mock_function, mock_dataset, sample_config):
        """Test that func attribute is callable."""
        context = TrialRunContext(
            func=mock_function,
            config=sample_config,
            dataset=mock_dataset,
            trial_number=1,
            session_id="session_1",
        )

        assert callable(context.func)
        result = context.func("test_input")
        assert result == {"output": "test_input"}

    def test_dataset_access(self, mock_function, mock_dataset, sample_config):
        """Test dataset access through context."""
        context = TrialRunContext(
            func=mock_function,
            config=sample_config,
            dataset=mock_dataset,
            trial_number=1,
            session_id="session_1",
        )

        assert context.dataset == mock_dataset
        assert len(context.dataset.examples) == 2

    def test_config_mutation(self, mock_function, mock_dataset, sample_config):
        """Test config can be mutated after context creation."""
        context = TrialRunContext(
            func=mock_function,
            config=sample_config,
            dataset=mock_dataset,
            trial_number=1,
            session_id="session_1",
        )

        # Config is mutable dict
        context.config["new_param"] = "new_value"
        assert "new_param" in context.config
        assert context.config["new_param"] == "new_value"

    def test_zero_trial_number(self, mock_function, mock_dataset, sample_config):
        """Test trial_number can be zero."""
        context = TrialRunContext(
            func=mock_function,
            config=sample_config,
            dataset=mock_dataset,
            trial_number=0,
            session_id="session_1",
        )

        assert context.trial_number == 0

    def test_negative_optuna_trial_id(self, mock_function, mock_dataset, sample_config):
        """Test optuna_trial_id can be negative."""
        context = TrialRunContext(
            func=mock_function,
            config=sample_config,
            dataset=mock_dataset,
            trial_number=1,
            session_id="session_1",
            optuna_trial_id=-1,
        )

        assert context.optuna_trial_id == -1

    def test_empty_config(self, mock_function, mock_dataset):
        """Test context with empty configuration."""
        context = TrialRunContext(
            func=mock_function,
            config={},
            dataset=mock_dataset,
            trial_number=1,
            session_id="session_1",
        )

        assert context.config == {}

    def test_repr_string(self, mock_function, mock_dataset, sample_config):
        """Test string representation of context."""
        context = TrialRunContext(
            func=mock_function,
            config=sample_config,
            dataset=mock_dataset,
            trial_number=1,
            session_id="session_1",
            optuna_trial_id=42,
        )

        repr_str = repr(context)
        assert "TrialRunContext" in repr_str
        assert "trial_number=1" in repr_str
