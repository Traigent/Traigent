"""Unit tests for LoggerFacade.

This test suite covers the logger facade that provides unified interface for
logging session starts, trial results, and checkpoints with consistent exception handling.

Tests cover:
- Initialization with various configurations
- Session start logging with comprehensive metadata
- Trial result logging with exception handling
- Checkpoint logging with optimizer state
- Exception handling for all operations
"""

from datetime import datetime
from unittest.mock import Mock, patch

import pytest

from traigent.api.types import TrialResult, TrialStatus
from traigent.core.logger_facade import LoggerFacade
from traigent.core.objectives import create_default_objectives


@pytest.fixture
def mock_optimization_logger():
    """Mock OptimizationLogger for testing."""
    with patch("traigent.core.logger_facade.OptimizationLogger") as mock:
        yield mock


@pytest.fixture
def logger_facade(mock_optimization_logger):
    """Create LoggerFacade instance with mocked logger."""
    facade = LoggerFacade(
        experiment_name="test_experiment",
        session_id="test_session_123",
        execution_mode="edge_analytics",
    )
    return facade


@pytest.fixture
def mock_dataset():
    """Mock dataset for testing."""
    dataset = Mock()
    dataset.__len__ = Mock(return_value=100)
    dataset.name = "test_dataset"
    return dataset


@pytest.fixture
def sample_trial_result():
    """Create sample trial result for testing."""
    return TrialResult(
        trial_id="trial_001",
        config={"temperature": 0.5, "model": "gpt-3.5"},
        metrics={"accuracy": 0.85, "cost": 0.05},
        status=TrialStatus.COMPLETED,
        duration=1.5,
        timestamp=datetime.now(),
        metadata={},
    )


class TestLoggerFacadeInitialization:
    """Tests for LoggerFacade initialization."""

    def test_initialization_creates_optimization_logger(self, mock_optimization_logger):
        """Test that initialization creates OptimizationLogger instance."""
        LoggerFacade(
            experiment_name="test_exp",
            session_id="session_123",
            execution_mode="backend_only",
        )

        mock_optimization_logger.assert_called_once_with(
            experiment_name="test_exp",
            session_id="session_123",
            execution_mode="backend_only",
        )

    def test_initialization_with_none_session_id(self, mock_optimization_logger):
        """Test initialization with None session_id uses default local-session."""
        LoggerFacade(
            experiment_name="local_exp",
            session_id=None,
            execution_mode="edge_analytics",
        )

        # When session_id is None, LoggerFacade defaults to "local-session"
        mock_optimization_logger.assert_called_once_with(
            experiment_name="local_exp",
            session_id="local-session",
            execution_mode="edge_analytics",
        )


class TestSessionStartLogging:
    """Tests for log_session_start method."""

    def test_log_session_start_with_objective_schema(
        self, logger_facade, mock_dataset, mock_optimization_logger
    ):
        """Test session start logging with ObjectiveSchema."""
        objective_schema = create_default_objectives(
            ["accuracy", "cost"],
            orientations={"accuracy": "maximize", "cost": "minimize"},
            weights={"accuracy": 0.7, "cost": 0.3},
        )

        config = {"execution_mode": "edge_analytics"}

        logger_facade.log_session_start(
            config=config,
            objectives=objective_schema,
            algorithm="OptunaOptimizer",
            dataset=mock_dataset,
        )

        # Verify log_session_start was called with correct arguments
        mock_logger_instance = mock_optimization_logger.return_value
        mock_logger_instance.log_session_start.assert_called_once()

        call_args = mock_logger_instance.log_session_start.call_args
        assert call_args[1]["config"] == config
        assert call_args[1]["objectives"] == objective_schema
        assert call_args[1]["algorithm"] == "OptunaOptimizer"
        assert call_args[1]["dataset_info"]["size"] == 100
        assert call_args[1]["dataset_info"]["name"] == "test_dataset"

    def test_log_session_start_with_objectives_list(
        self, logger_facade, mock_dataset, mock_optimization_logger
    ):
        """Test session start logging with objectives list."""
        objectives_list = ["accuracy", "latency"]
        config = {"max_trials": 50}

        logger_facade.log_session_start(
            config=config,
            objectives=objectives_list,
            algorithm="RandomOptimizer",
            dataset=mock_dataset,
        )

        mock_logger_instance = mock_optimization_logger.return_value
        call_args = mock_logger_instance.log_session_start.call_args
        assert call_args[1]["objectives"] == objectives_list

    def test_log_session_start_with_dataset_without_name(
        self, logger_facade, mock_optimization_logger
    ):
        """Test session start logging with dataset lacking name attribute."""
        dataset = Mock()
        dataset.__len__ = Mock(return_value=50)
        del dataset.name  # Remove name attribute

        logger_facade.log_session_start(
            config={},
            objectives=["accuracy"],
            algorithm="TestOptimizer",
            dataset=dataset,
        )

        mock_logger_instance = mock_optimization_logger.return_value
        call_args = mock_logger_instance.log_session_start.call_args
        assert call_args[1]["dataset_info"]["name"] == "unknown"
        assert call_args[1]["dataset_info"]["size"] == 50

    def test_log_session_start_exception_handling(
        self, logger_facade, mock_dataset, mock_optimization_logger
    ):
        """Test that exceptions during session start logging are caught."""
        mock_logger_instance = mock_optimization_logger.return_value
        mock_logger_instance.log_session_start.side_effect = Exception("Logging failed")

        # Should not raise exception - verify method completes without propagating error
        result = logger_facade.log_session_start(
            config={},
            objectives=["accuracy"],
            algorithm="TestOptimizer",
            dataset=mock_dataset,
        )
        assert result is None  # Method should complete silently


class TestTrialLogging:
    """Tests for log_trial method."""

    def test_log_trial_success(
        self, logger_facade, sample_trial_result, mock_optimization_logger
    ):
        """Test successful trial logging."""
        logger_facade.log_trial(sample_trial_result)

        mock_logger_instance = mock_optimization_logger.return_value
        mock_logger_instance.log_trial_result.assert_called_once_with(
            sample_trial_result
        )

    def test_log_trial_with_none_logger(self, sample_trial_result):
        """Test trial logging when logger is None."""
        # Create facade and manually set logger to None
        with patch("traigent.core.logger_facade.OptimizationLogger"):
            facade = LoggerFacade(
                experiment_name="test",
                session_id="session",
                execution_mode="edge_analytics",
            )
            facade._logger = None

            # Should not raise exception - verify method completes
            result = facade.log_trial(sample_trial_result)
            assert result is None

    def test_log_trial_exception_handling(
        self, logger_facade, sample_trial_result, mock_optimization_logger
    ):
        """Test that exceptions during trial logging are caught."""
        mock_logger_instance = mock_optimization_logger.return_value
        mock_logger_instance.log_trial_result.side_effect = Exception("Logging failed")

        # Should not raise exception - verify method completes silently
        result = logger_facade.log_trial(sample_trial_result)
        assert result is None

    def test_log_trial_with_failed_trial(self, logger_facade, mock_optimization_logger):
        """Test logging of failed trial result."""
        failed_trial = TrialResult(
            trial_id="trial_failed",
            config={"temperature": 1.0},
            metrics={},
            status=TrialStatus.FAILED,
            duration=0.5,
            timestamp=datetime.now(),
            error_message="Evaluation failed",
            metadata={},
        )

        logger_facade.log_trial(failed_trial)

        mock_logger_instance = mock_optimization_logger.return_value
        mock_logger_instance.log_trial_result.assert_called_once_with(failed_trial)


class TestCheckpointLogging:
    """Tests for log_checkpoint method."""

    def test_log_checkpoint_success(self, logger_facade, mock_optimization_logger):
        """Test successful checkpoint logging."""
        optimizer_state = {"iteration": 10, "best_config": {"temperature": 0.5}}
        trials_history = [
            TrialResult(
                trial_id=f"trial_{i}",
                config={"temperature": 0.5 + i * 0.1},
                metrics={"accuracy": 0.8 + i * 0.01},
                status=TrialStatus.COMPLETED,
                duration=1.0 + i * 0.1,
                timestamp=datetime.now(),
                metadata={},
            )
            for i in range(5)
        ]

        logger_facade.log_checkpoint(
            optimizer_state=optimizer_state,
            trials_history=trials_history,
            trial_count=5,
        )

        mock_logger_instance = mock_optimization_logger.return_value
        mock_logger_instance.save_checkpoint.assert_called_once_with(
            optimizer_state=optimizer_state,
            trials_history=trials_history,
            trial_count=5,
        )

    def test_log_checkpoint_with_none_logger(self):
        """Test checkpoint logging when logger is None."""
        with patch("traigent.core.logger_facade.OptimizationLogger"):
            facade = LoggerFacade(
                experiment_name="test",
                session_id="session",
                execution_mode="edge_analytics",
            )
            facade._logger = None

            # Should not raise exception - verify method completes
            result = facade.log_checkpoint(
                optimizer_state={},
                trials_history=[],
                trial_count=0,
            )
            assert result is None

    def test_log_checkpoint_without_save_checkpoint_method(
        self, logger_facade, mock_optimization_logger
    ):
        """Test checkpoint logging when logger lacks save_checkpoint method."""
        mock_logger_instance = mock_optimization_logger.return_value
        del mock_logger_instance.save_checkpoint  # Remove method

        # Should not raise exception - verify method completes
        result = logger_facade.log_checkpoint(
            optimizer_state={},
            trials_history=[],
            trial_count=0,
        )
        assert result is None

    def test_log_checkpoint_exception_handling(
        self, logger_facade, mock_optimization_logger
    ):
        """Test that exceptions during checkpoint logging are caught."""
        mock_logger_instance = mock_optimization_logger.return_value
        mock_logger_instance.save_checkpoint.side_effect = Exception(
            "Checkpoint failed"
        )

        # Should not raise exception - verify method completes silently
        result = logger_facade.log_checkpoint(
            optimizer_state={},
            trials_history=[],
            trial_count=10,
        )
        assert result is None

    def test_log_checkpoint_with_empty_state(
        self, logger_facade, mock_optimization_logger
    ):
        """Test checkpoint logging with empty optimizer state."""
        logger_facade.log_checkpoint(
            optimizer_state={},
            trials_history=[],
            trial_count=0,
        )

        mock_logger_instance = mock_optimization_logger.return_value
        mock_logger_instance.save_checkpoint.assert_called_once_with(
            optimizer_state={},
            trials_history=[],
            trial_count=0,
        )


class TestExceptionHandling:
    """Tests for consistent exception handling across all methods."""

    def test_all_methods_handle_exceptions_gracefully(self, mock_dataset):
        """Test that all methods handle exceptions without propagating them."""
        with patch("traigent.core.logger_facade.OptimizationLogger") as mock:
            mock_instance = mock.return_value
            mock_instance.log_session_start.side_effect = Exception("Error 1")
            mock_instance.log_trial_result.side_effect = Exception("Error 2")
            mock_instance.save_checkpoint.side_effect = Exception("Error 3")

            facade = LoggerFacade(
                experiment_name="test",
                session_id="session",
                execution_mode="edge_analytics",
            )

            # None of these should raise exceptions - verify all complete silently
            result1 = facade.log_session_start(
                config={},
                objectives=["accuracy"],
                algorithm="TestOptimizer",
                dataset=mock_dataset,
            )
            assert result1 is None

            result2 = facade.log_trial(
                TrialResult(
                    trial_id="test",
                    config={},
                    metrics={},
                    status=TrialStatus.COMPLETED,
                    duration=1.0,
                    timestamp=datetime.now(),
                    metadata={},
                )
            )
            assert result2 is None

            result3 = facade.log_checkpoint(
                optimizer_state={},
                trials_history=[],
                trial_count=0,
            )
            assert result3 is None
