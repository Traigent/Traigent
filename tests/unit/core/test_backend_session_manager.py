"""Unit tests for BackendSessionManager.

Tests the extracted backend session lifecycle manager with stub backend client.
"""

import re
from unittest.mock import AsyncMock, MagicMock, Mock

import pytest

from traigent.api.types import OptimizationResult, OptimizationStatus, TrialResult
from traigent.config.types import TraigentConfig
from traigent.core.backend_session_manager import BackendSessionManager
from traigent.core.objectives import create_default_objectives
from traigent.evaluators.base import Dataset, EvaluationExample
from traigent.utils.function_identity import resolve_function_descriptor


@pytest.fixture
def mock_backend_client():
    """Create mock backend client."""
    client = Mock()
    client.create_session = Mock(return_value="test-session-id")
    client.get_session_mapping = Mock(
        return_value=MagicMock(experiment_run_id="run_test_123")
    )
    client.upload_example_features = Mock(return_value=True)
    client.submit_result = Mock()
    client.register_trial_start = AsyncMock()
    client._submit_trial_result_via_session = AsyncMock(return_value=True)
    client.update_trial_weighted_scores = AsyncMock(return_value=True)
    client.finalize_session = Mock(return_value={"status": "completed"})
    client.finalize_session_sync = Mock(return_value={"status": "completed"})

    # Mock auth manager
    auth_manager = Mock()
    auth_manager.has_api_key = Mock(return_value=True)
    client.auth_manager = auth_manager

    return client


@pytest.fixture
def mock_optimizer():
    """Create mock optimizer."""
    optimizer = Mock()
    optimizer.objectives = ["accuracy", "cost"]
    optimizer.config_space = {"param1": [1, 2, 3]}
    return optimizer


@pytest.fixture
def traigent_config():
    """Create test config."""
    config = TraigentConfig()
    config.execution_mode = "edge_analytics"
    return config


@pytest.fixture
def objective_schema():
    """Create test objective schema."""
    return create_default_objectives(
        objective_names=["accuracy", "cost"],
        orientations={"accuracy": "maximize", "cost": "minimize"},
        weights={"accuracy": 0.7, "cost": 0.3},
    )


@pytest.fixture
def backend_session_manager(
    mock_backend_client, traigent_config, objective_schema, mock_optimizer
):
    """Create BackendSessionManager instance."""
    return BackendSessionManager(
        backend_client=mock_backend_client,
        traigent_config=traigent_config,
        objectives=["accuracy", "cost"],
        objective_schema=objective_schema,
        optimizer=mock_optimizer,
        optimization_id="test-opt-id",
        optimization_status=OptimizationStatus.RUNNING,
    )


@pytest.fixture
def mock_dataset():
    """Create mock dataset."""
    dataset = Mock(spec=Dataset)
    dataset.name = "test_dataset"
    dataset.examples = [
        EvaluationExample(
            input_data={"query": "What is AI?"},
            expected_output="AI is artificial intelligence.",
        ),
        EvaluationExample(
            input_data={"query": "What is ML?"},
            expected_output="ML is machine learning.",
        ),
    ]
    dataset.__len__ = Mock(return_value=len(dataset.examples))
    return dataset


@pytest.fixture
def mock_trial_result():
    """Create mock trial result."""
    from traigent.api.types import TrialStatus

    trial = Mock(spec=TrialResult)
    trial.trial_id = "trial-123"
    trial.config = {"param1": 1}
    trial.metrics = {"accuracy": 0.9, "cost": 0.5, "total_cost": 0.5}
    trial.is_successful = True
    trial.status = TrialStatus.COMPLETED
    trial.duration = 1.5
    trial.error_message = None
    trial.metadata = {}  # Add metadata attribute
    trial.get_metric = Mock(
        side_effect=lambda key, default=None: trial.metrics.get(key, default)
    )
    trial.evaluation_results = []
    return trial


class TestBackendSessionManagerCreation:
    """Test session creation."""

    def test_create_session_with_backend(
        self, backend_session_manager, mock_dataset, mock_backend_client
    ):
        """Test session creation when backend client is configured."""

        def func(x):
            return x

        func.__name__ = "test_func"

        descriptor = resolve_function_descriptor(func)

        session_ctx = backend_session_manager.create_session(
            func=func,
            dataset=mock_dataset,
            function_descriptor=descriptor,
            max_trials=10,
            start_time=1234567890.0,
        )

        # Verify session_id returned
        assert session_ctx.session_id == "test-session-id"
        assert session_ctx.dataset_name == "test_dataset"
        assert session_ctx.function_name == descriptor.identifier

        # Verify backend client called
        mock_backend_client.create_session.assert_called_once()
        call_kwargs = mock_backend_client.create_session.call_args[1]
        assert call_kwargs["function_name"] == descriptor.slug
        assert call_kwargs["optimization_goal"] == "maximize"
        assert (
            call_kwargs["metadata"]["function_display_name"] == descriptor.display_name
        )
        assert call_kwargs["metadata"]["function_name"] == descriptor.identifier
        assert call_kwargs["metadata"]["function_slug"] == descriptor.slug
        mock_backend_client.upload_example_features.assert_called_once()
        upload_args = mock_backend_client.upload_example_features.call_args[0]
        assert upload_args[0] == "run_test_123"
        assert upload_args[1] == "simhash_v1"
        assert len(upload_args[2]) == len(mock_dataset)

    def test_create_session_without_backend(
        self, traigent_config, objective_schema, mock_optimizer, mock_dataset
    ):
        """Test session creation when backend client is None."""
        manager = BackendSessionManager(
            backend_client=None,
            traigent_config=traigent_config,
            objectives=["accuracy"],
            objective_schema=objective_schema,
            optimizer=mock_optimizer,
            optimization_id="test-opt-id",
            optimization_status=OptimizationStatus.RUNNING,
        )

        def func(x):
            return x

        func.__name__ = "test_func"

        descriptor = resolve_function_descriptor(func)

        session_ctx = manager.create_session(
            func=func,
            dataset=mock_dataset,
            function_descriptor=descriptor,
            max_trials=5,
            start_time=1234567890.0,
        )

        # Verify no session_id when backend disabled
        assert session_ctx.session_id is None
        assert session_ctx.dataset_name == "test_dataset"
        assert session_ctx.function_name == descriptor.identifier


class TestBackendSessionManagerTrialSubmission:
    """Test trial submission."""

    @pytest.mark.asyncio
    async def test_submit_trial_success(
        self, backend_session_manager, mock_trial_result, mock_backend_client
    ):
        """Test successful trial submission."""
        result = await backend_session_manager.submit_trial(
            trial_result=mock_trial_result,
            session_id="test-session-id",
        )

        assert result is True

        # Verify backend calls
        mock_backend_client.submit_result.assert_called_once()
        mock_backend_client.register_trial_start.assert_called_once()
        mock_backend_client._submit_trial_result_via_session.assert_called_once()

    @pytest.mark.asyncio
    async def test_submit_trial_without_backend(
        self, traigent_config, objective_schema, mock_optimizer, mock_trial_result
    ):
        """Test trial submission when backend is None."""
        manager = BackendSessionManager(
            backend_client=None,
            traigent_config=traigent_config,
            objectives=["accuracy"],
            objective_schema=objective_schema,
            optimizer=mock_optimizer,
            optimization_id="test-opt-id",
            optimization_status=OptimizationStatus.RUNNING,
        )

        result = await manager.submit_trial(
            trial_result=mock_trial_result,
            session_id="test-session-id",
        )

        assert result is False

    @pytest.mark.asyncio
    async def test_submit_trial_without_session_id(
        self, backend_session_manager, mock_trial_result
    ):
        """Test trial submission when session_id is None."""
        result = await backend_session_manager.submit_trial(
            trial_result=mock_trial_result,
            session_id=None,
        )

        assert result is False

    @pytest.mark.asyncio
    async def test_submit_pruned_trial_with_correct_status(
        self, backend_session_manager, mock_backend_client
    ):
        """Test that pruned trials are submitted with PRUNED status."""
        from traigent.api.types import TrialStatus

        # Create a pruned trial result
        pruned_trial = Mock(spec=TrialResult)
        pruned_trial.trial_id = "trial-pruned-456"
        pruned_trial.config = {"param1": 1}
        pruned_trial.metrics = {"accuracy": 0.7, "cost": 0.3}
        pruned_trial.is_successful = False
        pruned_trial.status = TrialStatus.PRUNED
        pruned_trial.duration = 0.5
        pruned_trial.error_message = "Early stopping triggered"
        pruned_trial.metadata = {"pruned": True}
        pruned_trial.get_metric = Mock(
            side_effect=lambda key, default=None: pruned_trial.metrics.get(key, default)
        )

        await backend_session_manager.submit_trial(
            trial_result=pruned_trial,
            session_id="test-session-id",
        )

        # Verify _submit_trial_result_via_session was called with PRUNED status
        mock_backend_client._submit_trial_result_via_session.assert_called_once()
        call_kwargs = mock_backend_client._submit_trial_result_via_session.call_args
        assert call_kwargs.kwargs["status"] == "PRUNED"

    @pytest.mark.asyncio
    async def test_submit_trial_metrics_keys_are_measuresdict_compatible(
        self, backend_session_manager, mock_backend_client
    ):
        """Submitted metric keys follow MeasuresDict identifier requirements."""
        from traigent.api.types import TrialStatus

        trial = Mock(spec=TrialResult)
        trial.trial_id = "trial-hybrid-metrics"
        trial.config = {"param1": 1}
        trial.metrics = {"cost": 0.05, "success_rate": 1.0, "latency": 120.0}
        trial.is_successful = True
        trial.status = TrialStatus.COMPLETED
        trial.duration = 1.0
        trial.error_message = None
        trial.metadata = {}
        trial.get_metric = Mock(
            side_effect=lambda key, default=None: trial.metrics.get(key, default)
        )

        await backend_session_manager.submit_trial(
            trial_result=trial,
            session_id="test-session-id",
        )

        call_kwargs = (
            mock_backend_client._submit_trial_result_via_session.call_args.kwargs
        )
        metrics_payload = call_kwargs["metrics"]
        pattern = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")
        assert all(pattern.match(k) for k in metrics_payload.keys())


class TestBackendSessionManagerWeightedScores:
    """Test weighted score updates."""

    @pytest.mark.asyncio
    async def test_update_weighted_scores_multi_objective(
        self, backend_session_manager, mock_trial_result, mock_backend_client
    ):
        """Test weighted score updates for multi-objective optimization."""
        # Create mock optimization result
        result = Mock(spec=OptimizationResult)
        result.trials = [mock_trial_result]
        result.successful_trials = [mock_trial_result]
        result.calculate_weighted_scores = Mock(
            return_value={
                "weighted_scores": [(mock_trial_result, 0.85)],
                "normalization_ranges": {},
                "best_weighted_config": {"param1": 1},
                "best_weighted_score": 0.85,
            }
        )
        result.metadata = {}

        update_count = await backend_session_manager.update_weighted_scores(
            result=result,
            session_id="test-session-id",
        )

        assert update_count == 1
        mock_backend_client.update_trial_weighted_scores.assert_awaited_once()

        # Verify metadata attached
        assert "weighted_results" in result.metadata

    @pytest.mark.asyncio
    async def test_update_weighted_scores_skips_failed_trials(
        self, backend_session_manager, mock_backend_client
    ):
        """Ensure weighted score submissions align with successful trials only."""
        successful_trial = Mock(spec=TrialResult)
        successful_trial.trial_id = "trial-success"
        successful_trial.config = {"param": "good"}
        successful_trial.metrics = {"accuracy": 0.95}
        successful_trial.is_successful = True
        failed_trial = Mock(spec=TrialResult)
        failed_trial.trial_id = "trial-failed"
        failed_trial.config = {"param": "bad"}
        failed_trial.metrics = {}
        failed_trial.is_successful = False

        result = Mock(spec=OptimizationResult)
        result.trials = [successful_trial, failed_trial]
        result.successful_trials = [successful_trial]
        result.calculate_weighted_scores = Mock(
            return_value={
                "weighted_scores": [(successful_trial, 0.92)],
                "normalization_ranges": {},
                "best_weighted_config": (
                    successful_trial.config
                    if hasattr(successful_trial, "config")
                    else {}
                ),
                "best_weighted_score": 0.92,
            }
        )
        result.metadata = {}

        update_count = await backend_session_manager.update_weighted_scores(
            result=result,
            session_id="test-session-id",
        )

        assert update_count == 1
        mock_backend_client.update_trial_weighted_scores.assert_awaited_once()
        awaited = mock_backend_client.update_trial_weighted_scores.await_args
        assert awaited.kwargs["trial_id"] == "trial-success"
        assert awaited.kwargs["weighted_score"] == 0.92
        assert awaited.kwargs["normalization_info"] == {}
        assert "objective_weights" in awaited.kwargs
        assert "weighted_results" in result.metadata
        assert result.metadata["weighted_results"]["trials_updated"] == 1

    @pytest.mark.asyncio
    async def test_update_weighted_scores_single_objective(
        self, mock_backend_client, traigent_config, mock_optimizer
    ):
        """Test weighted scores skipped for single-objective optimization."""
        manager = BackendSessionManager(
            backend_client=mock_backend_client,
            traigent_config=traigent_config,
            objectives=["accuracy"],  # Single objective
            objective_schema=None,
            optimizer=mock_optimizer,
            optimization_id="test-opt-id",
            optimization_status=OptimizationStatus.RUNNING,
        )

        result = Mock(spec=OptimizationResult)
        result.trials = []

        update_count = await manager.update_weighted_scores(
            result=result,
            session_id="test-session-id",
        )

        assert update_count == 0
        mock_backend_client.update_trial_weighted_scores.assert_not_called()


class TestBackendSessionManagerFinalization:
    """Test session finalization."""

    def test_finalize_session_success(
        self, backend_session_manager, mock_backend_client
    ):
        """Test successful session finalization."""
        summary = backend_session_manager.finalize_session(
            session_id="test-session-id",
            optimization_status=OptimizationStatus.COMPLETED,
        )

        assert summary == {"status": "completed"}
        mock_backend_client.finalize_session_sync.assert_called_once_with(
            "test-session-id", True
        )

    def test_finalize_session_without_backend(
        self, traigent_config, objective_schema, mock_optimizer
    ):
        """Test finalization when backend is None."""
        manager = BackendSessionManager(
            backend_client=None,
            traigent_config=traigent_config,
            objectives=["accuracy"],
            objective_schema=objective_schema,
            optimizer=mock_optimizer,
            optimization_id="test-opt-id",
            optimization_status=OptimizationStatus.RUNNING,
        )

        summary = manager.finalize_session(
            session_id="test-session-id",
            optimization_status=OptimizationStatus.COMPLETED,
        )

        assert summary is None


class TestBackendSessionManagerWarningSuppression:
    """Test warning suppression logic."""

    def test_suppress_warnings_in_offline_mode(
        self, mock_backend_client, traigent_config, objective_schema, mock_optimizer
    ):
        """Test warnings are suppressed when TRAIGENT_OFFLINE_MODE is set."""
        import os
        from unittest.mock import patch

        manager = BackendSessionManager(
            backend_client=mock_backend_client,
            traigent_config=traigent_config,
            objectives=["accuracy"],
            objective_schema=objective_schema,
            optimizer=mock_optimizer,
            optimization_id="test-opt-id",
            optimization_status=OptimizationStatus.RUNNING,
        )

        with patch.dict(os.environ, {"TRAIGENT_OFFLINE_MODE": "true"}):
            assert manager._should_suppress_backend_warnings() is True

    def test_suppress_warnings_without_api_key(
        self, traigent_config, objective_schema, mock_optimizer
    ):
        """Test warnings are suppressed when no API key is configured."""
        # Create client with auth manager that has no API key
        client = Mock()
        auth_manager = Mock()
        auth_manager.has_api_key = Mock(return_value=False)
        client.auth = auth_manager

        manager = BackendSessionManager(
            backend_client=client,
            traigent_config=traigent_config,
            objectives=["accuracy"],
            objective_schema=objective_schema,
            optimizer=mock_optimizer,
            optimization_id="test-opt-id",
            optimization_status=OptimizationStatus.RUNNING,
        )

        assert manager._should_suppress_backend_warnings() is True

    def test_no_suppress_warnings_with_api_key(
        self, mock_backend_client, traigent_config, objective_schema, mock_optimizer
    ):
        """Test warnings are NOT suppressed when API key is configured."""
        import os
        from unittest.mock import patch

        # Mock auth with has_api_key returning True
        mock_backend_client.auth = Mock()
        mock_backend_client.auth.has_api_key = Mock(return_value=True)

        manager = BackendSessionManager(
            backend_client=mock_backend_client,
            traigent_config=traigent_config,
            objectives=["accuracy"],
            objective_schema=objective_schema,
            optimizer=mock_optimizer,
            optimization_id="test-opt-id",
            optimization_status=OptimizationStatus.RUNNING,
        )

        # Ensure offline mode is off
        with patch.dict(os.environ, {"TRAIGENT_OFFLINE_MODE": "false"}, clear=False):
            # Clear cache to ensure fresh check
            assert manager._should_suppress_backend_warnings() is False

    def test_suppress_warnings_no_backend_client(
        self, traigent_config, objective_schema, mock_optimizer
    ):
        """Test warnings suppression when backend_client is None."""
        import os
        from unittest.mock import patch

        manager = BackendSessionManager(
            backend_client=None,
            traigent_config=traigent_config,
            objectives=["accuracy"],
            objective_schema=objective_schema,
            optimizer=mock_optimizer,
            optimization_id="test-opt-id",
            optimization_status=OptimizationStatus.RUNNING,
        )

        # With no backend client and offline mode off, should return False
        with patch.dict(os.environ, {"TRAIGENT_OFFLINE_MODE": "false"}, clear=False):
            assert manager._should_suppress_backend_warnings() is False

    def test_suppress_warnings_client_without_auth_attr(
        self, traigent_config, objective_schema, mock_optimizer
    ):
        """Test warnings suppression when client has no auth attribute."""
        import os
        from unittest.mock import patch

        # Client without auth attribute
        client = Mock(spec=[])  # Empty spec = no attributes

        manager = BackendSessionManager(
            backend_client=client,
            traigent_config=traigent_config,
            objectives=["accuracy"],
            objective_schema=objective_schema,
            optimizer=mock_optimizer,
            optimization_id="test-opt-id",
            optimization_status=OptimizationStatus.RUNNING,
        )

        # No auth attribute means no API key, should suppress
        with patch.dict(os.environ, {"TRAIGENT_OFFLINE_MODE": "false"}, clear=False):
            assert manager._should_suppress_backend_warnings() is True

    @pytest.mark.asyncio
    async def test_weighted_score_update_failure_with_suppressed_warning(
        self, traigent_config, mock_optimizer
    ):
        """Test weighted score update logs debug (not warning) when suppressed."""
        import os
        from unittest.mock import patch

        from traigent.core.objectives import create_default_objectives

        # Client with no API key (warnings suppressed)
        client = Mock()
        client.auth = Mock()
        client.auth.has_api_key = Mock(return_value=False)
        client.update_trial_weighted_scores = AsyncMock(return_value=False)
        client.weighted_update_concurrency = 8

        objective_schema = create_default_objectives(
            objective_names=["accuracy", "cost"],
            orientations={"accuracy": "maximize", "cost": "minimize"},
            weights={"accuracy": 0.7, "cost": 0.3},
        )

        manager = BackendSessionManager(
            backend_client=client,
            traigent_config=traigent_config,
            objectives=["accuracy", "cost"],
            objective_schema=objective_schema,
            optimizer=mock_optimizer,
            optimization_id="test-opt-id",
            optimization_status=OptimizationStatus.RUNNING,
        )

        # Create mock trial and result
        mock_trial = Mock(spec=TrialResult)
        mock_trial.trial_id = "trial-fail-test"
        mock_trial.config = {"param": 1}
        mock_trial.metrics = {"accuracy": 0.9, "cost": 0.5}
        mock_trial.is_successful = True

        result = Mock(spec=OptimizationResult)
        result.trials = [mock_trial]
        result.successful_trials = [mock_trial]
        result.calculate_weighted_scores = Mock(
            return_value={
                "weighted_scores": [(mock_trial, 0.85)],
                "normalization_ranges": {},
                "best_weighted_config": {"param": 1},
                "best_weighted_score": 0.85,
            }
        )
        result.metadata = {}

        with patch.dict(os.environ, {"TRAIGENT_MOCK_LLM": "false"}):
            update_count = await manager.update_weighted_scores(
                result=result, session_id="test-session"
            )

        # Update failed but warning was suppressed (logged as debug)
        assert update_count == 0
        client.update_trial_weighted_scores.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_weighted_score_update_exception_with_suppressed_warning(
        self, traigent_config, mock_optimizer
    ):
        """Test weighted score exception logs debug when warnings suppressed."""
        import os
        from unittest.mock import patch

        from traigent.core.objectives import create_default_objectives

        # Client with no API key (warnings suppressed) that raises exception
        client = Mock()
        client.auth = Mock()
        client.auth.has_api_key = Mock(return_value=False)
        client.update_trial_weighted_scores = AsyncMock(
            side_effect=Exception("Connection failed")
        )
        client.weighted_update_concurrency = 8

        objective_schema = create_default_objectives(
            objective_names=["accuracy", "cost"],
            orientations={"accuracy": "maximize", "cost": "minimize"},
            weights={"accuracy": 0.7, "cost": 0.3},
        )

        manager = BackendSessionManager(
            backend_client=client,
            traigent_config=traigent_config,
            objectives=["accuracy", "cost"],
            objective_schema=objective_schema,
            optimizer=mock_optimizer,
            optimization_id="test-opt-id",
            optimization_status=OptimizationStatus.RUNNING,
        )

        mock_trial = Mock(spec=TrialResult)
        mock_trial.trial_id = "trial-exc-test"
        mock_trial.config = {"param": 1}
        mock_trial.metrics = {"accuracy": 0.9, "cost": 0.5}
        mock_trial.is_successful = True

        result = Mock(spec=OptimizationResult)
        result.trials = [mock_trial]
        result.successful_trials = [mock_trial]
        result.calculate_weighted_scores = Mock(
            return_value={
                "weighted_scores": [(mock_trial, 0.85)],
                "normalization_ranges": {},
                "best_weighted_config": {"param": 1},
                "best_weighted_score": 0.85,
            }
        )
        result.metadata = {}

        with patch.dict(os.environ, {"TRAIGENT_MOCK_LLM": "false"}):
            update_count = await manager.update_weighted_scores(
                result=result, session_id="test-session"
            )

        # Exception occurred but warning was suppressed
        assert update_count == 0


class TestBackendSessionManagerMetadata:
    """Test metadata attachment."""

    def test_attach_session_metadata(self, backend_session_manager):
        """Test attaching session metadata to result."""
        result = Mock(spec=OptimizationResult)
        result.metadata = {}

        backend_session_manager.attach_session_metadata(
            result=result,
            session_id="test-session-id",
            session_summary={"status": "completed", "trials": 10},
        )

        assert result.metadata["local_session_id"] == "test-session-id"
        assert result.metadata["local_session_summary"] == {
            "status": "completed",
            "trials": 10,
        }

    def test_attach_metadata_without_session_id(self, backend_session_manager):
        """Test metadata attachment when session_id is None."""
        result = Mock(spec=OptimizationResult)
        result.metadata = {}

        backend_session_manager.attach_session_metadata(
            result=result,
            session_id=None,
            session_summary=None,
        )

        # Metadata should remain empty
        assert result.metadata == {}


class TestStatisticalSignificanceWiring:
    """Verify significance data flows through submit_session_aggregation."""

    def test_significance_included_in_aggregation_payload(
        self, mock_backend_client, mock_optimizer, objective_schema
    ):
        """Significance metadata propagates to backend submit_result call."""
        # Non-edge config so submit_session_aggregation actually submits
        config = TraigentConfig()
        config.execution_mode = "standard"

        manager = BackendSessionManager(
            backend_client=mock_backend_client,
            traigent_config=config,
            objectives=["accuracy"],
            objective_schema=objective_schema,
            optimizer=mock_optimizer,
            optimization_id="test-sig-wiring",
            optimization_status=OptimizationStatus.COMPLETED,
        )

        sig_data = {
            "accuracy": {
                "winners": [0],
                "top_group": [0],
                "rest_group": [1],
                "badge_name": "accuracy",
            }
        }

        result = Mock(spec=OptimizationResult)
        result.trials = []
        result.best_config = {"model": "gpt-4o"}
        result.best_score = 0.95
        result.duration = 10.0
        result.success_rate = 1.0
        result.metrics = {"accuracy": 0.95}
        result.metadata = {
            "session_summary": {
                "metrics": {"accuracy": 0.95},
                "samples_per_config": {"gpt-4o": 5},
            },
            "statistical_significance": sig_data,
        }

        manager.submit_session_aggregation(result, "test-session-id")

        mock_backend_client.submit_result.assert_called_once()
        call_kwargs = mock_backend_client.submit_result.call_args
        payload_metadata = call_kwargs.kwargs.get(
            "metadata", call_kwargs[1].get("metadata", {})
        )
        summary_stats = payload_metadata.get("summary_stats", {})
        inner_metadata = summary_stats.get("metadata", {})
        assert "statistical_significance" in inner_metadata
        assert inner_metadata["statistical_significance"] == sig_data

    def test_aggregation_works_without_significance(
        self, mock_backend_client, mock_optimizer, objective_schema
    ):
        """Aggregation still works when no significance data is present."""
        config = TraigentConfig()
        config.execution_mode = "standard"

        manager = BackendSessionManager(
            backend_client=mock_backend_client,
            traigent_config=config,
            objectives=["accuracy"],
            objective_schema=objective_schema,
            optimizer=mock_optimizer,
            optimization_id="test-no-sig",
            optimization_status=OptimizationStatus.COMPLETED,
        )

        result = Mock(spec=OptimizationResult)
        result.trials = []
        result.best_config = {"model": "gpt-4o"}
        result.best_score = 0.95
        result.duration = 10.0
        result.success_rate = 1.0
        result.metrics = {"accuracy": 0.95}
        result.metadata = {
            "session_summary": {
                "metrics": {"accuracy": 0.95},
                "samples_per_config": {"gpt-4o": 5},
            },
        }

        manager.submit_session_aggregation(result, "test-session-id")

        mock_backend_client.submit_result.assert_called_once()
        call_kwargs = mock_backend_client.submit_result.call_args
        payload_metadata = call_kwargs.kwargs.get(
            "metadata", call_kwargs[1].get("metadata", {})
        )
        summary_stats = payload_metadata.get("summary_stats", {})
        inner_metadata = summary_stats.get("metadata", {})
        assert "statistical_significance" not in inner_metadata


class TestTrialSkipWarningIncludesSignupUrl:
    """Verify the no-API-key trial-skip warning includes the signup URL."""

    @pytest.mark.asyncio
    async def test_log_trial_warns_with_signup_url_when_no_api_key(
        self, traigent_config, objective_schema, mock_optimizer, caplog
    ):
        """_log_trial_to_backend must emit WARNING with signup URL when no key."""
        import logging
        from unittest.mock import patch

        import traigent.core.backend_session_manager as bsm
        from traigent.config.backend_config import SIGNUP_URL

        # Reset the once-per-process guard
        bsm._warned_no_api_key = False

        client = Mock()
        auth_manager = Mock()
        auth_manager.has_api_key = Mock(return_value=False)
        client.auth_manager = auth_manager
        client.log_trial = AsyncMock()

        manager = BackendSessionManager(
            backend_client=client,
            traigent_config=traigent_config,
            objectives=["accuracy"],
            objective_schema=objective_schema,
            optimizer=mock_optimizer,
            optimization_id="test-opt-id",
            optimization_status=OptimizationStatus.RUNNING,
        )

        trial_result = Mock(spec=TrialResult)
        trial_result.trial_id = "t1"

        with (
            patch(
                "traigent.core.backend_session_manager.is_backend_offline",
                return_value=False,
            ),
            caplog.at_level(
                logging.WARNING, logger="traigent.core.backend_session_manager"
            ),
        ):
            await manager._log_trial_to_backend(
                session_id="s1", trial_result=trial_result, score=0.9, metadata={}
            )

        assert any(
            SIGNUP_URL in msg for msg in caplog.messages
        ), f"Expected {SIGNUP_URL!r} in warning: {caplog.messages}"
        # Verify it's WARNING level, not DEBUG
        warning_records = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warning_records) >= 1
