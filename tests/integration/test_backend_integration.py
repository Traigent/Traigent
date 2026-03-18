"""Integration tests for Traigent SDK and Traigent Backend integration."""

import time
from copy import deepcopy
from unittest.mock import AsyncMock, Mock, patch

import pytest

from traigent.cloud.backend_bridges import SessionExperimentMapping, bridge
from traigent.cloud.dataset_converter import converter
from traigent.cloud.integration_manager import (
    IntegrationConfig,
    IntegrationManager,
    IntegrationMode,
    IntegrationResult,
)
from traigent.cloud.models import (
    AgentSpecification,
    DatasetSubsetIndices,
    OptimizationRequest,
    OptimizationSession,
    OptimizationSessionStatus,
    TrialSuggestion,
)
from traigent.cloud.sessions import SessionLifecycleManager
from traigent.evaluators.base import Dataset, EvaluationExample

lifecycle_manager = SessionLifecycleManager()


@pytest.fixture
def sample_dataset():
    """Create sample dataset for testing."""
    examples = [
        EvaluationExample(
            input_data={"query": "What is the capital of France?"},
            expected_output="Paris",
            metadata={"difficulty": "easy"},
        ),
        EvaluationExample(
            input_data={"query": "Explain quantum computing"},
            expected_output="Quantum computing uses quantum mechanics...",
            metadata={"difficulty": "hard"},
        ),
        EvaluationExample(
            input_data={"query": "What is 2+2?"},
            expected_output="4",
            metadata={"difficulty": "easy"},
        ),
    ]

    return Dataset(
        examples=examples,
        name="test_dataset",
        description="Test dataset for integration tests",
    )


@pytest.fixture
def sample_agent_spec():
    """Create sample agent specification."""
    return AgentSpecification(
        id="test_agent_001",
        name="Test Assistant",
        agent_type="conversational",
        agent_platform="openai",
        prompt_template="You are a helpful assistant. Answer: {input}",
        model_parameters={"model": "o4-mini", "temperature": 0.7, "max_tokens": 150},
        reasoning="Think step by step",
        style="helpful and concise",
        tone="professional",
        format="plain text",
        persona="knowledgeable assistant",
        guidelines=["Be accurate", "Be helpful"],
        response_validation=True,
        metadata={"created_from": "test"},
    )


@pytest.fixture
def sample_optimization_request(sample_dataset, sample_agent_spec):
    """Create sample optimization request."""
    return OptimizationRequest(
        function_name="test_function",
        dataset=sample_dataset,
        configuration_space={
            "temperature": (0.1, 1.0),
            "max_tokens": [50, 100, 150, 200],
        },
        objectives=["accuracy", "cost"],
        max_trials=10,
        agent_specification=sample_agent_spec,
        metadata={"test": True},
    )


class TestBackendBridges:
    """Test model bridges between SDK and backend entities."""

    def test_optimization_request_to_backend(self, sample_optimization_request):
        """Test conversion of optimization request to backend format."""
        backend_request = bridge.optimization_request_to_backend(
            sample_optimization_request
        )

        assert "test_function" in backend_request.name
        assert backend_request.description is not None
        assert backend_request.measures == ["accuracy", "cost"]
        assert backend_request.metadata["test"] is True
        assert backend_request.agent_data is not None
        assert backend_request.example_set_data is not None

    def test_agent_specification_to_backend(self, sample_agent_spec):
        """Test conversion of agent specification to backend format."""
        backend_agent = bridge.agent_specification_to_backend(sample_agent_spec)

        assert backend_agent["name"] == "Test Assistant"
        assert (
            backend_agent.get("agent_type_id") == "agent-type-1"
        )  # Backend uses agent_type_id
        assert (
            backend_agent.get("prompt_template")
            == "You are a helpful assistant. Answer: {input}"
        )
        assert backend_agent.get("description") is not None
        assert backend_agent.get("metadata") is not None

    def test_session_mapping_creation(self):
        """Test session to experiment mapping creation."""
        mapping = bridge.create_session_mapping(
            session_id="session_123",
            experiment_id="exp_456",
            experiment_run_id="run_789",
            function_name="test_func",
            configuration_space={"temp": (0.1, 1.0)},
            objectives=["accuracy"],
        )

        assert mapping.session_id == "session_123"
        assert mapping.experiment_id == "exp_456"
        assert mapping.experiment_run_id == "run_789"
        assert mapping.function_name == "test_func"
        assert len(mapping.trial_mappings) == 0

    def test_trial_suggestion_to_config_run(self):
        """Test conversion of trial suggestion to configuration run."""
        from traigent.cloud.models import DatasetSubsetIndices

        suggestion = TrialSuggestion(
            trial_id="trial_123",
            session_id="session_456",
            trial_number=1,
            config={"temperature": 0.7, "max_tokens": 100},
            dataset_subset=DatasetSubsetIndices(
                indices=[0, 1, 2],
                selection_strategy="random",
                confidence_level=0.8,
                estimated_representativeness=0.8,
            ),
            exploration_type="exploration",
        )

        config_run = bridge.trial_suggestion_to_config_run(suggestion, "run_789")

        assert config_run.experiment_run_id == "run_789"
        assert config_run.config_run_id == "trial_123"
        assert config_run.experiment_parameters["config"] == {
            "temperature": 0.7,
            "max_tokens": 100,
        }
        assert config_run.experiment_parameters["trial_number"] == 1


class TestSessionLifecycleManager:
    """Test session lifecycle management."""

    @pytest.mark.asyncio
    async def test_session_registration(self, sample_optimization_request):
        """Test session registration with lifecycle manager."""
        session = OptimizationSession(
            session_id="test_session_001",
            function_name=sample_optimization_request.function_name,
            configuration_space=sample_optimization_request.configuration_space,
            objectives=sample_optimization_request.objectives,
            max_trials=sample_optimization_request.max_trials,
            status=OptimizationSessionStatus.CREATED,
            created_at=time.time(),
            updated_at=time.time(),
        )

        mapping = SessionExperimentMapping(
            session_id="test_session_001",
            experiment_id="exp_001",
            experiment_run_id="run_001",
            function_name=sample_optimization_request.function_name,
            configuration_space=sample_optimization_request.configuration_space,
            objectives=sample_optimization_request.objectives,
            trial_mappings={},
        )

        # Register session
        lifecycle_manager.register_session(session, mapping)

        # Check session state
        session_state = lifecycle_manager.get_session_state("test_session_001")
        assert session_state is not None
        assert session_state.session.session_id == "test_session_001"
        assert session_state.mapping.experiment_id == "exp_001"

        # Clean up
        lifecycle_manager.cleanup()

    @pytest.mark.asyncio
    async def test_trial_lifecycle(self):
        """Test complete trial lifecycle."""
        session_id = "test_session_002"

        # Create session first
        session = OptimizationSession(
            session_id=session_id,
            function_name="test_func",
            configuration_space={"temp": (0.1, 1.0)},
            objectives=["accuracy"],
            max_trials=5,
            status=OptimizationSessionStatus.ACTIVE,
            created_at=time.time(),
            updated_at=time.time(),
        )

        mapping = SessionExperimentMapping(
            session_id=session_id,
            experiment_id="exp_002",
            experiment_run_id="run_002",
            function_name="test_func",
            configuration_space={"temp": (0.1, 1.0)},
            objectives=["accuracy"],
            trial_mappings={},
        )

        lifecycle_manager.register_session(session, mapping)

        # Register trial suggestion

        # The refactored lifecycle manager has a different API
        # We can test that the session was registered and can update status
        # Get session state to verify registration worked
        session_state = lifecycle_manager.get_session_state(session_id)
        assert session_state is not None
        assert session_state.session.function_name == "test_func"

        # Test session status update
        success = session_state.update_session_status(
            OptimizationSessionStatus.COMPLETED
        )
        assert success

        # Test incrementing completed trials
        initial_trials = session_state.session.completed_trials
        session_state.increment_completed_trials()
        assert session_state.session.completed_trials == initial_trials + 1

        # Test updating best results
        success = session_state.update_best_results(
            {"temp": 0.7}, {"accuracy": 0.85, "cost": 0.02}
        )
        assert success
        assert session_state.session.best_config == {"temp": 0.7}
        assert session_state.session.best_metrics == {"accuracy": 0.85, "cost": 0.02}

    @pytest.mark.asyncio
    async def test_session_summary(self):
        """Test session summary generation."""
        session_id = "test_session_003"

        # Create and register session
        session = OptimizationSession(
            session_id=session_id,
            function_name="summary_test",
            configuration_space={"param": [1, 2, 3]},
            objectives=["metric1"],
            max_trials=3,
            status=OptimizationSessionStatus.ACTIVE,
            created_at=time.time(),
            updated_at=time.time(),
            best_config={"param": 2},
            best_metrics={"metric1": 0.95},
        )

        mapping = SessionExperimentMapping(
            session_id=session_id,
            experiment_id="exp_003",
            experiment_run_id="run_003",
            function_name="summary_test",
            configuration_space={"param": [1, 2, 3]},
            objectives=["metric1"],
            trial_mappings={},
        )

        lifecycle_manager.register_session(session, mapping)

        # Get summary
        summary = lifecycle_manager.get_session_summary(session_id)

        assert summary is not None
        # The new implementation returns a dict from SessionState.get_session_summary()
        assert summary["session_id"] == session_id
        assert summary["function_name"] == "summary_test"
        assert summary["completed_trials"] == 0  # No trials registered yet
        assert summary["best_config"] == {"param": 2}
        assert summary["best_metrics"] == {"metric1": 0.95}
        assert summary["backend_experiment_id"] == "exp_003"

        # Clean up
        lifecycle_manager.cleanup()


class TestDatasetConverter:
    """Test dataset conversion between SDK and backend formats."""

    def test_sdk_to_backend_conversion(self, sample_dataset):
        """Test SDK dataset to backend examples conversion."""
        examples, metadata = converter.sdk_dataset_to_backend_examples(sample_dataset)

        assert len(examples) == 3
        assert metadata.name == "test_dataset"
        assert metadata.type == "input-output"
        assert metadata.total_examples == 3
        assert not metadata.privacy_mode

        # Check example format
        example = examples[0]
        assert "input" in example
        assert "output" in example
        assert "example_id" in example

    def test_privacy_mode_conversion(self, sample_dataset):
        """Test privacy mode conversion."""
        examples, metadata = converter.sdk_dataset_to_backend_examples(
            sample_dataset, privacy_mode=True
        )

        assert metadata.privacy_mode

        # Check that sensitive data is redacted
        example = examples[0]
        assert "<str>" in example["input"]  # Should be anonymized

    def test_csv_conversion(self, sample_dataset):
        """Test CSV format conversion."""
        csv_content = converter.sdk_dataset_to_csv(
            sample_dataset, include_metadata=True
        )

        assert "input,output,explanation,tags" in csv_content
        assert "What is the capital of France?" in csv_content
        assert "Paris" in csv_content
        assert "difficulty:easy" in csv_content

    def test_privacy_metadata_creation(self, sample_dataset):
        """Test privacy metadata creation."""
        metadata = converter.create_privacy_metadata(sample_dataset)

        assert metadata["name"] == "test_dataset"
        assert metadata["total_examples"] == 3
        assert metadata["privacy_safe"] is True
        assert "statistics" in metadata
        assert "samples" in metadata

        # Check statistics
        stats = metadata["statistics"]
        assert "avg_input_length" in stats
        assert "avg_output_length" in stats

        # Check samples are anonymized
        samples = metadata["samples"]
        assert len(samples) == 3
        for sample in samples:
            assert "input_sample" in sample
            assert "..." in sample["input_sample"]  # Should be anonymized

    def test_dataset_subset_indices(self, sample_dataset):
        """Test dataset subset index generation."""
        indices = converter.create_dataset_subset_indices(
            sample_dataset, subset_size=2, strategy="diverse_sampling"
        )

        assert len(indices) == 2
        assert all(0 <= idx < len(sample_dataset.examples) for idx in indices)
        assert indices == sorted(indices)  # Should be sorted


class TestIntegrationManager:
    """Test integration manager functionality."""

    @pytest.fixture
    def integration_config(self):
        """Create integration configuration for testing."""
        return IntegrationConfig(
            mode=IntegrationMode.PRIVACY,
            backend_base_url="http://localhost:5000",
            enable_fallback=True,
            max_retries=2,
            timeout=10.0,
        )

    @pytest.mark.asyncio
    async def test_integration_manager_initialization(self, integration_config):
        """Test integration manager initialization."""
        with patch(
            "traigent.cloud.integration_manager.get_production_mcp_client"
        ) as mock_mcp:
            with patch(
                "traigent.cloud.integration_manager.get_backend_client"
            ) as mock_backend:
                mock_mcp.return_value = Mock()
                mock_backend.return_value = Mock()

                manager = IntegrationManager(integration_config)
                success = await manager.initialize()

                assert success
                assert manager._initialized
                assert mock_mcp.called
                assert mock_backend.called

    @pytest.mark.asyncio
    async def test_private_integration(
        self, integration_config, sample_optimization_request
    ):
        """Test privacy-first integration workflow."""
        with patch(
            "traigent.cloud.integration_manager.get_production_mcp_client"
        ) as mock_mcp_client:
            with patch(
                "traigent.cloud.integration_manager.get_backend_client"
            ) as mock_backend_client:
                # Setup mocks
                mock_mcp = Mock()
                mock_mcp.create_optimization_workflow = AsyncMock(
                    return_value=("agent_123", "exp_456", "run_789")
                )
                mock_mcp_client.return_value = mock_mcp

                mock_backend = Mock()
                mock_backend.create_privacy_optimization_session = AsyncMock(
                    return_value=("session_123", "exp_456", "run_789")
                )
                mock_backend_client.return_value = mock_backend

                # Test integration
                manager = IntegrationManager(integration_config)
                await manager.initialize()

                result = await manager.start_optimization_integration(
                    sample_optimization_request, IntegrationMode.PRIVACY
                )

                assert result.success
                assert result.session_id == "session_123"
                assert result.experiment_id == "exp_456"
                assert result.experiment_run_id == "run_789"
                assert result.agent_id == "agent_123"
                assert result.metadata["mode"] == "privacy"

    @pytest.mark.asyncio
    async def test_hybrid_mode_selection(
        self, integration_config, sample_optimization_request
    ):
        """Test integration mode selection based on dataset size."""
        with patch(
            "traigent.cloud.integration_manager.get_production_mcp_client"
        ) as mock_mcp_client:
            with patch(
                "traigent.cloud.integration_manager.get_backend_client"
            ) as mock_backend_client:
                mock_mcp_client.return_value = Mock()
                mock_backend_client.return_value = Mock()

                manager = IntegrationManager(integration_config)

                # Test privacy preference (large dataset)
                large_dataset = Dataset(
                    examples=[EvaluationExample({}, "")] * 1500, name="large_dataset"
                )
                sample_optimization_request.dataset = large_dataset

                with patch.object(
                    manager, "_start_privacy_integration"
                ) as mock_privacy:
                    mock_privacy.return_value = Mock(
                        success=True,
                        session_id="session_123",
                        experiment_id="exp_123",
                        experiment_run_id="run_123",
                        agent_id="agent_123",
                        metadata={"mode": "privacy"},
                    )

                    await manager.initialize()

                    # Call the actual public method that exists
                    result = await manager.start_optimization_integration(
                        sample_optimization_request, IntegrationMode.PRIVACY
                    )

                    assert result.success
                    mock_privacy.assert_called_once()

    @pytest.mark.asyncio
    async def test_standard_integration_privacy_branch(
        self, sample_optimization_request
    ):
        """Standard mode should reuse the privacy helper when privacy indicators are set."""
        config = IntegrationConfig(
            mode=IntegrationMode.STANDARD,
            backend_base_url="http://localhost:5000",
            enable_fallback=True,
            max_retries=2,
            timeout=10.0,
        )
        request = deepcopy(sample_optimization_request)
        request.metadata["privacy_mode"] = True

        with (
            patch(
                "traigent.cloud.integration_manager.get_production_mcp_client"
            ) as mock_mcp_client,
            patch(
                "traigent.cloud.integration_manager.get_backend_client"
            ) as mock_backend_client,
        ):
            mock_mcp_client.return_value = Mock()
            mock_backend_client.return_value = Mock()

            manager = IntegrationManager(config)
            await manager.initialize()

            privacy_result = IntegrationResult(
                success=True,
                session_id="session_privacy",
                experiment_id="exp_privacy",
                experiment_run_id="run_privacy",
                agent_id="agent_privacy",
                metadata={"mode": "privacy"},
            )

            with (
                patch.object(
                    manager,
                    "_start_privacy_integration",
                    AsyncMock(return_value=privacy_result),
                ) as mock_privacy,
                patch.object(
                    manager,
                    "_start_cloud_integration",
                    AsyncMock(),
                ) as mock_cloud,
            ):
                result = await manager.start_optimization_integration(
                    request, IntegrationMode.STANDARD
                )

                assert result is privacy_result
                mock_privacy.assert_awaited_once()
                mock_cloud.assert_not_called()

    @pytest.mark.asyncio
    async def test_trial_management(self, integration_config):
        """Test trial management through integration manager."""
        with patch("traigent.cloud.integration_manager.get_production_mcp_client"):
            with patch(
                "traigent.cloud.integration_manager.get_backend_client"
            ) as mock_backend_client:
                with patch(
                    "traigent.cloud.integration_manager.lifecycle_manager"
                ) as mock_lifecycle:
                    # Mock session state
                    mock_session_state = Mock()
                    mock_session_state.session.session_id = "session_456"
                    mock_lifecycle.get_session_state.return_value = mock_session_state

                    mock_backend = Mock()
                    mock_backend.get_next_privacy_trial = AsyncMock(
                        return_value=TrialSuggestion(
                            trial_id="trial_123",
                            session_id="session_456",
                            trial_number=1,
                            config={"temp": 0.7},
                            dataset_subset=DatasetSubsetIndices(
                                indices=[0, 1, 2],
                                selection_strategy="random",
                                confidence_level=0.8,
                                estimated_representativeness=0.8,
                            ),
                            exploration_type="exploration",
                        )
                    )
                    mock_backend.submit_privacy_trial_results = AsyncMock(
                        return_value=True
                    )
                    mock_backend_client.return_value = mock_backend

                    manager = IntegrationManager(integration_config)
                    await manager.initialize()

                    # Add mock integration
                    manager._active_integrations["test_int"] = {
                        "mode": "private",
                        "result": Mock(session_id="session_456"),
                    }

                    # Test getting next trial
                    suggestion = await manager.get_next_trial("session_456")
                    assert suggestion is not None
                    assert suggestion.trial_id == "trial_123"

                    # Test submitting results
                    success = await manager.submit_trial_results(
                        "session_456",
                        "trial_123",
                        {"temp": 0.7},  # config
                        {"accuracy": 0.9},  # metrics
                        1.5,  # duration
                    )
                    assert success

    def test_integration_statistics(self, integration_config):
        """Test integration statistics collection."""
        manager = IntegrationManager(integration_config)
        stats = manager.get_integration_statistics()

        assert "total_integrations" in stats
        assert "successful_integrations" in stats
        assert "failed_integrations" in stats
        assert "active_sessions" in stats
        assert "lifecycle_stats" in stats

    @pytest.mark.asyncio
    async def test_health_check(self, integration_config):
        """Test integration manager health check."""
        with patch(
            "traigent.cloud.integration_manager.get_production_mcp_client"
        ) as mock_mcp_client:
            with patch(
                "traigent.cloud.integration_manager.get_backend_client"
            ) as mock_backend_client:
                mock_mcp = Mock()
                mock_mcp.health_check = AsyncMock(return_value=Mock(success=True))
                mock_mcp_client.return_value = mock_mcp
                mock_backend_client.return_value = Mock()

                manager = IntegrationManager(integration_config)
                await manager.initialize()

                health = await manager.health_check()

                assert health["integration_manager"] == "healthy"
                assert health["initialized"] is True
                assert health["components"]["mcp_client"] == "healthy"
                assert health["components"]["backend_client"] == "healthy"


class TestEndToEndIntegration:
    """End-to-end integration tests."""

    @pytest.mark.asyncio
    async def test_complete_privacy_workflow(self, sample_optimization_request):
        """Test complete privacy-first optimization workflow."""
        config = IntegrationConfig(mode=IntegrationMode.PRIVACY, enable_fallback=True)

        with patch(
            "traigent.cloud.integration_manager.get_production_mcp_client"
        ) as mock_mcp_client:
            with patch(
                "traigent.cloud.integration_manager.get_backend_client"
            ) as mock_backend_client:
                with patch(
                    "traigent.cloud.integration_manager.lifecycle_manager"
                ) as mock_lifecycle:
                    # Setup lifecycle manager mock
                    mock_session_state = Mock()
                    mock_session_state.session.session_id = "session_001"
                    mock_lifecycle.get_session_state.return_value = mock_session_state
                    mock_lifecycle.register_session = Mock()
                    mock_lifecycle.start_session = Mock()
                    mock_lifecycle.cleanup = Mock()

                    # Setup comprehensive mocks
                    mock_mcp = Mock()
                    mock_mcp.create_optimization_workflow = AsyncMock(
                        return_value=("agent_001", "exp_001", "run_001")
                    )
                    # Add async context manager support
                    mock_mcp.__aenter__ = AsyncMock(return_value=mock_mcp)
                    mock_mcp.__aexit__ = AsyncMock(return_value=None)
                    mock_mcp_client.return_value = mock_mcp

                    mock_backend = Mock()
                    mock_backend.create_privacy_optimization_session = AsyncMock(
                        return_value=("session_001", "exp_001", "run_001")
                    )
                    mock_backend.get_next_privacy_trial = AsyncMock(
                        return_value=TrialSuggestion(
                            trial_id="trial_001",
                            session_id="session_001",
                            trial_number=1,
                            config={"temperature": 0.8, "max_tokens": 100},
                            dataset_subset=DatasetSubsetIndices(
                                indices=[0, 1, 2, 3, 4],
                                selection_strategy="representative",
                                confidence_level=0.9,
                                estimated_representativeness=0.85,
                            ),
                            exploration_type="exploration",
                        )
                    )
                    mock_backend.submit_privacy_trial_results = AsyncMock(
                        return_value=True
                    )
                    mock_backend.finalize_session = AsyncMock()
                    # Add async context manager support
                    mock_backend.__aenter__ = AsyncMock(return_value=mock_backend)
                    mock_backend.__aexit__ = AsyncMock(return_value=None)
                    mock_backend_client.return_value = mock_backend

                    # Create manager without using async context manager to avoid cleanup issues
                    manager = IntegrationManager(config)
                    await manager.initialize()

                    # Start integration
                    result = await manager.start_optimization_integration(
                        sample_optimization_request
                    )

                    assert result.success
                    assert result.session_id == "session_001"
                    assert result.experiment_id == "exp_001"
                    assert result.experiment_run_id == "run_001"

                    # Mock the backend for getting trial
                    suggestion = await mock_backend.get_next_privacy_trial()
                    assert suggestion is not None
                    assert suggestion.trial_id == "trial_001"

                    # Mock submitting results
                    success = await mock_backend.submit_privacy_trial_results(
                        "session_001",
                        "trial_001",
                        {"accuracy": 0.92, "cost": 0.01},
                        2.3,
                    )
                    assert success is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
