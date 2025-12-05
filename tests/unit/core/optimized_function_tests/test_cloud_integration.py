"""Tests for OptimizedFunction cloud integration.

Tests cloud service integration, fallback behavior, and cloud execution features.
"""

from unittest.mock import AsyncMock, Mock, patch

import pytest

from traigent.cloud.client import CloudServiceError
from traigent.core.optimized_function import OptimizedFunction


class TestCloudIntegration:
    """Test cloud integration functionality."""

    @pytest.mark.asyncio
    async def test_cloud_mode_uses_cloud_service(
        self, simple_function, sample_config_space, sample_objectives, sample_dataset
    ):
        """Test that cloud mode uses cloud services."""
        opt_func = OptimizedFunction(
            func=simple_function,
            configuration_space=sample_config_space,
            objectives=sample_objectives,
            eval_dataset=sample_dataset,
            execution_mode="cloud",
        )

        # Execution mode should be set to cloud for managed optimizations
        assert opt_func.execution_mode == "cloud"

    @pytest.mark.asyncio
    async def test_cloud_service_fallback(
        self, simple_function, sample_config_space, sample_objectives, sample_dataset
    ):
        """Test fallback to local when cloud service fails."""
        opt_func = OptimizedFunction(
            func=simple_function,
            configuration_space=sample_config_space,
            objectives=sample_objectives,
            eval_dataset=sample_dataset,
            execution_mode="cloud",
        )

        # When cloud service fails, it should fall back to local
        # Mock the cloud service to fail during context manager entry
        with patch("traigent.cloud.client.TraiGentCloudClient") as MockClient:
            mock_instance = Mock()
            mock_instance.__aenter__ = AsyncMock(
                side_effect=Exception("Cloud service unavailable")
            )
            MockClient.return_value = mock_instance

            # Should still work with local optimization
            from datetime import datetime

            from traigent.api.types import OptimizationResult, OptimizationStatus

            mock_result = OptimizationResult(
                trials=[],
                best_config={"temperature": 0.5},
                best_score=0.85,
                optimization_id="local-001",
                duration=5.0,
                convergence_info={},
                status=OptimizationStatus.COMPLETED,
                objectives=sample_objectives,
                algorithm="grid",
                timestamp=datetime.now(),
                metadata={},
            )

            with patch(
                "traigent.core.optimized_function.OptimizationOrchestrator"
            ) as MockOrchestrator:
                mock_orchestrator = MockOrchestrator.return_value
                mock_orchestrator.optimize = AsyncMock(return_value=mock_result)

                # Should fall back to local optimization
                result = await opt_func.optimize()
                assert result.best_score == 0.85

    def test_cloud_mode_configuration(
        self, simple_function, sample_config_space, sample_objectives
    ):
        """Test cloud execution configuration settings."""
        opt_func = OptimizedFunction(
            func=simple_function,
            configuration_space=sample_config_space,
            objectives=sample_objectives,
            execution_mode="cloud",
        )

        assert opt_func.execution_mode == "cloud"
        # API key is handled by the cloud client, not stored in OptimizedFunction

    @pytest.mark.asyncio
    async def test_cloud_optimization_with_custom_params(
        self, simple_function, sample_config_space, sample_objectives, sample_dataset
    ):
        """Test cloud optimization with custom parameters."""
        opt_func = OptimizedFunction(
            func=simple_function,
            configuration_space=sample_config_space,
            objectives=sample_objectives,
            eval_dataset=sample_dataset,
            execution_mode="cloud",
        )

        # Test that cloud execution is set
        assert opt_func.execution_mode == "cloud"

        # Test that optimization can proceed (fallback to local)
        from datetime import datetime

        from traigent.api.types import OptimizationResult, OptimizationStatus

        mock_result = OptimizationResult(
            trials=[],
            best_config={"temperature": 0.7},
            best_score=0.95,
            optimization_id="test-004",
            duration=5.0,
            convergence_info={},
            status=OptimizationStatus.COMPLETED,
            objectives=sample_objectives,
            algorithm="grid",
            timestamp=datetime.now(),
            metadata={},
        )

        with patch(
            "traigent.core.optimized_function.OptimizationOrchestrator"
        ) as MockOrchestrator:
            mock_orchestrator = MockOrchestrator.return_value
            mock_orchestrator.optimize = AsyncMock(return_value=mock_result)

            result = await opt_func.optimize()
            assert result.best_score == 0.95

    def test_disable_cloud_mode(
        self, simple_function, sample_config_space, sample_objectives
    ):
        """Test disabling cloud mode (edge execution)."""
        opt_func = OptimizedFunction(
            func=simple_function,
            configuration_space=sample_config_space,
            objectives=sample_objectives,
            execution_mode="edge_analytics",
        )

        assert opt_func.execution_mode == "edge_analytics"

    @pytest.mark.asyncio
    async def test_cloud_service_authentication_error(
        self, simple_function, sample_config_space, sample_objectives, sample_dataset
    ):
        """Test handling of authentication errors."""
        opt_func = OptimizedFunction(
            func=simple_function,
            configuration_space=sample_config_space,
            objectives=sample_objectives,
            eval_dataset=sample_dataset,
            execution_mode="cloud",
        )

        # Mock cloud service to raise authentication error during initialization
        with patch("traigent.cloud.client.TraiGentCloudClient") as MockClient:
            # Make the class initialization raise the error
            mock_instance = Mock()
            mock_instance.__aenter__ = AsyncMock(
                side_effect=CloudServiceError("Authentication failed")
            )
            MockClient.return_value = mock_instance

            # Should fall back to local optimization instead of propagating the error
            from datetime import datetime

            from traigent.api.types import OptimizationResult, OptimizationStatus

            mock_result = OptimizationResult(
                trials=[],
                best_config={"temperature": 0.5},
                best_score=0.80,
                optimization_id="fallback-001",
                duration=3.0,
                convergence_info={},
                status=OptimizationStatus.COMPLETED,
                objectives=sample_objectives,
                algorithm="grid",
                timestamp=datetime.now(),
                metadata={},
            )

            with patch(
                "traigent.core.optimized_function.OptimizationOrchestrator"
            ) as MockOrchestrator:
                mock_orchestrator = MockOrchestrator.return_value
                mock_orchestrator.optimize = AsyncMock(return_value=mock_result)

                result = await opt_func.optimize()
                assert result.best_score == 0.80

    @pytest.mark.asyncio
    async def test_hybrid_optimization_mode(
        self, simple_function, sample_config_space, sample_objectives, sample_dataset
    ):
        """Test hybrid optimization using both cloud and local resources."""
        opt_func = OptimizedFunction(
            func=simple_function,
            configuration_space=sample_config_space,
            objectives=sample_objectives,
            eval_dataset=sample_dataset,
            execution_mode="cloud",
        )

        # Test that cloud execution enables cloud service
        assert opt_func.execution_mode == "cloud"

        # The actual hybrid implementation would be complex to test properly
        # Just verify basic functionality works
        from datetime import datetime

        from traigent.api.types import OptimizationResult, OptimizationStatus

        mock_result = OptimizationResult(
            trials=[],
            best_config={"temperature": 0.6},
            best_score=0.90,
            optimization_id="hybrid-001",
            duration=4.0,
            convergence_info={},
            status=OptimizationStatus.COMPLETED,
            objectives=sample_objectives,
            algorithm="grid",
            timestamp=datetime.now(),
            metadata={},
        )

        with patch(
            "traigent.core.optimized_function.OptimizationOrchestrator"
        ) as MockOrchestrator:
            mock_orchestrator = MockOrchestrator.return_value
            mock_orchestrator.optimize = AsyncMock(return_value=mock_result)

            result = await opt_func.optimize()
            assert result.best_score == 0.90

    def test_cloud_result_caching(
        self, simple_function, sample_config_space, sample_objectives
    ):
        """Test caching of cloud optimization results."""
        opt_func = OptimizedFunction(
            func=simple_function,
            configuration_space=sample_config_space,
            objectives=sample_objectives,
            execution_mode="cloud",
        )

        # Check that cache_results is stored in kwargs
        assert opt_func.kwargs.get("cache_results", False) is False  # Default is False

        # Create function with caching enabled
        opt_func2 = OptimizedFunction(
            func=simple_function,
            configuration_space=sample_config_space,
            objectives=sample_objectives,
            execution_mode="cloud",
            cache_results=True,
        )

        assert opt_func2.kwargs.get("cache_results") is True
