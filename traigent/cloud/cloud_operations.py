"""Cloud SaaS optimization operations for Traigent Cloud Client.

This module handles cloud-based optimization operations where agents are
executed in the cloud with full data transmission.
"""

# Traceability: CONC-Layer-Infra CONC-Quality-Reliability FUNC-CLOUD-HYBRID FUNC-AGENTS REQ-CLOUD-009 REQ-AGNT-013

from typing import TYPE_CHECKING, Any

from traigent.cloud.client import CloudServiceError
from traigent.cloud.models import (
    AgentExecutionRequest,
    AgentExecutionResponse,
    AgentOptimizationRequest,
    AgentOptimizationResponse,
    AgentSpecification,
)
from traigent.evaluators.base import Dataset
from traigent.utils.logging import get_logger

if TYPE_CHECKING:
    from traigent.cloud.backend_client import BackendIntegratedClient

logger = get_logger(__name__)


class CloudOperations:
    """Handles cloud SaaS optimization operations."""

    def __init__(self, client: "BackendIntegratedClient"):
        """Initialize cloud operations handler.

        Args:
            client: Parent BackendIntegratedClient instance
        """
        self.client = client

    async def start_agent_optimization(
        self,
        agent_spec: AgentSpecification,
        dataset: Dataset,
        configuration_space: dict[str, Any],
        objectives: list[str],
        max_trials: int = 50,
        user_id: str | None = None,
    ) -> AgentOptimizationResponse:
        """Start cloud SaaS optimization with full agent execution.

        Args:
            agent_spec: Complete agent specification
            dataset: Full evaluation dataset (transmitted to cloud)
            configuration_space: Parameter search space
            objectives: Optimization objectives
            max_trials: Maximum optimization trials
            user_id: Optional user identifier

        Returns:
            Agent optimization response with session details
        """
        logger.info(f"Starting cloud SaaS optimization for agent {agent_spec.name}")

        try:
            # Create backend agent and experiment
            (
                experiment_id,
                experiment_run_id,
            ) = await self.client._create_backend_agent_experiment(
                agent_spec, dataset, configuration_space, objectives, max_trials
            )

            # Submit to cloud for execution
            request = AgentOptimizationRequest(
                agent_spec=agent_spec,
                dataset=dataset,
                configuration_space=configuration_space,
                objectives=objectives,
                max_trials=max_trials,
                user_id=user_id,
                billing_tier="cloud",
            )

            response = await self.client._submit_agent_optimization(request)

            # Create session mapping
            assert agent_spec.name is not None, "Agent name is required"
            self.client.session_bridge.create_session_mapping(
                session_id=response.session_id,
                experiment_id=experiment_id,
                experiment_run_id=experiment_run_id,
                function_name=agent_spec.name,
                configuration_space=configuration_space,
                objectives=objectives,
            )

            logger.info(
                f"Started cloud optimization: {response.session_id} -> {experiment_id}"
            )
            return response

        except Exception as e:
            logger.error(f"Failed to start agent optimization: {e}")
            raise CloudServiceError(f"Failed to start optimization: {e}") from None

    async def execute_agent(
        self,
        agent_spec: AgentSpecification,
        input_data: dict[str, Any],
        config_overrides: dict[str, Any] | None = None,
    ) -> AgentExecutionResponse:
        """Execute agent with specified configuration.

        Args:
            agent_spec: Agent specification
            input_data: Input data for execution
            config_overrides: Optional configuration overrides

        Returns:
            Agent execution response
        """
        logger.debug(f"Executing agent {agent_spec.name}")

        try:
            request = AgentExecutionRequest(
                agent_spec=agent_spec,
                input_data=input_data,
                config_overrides=config_overrides,
            )

            return await self.client._execute_cloud_agent(request)

        except Exception as e:
            logger.error(f"Failed to execute agent: {e}")
            raise CloudServiceError(f"Failed to execute agent: {e}") from None
