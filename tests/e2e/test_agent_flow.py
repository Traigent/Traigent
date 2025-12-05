"""End-to-end tests for Model 2: Agent Optimization (Cloud-based execution)."""

import asyncio
from datetime import datetime
from unittest.mock import AsyncMock

import pytest

from traigent.agents import (
    apply_config_to_agent,
    validate_config_compatibility,
)
from traigent.cloud.client import CloudServiceError, TraiGentCloudClient
from traigent.cloud.models import (
    AgentExecutionResponse,
    AgentOptimizationResponse,
    AgentOptimizationStatus,
    AgentSpecification,
    OptimizationSessionStatus,
)
from traigent.evaluators.base import Dataset, EvaluationExample


@pytest.fixture
def sample_agent_spec():
    """Create a sample agent specification."""
    return AgentSpecification(
        id="e2e-test-agent",
        name="E2E Test Agent",
        agent_type="conversational",
        agent_platform="openai",
        prompt_template="""You are a helpful assistant.

Question: {question}
Context: {context}

Provide a clear, accurate answer.""",
        model_parameters={"model": "o4-mini", "temperature": 0.7, "max_tokens": 150},
        persona="knowledgeable and helpful assistant",
        guidelines=["Be accurate", "Be concise", "Cite sources when available"],
        metadata={"test_version": "1.0"},
    )


@pytest.fixture
def sample_dataset():
    """Create evaluation dataset."""
    return Dataset(
        [
            EvaluationExample(
                input_data={
                    "question": "What is Python?",
                    "context": "Python is a high-level programming language.",
                },
                expected_output="Python is a high-level, interpreted programming language known for its simplicity and readability.",
            ),
            EvaluationExample(
                input_data={
                    "question": "What is machine learning?",
                    "context": "ML is a subset of AI that uses data to learn.",
                },
                expected_output="Machine learning is a subset of AI that enables systems to learn and improve from data without explicit programming.",
            ),
            EvaluationExample(
                input_data={
                    "question": "Explain cloud computing",
                    "context": "Cloud computing provides on-demand computing resources.",
                },
                expected_output="Cloud computing delivers computing services over the internet, providing on-demand access to resources like storage and processing power.",
            ),
        ]
    )


@pytest.fixture
def mock_cloud_client_with_agent_support():
    """Create mock cloud client with agent optimization support."""
    client = AsyncMock(spec=TraiGentCloudClient)

    # Track optimization state
    client._optimizations = {}
    client._optimization_counter = 0

    async def optimize_agent(
        agent_spec,
        dataset,
        configuration_space,
        objectives=None,
        max_trials=50,
        target_cost_reduction=0.65,
        optimization_strategy=None,
    ):
        client._optimization_counter += 1
        opt_id = f"opt-{client._optimization_counter}"
        session_id = f"session-{client._optimization_counter}"

        # Store optimization details
        client._optimizations[opt_id] = {
            "agent_spec": agent_spec,
            "dataset": dataset,
            "config_space": configuration_space,
            "objectives": objectives or ["accuracy"],
            "max_trials": max_trials,
            "status": OptimizationSessionStatus.ACTIVE,
            "progress": 0.0,
            "completed_trials": 0,
            "start_time": datetime.now(),
        }

        return AgentOptimizationResponse(
            session_id=session_id,
            optimization_id=opt_id,
            status="started",
            estimated_cost=5.0,
            estimated_duration=300.0,
            next_steps=["Monitor progress", "Wait for completion"],
        )

    async def get_agent_optimization_status(optimization_id):
        if optimization_id not in client._optimizations:
            raise CloudServiceError(f"Optimization {optimization_id} not found")

        opt = client._optimizations[optimization_id]

        # Simulate progress
        opt["completed_trials"] = min(opt["completed_trials"] + 5, opt["max_trials"])
        opt["progress"] = opt["completed_trials"] / opt["max_trials"]

        if opt["progress"] >= 1.0:
            opt["status"] = OptimizationSessionStatus.COMPLETED

        # Simulate improving metrics
        current_accuracy = 0.7 + (0.2 * opt["progress"])

        return AgentOptimizationStatus(
            optimization_id=optimization_id,
            status=opt["status"],
            progress=opt["progress"],
            completed_trials=opt["completed_trials"],
            total_trials=opt["max_trials"],
            current_best_metrics={
                "accuracy": current_accuracy,
                "cost": 0.01 * (1 - opt["progress"] * 0.3),
                "latency": 1.5 - (0.5 * opt["progress"]),
            },
            estimated_time_remaining=300 * (1 - opt["progress"]),
        )

    async def execute_agent(
        agent_spec, input_data, config_overrides=None, execution_context=None
    ):
        # Simulate agent execution
        await asyncio.sleep(0.01)

        # Generate response based on input
        question = input_data.get("question", "")
        context = input_data.get("context", "")

        response = f"Based on the context that {context.lower()}, "
        response += f"the answer to '{question}' is: [simulated response]"

        return AgentExecutionResponse(
            output=response,
            duration=0.5,
            tokens_used=45,
            cost=0.002,
            metadata={
                "model": agent_spec.model_parameters.get("model", "unknown"),
                "platform": agent_spec.agent_platform,
            },
        )

    async def cancel_agent_optimization(optimization_id):
        if optimization_id not in client._optimizations:
            raise CloudServiceError(f"Optimization {optimization_id} not found")

        client._optimizations[optimization_id][
            "status"
        ] = OptimizationSessionStatus.CANCELLED

        return {
            "optimization_id": optimization_id,
            "status": "cancelled",
            "message": "Optimization cancelled successfully",
        }

    # Async context manager support
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass

    client.optimize_agent = optimize_agent
    client.get_agent_optimization_status = get_agent_optimization_status
    client.execute_agent = execute_agent
    client.cancel_agent_optimization = cancel_agent_optimization
    client.__aenter__ = __aenter__
    client.__aexit__ = __aexit__

    return client


class TestAgentOptimizationE2E:
    """End-to-end tests for agent optimization workflow."""

    @pytest.mark.asyncio
    async def test_complete_agent_optimization_workflow(
        self, mock_cloud_client_with_agent_support, sample_agent_spec, sample_dataset
    ):
        """Test complete agent optimization from start to finish."""

        client = mock_cloud_client_with_agent_support

        # Define configuration space
        config_space = {
            "model": ["o4-mini", "GPT-4o"],
            "temperature": (0.0, 1.0),
            "max_tokens": [100, 200, 300],
            "top_p": (0.8, 1.0),
            "frequency_penalty": (0.0, 0.2),
        }

        # Phase 1: Start optimization
        async with client:
            response = await client.optimize_agent(
                agent_spec=sample_agent_spec,
                dataset=sample_dataset,
                configuration_space=config_space,
                objectives=["accuracy", "cost", "latency"],
                max_trials=20,
                optimization_strategy={
                    "exploration_ratio": 0.3,
                    "early_stopping": True,
                    "parallel_trials": 3,
                },
            )

            assert response.optimization_id.startswith("opt-")
            assert response.status == "started"
            assert response.estimated_cost == 5.0

            optimization_id = response.optimization_id

            # Phase 2: Monitor progress
            statuses = []
            final_status = None

            for _i in range(5):  # Check status 5 times
                status = await client.get_agent_optimization_status(optimization_id)
                statuses.append(status)

                assert status.progress >= 0.0
                assert status.progress <= 1.0
                assert status.completed_trials <= status.total_trials

                if status.status == OptimizationSessionStatus.COMPLETED:
                    final_status = status
                    break

                await asyncio.sleep(0.01)  # Small delay between checks

            # Verify progression
            assert len(statuses) >= 2
            assert statuses[0].progress < statuses[-1].progress
            assert final_status is not None
            assert final_status.status == OptimizationSessionStatus.COMPLETED

            # Phase 3: Apply optimized configuration
            best_config = {
                "model": "GPT-4o",
                "temperature": 0.3,
                "max_tokens": 200,
                "top_p": 0.9,
                "frequency_penalty": 0.1,
            }

            optimized_agent = apply_config_to_agent(
                sample_agent_spec, best_config, preserve_original=True
            )

            assert optimized_agent.model_parameters["model"] == "GPT-4o"
            assert optimized_agent.model_parameters["temperature"] == 0.3

            # Phase 4: Test optimized agent
            test_inputs = [
                {"question": "What is AI?", "context": "AI is artificial intelligence"},
                {"question": "Explain ML", "context": "ML stands for machine learning"},
            ]

            execution_results = []
            for input_data in test_inputs:
                result = await client.execute_agent(
                    agent_spec=optimized_agent,
                    input_data=input_data,
                    execution_context={"test_run": True},
                )

                execution_results.append(result)
                assert result.output is not None
                assert result.duration > 0
                assert result.cost > 0

            # Verify executions
            assert len(execution_results) == 2
            assert all(r.tokens_used > 0 for r in execution_results)

    @pytest.mark.asyncio
    async def test_agent_optimization_with_validation(
        self, mock_cloud_client_with_agent_support, sample_agent_spec, sample_dataset
    ):
        """Test agent optimization with configuration validation."""

        # Configuration space with some invalid options
        config_space = {
            "model": ["o4-mini", "GPT-4o", "invalid-model"],
            "temperature": (0.0, 2.5),  # Max too high
            "max_tokens": [50, 100, 200],
            "unknown_param": ["value1", "value2"],  # Unknown parameter
        }

        # Validate before optimization
        validation_result = validate_config_compatibility(
            sample_agent_spec, config_space
        )

        # Should have warnings but still be compatible
        assert validation_result["compatible"] is True
        assert len(validation_result["warnings"]) > 0  # Should have warnings
        # Check that unknown_param warning is present in any of the warnings
        unknown_param_warning_found = any(
            "unknown_param" in warning for warning in validation_result["warnings"]
        )
        assert (
            unknown_param_warning_found
        ), f"Expected unknown_param warning not found in: {validation_result['warnings']}"

        # Run optimization anyway
        async with mock_cloud_client_with_agent_support as client:
            response = await client.optimize_agent(
                agent_spec=sample_agent_spec,
                dataset=sample_dataset,
                configuration_space=config_space,
                objectives=["accuracy"],
            )

            assert response.optimization_id is not None

    @pytest.mark.asyncio
    async def test_multi_platform_agent_optimization(
        self, mock_cloud_client_with_agent_support, sample_dataset
    ):
        """Test optimizing agents for different platforms."""

        # Create agents for different platforms
        agents = [
            AgentSpecification(
                id="openai-agent",
                name="OpenAI Agent",
                agent_type="conversational",
                agent_platform="openai",
                prompt_template="Answer: {question}",
                model_parameters={"model": "o4-mini"},
            ),
            AgentSpecification(
                id="langchain-agent",
                name="LangChain Agent",
                agent_type="task",
                agent_platform="langchain",
                prompt_template="Task: {task}\nResponse:",
                model_parameters={"model": "o4-mini"},
            ),
        ]

        optimization_ids = []

        async with mock_cloud_client_with_agent_support as client:
            # Start optimizations for each platform
            for agent in agents:
                response = await client.optimize_agent(
                    agent_spec=agent,
                    dataset=sample_dataset,
                    configuration_space={
                        "temperature": (0.0, 1.0),
                        "max_tokens": [100, 200],
                    },
                    max_trials=10,
                )

                optimization_ids.append(response.optimization_id)

            # Monitor both optimizations
            for opt_id in optimization_ids:
                status = await client.get_agent_optimization_status(opt_id)
                assert status.optimization_id == opt_id
                assert status.total_trials == 10

    @pytest.mark.asyncio
    async def test_optimization_cancellation(
        self, mock_cloud_client_with_agent_support, sample_agent_spec, sample_dataset
    ):
        """Test cancelling an ongoing optimization."""

        async with mock_cloud_client_with_agent_support as client:
            # Start optimization
            response = await client.optimize_agent(
                agent_spec=sample_agent_spec,
                dataset=sample_dataset,
                configuration_space={"temperature": (0.0, 1.0)},
                max_trials=100,  # Long running
            )

            opt_id = response.optimization_id

            # Check initial status
            status1 = await client.get_agent_optimization_status(opt_id)
            assert status1.status == OptimizationSessionStatus.ACTIVE

            # Cancel optimization
            cancel_result = await client.cancel_agent_optimization(opt_id)
            assert cancel_result["status"] == "cancelled"

            # Verify cancellation
            assert (
                client._optimizations[opt_id]["status"]
                == OptimizationSessionStatus.CANCELLED
            )

    @pytest.mark.asyncio
    async def test_agent_configuration_evolution(
        self, mock_cloud_client_with_agent_support, sample_agent_spec, sample_dataset
    ):
        """Test tracking configuration changes during optimization."""

        # Track configuration evolution
        config_history = []

        async with mock_cloud_client_with_agent_support as client:
            # Start with base configuration
            base_config = sample_agent_spec.model_parameters.copy()
            config_history.append(("initial", base_config))

            # Run optimization
            await client.optimize_agent(
                agent_spec=sample_agent_spec,
                dataset=sample_dataset,
                configuration_space={
                    "model": ["o4-mini", "GPT-4o"],
                    "temperature": (0.0, 1.0),
                    "max_tokens": [100, 200, 300],
                },
                max_trials=15,
            )

            # Simulate configuration updates during optimization
            trial_configs = [
                {"model": "o4-mini", "temperature": 0.5, "max_tokens": 100},
                {"model": "GPT-4o", "temperature": 0.3, "max_tokens": 200},
                {"model": "GPT-4o", "temperature": 0.4, "max_tokens": 250},
            ]

            for i, config in enumerate(trial_configs):
                # Apply configuration
                updated_agent = apply_config_to_agent(sample_agent_spec, config)
                config_history.append(
                    (f"trial_{i + 1}", updated_agent.model_parameters)
                )

                # Execute to test
                result = await client.execute_agent(
                    agent_spec=updated_agent,
                    input_data={"question": "Test question", "context": "Test context"},
                )

                assert result.output is not None

            # Analyze evolution
            assert len(config_history) == 4  # initial + 3 trials

            # Verify progression
            initial_config = config_history[0][1]
            final_config = config_history[-1][1]

            # Model upgraded
            assert initial_config["model"] == "o4-mini"
            assert final_config["model"] == "GPT-4o"

            # Temperature optimized
            assert initial_config["temperature"] == 0.7
            assert final_config["temperature"] == 0.4

    @pytest.mark.asyncio
    async def test_cost_aware_optimization(
        self, mock_cloud_client_with_agent_support, sample_agent_spec, sample_dataset
    ):
        """Test optimization with cost constraints."""

        async with mock_cloud_client_with_agent_support as client:
            # Run cost-aware optimization
            response = await client.optimize_agent(
                agent_spec=sample_agent_spec,
                dataset=sample_dataset,
                configuration_space={
                    "model": ["o4-mini", "GPT-4o"],  # Different costs
                    "temperature": (0.0, 1.0),
                    "max_tokens": [100, 500],  # Affects cost
                },
                objectives=["accuracy", "cost"],  # Multi-objective
                max_trials=20,  # Set explicit max_trials for faster completion
                optimization_strategy={
                    "max_cost_budget": 10.0,
                    "cost_weight": 0.3,  # 30% weight on cost
                    "accuracy_weight": 0.7,  # 70% weight on accuracy
                },
            )

            opt_id = response.optimization_id

            # Monitor optimization
            final_status = None
            for _ in range(
                5
            ):  # Should complete with max_trials=20 (5 trials per iteration)
                status = await client.get_agent_optimization_status(opt_id)
                if status.status == OptimizationSessionStatus.COMPLETED:
                    final_status = status
                    break

            assert final_status is not None

            # Verify cost optimization worked
            best_metrics = final_status.current_best_metrics
            assert "cost" in best_metrics
            assert best_metrics["cost"] < 0.01  # Should optimize for lower cost
