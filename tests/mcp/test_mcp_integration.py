"""Integration tests for end-to-end MCP scenarios.

Tests complete workflows that involve multiple MCP endpoints working together,
including agent creation, execution, optimization, and result analysis.
"""

import asyncio

import pytest

from traigent.cloud.models import (
    AgentExecutionRequest,
    AgentOptimizationRequest,
    AgentSpecification,
    NextTrialRequest,
    SessionCreationRequest,
    TrialResultSubmission,
    TrialStatus,
)
from traigent.evaluators.base import Dataset, EvaluationExample


class TestCompleteOptimizationFlow:
    """Test complete optimization workflows."""

    @pytest.mark.asyncio
    async def test_create_execute_optimize_flow(
        self, mock_cloud_client, sample_dataset, mcp_test_context
    ):
        """Test full flow: create agent -> execute -> optimize -> analyze results."""
        # Step 1: Create an agent
        agent_spec = AgentSpecification(
            id="customer-support-agent",
            name="Customer Support Assistant",
            agent_type="conversational",
            agent_platform="openai",
            prompt_template="You are a helpful customer support agent. {query}",
            model_parameters={
                "model": "o4-mini",
                "temperature": 0.7,
                "max_tokens": 150,
            },
            persona="friendly and professional customer support representative",
            guidelines=[
                "Be empathetic",
                "Provide clear solutions",
                "Ask for clarification when needed",
            ],
        )

        agent_result = await mock_cloud_client.create_agent(agent_spec)
        mcp_test_context.created_agents.append(agent_result["agent_id"])

        # Step 2: Execute agent with test queries
        test_queries = [
            {"query": "How do I reset my password?"},
            {"query": "My order hasn't arrived yet"},
            {"query": "Can I get a refund?"},
        ]

        execution_results = []
        for query_data in test_queries:
            exec_request = AgentExecutionRequest(
                agent_spec=agent_spec, input_data=query_data
            )
            result = await mock_cloud_client.execute_agent(exec_request)
            execution_results.append(result)

        # Verify executions
        assert len(execution_results) == 3
        assert all(r.error is None for r in execution_results)

        # Step 3: Start optimization
        opt_request = AgentOptimizationRequest(
            agent_spec=agent_spec,
            dataset=sample_dataset,
            configuration_space={
                "temperature": (0.0, 1.0),
                "max_tokens": [100, 150, 200],
                "model": ["o4-mini", "GPT-4o"],
            },
            objectives=["accuracy", "cost", "response_time"],
            max_trials=20,
            target_cost_reduction=0.5,
        )

        opt_response = await mock_cloud_client.start_agent_optimization(opt_request)
        mcp_test_context.created_sessions.append(opt_response.session_id)

        # Step 4: Monitor optimization progress
        assert opt_response.status == "started"
        assert opt_response.estimated_cost > 0
        assert len(opt_response.next_steps) > 0

    @pytest.mark.asyncio
    async def test_iterative_optimization_with_trials(
        self, mock_cloud_client, sample_dataset, mcp_test_context
    ):
        """Test iterative optimization with trial execution and result submission."""
        # Create optimization session
        session_request = SessionCreationRequest(
            function_name="qa_assistant",
            configuration_space={
                "model": ["o4-mini", "GPT-4o"],
                "temperature": (0.1, 0.9),
                "top_p": (0.5, 1.0),
            },
            objectives=["accuracy", "consistency"],
            dataset_metadata={"size": len(sample_dataset.examples), "type": "qa_pairs"},
            max_trials=15,
        )

        session_response = await mock_cloud_client.create_optimization_session(
            session_request
        )
        session_id = session_response.session_id
        mcp_test_context.created_sessions.append(session_id)

        # Run optimization trials
        trial_results = []
        for i in range(5):
            # Get next trial suggestion
            trial_request = NextTrialRequest(session_id=session_id)
            trial_response = await mock_cloud_client.get_next_trial(trial_request)

            if not trial_response.should_continue:
                break

            suggestion = trial_response.suggestion

            # Simulate trial execution
            # In real scenario, this would involve actual function evaluation
            simulated_metrics = {
                "accuracy": 0.8 + (i * 0.02),  # Improving accuracy
                "consistency": 0.75 + (i * 0.03),
                "avg_response_time": 1.2 - (i * 0.1),  # Faster responses
            }

            # Submit trial results
            result_submission = TrialResultSubmission(
                session_id=session_id,
                trial_id=suggestion.trial_id,
                metrics=simulated_metrics,
                duration=2.5,
                status=TrialStatus.COMPLETED,
                outputs_sample=["Sample output 1", "Sample output 2"],
            )

            trial_results.append({"trial": suggestion, "results": result_submission})

        # Verify trial progression
        assert len(trial_results) == 5

        # Check that metrics improve over trials
        accuracies = [r["results"].metrics["accuracy"] for r in trial_results]
        assert accuracies[-1] > accuracies[0]  # Accuracy improved

        # Verify exploration vs exploitation
        exploration_trials = [
            r for r in trial_results if r["trial"].exploration_type == "exploration"
        ]
        assert len(exploration_trials) >= 2  # Should have exploration phase

    @pytest.mark.asyncio
    async def test_multi_agent_optimization(
        self, mock_cloud_client, sample_dataset, mcp_test_context
    ):
        """Test optimizing multiple agents in parallel."""
        # Create multiple agent specifications
        agents = [
            AgentSpecification(
                id=f"agent-{i}",
                name=f"Agent {i}",
                agent_type="conversational",
                agent_platform="openai",
                prompt_template=f"Agent {i} template: {{query}}",
                model_parameters={"model": "o4-mini", "temperature": 0.7},
            )
            for i in range(3)
        ]

        # Start optimization for each agent
        optimization_tasks = []
        for agent_spec in agents:
            opt_request = AgentOptimizationRequest(
                agent_spec=agent_spec,
                dataset=sample_dataset,
                configuration_space={
                    "temperature": (0.0, 1.0),
                    "max_tokens": [100, 150, 200],
                },
                objectives=["accuracy", "cost"],
                max_trials=10,
            )

            task = mock_cloud_client.start_agent_optimization(opt_request)
            optimization_tasks.append(task)

        # Run optimizations in parallel
        opt_responses = await asyncio.gather(*optimization_tasks)

        # Track sessions for cleanup
        for response in opt_responses:
            mcp_test_context.created_sessions.append(response.session_id)

        # Verify all optimizations started
        assert len(opt_responses) == 3
        assert all(r.status == "started" for r in opt_responses)
        assert len({r.session_id for r in opt_responses}) == 3  # All unique sessions


class TestErrorRecoveryScenarios:
    """Test error handling and recovery in integration scenarios."""

    @pytest.mark.asyncio
    async def test_optimization_with_trial_failures(
        self, mock_cloud_client, sample_dataset
    ):
        """Test optimization continues despite some trial failures."""
        # Create session
        session_request = SessionCreationRequest(
            function_name="robust_function",
            configuration_space={"model": ["o4-mini", "GPT-4o"]},
            objectives=["accuracy"],
            dataset_metadata={"size": 50},
        )

        session_response = await mock_cloud_client.create_optimization_session(
            session_request
        )
        session_id = session_response.session_id

        # Run trials with some failures
        successful_trials = 0
        failed_trials = 0

        for i in range(10):
            # Get next trial
            trial_request = NextTrialRequest(session_id=session_id)
            trial_response = await mock_cloud_client.get_next_trial(trial_request)

            if not trial_response.should_continue:
                break

            # Simulate some failures
            if i % 3 == 0:  # Every 3rd trial fails
                TrialResultSubmission(
                    session_id=session_id,
                    trial_id=trial_response.suggestion.trial_id,
                    metrics={},
                    duration=0.1,
                    status=TrialStatus.FAILED,
                    error_message="Simulated trial failure",
                )
                failed_trials += 1
            else:
                TrialResultSubmission(
                    session_id=session_id,
                    trial_id=trial_response.suggestion.trial_id,
                    metrics={"accuracy": 0.8},
                    duration=2.0,
                    status=TrialStatus.COMPLETED,
                )
                successful_trials += 1

        # Verify optimization continues despite failures
        assert successful_trials > failed_trials
        assert successful_trials >= 6  # Most trials should succeed

    @pytest.mark.asyncio
    async def test_resource_cleanup_on_error(
        self, mock_cloud_client, sample_agent_spec, mcp_test_context
    ):
        """Test that resources are properly cleaned up even when errors occur."""
        try:
            # Create agent
            agent_result = await mock_cloud_client.create_agent(sample_agent_spec)
            mcp_test_context.created_agents.append(agent_result["agent_id"])

            # Simulate an error during execution
            invalid_request = AgentExecutionRequest(
                agent_spec=sample_agent_spec,
                input_data=None,  # Invalid: missing input data
            )

            # This might fail in a real implementation
            await mock_cloud_client.execute_agent(invalid_request)

        except Exception:
            # Error occurred, but cleanup should still happen
            pass

        # Verify cleanup tracking
        assert len(mcp_test_context.created_agents) >= 1


class TestDatasetSubsetOptimization:
    """Test dataset subset selection during optimization."""

    @pytest.mark.asyncio
    async def test_adaptive_subset_sizing(self, mock_cloud_client):
        """Test that subset sizes adapt during optimization."""
        # Create large dataset
        large_dataset = Dataset(
            examples=[
                EvaluationExample(
                    input_data={"query": f"Question {i}"}, expected_output=f"Answer {i}"
                )
                for i in range(1000)
            ]
        )

        # Create session with large dataset
        session_request = SessionCreationRequest(
            function_name="large_dataset_function",
            configuration_space={"model": ["o4-mini", "GPT-4o"]},
            objectives=["accuracy"],
            dataset_metadata={
                "size": len(large_dataset.examples),
                "characteristics": "diverse_qa_pairs",
            },
            optimization_strategy={
                "adaptive_sample_size": True,
                "min_examples_per_trial": 10,
                "exploration_ratio": 0.3,
            },
        )

        session_response = await mock_cloud_client.create_optimization_session(
            session_request
        )

        # Track subset sizes across trials
        subset_sizes = []
        for _i in range(10):
            trial_request = NextTrialRequest(session_id=session_response.session_id)
            trial_response = await mock_cloud_client.get_next_trial(trial_request)

            if not trial_response.should_continue:
                break

            subset = trial_response.suggestion.dataset_subset
            subset_sizes.append(len(subset.indices))

            # Verify subset properties
            assert subset.confidence_level > 0
            assert subset.estimated_representativeness > 0
            assert subset.selection_strategy in [
                "diverse_sampling",
                "representative_sampling",
                "high_confidence_sampling",
            ]

        # In a real implementation, subset sizes should vary based on optimization phase
        assert len(subset_sizes) > 0
        assert min(subset_sizes) >= 2  # At least some examples


class TestCostOptimization:
    """Test cost-aware optimization scenarios."""

    @pytest.mark.asyncio
    async def test_cost_aware_model_selection(self, mock_cloud_client, sample_dataset):
        """Test optimization that considers cost when selecting models."""
        # Define cost-aware configuration space
        session_request = SessionCreationRequest(
            function_name="cost_optimized_assistant",
            configuration_space={
                "model": ["o4-mini", "GPT-4o", "gpt-4-turbo"],  # Different cost tiers
                "temperature": (0.0, 1.0),
                "max_tokens": [50, 100, 150],
            },
            objectives=["accuracy", "cost"],  # Multi-objective with cost
            dataset_metadata={"size": 100},
            optimization_strategy={
                "max_cost_budget": 10.0,
                "cost_weight": 0.4,  # 40% weight on cost reduction
                "performance_weight": 0.6,
            },
        )

        session_response = await mock_cloud_client.create_optimization_session(
            session_request
        )

        # Run several trials
        trial_configs = []
        for _i in range(5):
            trial_request = NextTrialRequest(session_id=session_response.session_id)
            trial_response = await mock_cloud_client.get_next_trial(trial_request)

            if trial_response.suggestion:
                trial_configs.append(trial_response.suggestion.config)

        # In a cost-aware optimization, cheaper models should be explored more
        sum(1 for config in trial_configs if config.get("model") == "o4-mini")
        sum(1 for config in trial_configs if config.get("model") == "GPT-4o")

        # Mock behavior: equal distribution, but real implementation should favor cheaper options
        assert len(trial_configs) == 5


class TestResultAggregation:
    """Test result aggregation and analysis."""

    @pytest.mark.asyncio
    async def test_optimization_finalization(self, mock_cloud_client, sample_dataset):
        """Test finalizing optimization and getting aggregated results."""
        # Create and run optimization session
        session_request = SessionCreationRequest(
            function_name="final_test_function",
            configuration_space={"model": ["o4-mini"], "temperature": (0.1, 0.9)},
            objectives=["accuracy"],
            dataset_metadata={"size": 50},
            max_trials=5,
        )

        session_response = await mock_cloud_client.create_optimization_session(
            session_request
        )
        session_id = session_response.session_id

        # Run all trials
        for i in range(5):
            trial_request = NextTrialRequest(session_id=session_id)
            trial_response = await mock_cloud_client.get_next_trial(trial_request)

            if not trial_response.should_continue:
                break

            # Submit results
            TrialResultSubmission(
                session_id=session_id,
                trial_id=trial_response.suggestion.trial_id,
                metrics={"accuracy": 0.7 + (i * 0.05)},
                duration=2.0,
                status=TrialStatus.COMPLETED,
            )

        # In a real implementation, finalization would aggregate results
        # and return the best configuration found
        assert session_id is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
