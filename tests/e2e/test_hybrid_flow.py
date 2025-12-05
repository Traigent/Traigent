"""End-to-end tests for hybrid optimization combining both models."""

import asyncio
import json
import tempfile
from datetime import datetime

import pytest

from traigent.agents import apply_config_to_agent
from traigent.cloud.models import (
    AgentExecutionResponse,
    AgentOptimizationResponse,
    AgentOptimizationStatus,
    AgentSpecification,
    DatasetSubsetIndices,
    NextTrialResponse,
    OptimizationFinalizationResponse,
    OptimizationSessionStatus,
    SessionCreationResponse,
    TrialSuggestion,
)
from traigent.evaluators.base import Dataset, EvaluationExample
from traigent.optimizers.interactive_optimizer import InteractiveOptimizer


@pytest.fixture
def hybrid_dataset():
    """Create dataset for hybrid optimization."""
    return Dataset(
        [
            EvaluationExample(
                input_data={
                    "question": "What is deep learning?",
                    "context": "Deep learning uses neural networks with multiple layers.",
                },
                expected_output="Deep learning is a subset of machine learning that uses neural networks with multiple layers to learn from data.",
            ),
            EvaluationExample(
                input_data={
                    "question": "Explain transformers",
                    "context": "Transformers are a neural network architecture.",
                },
                expected_output="Transformers are a neural network architecture that uses self-attention mechanisms for processing sequential data.",
            ),
            EvaluationExample(
                input_data={
                    "question": "What is GPT?",
                    "context": "GPT stands for Generative Pre-trained Transformer.",
                },
                expected_output="GPT (Generative Pre-trained Transformer) is a language model that generates human-like text.",
            ),
            EvaluationExample(
                input_data={
                    "question": "Define reinforcement learning",
                    "context": "RL involves learning through rewards and penalties.",
                },
                expected_output="Reinforcement learning is a type of machine learning where agents learn by receiving rewards or penalties for their actions.",
            ),
        ]
    )


class MockHybridCloudClient:
    """Mock cloud client supporting both optimization models."""

    def __init__(self):
        # State tracking
        self._session_state = {}
        self._agent_optimizations = {}
        self._trial_counter = 0

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        return None

    # Interactive optimization methods (Model 1)
    async def create_session(self, request):
        session_id = f"hybrid-session-{len(self._session_state) + 1}"
        self._session_state[session_id] = {
            "config_space": request.configuration_space,
            "trials_completed": 0,
            "max_trials": request.max_trials,
            "best_config": None,
            "best_metrics": None,
        }

        return SessionCreationResponse(
            session_id=session_id,
            status=OptimizationSessionStatus.ACTIVE,
            optimization_strategy={"mode": "hybrid", "phase": "local_exploration"},
        )

    async def get_next_trial(self, request):
        session = self._session_state.get(request.session_id)
        if not session:
            return NextTrialResponse(suggestion=None, should_continue=False)

        self._trial_counter += 1
        session["trials_completed"] += 1

        if session["trials_completed"] > 3:  # Quick exploration
            return NextTrialResponse(
                suggestion=None,
                should_continue=False,
                reason="Local exploration complete",
            )

        # Suggest configurations for local exploration
        configs = [
            {"temperature": 0.3, "model": "o4-mini"},
            {"temperature": 0.7, "model": "o4-mini"},
            {"temperature": 0.5, "model": "GPT-4o"},
        ]

        config = configs[session["trials_completed"] - 1]

        suggestion = TrialSuggestion(
            trial_id=f"hybrid-trial-{self._trial_counter}",
            session_id=request.session_id,
            trial_number=session["trials_completed"],
            config=config,
            dataset_subset=DatasetSubsetIndices(
                indices=[0, 1],  # Small subset for quick local testing
                selection_strategy="diverse",
                confidence_level=0.7,
                estimated_representativeness=0.8,
            ),
            exploration_type="exploration",
        )

        return NextTrialResponse(suggestion=suggestion, should_continue=True)

    async def submit_result(self, result):
        session = self._session_state.get(result.session_id)
        if session:
            current_best = session.get("best_metrics") or {}
            current_accuracy = current_best.get("accuracy", 0)
            new_accuracy = result.metrics.get("accuracy", 0)

            if new_accuracy > current_accuracy:
                session["best_metrics"] = result.metrics
                # Extract config from trial_id (simplified)
                session["best_config"] = {"temperature": 0.5, "model": "GPT-4o"}

    async def finalize_session(self, request):
        session = self._session_state.get(request.session_id)
        return OptimizationFinalizationResponse(
            session_id=request.session_id,
            best_config=session.get(
                "best_config", {"temperature": 0.5, "model": "GPT-4o"}
            ),
            best_metrics=session.get("best_metrics", {"accuracy": 0.85}),
            total_trials=session.get("trials_completed", 3),
            successful_trials=session.get("trials_completed", 3),
            total_duration=30.0,
            cost_savings=0.7,
        )

    # Agent optimization methods (Model 2)
    async def optimize_agent(self, agent_spec, dataset, configuration_space, **kwargs):
        opt_id = f"hybrid-opt-{len(self._agent_optimizations) + 1}"
        self._agent_optimizations[opt_id] = {
            "agent_spec": agent_spec,
            "status": OptimizationSessionStatus.ACTIVE,
            "progress": 0.0,
            "trials": 0,
            "max_trials": kwargs.get("max_trials", 20),
        }

        return AgentOptimizationResponse(
            session_id=f"agent-session-{opt_id}",
            optimization_id=opt_id,
            status="started",
            estimated_cost=3.0,
            estimated_duration=180.0,
        )

    async def get_agent_optimization_status(self, optimization_id):
        opt = self._agent_optimizations.get(optimization_id)
        if not opt:
            raise ValueError(f"Unknown optimization: {optimization_id}")

        # Simulate progress
        opt["trials"] = min(opt["trials"] + 5, opt["max_trials"])
        opt["progress"] = opt["trials"] / opt["max_trials"]

        if opt["progress"] >= 1.0:
            opt["status"] = OptimizationSessionStatus.COMPLETED

        return AgentOptimizationStatus(
            optimization_id=optimization_id,
            status=opt["status"],
            progress=opt["progress"],
            completed_trials=opt["trials"],
            total_trials=opt["max_trials"],
            current_best_metrics={
                "accuracy": 0.85 + (0.1 * opt["progress"]),  # Improve with progress
                "cost": 0.005,
                "latency": 0.8,
            },
        )

    async def execute_agent(self, agent_spec, input_data, **kwargs):
        # Simulate cloud execution
        await asyncio.sleep(0.01)

        question = input_data.get("question", "")
        accuracy_boost = (
            0.1 if agent_spec.model_parameters.get("model") == "GPT-4o" else 0
        )

        return AgentExecutionResponse(
            output=f"Cloud response for: {question}",
            duration=0.8,
            tokens_used=60,
            cost=0.003,
            metadata={"phase": "validation", "accuracy_estimate": 0.9 + accuracy_boost},
        )


@pytest.fixture
def mock_hybrid_cloud_client():
    """Create mock cloud client supporting both optimization models."""
    return MockHybridCloudClient()


class TestHybridOptimizationE2E:
    """End-to-end tests for hybrid optimization workflow."""

    @pytest.mark.asyncio
    async def test_complete_hybrid_workflow(
        self, mock_hybrid_cloud_client, hybrid_dataset
    ):
        """Test complete hybrid optimization workflow from local to cloud."""

        # Create agent specification
        agent_spec = AgentSpecification(
            id="hybrid-test-agent",
            name="Hybrid Test Agent",
            agent_type="conversational",
            agent_platform="openai",
            prompt_template="Question: {question}\nContext: {context}\nAnswer:",
            model_parameters={
                "model": "o4-mini",
                "temperature": 0.7,
                "max_tokens": 150,
            },
        )

        # Configuration space
        config_space = {
            "model": ["o4-mini", "GPT-4o"],
            "temperature": (0.0, 1.0),
            "max_tokens": [100, 200, 300],
        }

        optimization_results = {
            "phase1_best": None,
            "phase2_best": None,
            "validation_results": [],
        }

        async with mock_hybrid_cloud_client as client:
            # Phase 1: Local optimization with remote guidance
            print("\n=== Phase 1: Local Optimization ===")

            optimizer = InteractiveOptimizer(
                config_space=config_space,
                objectives=["accuracy", "latency"],
                remote_service=client,
                dataset_metadata={"size": len(hybrid_dataset.examples)},
            )

            session = await optimizer.initialize_session(
                function_name="hybrid_agent", max_trials=10
            )

            assert session.session_id.startswith("hybrid-session-")

            # Mock local executor
            async def mock_local_executor(agent_spec, input_data):
                await asyncio.sleep(0.001)
                # Simulate quality based on config
                quality = 0.7
                if agent_spec.model_parameters.get("model") == "GPT-4o":
                    quality += 0.15
                if agent_spec.model_parameters.get("temperature", 0.7) < 0.5:
                    quality += 0.05

                return {
                    "output": f"Local response for: {input_data.get('question')}",
                    "quality": quality,
                }

            # Run local trials
            local_trial_count = 0
            while True:
                suggestion = await optimizer.get_next_suggestion(
                    len(hybrid_dataset.examples)
                )
                if not suggestion:
                    break

                local_trial_count += 1

                # Apply config and execute locally
                trial_agent = apply_config_to_agent(agent_spec, suggestion.config)

                # Execute on subset
                correct = 0
                total_time = 0
                for idx in suggestion.dataset_subset.indices:
                    example = hybrid_dataset.examples[idx]
                    start = asyncio.get_event_loop().time()

                    result = await mock_local_executor(trial_agent, example.input_data)

                    elapsed = asyncio.get_event_loop().time() - start
                    total_time += elapsed

                    if result["quality"] > 0.8:
                        correct += 1

                # Report metrics
                metrics = {
                    "accuracy": correct / len(suggestion.dataset_subset.indices),
                    "latency": total_time / len(suggestion.dataset_subset.indices),
                }

                await optimizer.report_results(
                    trial_id=suggestion.trial_id, metrics=metrics, duration=total_time
                )

            # Finalize local optimization
            local_results = await optimizer.finalize_optimization()
            optimization_results["phase1_best"] = local_results.best_config

            assert local_trial_count == 3
            assert local_results.best_config is not None
            assert local_results.cost_savings >= 0.5

            # Phase 2: Cloud-based refinement
            print("\n=== Phase 2: Cloud Refinement ===")

            # Create refined agent with best local config
            refined_agent = apply_config_to_agent(
                agent_spec, local_results.best_config, preserve_original=True
            )

            # Refine configuration space around best config
            refined_space = {
                "model": [local_results.best_config.get("model", "o4-mini")],
                "temperature": (
                    max(0, local_results.best_config.get("temperature", 0.5) - 0.2),
                    min(1, local_results.best_config.get("temperature", 0.5) + 0.2),
                ),
                "max_tokens": [150, 200, 250],
            }

            # Start cloud optimization
            cloud_response = await client.optimize_agent(
                agent_spec=refined_agent,
                dataset=hybrid_dataset,
                configuration_space=refined_space,
                objectives=["accuracy", "response_quality"],
                max_trials=10,
                optimization_strategy={
                    "start_from_best": True,
                    "exploration_ratio": 0.2,  # Less exploration
                },
            )

            assert cloud_response.optimization_id.startswith("hybrid-opt-")

            # Monitor cloud optimization
            cloud_status = None
            for _ in range(3):
                status = await client.get_agent_optimization_status(
                    cloud_response.optimization_id
                )
                cloud_status = status

                if status.status == OptimizationSessionStatus.COMPLETED:
                    break

                await asyncio.sleep(0.01)

            assert cloud_status is not None
            assert cloud_status.progress == 1.0
            assert cloud_status.current_best_metrics["accuracy"] >= 0.9

            optimization_results["phase2_best"] = cloud_status.current_best_metrics

            # Phase 3: Validation
            print("\n=== Phase 3: Deployment Validation ===")

            # Create final optimized agent
            final_agent = apply_config_to_agent(
                refined_agent,
                {"temperature": 0.4, "max_tokens": 200},  # Simulated final config
                preserve_original=True,
            )

            # Validate on holdout examples
            holdout_indices = [2, 3]  # Last examples as holdout

            for idx in holdout_indices:
                example = hybrid_dataset.examples[idx]

                result = await client.execute_agent(
                    agent_spec=final_agent,
                    input_data=example.input_data,
                    execution_context={"validation": True},
                )

                optimization_results["validation_results"].append(
                    {
                        "input": example.input_data,
                        "output": result.output,
                        "cost": result.cost,
                        "accuracy_estimate": result.metadata.get(
                            "accuracy_estimate", 0
                        ),
                    }
                )

            # Verify complete workflow
            assert optimization_results["phase1_best"] is not None
            assert optimization_results["phase2_best"] is not None
            assert len(optimization_results["validation_results"]) == 2

            # Phase 2 should improve on Phase 1
            assert optimization_results["phase2_best"]["accuracy"] > 0.85

            # Validation should show good results
            avg_validation_accuracy = sum(
                r["accuracy_estimate"]
                for r in optimization_results["validation_results"]
            ) / len(optimization_results["validation_results"])
            assert avg_validation_accuracy >= 0.9

    @pytest.mark.asyncio
    async def test_hybrid_with_early_stopping(
        self, mock_hybrid_cloud_client, hybrid_dataset
    ):
        """Test hybrid optimization with early stopping in different phases."""

        AgentSpecification(
            id="early-stop-agent",
            name="Early Stop Test",
            agent_type="conversational",
            agent_platform="openai",
            prompt_template="{question}",
            model_parameters={"model": "o4-mini"},
        )

        async with mock_hybrid_cloud_client as client:
            # Phase 1 with early stopping
            optimizer = InteractiveOptimizer(
                config_space={"temperature": (0.0, 1.0)},
                objectives=["accuracy"],
                remote_service=client,
                optimization_strategy={
                    "early_stopping": True,
                    "patience": 2,
                    "min_improvement": 0.05,
                },
            )

            await optimizer.initialize_session("test_function", max_trials=20)

            # Simulate trials with no improvement
            trial_count = 0
            last_accuracy = 0.8

            while True:
                suggestion = await optimizer.get_next_suggestion(
                    len(hybrid_dataset.examples)
                )
                if not suggestion:
                    break

                trial_count += 1

                # Report similar metrics (no improvement)
                await optimizer.report_results(
                    trial_id=suggestion.trial_id,
                    metrics={"accuracy": last_accuracy + 0.01},  # Minimal improvement
                    duration=1.0,
                )

            # Should stop early
            assert trial_count <= 5  # Less than max_trials

    @pytest.mark.asyncio
    async def test_hybrid_cost_optimization(
        self, mock_hybrid_cloud_client, hybrid_dataset
    ):
        """Test hybrid optimization with cost awareness."""

        # Track costs
        total_costs = {"phase1_local": 0.0, "phase2_cloud": 0.0, "total": 0.0}

        agent_spec = AgentSpecification(
            id="cost-aware-agent",
            name="Cost Aware Agent",
            agent_type="conversational",
            agent_platform="openai",
            prompt_template="{question}",
            model_parameters={"model": "o4-mini"},
        )

        async with mock_hybrid_cloud_client as client:
            # Phase 1: Local execution (low cost)
            optimizer = InteractiveOptimizer(
                config_space={
                    "model": ["o4-mini", "GPT-4o"],
                    "temperature": (0.0, 1.0),
                },
                objectives=["accuracy", "cost"],
                remote_service=client,
            )

            await optimizer.initialize_session("cost_function", max_trials=10)

            while True:
                suggestion = await optimizer.get_next_suggestion(
                    len(hybrid_dataset.examples)
                )
                if not suggestion:
                    break

                # Local execution cost (minimal)
                local_cost = 0.0001 * len(suggestion.dataset_subset.indices)
                total_costs["phase1_local"] += local_cost

                await optimizer.report_results(
                    trial_id=suggestion.trial_id,
                    metrics={"accuracy": 0.85, "cost": local_cost},
                    duration=0.5,
                )

            local_results = await optimizer.finalize_optimization()

            # Phase 2: Cloud execution (higher cost)
            cloud_response = await client.optimize_agent(
                agent_spec=agent_spec,
                dataset=hybrid_dataset,
                configuration_space={"temperature": (0.3, 0.7)},
                objectives=["accuracy", "cost"],
                max_trials=5,
                optimization_strategy={"max_cost_budget": 1.0, "cost_aware": True},
            )

            # Simulate cloud costs
            for _ in range(2):
                status = await client.get_agent_optimization_status(
                    cloud_response.optimization_id
                )
                cloud_trial_cost = 0.003 * status.completed_trials
                total_costs["phase2_cloud"] = cloud_trial_cost

            # Calculate total
            total_costs["total"] = (
                total_costs["phase1_local"] + total_costs["phase2_cloud"]
            )

            # Verify cost efficiency
            assert total_costs["phase1_local"] < 0.001  # Local is cheap
            assert total_costs["phase2_cloud"] < 0.1  # Cloud controlled
            assert total_costs["total"] < 0.2  # Total reasonable

            # Verify we achieved cost savings
            assert local_results.cost_savings >= 0.6  # At least 60% savings

    @pytest.mark.asyncio
    async def test_hybrid_workflow_persistence(
        self, mock_hybrid_cloud_client, hybrid_dataset
    ):
        """Test saving and loading hybrid optimization results."""

        agent_spec = AgentSpecification(
            id="persistent-agent",
            name="Persistent Agent",
            agent_type="conversational",
            agent_platform="openai",
            prompt_template="{question}",
            model_parameters={"model": "o4-mini"},
        )

        # Use temp file for results
        with tempfile.NamedTemporaryFile(mode="w+", suffix=".json", delete=False) as f:
            results_file = f.name

        try:
            async with mock_hybrid_cloud_client as client:
                # Run simplified hybrid optimization
                optimizer = InteractiveOptimizer(
                    config_space={"temperature": (0.0, 1.0)},
                    objectives=["accuracy"],
                    remote_service=client,
                )

                await optimizer.initialize_session("test", max_trials=5)

                # Run one trial
                suggestion = await optimizer.get_next_suggestion(
                    len(hybrid_dataset.examples)
                )
                await optimizer.report_results(
                    trial_id=suggestion.trial_id,
                    metrics={"accuracy": 0.9},
                    duration=1.0,
                )

                local_results = await optimizer.finalize_optimization()

                # Save results
                results_data = {
                    "timestamp": datetime.now().isoformat(),
                    "agent_id": agent_spec.id,
                    "phase1_results": {
                        "best_config": local_results.best_config,
                        "best_metrics": local_results.best_metrics,
                        "trials": local_results.total_trials,
                    },
                    "optimization_complete": True,
                }

                with open(results_file, "w") as f:
                    json.dump(results_data, f, indent=2)

                # Load and verify
                with open(results_file) as f:
                    loaded_data = json.load(f)

                assert loaded_data["agent_id"] == "persistent-agent"
                assert loaded_data["phase1_results"]["best_config"] is not None
                assert loaded_data["optimization_complete"] is True

        finally:
            # Clean up
            import os

            if os.path.exists(results_file):
                os.remove(results_file)
