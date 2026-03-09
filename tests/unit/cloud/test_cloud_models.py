"""Unit tests for cloud service data models."""

from datetime import datetime

from traigent.cloud.models import (
    AgentOptimizationRequest,
    AgentOptimizationStatus,
    AgentSpecification,
    DatasetSubsetIndices,
    NextTrialResponse,
    OptimizationFinalizationRequest,
    OptimizationFinalizationResponse,
    OptimizationSession,
    OptimizationSessionStatus,
    SessionCreationRequest,
    SessionObjectiveDefinition,
    TrialResultSubmission,
    TrialStatus,
    TrialSuggestion,
)
from traigent.evaluators.base import Dataset, EvaluationExample


class TestOptimizationSession:
    """Test OptimizationSession model."""

    def test_session_creation(self):
        """Test creating an optimization session."""
        session = OptimizationSession(
            session_id="test-123",
            function_name="my_function",
            configuration_space={"param1": [1, 2, 3]},
            objectives=["accuracy"],
            max_trials=50,
            status=OptimizationSessionStatus.ACTIVE,
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )

        assert session.session_id == "test-123"
        assert session.function_name == "my_function"
        assert session.completed_trials == 0
        assert session.best_config is None
        assert session.best_metrics is None

    def test_is_complete(self):
        """Test checking if session is complete."""
        session = OptimizationSession(
            session_id="test-123",
            function_name="my_function",
            configuration_space={},
            objectives=["accuracy"],
            max_trials=50,
            status=OptimizationSessionStatus.ACTIVE,
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )

        assert not session.is_complete()

        session.status = OptimizationSessionStatus.COMPLETED
        assert session.is_complete()

        session.status = OptimizationSessionStatus.FAILED
        assert session.is_complete()

        session.status = OptimizationSessionStatus.CANCELLED
        assert session.is_complete()

    def test_can_continue(self):
        """Test checking if session can continue."""
        session = OptimizationSession(
            session_id="test-123",
            function_name="my_function",
            configuration_space={},
            objectives=["accuracy"],
            max_trials=50,
            status=OptimizationSessionStatus.ACTIVE,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            completed_trials=10,
        )

        assert session.can_continue()

        session.completed_trials = 50
        assert not session.can_continue()

        session.completed_trials = 10
        session.status = OptimizationSessionStatus.COMPLETED
        assert not session.can_continue()


class TestTrialSuggestion:
    """Test TrialSuggestion model."""

    def test_trial_suggestion_creation(self):
        """Test creating a trial suggestion."""
        subset = DatasetSubsetIndices(
            indices=[0, 5, 10, 15],
            selection_strategy="diverse_sampling",
            confidence_level=0.8,
            estimated_representativeness=0.85,
        )

        suggestion = TrialSuggestion(
            trial_id="trial-001",
            session_id="session-123",
            trial_number=1,
            config={"temperature": 0.7},
            dataset_subset=subset,
            exploration_type="exploration",
        )

        assert suggestion.trial_id == "trial-001"
        assert suggestion.config == {"temperature": 0.7}
        assert len(suggestion.dataset_subset.indices) == 4
        assert suggestion.priority == 1


class TestTrialResultSubmission:
    """Test TrialResultSubmission model."""

    def test_result_submission_success(self):
        """Test creating a successful trial result."""
        result = TrialResultSubmission(
            session_id="session-123",
            trial_id="trial-001",
            metrics={"accuracy": 0.95, "latency": 0.12},
            duration=45.2,
            status=TrialStatus.COMPLETED,
        )

        assert result.metrics["accuracy"] == 0.95
        assert result.error_message is None
        assert result.status == TrialStatus.COMPLETED

    def test_result_submission_failure(self):
        """Test creating a failed trial result."""
        result = TrialResultSubmission(
            session_id="session-123",
            trial_id="trial-001",
            metrics={},
            duration=0.5,
            status=TrialStatus.FAILED,
            error_message="Connection timeout",
        )

        assert result.metrics == {}
        assert result.error_message == "Connection timeout"
        assert result.status == TrialStatus.FAILED


class TestSessionRequests:
    """Test session request/response models."""

    def test_session_creation_request(self):
        """Test session creation request."""
        request = SessionCreationRequest(
            function_name="optimize_llm",
            configuration_space={
                "temperature": (0.0, 1.0),
                "model": ["gpt-3.5", "GPT-4o"],
            },
            objectives=["accuracy", "cost"],
            dataset_metadata={"size": 1000, "type": "qa"},
            max_trials=100,
            budget={"max_cost_usd": 5.0},
            constraints={"derived": [{"require": "metrics.accuracy >= 0.8"}]},
        )

        assert request.function_name == "optimize_llm"
        assert request.max_trials == 100
        assert request.billing_tier == "standard"
        assert "temperature" in request.configuration_space
        assert request.budget == {"max_cost_usd": 5.0}
        assert request.constraints == {
            "derived": [{"require": "metrics.accuracy >= 0.8"}]
        }

    def test_session_creation_request_accepts_typed_objectives(self):
        request = SessionCreationRequest(
            function_name="optimize_llm",
            configuration_space={
                "model": {"type": "categorical", "choices": ["cheap", "accurate"]},
                "max_tokens": {
                    "type": "int",
                    "low": 64,
                    "high": 256,
                    "conditions": {"model": "accurate"},
                    "default": 64,
                },
            },
            objectives=[
                SessionObjectiveDefinition(
                    metric="accuracy",
                    direction="maximize",
                    weight=2.0,
                ),
                {"metric": "latency", "direction": "minimize", "weight": 1.0},
            ],
            budget={"max_wallclock_ms": 30_000},
            constraints={
                "structural": [
                    {
                        "when": 'params.model == "accurate"',
                        "then": "params.max_tokens >= 64",
                    }
                ]
            },
        )

        assert isinstance(request.objectives[0], SessionObjectiveDefinition)
        assert request.configuration_space["max_tokens"]["default"] == 64
        assert request.budget == {"max_wallclock_ms": 30_000}
        assert request.constraints["structural"][0]["then"] == "params.max_tokens >= 64"

    def test_session_creation_request_accepts_banded_objectives_and_policy(self):
        request = SessionCreationRequest(
            function_name="optimize_llm",
            configuration_space={
                "retrieval_pair": {
                    "type": "categorical",
                    "choices": ["choice_0", "choice_1"],
                    "value_map": {
                        "choice_0": ["dense", "rerank"],
                        "choice_1": ["bm25", "none"],
                    },
                }
            },
            objectives=[
                SessionObjectiveDefinition(
                    metric="response_length",
                    band={"low": 120, "high": 180},
                    test="TOST",
                    alpha=0.05,
                    weight=2.0,
                )
            ],
            default_config={"temperature": 0.7},
            promotion_policy={"dominance": "epsilon_pareto", "alpha": 0.05},
        )

        objective = request.objectives[0]
        assert isinstance(objective, SessionObjectiveDefinition)
        assert objective.band == {"low": 120, "high": 180}
        assert request.default_config == {"temperature": 0.7}
        assert request.promotion_policy == {
            "dominance": "epsilon_pareto",
            "alpha": 0.05,
        }

    def test_next_trial_response(self):
        """Test next trial response."""
        suggestion = TrialSuggestion(
            trial_id="trial-001",
            session_id="session-123",
            trial_number=1,
            config={"temperature": 0.5},
            dataset_subset=DatasetSubsetIndices(
                indices=[1, 2, 3],
                selection_strategy="random",
                confidence_level=0.5,
                estimated_representativeness=0.5,
            ),
            exploration_type="exploration",
        )

        response = NextTrialResponse(
            suggestion=suggestion,
            should_continue=True,
            stop_reason=None,
            session_status=OptimizationSessionStatus.ACTIVE,
        )

        assert response.should_continue
        assert response.suggestion.config["temperature"] == 0.5
        assert response.reason is None
        assert response.stop_reason is None


class TestAgentModels:
    """Test agent-specific models."""

    def test_agent_specification(self):
        """Test agent specification creation."""
        agent = AgentSpecification(
            id="agent-001",
            name="Customer Support Agent",
            agent_type="conversational",
            agent_platform="openai",
            prompt_template="You are a helpful assistant. Query: {query}",
            model_parameters={
                "temperature": 0.7,
                "max_tokens": 1000,
                "model": "GPT-4o",
            },
            reasoning="chain-of-thought",
            style="professional",
            tone="friendly",
        )

        assert agent.id == "agent-001"
        assert agent.model_parameters["temperature"] == 0.7
        assert "{query}" in agent.prompt_template
        assert agent.response_validation is False

    def test_agent_optimization_request(self):
        """Test agent optimization request."""
        agent = AgentSpecification(
            id="agent-001",
            name="Test Agent",
            agent_type="task",
            agent_platform="langchain",
            prompt_template="Process: {input}",
            model_parameters={"temperature": 0.5},
        )

        dataset = Dataset(
            name="test_dataset",
            examples=[
                EvaluationExample(
                    input_data={"query": "test"}, expected_output="response"
                )
            ],
        )

        request = AgentOptimizationRequest(
            agent_spec=agent,
            dataset=dataset,
            configuration_space={
                "model_parameters.temperature": (0.0, 1.0),
                "model_parameters.top_p": (0.1, 0.9),
            },
            objectives=["accuracy"],
            max_trials=25,
        )

        assert request.agent_spec.id == "agent-001"
        assert len(request.dataset.examples) == 1
        assert "model_parameters.temperature" in request.configuration_space
        assert request.target_cost_reduction == 0.65

    def test_agent_optimization_status(self):
        """Test agent optimization status."""
        status = AgentOptimizationStatus(
            optimization_id="opt-123",
            status=OptimizationSessionStatus.ACTIVE,
            progress=0.6,
            completed_trials=30,
            total_trials=50,
            current_best_metrics={"accuracy": 0.92},
            estimated_time_remaining=300.0,
        )

        assert status.progress == 0.6
        assert status.completed_trials == 30
        assert status.current_best_metrics["accuracy"] == 0.92


class TestDatasetSubsetIndices:
    """Test DatasetSubsetIndices model."""

    def test_subset_indices_creation(self):
        """Test creating dataset subset indices."""
        subset = DatasetSubsetIndices(
            indices=[0, 10, 20, 30, 40],
            selection_strategy="stratified_sampling",
            confidence_level=0.95,
            estimated_representativeness=0.9,
            metadata={"strata": ["A", "B", "C", "D", "E"]},
        )

        assert len(subset.indices) == 5
        assert subset.confidence_level == 0.95
        assert subset.selection_strategy == "stratified_sampling"
        assert "strata" in subset.metadata


class TestOptimizationFinalization:
    """Test optimization finalization models."""

    def test_finalization_request(self):
        """Test finalization request."""
        request = OptimizationFinalizationRequest(
            session_id="session-123",
            include_full_history=True,
            metadata={"reason": "user_requested"},
        )

        assert request.session_id == "session-123"
        assert request.include_full_history is True
        assert request.metadata["reason"] == "user_requested"

    def test_finalization_response(self):
        """Test finalization response."""
        response = OptimizationFinalizationResponse(
            session_id="session-123",
            best_config={"temperature": 0.7, "model": "GPT-4o"},
            best_metrics={"accuracy": 0.96, "cost": 0.05},
            total_trials=50,
            successful_trials=48,
            total_duration=3600.0,
            cost_savings=0.65,
            stop_reason="max_trials_reached",
        )

        assert response.best_config["temperature"] == 0.7
        assert response.best_metrics["accuracy"] == 0.96
        assert response.successful_trials == 48
        assert response.cost_savings == 0.65
        assert response.stop_reason == "max_trials_reached"
        assert response.convergence_history is None
