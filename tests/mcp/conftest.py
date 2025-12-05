"""Pytest configuration and fixtures for MCP tests.

Provides shared fixtures, cleanup mechanisms, and test utilities
for MCP endpoint testing.
"""

import json
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest

from traigent.cloud.models import (
    AgentExecutionRequest,
    AgentExecutionResponse,
    AgentOptimizationRequest,
    AgentOptimizationResponse,
    AgentSpecification,
    DatasetSubsetIndices,
    NextTrialRequest,
    NextTrialResponse,
    OptimizationSessionStatus,
    SessionCreationRequest,
    SessionCreationResponse,
    TrialSuggestion,
)
from traigent.evaluators.base import Dataset, EvaluationExample
from traigent.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class MCPTestContext:
    """Context for MCP tests with cleanup tracking."""

    created_agents: list[str]
    created_sessions: list[str]
    temp_files: list[Path]
    active_resources: dict[str, Any]

    def __post_init__(self):
        self.created_agents = []
        self.created_sessions = []
        self.temp_files = []
        self.active_resources = {}


@pytest.fixture
def mcp_test_context():
    """Provide test context with automatic cleanup."""
    context = MCPTestContext([], [], [], {})
    yield context

    # Cleanup all created resources
    for agent_id in context.created_agents:
        logger.debug(f"Cleaning up agent: {agent_id}")

    for session_id in context.created_sessions:
        logger.debug(f"Cleaning up session: {session_id}")

    for temp_file in context.temp_files:
        if temp_file.exists():
            temp_file.unlink()
            logger.debug(f"Removed temp file: {temp_file}")


@pytest.fixture
def sample_dataset():
    """Create a sample dataset for testing."""
    examples = [
        EvaluationExample(
            input_data={"query": "What is Python?"},
            expected_output="Python is a high-level programming language.",
        ),
        EvaluationExample(
            input_data={"query": "How do I install numpy?"},
            expected_output="You can install numpy using pip: pip install numpy",
        ),
        EvaluationExample(
            input_data={"query": "What is machine learning?"},
            expected_output="Machine learning is a subset of AI that enables systems to learn from data.",
        ),
    ]
    return Dataset(examples=examples)


@pytest.fixture
def sample_agent_spec():
    """Create a sample agent specification."""
    return AgentSpecification(
        id="test-agent-001",
        name="Python Q&A Agent",
        agent_type="conversational",
        agent_platform="openai",
        prompt_template="You are a helpful Python assistant. Answer this question: {query}",
        model_parameters={"model": "o4-mini", "temperature": 0.7, "max_tokens": 150},
        persona="knowledgeable Python expert",
        guidelines=["Be concise", "Provide examples when helpful"],
        response_validation=True,
    )


@pytest.fixture
def mock_mcp_service():
    """Mock MCP service endpoints for testing."""

    class MockMCPService:
        def __init__(self):
            self.agents = {}
            self.sessions = {}
            self.trial_counter = 0
            self.call_history = []

        async def create_agent(self, spec: AgentSpecification) -> dict[str, Any]:
            """Mock agent creation."""
            self.call_history.append(("create_agent", spec))
            agent_id = f"agent-{len(self.agents) + 1}"
            self.agents[agent_id] = spec
            return {
                "agent_id": agent_id,
                "status": "created",
                "platform": spec.agent_platform,
            }

        async def execute_agent(
            self, request: AgentExecutionRequest
        ) -> AgentExecutionResponse:
            """Mock agent execution."""
            self.call_history.append(("execute_agent", request))

            # Simulate execution based on input
            output = f"Mock response for: {request.input_data.get('query', 'unknown')}"

            return AgentExecutionResponse(
                output=output,
                duration=0.5,
                tokens_used=50,
                cost=0.001,
                metadata={"mock": True},
            )

        async def create_optimization_session(
            self, request: SessionCreationRequest
        ) -> SessionCreationResponse:
            """Mock session creation."""
            self.call_history.append(("create_optimization_session", request))
            session_id = f"session-{len(self.sessions) + 1}"

            self.sessions[session_id] = {
                "request": request,
                "status": OptimizationSessionStatus.CREATED,
                "trials": [],
            }

            return SessionCreationResponse(
                session_id=session_id,
                status=OptimizationSessionStatus.CREATED,
                optimization_strategy={
                    "initial_exploration": 5,
                    "exploitation_ratio": 0.7,
                },
            )

        async def get_next_trial(self, request: NextTrialRequest) -> NextTrialResponse:
            """Mock next trial suggestion."""
            self.call_history.append(("get_next_trial", request))

            session = self.sessions.get(request.session_id)
            if not session:
                return NextTrialResponse(
                    suggestion=None, should_continue=False, reason="Session not found"
                )

            self.trial_counter += 1

            # Create mock trial suggestion
            suggestion = TrialSuggestion(
                trial_id=f"trial-{self.trial_counter}",
                session_id=request.session_id,
                trial_number=self.trial_counter,
                config={
                    "model": "o4-mini" if self.trial_counter % 2 == 0 else "GPT-4o",
                    "temperature": 0.5 + (self.trial_counter * 0.1),
                },
                dataset_subset=DatasetSubsetIndices(
                    indices=[0, 1],
                    selection_strategy="diverse_sampling",
                    confidence_level=0.8,
                    estimated_representativeness=0.85,
                ),
                exploration_type=(
                    "exploration" if self.trial_counter < 5 else "exploitation"
                ),
            )

            session["trials"].append(suggestion)

            return NextTrialResponse(
                suggestion=suggestion,
                should_continue=self.trial_counter < 10,
                session_status=OptimizationSessionStatus.ACTIVE,
            )

        async def start_agent_optimization(
            self, request: AgentOptimizationRequest
        ) -> AgentOptimizationResponse:
            """Mock agent optimization start."""
            self.call_history.append(("start_agent_optimization", request))

            # Generate unique IDs based on current length + timestamp to ensure uniqueness
            import time

            unique_suffix = (
                f"{len(self.sessions) + 1}_{int(time.time() * 1000) % 10000}"
            )
            session_id = f"opt-session-{unique_suffix}"
            optimization_id = f"opt-{unique_suffix}"

            # Store the session
            self.sessions[session_id] = {
                "request": request,
                "status": "started",
                "optimization_id": optimization_id,
            }

            return AgentOptimizationResponse(
                session_id=session_id,
                optimization_id=optimization_id,
                status="started",
                estimated_cost=5.0,
                estimated_duration=300.0,
                next_steps=[
                    "Analyzing dataset",
                    "Generating trial configurations",
                    "Starting optimization",
                ],
            )

        def get_call_history(self) -> list[tuple]:
            """Get history of all calls made."""
            return self.call_history

        def clear_history(self):
            """Clear call history."""
            self.call_history = []

    return MockMCPService()


@pytest.fixture
def mock_cloud_client(mock_mcp_service):
    """Create a mock cloud client with MCP service."""
    from traigent.cloud.client import TraiGentCloudClient

    # Create client without real HTTP
    client = TraiGentCloudClient(api_key="test-key", enable_fallback=False)

    # Mock the internal methods to use our mock service
    client._submit_optimization = AsyncMock(
        side_effect=mock_mcp_service.start_agent_optimization
    )
    client.create_agent = AsyncMock(side_effect=mock_mcp_service.create_agent)
    client.execute_agent = AsyncMock(side_effect=mock_mcp_service.execute_agent)
    client.create_optimization_session = AsyncMock(
        side_effect=mock_mcp_service.create_optimization_session
    )
    client.get_next_trial = AsyncMock(side_effect=mock_mcp_service.get_next_trial)
    client.start_agent_optimization = AsyncMock(
        side_effect=mock_mcp_service.start_agent_optimization
    )

    # Store reference to mock service
    client._mock_service = mock_mcp_service

    return client


@pytest.fixture
def sample_task_descriptions():
    """Sample natural language task descriptions for LLM interpretation testing."""
    return [
        {
            "description": "Create an agent that answers questions about Python programming",
            "expected_mapping": {
                "action": "create_agent",
                "agent_type": "conversational",
                "platform": "openai",
                "topic": "Python programming",
            },
        },
        {
            "description": "I need to optimize my customer support chatbot to reduce costs",
            "expected_mapping": {
                "action": "optimize_agent",
                "optimization_goal": "cost_reduction",
                "agent_type": "conversational",
            },
        },
        {
            "description": "Execute the Python Q&A agent with the question 'How do I read a file?'",
            "expected_mapping": {
                "action": "execute_agent",
                "agent_name": "Python Q&A agent",
                "input": {"query": "How do I read a file?"},
            },
        },
        {
            "description": "Create a task agent that can search documentation and answer technical questions",
            "expected_mapping": {
                "action": "create_agent",
                "agent_type": "task",
                "capabilities": ["search", "documentation", "technical Q&A"],
            },
        },
        {
            "description": "Start optimizing my translation agent to improve accuracy while keeping costs low",
            "expected_mapping": {
                "action": "optimize_agent",
                "objectives": ["accuracy", "cost"],
                "agent_function": "translation",
            },
        },
    ]


@pytest.fixture
def temp_dataset_file(tmp_path):
    """Create a temporary dataset file for testing."""
    dataset_path = tmp_path / "test_dataset.jsonl"

    examples = [
        {"input": {"text": "Hello"}, "output": "Hi there!"},
        {"input": {"text": "How are you?"}, "output": "I'm doing well, thanks!"},
        {"input": {"text": "Goodbye"}, "output": "See you later!"},
    ]

    with open(dataset_path, "w") as f:
        for example in examples:
            f.write(json.dumps(example) + "\n")

    return dataset_path


@asynccontextmanager
async def cleanup_mcp_resources(context: MCPTestContext):
    """Context manager for automatic MCP resource cleanup."""
    try:
        yield context
    finally:
        # Cleanup all tracked resources
        for resource_type, resource_id in context.active_resources.items():
            logger.info(f"Cleaning up {resource_type}: {resource_id}")

        context.active_resources.clear()


@pytest.fixture
def mock_llm_interpreter():
    """Mock LLM for task interpretation testing."""

    class MockLLMInterpreter:
        def __init__(self):
            self.interpretation_rules = {
                "create.*agent.*answer.*question": {
                    "action": "create_agent",
                    "agent_type": "conversational",
                },
                "optimize.*reduce.*cost": {
                    "action": "optimize_agent",
                    "optimization_goal": "cost_reduction",
                },
                "execute.*agent.*question": {"action": "execute_agent"},
                "run.*agent.*answer": {"action": "execute_agent"},
                "create.*task.*agent": {"action": "create_agent", "agent_type": "task"},
                "need.*task.*agent": {"action": "create_agent", "agent_type": "task"},
                "need.*agent.*customer.*support": {
                    "action": "create_agent",
                    "agent_type": "task",
                },
                "build.*conversational": {
                    "action": "create_agent",
                    "agent_type": "conversational",
                },
                "build.*conversational.*ai": {
                    "action": "create_agent",
                    "agent_type": "conversational",
                },
                "improve.*accuracy": {
                    "action": "optimize_agent",
                    "objectives": ["accuracy"],
                },
                "optimize.*speed.*accuracy": {
                    "action": "optimize_agent",
                    "objectives": ["speed", "accuracy"],
                },
                "run.*customer.*support.*agent": {"action": "execute_agent"},
                "execute.*python.*q.*a.*agent": {"action": "execute_agent"},
                # Additional patterns for test coverage
                "set up.*agent.*code review": {
                    "action": "create_agent",
                    "agent_type": "task",
                },
                "optimize.*translation.*agent": {
                    "action": "optimize_agent",
                    "objectives": ["speed", "accuracy"],
                },
                "create.*code review.*agent": {
                    "action": "create_agent",
                    "agent_type": "task",
                },
                "create.*python.*tutoring.*agent": {
                    "action": "create_agent",
                    "agent_type": "conversational",
                },
                "if.*accuracy.*below.*optimize": {"action": "optimize_agent"},
            }

        async def interpret_task(self, description: str) -> dict[str, Any]:
            """Interpret natural language task description."""
            description_lower = description.lower()

            # Simple pattern matching for testing
            for pattern, mapping in self.interpretation_rules.items():
                import re

                if re.search(pattern, description_lower):
                    result = mapping.copy()

                    # Extract additional context
                    if "python" in description_lower:
                        result["topic"] = "Python programming"
                    if "chatbot" in description_lower:
                        result["agent_type"] = "conversational"
                    if "accuracy" in description_lower:
                        result["objectives"] = result.get("objectives", []) + [
                            "accuracy"
                        ]

                    return result

            return {"action": "unknown", "raw_description": description}

    return MockLLMInterpreter()


# Utility functions for test data generation


def generate_test_dataset(num_examples: int = 10) -> Dataset:
    """Generate a test dataset with specified number of examples."""
    examples = []
    for i in range(num_examples):
        examples.append(
            EvaluationExample(
                input_data={"query": f"Test question {i}"},
                expected_output=f"Test answer {i}",
            )
        )
    return Dataset(examples=examples)


def generate_test_agent_spec(
    name: str = "Test Agent",
    agent_type: str = "conversational",
    platform: str = "openai",
) -> AgentSpecification:
    """Generate a test agent specification."""
    return AgentSpecification(
        id=f"test-{name.lower().replace(' ', '-')}",
        name=name,
        agent_type=agent_type,
        agent_platform=platform,
        prompt_template="Test prompt: {input}",
        model_parameters={"model": "o4-mini", "temperature": 0.7},
    )
