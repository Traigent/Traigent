"""Tests for LLM task interpretation and MCP endpoint mapping.

Verifies that natural language task descriptions are correctly interpreted
and mapped to appropriate MCP calls.
"""

from typing import Any, Dict, List

import pytest

from traigent.cloud.models import (
    AgentExecutionRequest,
    AgentOptimizationRequest,
    AgentSpecification,
)


class TestTaskInterpretation:
    """Test natural language task interpretation."""

    @pytest.mark.asyncio
    async def test_interpret_agent_creation_tasks(self, mock_llm_interpreter):
        """Test interpreting various agent creation requests."""
        test_cases = [
            {
                "description": "Create an agent that answers questions about Python programming",
                "expected": {
                    "action": "create_agent",
                    "agent_type": "conversational",
                    "topic": "Python programming",
                },
            },
            {
                "description": "I need a task agent that can search documentation and answer technical questions",
                "expected": {"action": "create_agent", "agent_type": "task"},
            },
            {
                "description": "Build a conversational AI that helps with customer support",
                "expected": {"action": "create_agent", "agent_type": "conversational"},
            },
            {
                "description": "Set up an agent for code review and analysis",
                "expected": {"action": "create_agent", "agent_type": "task"},
            },
        ]

        for test_case in test_cases:
            result = await mock_llm_interpreter.interpret_task(test_case["description"])

            # Verify core action is correct
            assert result["action"] == test_case["expected"]["action"]

            # Verify other attributes when specified
            if "agent_type" in test_case["expected"]:
                assert result.get("agent_type") == test_case["expected"]["agent_type"]

    @pytest.mark.asyncio
    async def test_interpret_optimization_tasks(self, mock_llm_interpreter):
        """Test interpreting optimization requests."""
        test_cases = [
            {
                "description": "Optimize my chatbot to reduce costs while maintaining quality",
                "expected": {
                    "action": "optimize_agent",
                    "optimization_goal": "cost_reduction",
                    "constraints": ["maintain_quality"],
                },
            },
            {
                "description": "I want to improve the accuracy of my Q&A agent",
                "expected": {"action": "optimize_agent", "objectives": ["accuracy"]},
            },
            {
                "description": "Optimize the translation agent for both speed and accuracy",
                "expected": {
                    "action": "optimize_agent",
                    "objectives": ["speed", "accuracy"],
                },
            },
        ]

        for test_case in test_cases:
            result = await mock_llm_interpreter.interpret_task(test_case["description"])
            assert result["action"] == "optimize_agent"

    @pytest.mark.asyncio
    async def test_interpret_execution_tasks(self, mock_llm_interpreter):
        """Test interpreting agent execution requests."""
        test_cases = [
            {
                "description": "Execute the Python Q&A agent with the question 'How do I read a file?'",
                "expected": {
                    "action": "execute_agent",
                    "input_text": "How do I read a file?",
                },
            },
            {
                "description": "Run my customer support agent to answer 'How can I reset my password?'",
                "expected": {
                    "action": "execute_agent",
                    "input_text": "How can I reset my password?",
                },
            },
        ]

        for test_case in test_cases:
            result = await mock_llm_interpreter.interpret_task(test_case["description"])
            assert result["action"] == "execute_agent"

    @pytest.mark.asyncio
    async def test_ambiguous_task_handling(self, mock_llm_interpreter):
        """Test handling of ambiguous or unclear task descriptions."""
        ambiguous_tasks = [
            "Do something with my agent",
            "Make it better",
            "Help me with AI",
            "I need assistance",
        ]

        for task in ambiguous_tasks:
            result = await mock_llm_interpreter.interpret_task(task)
            # Should either return unknown or ask for clarification
            assert result["action"] == "unknown" or "clarification_needed" in result


class TestEndToEndTaskExecution:
    """Test complete task execution from natural language to MCP calls."""

    @pytest.mark.asyncio
    async def test_create_agent_from_description(
        self, mock_cloud_client, mock_llm_interpreter, mcp_test_context
    ):
        """Test creating an agent from natural language description."""
        # Natural language task
        task_description = (
            "Create an agent that answers questions about Python programming"
        )

        # Step 1: Interpret the task
        interpretation = await mock_llm_interpreter.interpret_task(task_description)
        assert interpretation["action"] == "create_agent"

        # Step 2: Convert interpretation to agent specification
        agent_spec = self._interpretation_to_agent_spec(
            interpretation, task_description
        )

        # Step 3: Execute MCP call
        result = await mock_cloud_client.create_agent(agent_spec)
        mcp_test_context.created_agents.append(result["agent_id"])

        # Verify agent was created
        assert result["status"] == "created"
        assert result["agent_id"] is not None

    @pytest.mark.asyncio
    async def test_optimize_agent_from_description(
        self, mock_cloud_client, mock_llm_interpreter, sample_dataset
    ):
        """Test optimizing an agent from natural language description."""
        # Natural language task
        task_description = "Optimize my customer support chatbot to reduce costs by 50% while maintaining accuracy"

        # Step 1: Interpret the task
        interpretation = await mock_llm_interpreter.interpret_task(task_description)
        assert interpretation["action"] == "optimize_agent"
        assert interpretation["optimization_goal"] == "cost_reduction"

        # Step 2: Create optimization request
        # First, create a dummy agent
        agent_spec = AgentSpecification(
            id="support-bot",
            name="Customer Support Bot",
            agent_type="conversational",
            agent_platform="openai",
            prompt_template="Support template: {query}",
            model_parameters={"model": "GPT-4o", "temperature": 0.7},
        )

        # Create optimization request based on interpretation
        opt_request = AgentOptimizationRequest(
            agent_spec=agent_spec,
            dataset=sample_dataset,
            configuration_space={
                "model": ["o4-mini", "GPT-4o"],  # Include cheaper model
                "temperature": (0.3, 0.8),
                "max_tokens": [100, 150, 200],
            },
            objectives=["accuracy", "cost"],
            target_cost_reduction=0.5,  # 50% as mentioned in task
            max_trials=30,
        )

        # Step 3: Execute optimization
        result = await mock_cloud_client.start_agent_optimization(opt_request)
        assert result.status == "started"
        assert result.estimated_cost is not None

    @pytest.mark.asyncio
    async def test_execute_agent_from_description(
        self, mock_cloud_client, mock_llm_interpreter
    ):
        """Test executing an agent from natural language description."""
        # Natural language task
        task_description = (
            "Run the Python Q&A agent to answer 'What are decorators in Python?'"
        )

        # Step 1: Interpret the task
        interpretation = await mock_llm_interpreter.interpret_task(task_description)
        assert interpretation["action"] == "execute_agent"

        # Step 2: Create execution request
        # Create a dummy agent spec for execution
        agent_spec = AgentSpecification(
            id="python-qa",
            name="Python Q&A Agent",
            agent_type="conversational",
            agent_platform="openai",
            prompt_template="Answer this Python question: {query}",
            model_parameters={"model": "o4-mini", "temperature": 0.5},
        )

        # Extract query from task description
        query = "What are decorators in Python?"

        exec_request = AgentExecutionRequest(
            agent_spec=agent_spec, input_data={"query": query}
        )

        # Step 3: Execute
        result = await mock_cloud_client.execute_agent(exec_request)
        assert result.error is None
        assert result.output is not None

    def _interpretation_to_agent_spec(
        self, interpretation: Dict[str, Any], original_task: str
    ) -> AgentSpecification:
        """Convert task interpretation to agent specification."""
        # Extract agent properties from interpretation
        agent_type = interpretation.get("agent_type", "conversational")
        topic = interpretation.get("topic", "general")

        # Generate agent specification
        return AgentSpecification(
            id=f"{topic.lower().replace(' ', '-')}-agent",
            name=f"{topic} Agent",
            agent_type=agent_type,
            agent_platform="openai",  # Default platform
            prompt_template=self._generate_prompt_template(agent_type, topic),
            model_parameters={
                "model": "o4-mini",
                "temperature": 0.7,
                "max_tokens": 150,
            },
            persona=(
                f"knowledgeable {topic} expert"
                if topic != "general"
                else "helpful assistant"
            ),
            guidelines=self._generate_guidelines(agent_type, topic),
        )

    def _generate_prompt_template(self, agent_type: str, topic: str) -> str:
        """Generate appropriate prompt template based on agent type and topic."""
        if agent_type == "conversational":
            return f"You are an expert in {topic}. Answer this question: {{query}}"
        elif agent_type == "task":
            return f"Complete this {topic} task: {{task_description}}"
        else:
            return "Process this input: {input}"

    def _generate_guidelines(self, agent_type: str, topic: str) -> List[str]:
        """Generate appropriate guidelines for the agent."""
        base_guidelines = ["Be helpful and accurate", "Provide clear explanations"]

        if agent_type == "conversational":
            base_guidelines.extend(
                ["Be conversational", "Ask for clarification if needed"]
            )
        elif agent_type == "task":
            base_guidelines.extend(["Focus on task completion", "Be efficient"])

        if "programming" in topic.lower() or "code" in topic.lower():
            base_guidelines.append("Include code examples when relevant")

        return base_guidelines


class TestComplexTaskInterpretation:
    """Test interpretation of complex, multi-step tasks."""

    @pytest.mark.asyncio
    async def test_compound_task_interpretation(self, mock_llm_interpreter):
        """Test interpreting tasks that involve multiple actions."""
        compound_task = (
            "Create a Python tutoring agent, test it with some basic questions, "
            "then optimize it for better accuracy and lower costs"
        )

        # In a real implementation, this might return multiple actions
        result = await mock_llm_interpreter.interpret_task(compound_task)

        # The mock returns single action, but real implementation could return:
        # actions = [
        #     {"action": "create_agent", ...},
        #     {"action": "execute_agent", ...},
        #     {"action": "optimize_agent", ...}
        # ]
        assert result["action"] in ["create_agent", "optimize_agent", "execute_agent"]

    @pytest.mark.asyncio
    async def test_conditional_task_interpretation(self, mock_llm_interpreter):
        """Test interpreting tasks with conditions."""
        conditional_task = (
            "If the customer support agent's accuracy is below 80%, "
            "optimize it for better performance"
        )

        result = await mock_llm_interpreter.interpret_task(conditional_task)

        # Should recognize optimization intent
        assert "optimize" in result["action"] or result["action"] == "optimize_agent"

    @pytest.mark.asyncio
    async def test_task_with_specific_requirements(self, mock_llm_interpreter):
        """Test interpreting tasks with specific requirements."""
        specific_task = (
            "Create a code review agent that uses GPT-4, "
            "has a temperature of 0.2, and focuses on Python best practices"
        )

        result = await mock_llm_interpreter.interpret_task(specific_task)
        assert result["action"] == "create_agent"

        # In a real implementation, it should extract:
        # - model: "GPT-4o"
        # - temperature: 0.2
        # - focus: "Python best practices"


class TestTaskValidation:
    """Test validation of interpreted tasks before execution."""

    def test_validate_agent_creation_params(self):
        """Test validation of agent creation parameters."""
        valid_params = {
            "action": "create_agent",
            "agent_type": "conversational",
            "platform": "openai",
            "name": "Test Agent",
        }

        # Should pass validation
        assert self._validate_task_params(valid_params, "create_agent")

        # Missing required field - but still has minimum required
        minimal_params = {
            "action": "create_agent",
            "agent_type": "conversational",
            # Missing platform but still valid
        }

        assert self._validate_task_params(minimal_params, "create_agent")

        # Actually invalid - missing agent_type
        invalid_params = {
            "action": "create_agent"
            # Missing agent_type
        }

        assert not self._validate_task_params(invalid_params, "create_agent")

    def test_validate_optimization_params(self):
        """Test validation of optimization parameters."""
        valid_params = {
            "action": "optimize_agent",
            "objectives": ["accuracy", "cost"],
            "max_trials": 30,
        }

        assert self._validate_task_params(valid_params, "optimize_agent")

        # Invalid objective
        invalid_params = {
            "action": "optimize_agent",
            "objectives": [],  # Empty objectives
        }

        assert not self._validate_task_params(invalid_params, "optimize_agent")

    def _validate_task_params(
        self, params: Dict[str, Any], expected_action: str
    ) -> bool:
        """Validate task parameters for a given action."""
        if params.get("action") != expected_action:
            return False

        if expected_action == "create_agent":
            required = ["agent_type"]
            return all(field in params for field in required)

        elif expected_action == "optimize_agent":
            return "objectives" in params and len(params["objectives"]) > 0

        elif expected_action == "execute_agent":
            return True  # Minimal validation for execution

        return False


class TestErrorMessages:
    """Test helpful error messages for failed interpretations."""

    @pytest.mark.asyncio
    async def test_unclear_task_error_messages(self, mock_llm_interpreter):
        """Test that unclear tasks produce helpful error messages."""
        unclear_tasks = [
            (
                "Make it work",
                "Task is too vague. Please specify what you want to create, optimize, or execute.",
            ),
            (
                "Do AI stuff",
                "Please be more specific about the AI task you want to perform.",
            ),
            ("Help", "Please describe what kind of help you need with your AI agent."),
        ]

        for task, _expected_hint in unclear_tasks:
            result = await mock_llm_interpreter.interpret_task(task)

            # In a real implementation, should provide helpful hints
            if result["action"] == "unknown":
                # Check that some form of guidance is provided
                assert "raw_description" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
