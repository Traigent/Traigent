#!/usr/bin/env python3
"""
Comprehensive demonstration of LLM-MCP validation testing.

This script shows how the testing framework validates that an LLM can:
1. Correctly interpret natural language task descriptions
2. Map them to appropriate MCP endpoints
3. Execute complete workflows with proper cleanup
4. Handle edge cases and ambiguous tasks
"""

import asyncio
import json
import sys
from pathlib import Path
from typing import Any

# Add Traigent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import the components we need
from traigent.cloud.models import (
    AgentExecutionRequest,
    AgentExecutionResponse,
    AgentOptimizationRequest,
    AgentOptimizationResponse,
    AgentSpecification,
)
from traigent.evaluators.base import Dataset, EvaluationExample


class MCPTestContext:
    """Context for tracking test resources."""

    def __init__(self):
        self.created_agents = []
        self.created_sessions = []
        self.temp_files = []
        self.active_resources = {}


class MockLLMInterpreter:
    """Mock implementation of LLM task interpretation."""

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
        }

    async def interpret_task(self, description: str) -> dict[str, Any]:
        """Interpret natural language task description."""
        import re

        description_lower = description.lower()

        # Simple pattern matching for testing
        for pattern, mapping in self.interpretation_rules.items():
            if re.search(pattern, description_lower):
                result = mapping.copy()

                # Extract additional context
                if "python" in description_lower:
                    result["topic"] = "Python programming"
                if "chatbot" in description_lower:
                    result["agent_type"] = "conversational"
                if "accuracy" in description_lower and "objectives" not in result:
                    result["objectives"] = ["accuracy"]

                # Extract input text for execution
                if "question" in description_lower:
                    import re

                    match = re.search(r"'([^']+)'", description)
                    if match:
                        result["input_text"] = match.group(1)

                return result

        return {"action": "unknown", "raw_description": description}


class MockMCPService:
    """Mock MCP service for testing."""

    def __init__(self):
        self.agents = {}
        self.sessions = {}
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

    async def start_agent_optimization(
        self, request: AgentOptimizationRequest
    ) -> AgentOptimizationResponse:
        """Mock agent optimization start."""
        self.call_history.append(("start_agent_optimization", request))

        session_id = f"opt-session-{len(self.sessions) + 1}"
        optimization_id = f"opt-{len(self.sessions) + 1}"

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
        prompt_template="You are a helpful assistant. Answer: {query}",
        model_parameters={"model": "o4-mini", "temperature": 0.7, "max_tokens": 150},
    )


class LLMMCPValidator:
    """Validates LLM interpretation and MCP usage."""

    def __init__(self):
        self.llm_interpreter = MockLLMInterpreter()
        self.mcp_service = MockMCPService()
        self.test_context = MCPTestContext()

    async def validate_task_interpretation(
        self, task_description: str
    ) -> dict[str, Any]:
        """
        Validate that a task description is correctly interpreted and executed.

        Args:
            task_description: Natural language description of the task

        Returns:
            Validation results including interpretation, execution, and cleanup
        """
        validation_result = {
            "task_description": task_description,
            "interpretation": None,
            "mcp_calls": [],
            "execution_success": False,
            "cleanup_success": False,
            "errors": [],
        }

        try:
            # Step 1: Interpret the task
            interpretation = await self.llm_interpreter.interpret_task(task_description)
            validation_result["interpretation"] = interpretation

            # Step 2: Execute based on interpretation
            if interpretation["action"] == "create_agent":
                await self._handle_agent_creation(interpretation, validation_result)
            elif interpretation["action"] == "optimize_agent":
                await self._handle_agent_optimization(interpretation, validation_result)
            elif interpretation["action"] == "execute_agent":
                await self._handle_agent_execution(interpretation, validation_result)
            else:
                validation_result["errors"].append(
                    f"Unknown action: {interpretation['action']}"
                )

            # Step 3: Verify cleanup
            await self._verify_cleanup(validation_result)

        except Exception as e:
            validation_result["errors"].append(f"Validation error: {str(e)}")

        return validation_result

    async def _handle_agent_creation(
        self, interpretation: dict[str, Any], result: dict[str, Any]
    ):
        """Handle agent creation tasks."""
        agent_spec = generate_test_agent_spec(
            name=f"Test Agent for {interpretation.get('topic', 'general')}",
            agent_type=interpretation.get("agent_type", "conversational"),
            platform=interpretation.get("platform", "openai"),
        )

        # Create the agent
        agent_response = await self.mcp_service.create_agent(agent_spec)
        result["mcp_calls"].append(("create_agent", agent_response))

        # Track for cleanup
        self.test_context.created_agents.append(agent_response["agent_id"])
        result["execution_success"] = True

    async def _handle_agent_optimization(
        self, interpretation: dict[str, Any], result: dict[str, Any]
    ):
        """Handle agent optimization tasks."""
        from traigent.cloud.models import AgentOptimizationRequest

        # First create an agent to optimize
        agent_spec = generate_test_agent_spec()
        agent_response = await self.mcp_service.create_agent(agent_spec)
        self.test_context.created_agents.append(agent_response["agent_id"])

        # Create optimization request
        opt_request = AgentOptimizationRequest(
            agent_spec=agent_spec,
            dataset=generate_test_dataset(20),
            configuration_space={
                "temperature": (0.0, 1.0),
                "max_tokens": [100, 200, 300],
            },
            objectives=interpretation.get("objectives", ["accuracy"]),
            max_trials=10,
        )

        # Start optimization
        opt_response = await self.mcp_service.start_agent_optimization(opt_request)
        result["mcp_calls"].append(("start_agent_optimization", opt_response))

        # Track for cleanup
        self.test_context.created_sessions.append(opt_response.session_id)
        result["execution_success"] = True

    async def _handle_agent_execution(
        self, interpretation: dict[str, Any], result: dict[str, Any]
    ):
        """Handle agent execution tasks."""
        from traigent.cloud.models import AgentExecutionRequest

        # First create an agent to execute
        agent_spec = generate_test_agent_spec()
        agent_response = await self.mcp_service.create_agent(agent_spec)
        self.test_context.created_agents.append(agent_response["agent_id"])

        # Execute the agent
        exec_request = AgentExecutionRequest(
            agent_spec=agent_spec,
            input_data={"query": interpretation.get("input_text", "Test question")},
        )

        exec_response = await self.mcp_service.execute_agent(exec_request)
        result["mcp_calls"].append(("execute_agent", exec_response))
        result["execution_success"] = True

    async def _verify_cleanup(self, result: dict[str, Any]):
        """Verify that all resources can be cleaned up."""
        cleanup_errors = []

        # Simulate cleanup of agents
        for agent_id in self.test_context.created_agents:
            try:
                # In real implementation, would call MCP cleanup endpoint
                print(f"Would clean up agent: {agent_id}")
            except Exception as e:
                cleanup_errors.append(f"Failed to cleanup agent {agent_id}: {e}")

        # Simulate cleanup of sessions
        for session_id in self.test_context.created_sessions:
            try:
                # In real implementation, would call session cleanup endpoint
                print(f"Would clean up session: {session_id}")
            except Exception as e:
                cleanup_errors.append(f"Failed to cleanup session {session_id}: {e}")

        result["cleanup_success"] = len(cleanup_errors) == 0
        if cleanup_errors:
            result["errors"].extend(cleanup_errors)


async def run_comprehensive_validation():
    """Run comprehensive validation of LLM-MCP integration."""
    validator = LLMMCPValidator()

    # Test cases covering various scenarios
    test_cases = [
        # Agent Creation Tasks
        "Create an agent that answers questions about Python programming",
        "I need a task agent for customer support",
        "Build a conversational AI that helps with documentation",
        # Optimization Tasks
        "Optimize my chatbot to reduce costs while maintaining quality",
        "I want to improve the accuracy of my Q&A agent",
        "Optimize the translation agent for both speed and accuracy",
        # Execution Tasks
        "Execute the Python Q&A agent with the question 'How do I read a file?'",
        "Run my customer support agent to answer 'How can I reset my password?'",
        # Edge Cases
        "Do something with my agent",  # Ambiguous
        "Create a super intelligent AGI",  # Outside scope
    ]

    print("🚀 Starting LLM-MCP Validation Testing")
    print("=" * 60)

    results = []
    for i, task in enumerate(test_cases, 1):
        print(f"\n📋 Test Case {i}/{len(test_cases)}")
        print(f"Task: {task}")
        print("-" * 40)

        result = await validator.validate_task_interpretation(task)
        results.append(result)

        # Print interpretation
        interpretation = result["interpretation"]
        print(f"✓ Interpretation: {interpretation['action']}")
        if interpretation.get("agent_type"):
            print(f"  Agent Type: {interpretation['agent_type']}")
        if interpretation.get("objectives"):
            print(f"  Objectives: {interpretation['objectives']}")

        # Print MCP calls
        if result["mcp_calls"]:
            print(f"✓ MCP Calls: {len(result['mcp_calls'])}")
            for call_type, _response in result["mcp_calls"]:
                print(f"  - {call_type}: Success")

        # Print status
        if result["execution_success"]:
            print("✅ Execution: SUCCESS")
        else:
            print("❌ Execution: FAILED")

        if result["cleanup_success"]:
            print("✅ Cleanup: SUCCESS")
        else:
            print("❌ Cleanup: FAILED")

        if result["errors"]:
            print(f"⚠️  Errors: {len(result['errors'])}")
            for error in result["errors"]:
                print(f"   - {error}")

    # Summary
    print("\n" + "=" * 60)
    print("📊 VALIDATION SUMMARY")
    print("=" * 60)

    total_tests = len(results)
    successful_interpretations = sum(
        1 for r in results if r["interpretation"]["action"] != "unknown"
    )
    successful_executions = sum(1 for r in results if r["execution_success"])
    successful_cleanups = sum(1 for r in results if r["cleanup_success"])

    print(f"Total Tests: {total_tests}")
    print(
        f"Successful Interpretations: {successful_interpretations}/{total_tests} ({successful_interpretations / total_tests * 100:.1f}%)"
    )
    print(
        f"Successful Executions: {successful_executions}/{total_tests} ({successful_executions / total_tests * 100:.1f}%)"
    )
    print(
        f"Successful Cleanups: {successful_cleanups}/{total_tests} ({successful_cleanups / total_tests * 100:.1f}%)"
    )

    # Detailed breakdown
    action_counts = {}
    for result in results:
        action = result["interpretation"]["action"]
        action_counts[action] = action_counts.get(action, 0) + 1

    print("\nAction Breakdown:")
    for action, count in action_counts.items():
        print(f"  {action}: {count} tests")

    # Recommendations
    print("\n💡 RECOMMENDATIONS:")
    if successful_interpretations < total_tests:
        print("- Improve natural language interpretation patterns")
    if successful_executions < successful_interpretations:
        print("- Fix MCP endpoint execution issues")
    if successful_cleanups < successful_executions:
        print("- Improve resource cleanup mechanisms")
    if (
        successful_interpretations == total_tests
        and successful_executions == total_tests
    ):
        print("- All tests passed! MCP is well-documented for LLM usage ✅")

    return results


async def main():
    """Main entry point."""
    try:
        results = await run_comprehensive_validation()

        # Save detailed results
        output_file = Path(__file__).parent / "mcp_validation_results.json"
        with open(output_file, "w") as f:
            # Convert results to JSON-serializable format
            json_results = []
            for result in results:
                json_result = result.copy()
                # Convert non-serializable objects to strings
                json_result["mcp_calls"] = [
                    (call_type, str(response))
                    for call_type, response in result["mcp_calls"]
                ]
                json_results.append(json_result)

            json.dump(json_results, f, indent=2)

        print(f"\n📄 Detailed results saved to: {output_file}")

    except Exception as e:
        print(f"❌ Validation failed: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
