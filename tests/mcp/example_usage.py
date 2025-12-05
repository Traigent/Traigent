"""Example usage of the MCP testing framework.

This file demonstrates how to use the MCP testing framework
for testing natural language task interpretation and execution.
"""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tests.mcp.conftest import (
    MockLLMInterpreter,
    MockMCPService,
    generate_test_agent_spec,
    generate_test_dataset,
)
from traigent.cloud.models import (
    AgentExecutionRequest,
    AgentOptimizationRequest,
    AgentSpecification,
)


async def example_natural_language_to_mcp():
    """Example: Convert natural language to MCP calls."""
    print("Example 1: Natural Language to MCP Calls")
    print("=" * 50)

    # Initialize mock LLM interpreter
    interpreter = MockLLMInterpreter()
    mcp_service = MockMCPService()

    # Natural language task
    task = "Create an agent that answers questions about Python programming"
    print(f"Task: {task}")

    # Step 1: Interpret the task
    interpretation = await interpreter.interpret_task(task)
    print(f"\nInterpretation: {interpretation}")

    # Step 2: Convert to agent specification
    if interpretation["action"] == "create_agent":
        agent_spec = AgentSpecification(
            id="python-qa-agent",
            name="Python Q&A Agent",
            agent_type=interpretation.get("agent_type", "conversational"),
            agent_platform="openai",
            prompt_template="You are a Python expert. Answer: {query}",
            model_parameters={"model": "o4-mini", "temperature": 0.7},
            persona="knowledgeable Python expert",
        )

        # Step 3: Create the agent
        result = await mcp_service.create_agent(agent_spec)
        print(f"\nAgent created: {result}")

        # Step 4: Test the agent
        exec_request = AgentExecutionRequest(
            agent_spec=agent_spec, input_data={"query": "What are Python decorators?"}
        )

        exec_result = await mcp_service.execute_agent(exec_request)
        print(f"\nExecution result: {exec_result.output}")


async def example_optimization_workflow():
    """Example: Complete optimization workflow."""
    print("\n\nExample 2: Optimization Workflow")
    print("=" * 50)

    mcp_service = MockMCPService()

    # Create test dataset
    dataset = generate_test_dataset(num_examples=20)

    # Create agent to optimize
    agent_spec = generate_test_agent_spec(
        name="Customer Support Bot", agent_type="conversational", platform="openai"
    )

    # Start optimization
    opt_request = AgentOptimizationRequest(
        agent_spec=agent_spec,
        dataset=dataset,
        configuration_space={
            "model": ["o4-mini", "GPT-4o"],
            "temperature": (0.1, 0.9),
            "max_tokens": [100, 150, 200],
        },
        objectives=["accuracy", "cost"],
        max_trials=10,
        target_cost_reduction=0.5,
    )

    opt_response = await mcp_service.start_agent_optimization(opt_request)
    print(f"Optimization started: {opt_response}")

    # Get optimization session details
    print(f"\nSession ID: {opt_response.session_id}")
    print(f"Estimated cost: ${opt_response.estimated_cost}")
    print(f"Estimated duration: {opt_response.estimated_duration}s")
    print(f"Next steps: {opt_response.next_steps}")


async def example_batch_testing():
    """Example: Batch testing multiple queries."""
    print("\n\nExample 3: Batch Testing")
    print("=" * 50)

    mcp_service = MockMCPService()

    # Create agent
    agent_spec = generate_test_agent_spec(name="Multi-Query Agent")

    # Test queries
    test_queries = [
        {"query": "What is Python?"},
        {"query": "How do I use lists?"},
        {"query": "Explain generators"},
        {"query": "What are decorators?"},
        {"query": "How does async/await work?"},
    ]

    print(f"Testing {len(test_queries)} queries...")

    # Execute batch
    results = []
    for i, query_data in enumerate(test_queries):
        exec_request = AgentExecutionRequest(
            agent_spec=agent_spec, input_data=query_data
        )

        result = await mcp_service.execute_agent(exec_request)
        results.append(result)
        print(f"  Query {i + 1}: ✓ ({result.duration:.2f}s)")

    # Summary
    total_tokens = sum(r.tokens_used or 0 for r in results)
    total_cost = sum(r.cost or 0 for r in results)
    avg_duration = sum(r.duration for r in results) / len(results)

    print("\nBatch Summary:")
    print(f"  Total tokens: {total_tokens}")
    print(f"  Total cost: ${total_cost:.4f}")
    print(f"  Average duration: {avg_duration:.2f}s")


async def example_call_tracking():
    """Example: Track all MCP calls made during testing."""
    print("\n\nExample 4: Call Tracking")
    print("=" * 50)

    mcp_service = MockMCPService()

    # Make various calls
    agent_spec = generate_test_agent_spec()

    # Create agent
    await mcp_service.create_agent(agent_spec)

    # Execute agent
    await mcp_service.execute_agent(
        AgentExecutionRequest(agent_spec=agent_spec, input_data={"query": "test"})
    )

    # Start optimization
    await mcp_service.start_agent_optimization(
        AgentOptimizationRequest(
            agent_spec=agent_spec,
            dataset=generate_test_dataset(5),
            configuration_space={"model": ["o4-mini"]},
            objectives=["accuracy"],
        )
    )

    # Get call history
    history = mcp_service.get_call_history()

    print("MCP Call History:")
    for i, (method, args) in enumerate(history):
        print(f"  {i + 1}. {method}")
        if hasattr(args, "id"):
            print(f"     - ID: {args.id}")
        if hasattr(args, "agent_spec"):
            print(f"     - Agent: {args.agent_spec.name}")


async def main():
    """Run all examples."""
    print("MCP Testing Framework Examples")
    print("=" * 70)

    await example_natural_language_to_mcp()
    await example_optimization_workflow()
    await example_batch_testing()
    await example_call_tracking()

    print("\n" + "=" * 70)
    print("Examples completed successfully!")


if __name__ == "__main__":
    asyncio.run(main())
