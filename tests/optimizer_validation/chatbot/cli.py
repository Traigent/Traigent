#!/usr/bin/env python3
"""CLI entry point for the test chatbot.

This module provides a command-line interface for interacting with
the test knowledge graph using Claude.

Usage:
    # Interactive mode
    python -m tests.optimizer_validation.chatbot.cli

    # Single query mode
    python -m tests.optimizer_validation.chatbot.cli -q "Which tests use optuna?"

    # Use specific model
    python -m tests.optimizer_validation.chatbot.cli --model claude-opus-4-20250514
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def get_tool_function(name: str) -> Any:
    """Get a tool function by name."""
    from tests.optimizer_validation.chatbot import tools

    tool_map = {
        "query_knowledge_graph": tools.query_knowledge_graph,
        "list_dimensions": tools.list_dimensions,
        "get_test_details": tools.get_test_details,
        "search_tests": tools.search_tests,
        "get_coverage_matrix": tools.get_coverage_matrix,
        "get_coverage_gaps": tools.get_coverage_gaps,
        "get_test_stats": tools.get_test_stats,
    }
    return tool_map.get(name)


def execute_tool(name: str, args: dict[str, Any]) -> str:
    """Execute a tool and return JSON result."""
    func = get_tool_function(name)
    if not func:
        return json.dumps({"error": f"Unknown tool: {name}"})

    try:
        result = func(**args)
        return json.dumps(result, indent=2, default=str)
    except Exception as e:
        return json.dumps({"error": str(e)})


def chat_loop(model: str = "claude-sonnet-4-20250514") -> None:
    """Run interactive chat loop.

    Args:
        model: The Claude model to use
    """
    try:
        from anthropic import Anthropic
    except ImportError:
        print("Error: anthropic package not installed.")
        print("Install with: pip install anthropic")
        sys.exit(1)

    from tests.optimizer_validation.chatbot.prompts import SYSTEM_PROMPT, TOOL_SCHEMAS

    client = Anthropic()
    messages: list[dict[str, Any]] = []

    print("=" * 60)
    print("TraiGent Test Suite Chatbot")
    print("=" * 60)
    print(f"Model: {model}")
    print("Type 'quit' or 'exit' to exit, 'help' for usage tips")
    print("-" * 60)

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        if user_input.lower() == "help":
            print_help()
            continue

        if user_input.lower() == "stats":
            # Quick stats shortcut
            from tests.optimizer_validation.chatbot.tools import get_test_stats

            stats = get_test_stats()
            print("\nTest Suite Stats:")
            print(f"  Total tests: {stats['total_tests']}")
            print(f"  Passed: {stats['passed']}")
            print(f"  Failed: {stats['failed']}")
            print(f"  Not run: {stats['not_run']}")
            continue

        if user_input.lower() == "dimensions":
            # Quick dimensions shortcut
            from tests.optimizer_validation.chatbot.tools import list_dimensions

            dims = list_dimensions()
            print("\nTest Dimensions:")
            for dim, values in dims.items():
                print(f"  {dim}: {', '.join(values)}")
            continue

        messages.append({"role": "user", "content": user_input})

        # Get response with potential tool use
        try:
            response = process_conversation(
                client, model, messages, SYSTEM_PROMPT, TOOL_SCHEMAS
            )
            print(f"\nAssistant: {response}")
        except Exception as e:
            print(f"\nError: {e}")
            messages.pop()  # Remove failed message


def process_conversation(
    client: Any,
    model: str,
    messages: list[dict[str, Any]],
    system_prompt: str,
    tool_schemas: list[dict[str, Any]],
) -> str:
    """Process a conversation turn, handling tool calls.

    Args:
        client: Anthropic client
        model: Model to use
        messages: Conversation history
        system_prompt: System prompt
        tool_schemas: Tool definitions

    Returns:
        Final text response
    """
    max_iterations = 10  # Prevent infinite loops

    for _ in range(max_iterations):
        response = client.messages.create(
            model=model,
            max_tokens=4096,
            system=system_prompt,
            tools=tool_schemas,
            messages=messages,
        )

        # Check for tool use
        tool_uses = [block for block in response.content if block.type == "tool_use"]

        if not tool_uses:
            # No tool use, return text response
            messages.append({"role": "assistant", "content": response.content})
            text_blocks = [
                block.text for block in response.content if hasattr(block, "text")
            ]
            return "\n".join(text_blocks)

        # Execute tools and continue
        messages.append({"role": "assistant", "content": response.content})

        tool_results = []
        for tool_use in tool_uses:
            print(f"  [Calling {tool_use.name}...]")
            result = execute_tool(tool_use.name, tool_use.input)
            tool_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": tool_use.id,
                    "content": result,
                }
            )

        messages.append({"role": "user", "content": tool_results})

    return "Max iterations reached. Please try a simpler query."


def single_query(query: str, model: str = "claude-sonnet-4-20250514") -> None:
    """Execute a single query and print result.

    Args:
        query: The query to execute
        model: The Claude model to use
    """
    try:
        from anthropic import Anthropic
    except ImportError:
        print("Error: anthropic package not installed.")
        print("Install with: pip install anthropic")
        sys.exit(1)

    from tests.optimizer_validation.chatbot.prompts import SYSTEM_PROMPT, TOOL_SCHEMAS

    client = Anthropic()
    messages = [{"role": "user", "content": query}]

    try:
        response = process_conversation(
            client, model, messages, SYSTEM_PROMPT, TOOL_SCHEMAS
        )
        print(response)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


def print_help() -> None:
    """Print usage help."""
    print(
        """
TraiGent Test Suite Chatbot - Help
==================================

Quick Commands:
  stats       - Show test suite statistics
  dimensions  - List all dimensions and values
  help        - Show this help message
  quit/exit   - Exit the chatbot

Example Questions:
  "Which tests use optuna?"
  "Show me all failed tests"
  "What coverage gaps exist between injection modes and algorithms?"
  "Tell me about the test called empty_dataset"
  "How many tests do we have for each algorithm?"
  "Find tests related to parallel execution"
  "Which tests use context injection with edge_analytics?"

Tips:
  - Be specific about what you're looking for
  - Use dimension names (InjectionMode, Algorithm, etc.) for precise queries
  - Ask about coverage gaps to find missing test scenarios
"""
    )


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="TraiGent Test Suite Chatbot",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                              # Interactive mode
  %(prog)s -q "Which tests use optuna?" # Single query
  %(prog)s --model claude-opus-4-20250514     # Use Opus model
        """,
    )
    parser.add_argument(
        "-q",
        "--query",
        help="Single query to execute (non-interactive mode)",
    )
    parser.add_argument(
        "--model",
        default="claude-sonnet-4-20250514",
        help="Claude model to use (default: claude-sonnet-4-20250514)",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Show test suite statistics and exit",
    )
    parser.add_argument(
        "--dimensions",
        action="store_true",
        help="List all dimensions and their values",
    )

    args = parser.parse_args()

    if args.stats:
        from tests.optimizer_validation.chatbot.tools import get_test_stats

        stats = get_test_stats()
        print(json.dumps(stats, indent=2))
        return

    if args.dimensions:
        from tests.optimizer_validation.chatbot.tools import list_dimensions

        dims = list_dimensions()
        print(json.dumps(dims, indent=2))
        return

    if args.query:
        single_query(args.query, args.model)
    else:
        chat_loop(args.model)


if __name__ == "__main__":
    main()
