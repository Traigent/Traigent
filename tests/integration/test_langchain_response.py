#!/usr/bin/env python3
"""Test what LangChain returns to understand response structure."""

import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Mock mode to avoid API calls
os.environ["TRAIGENT_MOCK_MODE"] = "true"

try:
    from langchain_anthropic import ChatAnthropic
    from langchain_core.messages import HumanMessage

    # Create a mock LLM
    llm = ChatAnthropic(
        model="claude-3-haiku-20240307",
        temperature=0.0,
        max_tokens=50,
    )

    # Try to invoke it
    messages = [HumanMessage(content="Test")]
    response = llm.invoke(messages)

    print(f"Response type: {type(response)}")
    print(f"Response attributes: {dir(response)}")

    # Check for usage/token info
    if hasattr(response, "usage_metadata"):
        print(f"Usage metadata: {response.usage_metadata}")
    if hasattr(response, "response_metadata"):
        print(f"Response metadata: {response.response_metadata}")

except Exception as e:
    print(f"Error: {e}")
    print("\nLangChain not available or mock mode doesn't work with it")
