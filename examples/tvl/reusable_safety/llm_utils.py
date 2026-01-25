"""LLM utilities using LiteLLM for unified Groq/OpenAI access.

This module provides a simple interface for making LLM calls via LiteLLM,
which supports multiple providers (Groq, OpenAI, Anthropic, etc.) through
a unified API.

Usage:
    from llm_utils import call_llm

    response = call_llm(
        messages=[{"role": "user", "content": "Hello!"}],
        model="groq/llama-3.3-70b-versatile",
        temperature=0.3,
    )
    print(response["text"])
"""

from __future__ import annotations

import logging
import os
from typing import Any

# Silence LiteLLM's verbose logging before importing
os.environ["LITELLM_LOG"] = "ERROR"  # Suppress info/debug messages
logging.getLogger("LiteLLM").setLevel(logging.ERROR)
logging.getLogger("litellm").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.WARNING)

# LiteLLM provides a unified interface for multiple LLM providers
try:
    import litellm
    from litellm import completion

    LITELLM_AVAILABLE = True
    litellm.set_verbose = False
    litellm.suppress_debug_info = True
except ImportError:
    LITELLM_AVAILABLE = False
    litellm = None  # type: ignore
    completion = None  # type: ignore

# Default models for agent and judge calls
DEFAULT_CHAT_MODEL = "groq/llama-3.3-70b-versatile"  # High quality, ~$0.59/M tokens
JUDGE_MODEL = "groq/llama-3.1-8b-instant"  # Fast + cheap, ~$0.05/M tokens


def get_groq_api_key() -> str:
    """Get Groq API key from environment.

    Returns:
        The GROQ_API_KEY value.

    Raises:
        ValueError: If GROQ_API_KEY is not set.
    """
    key = os.environ.get("GROQ_API_KEY")
    if not key:
        raise ValueError(
            "GROQ_API_KEY not set. Get a free key at https://console.groq.com"
        )
    return key


def call_llm(
    messages: list[dict[str, str]],
    model: str = DEFAULT_CHAT_MODEL,
    temperature: float = 0.3,
    max_tokens: int = 1024,
) -> dict[str, Any]:
    """Call LLM via LiteLLM with unified interface.

    Supports any model that LiteLLM supports:
    - Groq: "groq/llama-3.3-70b-versatile", "groq/llama-3.1-8b-instant"
    - OpenAI: "gpt-4o", "gpt-4o-mini"
    - Anthropic: "claude-3-5-sonnet-latest", "claude-3-haiku-20240307"

    Args:
        messages: List of message dicts with "role" and "content" keys.
        model: Model identifier (provider/model format for non-OpenAI).
        temperature: Sampling temperature (0.0-1.0).
        max_tokens: Maximum tokens to generate.

    Returns:
        Dict with keys:
        - text: Generated text response
        - tokens: Dict with input, output, total token counts
        - cost: Estimated cost in USD
        - model: Model identifier used

    Raises:
        ImportError: If litellm is not installed.
        ValueError: If required API key is not set.
    """
    if not LITELLM_AVAILABLE:
        raise ImportError("litellm not installed. Run: pip install litellm")

    # Validate API key for Groq models
    if model.startswith("groq/"):
        get_groq_api_key()  # Will raise if not set

    response = completion(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    # Extract response content
    text = response.choices[0].message.content or ""
    usage = response.usage

    # Calculate cost using LiteLLM's built-in cost tracking
    try:
        cost = litellm.completion_cost(completion_response=response)
    except Exception:
        cost = 0.0

    # Fallback: Manual cost calculation for Groq models (LiteLLM may not have pricing)
    if cost == 0.0 and usage:
        input_tokens = usage.prompt_tokens or 0
        output_tokens = usage.completion_tokens or 0

        # Groq pricing per 1M tokens (as of Jan 2025)
        groq_pricing = {
            "groq/llama-3.3-70b-versatile": {"input": 0.59, "output": 0.79},
            "groq/llama-3.1-8b-instant": {"input": 0.05, "output": 0.08},
            "groq/qwen/qwen3-32b": {"input": 0.29, "output": 0.39},
        }

        if model in groq_pricing:
            prices = groq_pricing[model]
            cost = (
                input_tokens * prices["input"] + output_tokens * prices["output"]
            ) / 1_000_000

    return {
        "text": text,
        "tokens": {
            "input": usage.prompt_tokens if usage else 0,
            "output": usage.completion_tokens if usage else 0,
            "total": usage.total_tokens if usage else 0,
        },
        "cost": cost,
        "model": model,
    }


def call_judge(
    prompt: str,
    model: str = JUDGE_MODEL,
    temperature: float = 0.1,
    max_tokens: int = 256,
) -> dict[str, Any]:
    """Convenience wrapper for judge/evaluator calls.

    Uses lower temperature for more consistent judgments.

    Args:
        prompt: The evaluation prompt.
        model: Judge model (defaults to cheap, fast model).
        temperature: Lower temperature for consistency.
        max_tokens: Typically short responses for judges.

    Returns:
        Same as call_llm().
    """
    return call_llm(
        messages=[{"role": "user", "content": prompt}],
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
    )
