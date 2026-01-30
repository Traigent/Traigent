"""Shared fixtures for validation tests.

This module provides fixtures for:
- API key validation (skip if missing)
- Traigent session initialization
- Cost tracking context manager
- Report generation helpers
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import pytest

# Provider API key environment variables
PROVIDER_API_KEYS = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "groq": "GROQ_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
}


def has_api_key(provider: str) -> bool:
    """Check if API key is available for provider."""
    env_var = PROVIDER_API_KEYS.get(provider)
    if env_var:
        return bool(os.environ.get(env_var))
    # LangChain and LiteLLM use underlying provider keys
    if provider in ("langchain", "litellm"):
        return has_api_key("openai") or has_api_key("anthropic")
    return False


def skip_if_no_api_key(provider: str):
    """Return pytest skip marker if API key is missing."""
    if not has_api_key(provider):
        env_var = PROVIDER_API_KEYS.get(provider, f"{provider.upper()}_API_KEY")
        return pytest.mark.skip(reason=f"Missing {env_var}")
    return pytest.mark.skipif(False, reason="")


@dataclass
class CostTracker:
    """Track costs during test execution."""

    costs: list[dict[str, Any]] = field(default_factory=list)
    start_time: datetime = field(default_factory=datetime.utcnow)

    def record(
        self,
        provider: str,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        cost: float,
        raw_response: dict[str, Any] | None = None,
    ) -> None:
        """Record a cost entry."""
        self.costs.append(
            {
                "provider": provider,
                "model": model,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "cost": cost,
                "timestamp": datetime.utcnow().isoformat(),
                "raw_response": raw_response or {},
            }
        )

    def get_last_cost(self) -> float:
        """Get the most recent recorded cost."""
        if not self.costs:
            return 0.0
        return self.costs[-1]["cost"]

    def total_cost(self) -> float:
        """Get total cost across all records."""
        return sum(c["cost"] for c in self.costs)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "costs": self.costs,
            "total_cost": self.total_cost(),
            "start_time": self.start_time.isoformat(),
            "end_time": datetime.utcnow().isoformat(),
        }


@pytest.fixture
def cost_tracker() -> CostTracker:
    """Provide a cost tracker for the test."""
    return CostTracker()


@pytest.fixture
def report_dir(tmp_path_factory):
    """Create a temporary directory for reports."""
    return tmp_path_factory.mktemp("reports")


# Provider-specific skip markers
requires_openai = pytest.mark.skipif(
    not has_api_key("openai"), reason="Missing OPENAI_API_KEY"
)
requires_anthropic = pytest.mark.skipif(
    not has_api_key("anthropic"), reason="Missing ANTHROPIC_API_KEY"
)
requires_groq = pytest.mark.skipif(
    not has_api_key("groq"), reason="Missing GROQ_API_KEY"
)
requires_openrouter = pytest.mark.skipif(
    not has_api_key("openrouter"), reason="Missing OPENROUTER_API_KEY"
)
