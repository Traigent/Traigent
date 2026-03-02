"""LLM backend abstraction for config generation.

Provides a thin protocol over LiteLLM for optional LLM-enriched config
generation.  When no LLM is available the ``NoOpLLMBackend`` silently
returns empty strings so the pipeline degrades gracefully to preset-only
mode.
"""

from __future__ import annotations

import logging
from typing import Protocol, runtime_checkable

logger = logging.getLogger(__name__)


class BudgetExhausted(Exception):
    """Raised when the LLM call budget has been exceeded."""


@runtime_checkable
class ConfigGenLLM(Protocol):
    """Protocol for LLM calls during config generation."""

    def complete(self, prompt: str, *, max_tokens: int = 1024) -> str:
        """Send a prompt and return the completion text."""
        ...  # pragma: no cover


class NoOpLLMBackend:
    """Fallback backend that returns empty strings for all calls.

    Used when no LLM API key is available or when running in preset-only
    mode.
    """

    @property
    def calls_made(self) -> int:
        return 0

    @property
    def total_cost_usd(self) -> float:
        return 0.0

    def complete(self, prompt: str, *, max_tokens: int = 1024) -> str:
        del prompt, max_tokens
        return ""


class LiteLLMBackend:
    """LiteLLM-based backend with budget tracking.

    Uses the cheapest capable model by default (gpt-4o-mini).  Tracks
    cumulative spend and raises ``BudgetExhausted`` when the budget is
    exceeded.

    Parameters
    ----------
    model:
        LiteLLM model identifier (e.g. ``"gpt-4o-mini"``).
    budget_usd:
        Maximum total spend for this backend instance.
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        budget_usd: float = 0.10,
    ) -> None:
        self._model = model
        self._budget_usd = budget_usd
        self._spent_usd: float = 0.0
        self._calls_made: int = 0

    @property
    def calls_made(self) -> int:
        return self._calls_made

    @property
    def total_cost_usd(self) -> float:
        return self._spent_usd

    def complete(self, prompt: str, *, max_tokens: int = 1024) -> str:
        """Call the LLM and return the completion text."""
        if self._spent_usd >= self._budget_usd:
            raise BudgetExhausted(
                f"Budget of ${self._budget_usd:.2f} exceeded "
                f"(spent ${self._spent_usd:.4f})"
            )

        try:
            import litellm
        except ImportError as exc:
            logger.warning("litellm not installed; falling back to preset-only mode")
            raise BudgetExhausted("litellm not installed") from exc

        response = litellm.completion(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0.0,  # Deterministic for config generation
        )

        self._calls_made += 1

        # Track cost from response usage
        usage = getattr(response, "usage", None)
        if usage:
            prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
            completion_tokens = getattr(usage, "completion_tokens", 0) or 0
            # Approximate cost for gpt-4o-mini: $0.15/1M input, $0.60/1M output
            cost = (prompt_tokens * 0.00000015) + (completion_tokens * 0.0000006)
            self._spent_usd += cost

        content = response.choices[0].message.content
        return content or ""
