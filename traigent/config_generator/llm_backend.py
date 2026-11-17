"""LLM backend abstraction for config generation.

Provides a thin protocol over LiteLLM for optional LLM-enriched config
generation.  When no LLM is available the ``NoOpLLMBackend`` silently
returns empty strings so the pipeline degrades gracefully to preset-only
mode.
"""

from __future__ import annotations

import logging
import math
import threading
from typing import Protocol, runtime_checkable

from traigent.utils.logging import configure_litellm_logging
from traigent.utils.money import format_usd

logger = logging.getLogger(__name__)


_FRAMING_TOKENS = 16  # per-message role/framing margin for the pre-call bound


class BudgetExhausted(Exception):
    """Raised when the LLM call budget has been exceeded."""


class UnmeasurableCostError(RuntimeError):
    """Raised when the cost of an LLM call cannot be measured.

    Deliberately *not* a ``BudgetExhausted``: callers swallow that one and
    degrade silently, which would hide that the spend cap cannot be enforced.
    """


def validate_budget_usd(budget_usd: float) -> float:
    """Return ``budget_usd`` if it is a finite number > 0, else raise ValueError."""
    if (
        isinstance(budget_usd, bool)
        or not isinstance(budget_usd, (int, float))
        or not math.isfinite(budget_usd)
        or budget_usd <= 0
    ):
        raise ValueError(f"budget_usd must be a finite number > 0, got {budget_usd!r}")
    return float(budget_usd)


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
        Maximum total spend for this backend instance (finite, > 0).

    Raises
    ------
    ValueError
        If ``budget_usd`` is not a finite number greater than zero.
    UnmeasurableCostError
        From ``complete`` if ``model`` has no known price, or a response
        carries no usable token usage, so the cap cannot be enforced.
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        budget_usd: float = 0.10,
    ) -> None:
        self._model = model
        self._budget_usd = validate_budget_usd(budget_usd)
        self._spent_usd: float = 0.0
        self._calls_made: int = 0
        self._lock = threading.Lock()

    @property
    def calls_made(self) -> int:
        with self._lock:
            return self._calls_made

    @property
    def total_cost_usd(self) -> float:
        with self._lock:
            return self._spent_usd

    def complete(self, prompt: str, *, max_tokens: int = 1024) -> str:
        """Call the LLM and return the completion text."""
        with self._lock:
            if self._spent_usd >= self._budget_usd:
                raise BudgetExhausted(
                    f"Budget of {format_usd(self._budget_usd)} exceeded "
                    f"(spent {format_usd(self._spent_usd)})"
                )

            try:
                import litellm
            except ImportError as exc:
                logger.warning(
                    "litellm not installed; falling back to preset-only mode"
                )
                raise BudgetExhausted("litellm not installed") from exc

            configure_litellm_logging(litellm_module=litellm)
            # Fail closed before spending: the cap is meaningless without a price.
            # Bound the call's cost from above (a token is never shorter than a
            # byte, and output is capped by max_tokens) so one call cannot
            # overshoot the budget.
            messages = [{"role": "user", "content": prompt}]
            # A token is never shorter than a byte; add a framing margin when
            # the real counter is unavailable, and never go below the byte bound.
            prompt_token_bound = len(prompt.encode("utf-8")) + _FRAMING_TOKENS
            try:
                prompt_token_bound = max(
                    prompt_token_bound,
                    int(litellm.token_counter(model=self._model, messages=messages)),
                )
            except Exception:
                logger.debug("token_counter failed; using byte bound", exc_info=True)
            try:
                prompt_bound, completion_bound = litellm.cost_per_token(
                    model=self._model,
                    prompt_tokens=prompt_token_bound,
                    completion_tokens=max_tokens,
                )
            except Exception as exc:
                raise UnmeasurableCostError(
                    f"No known price for model {self._model!r}; refusing to call "
                    "it because the --budget cap cannot be enforced"
                ) from exc
            worst_case = prompt_bound + completion_bound
            if self._spent_usd + worst_case > self._budget_usd:
                raise BudgetExhausted(
                    f"Budget of {format_usd(self._budget_usd)} would be exceeded "
                    f"(spent {format_usd(self._spent_usd)}, next call could cost "
                    f"up to {format_usd(worst_case)})"
                )

            try:
                response = litellm.completion(
                    model=self._model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=0.0,  # Deterministic for config generation
                    num_retries=0,  # hidden retries would be unbudgeted spend
                )
            except BaseException:
                # The request may have been billed even though it failed on our
                # side; reserve its worst-case cost so it cannot be spent twice.
                self._spent_usd += worst_case
                raise

            self._calls_made += 1

            usage = getattr(response, "usage", None)
            prompt_tokens = getattr(usage, "prompt_tokens", None)
            completion_tokens = getattr(usage, "completion_tokens", None)
            if not all(
                isinstance(n, int) and not isinstance(n, bool) and n >= 0
                for n in (prompt_tokens, completion_tokens)
            ):
                # Unknown spend must not be counted as free.
                self._spent_usd = math.inf
                raise UnmeasurableCostError(
                    f"Response from {self._model!r} reported no token usage; "
                    "cannot account for spend against the budget"
                )
            try:
                prompt_cost, completion_cost = litellm.cost_per_token(
                    model=self._model,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                )
            except Exception as exc:
                self._spent_usd = math.inf
                raise UnmeasurableCostError(
                    f"Could not price a call to {self._model!r}; "
                    "cannot account for spend against the budget"
                ) from exc
            self._spent_usd += prompt_cost + completion_cost

            content = response.choices[0].message.content
            return content or ""
