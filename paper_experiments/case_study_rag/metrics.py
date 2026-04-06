"""Metric helpers for the HotpotQA case study tests.

Token and cost tracking is handled automatically by Traigent when using LangChain.
Traigent passes token counts via the `llm_metrics` parameter to metric functions.
"""

from __future__ import annotations

import re
from typing import Any
from collections.abc import Mapping

__all__ = [
    "build_hotpot_metric_functions",
]


def _normalize_answer(answer: str) -> str:
    """Normalize answer for comparison: lowercase, strip, remove articles."""
    text = (answer or "").lower().strip()
    # Remove common articles
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    # Collapse whitespace
    text = " ".join(text.split())
    return text


def _compute_f1(prediction: str, ground_truth: str) -> float:
    """Compute token-level F1 score between prediction and ground truth."""
    pred_tokens = set(_normalize_answer(prediction).split())
    gold_tokens = set(_normalize_answer(ground_truth).split())

    if not pred_tokens or not gold_tokens:
        return 0.0

    common = pred_tokens & gold_tokens
    if not common:
        return 0.0

    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(gold_tokens)

    return 2 * precision * recall / (precision + recall)


def _compute_em(prediction: str, ground_truth: str) -> float:
    """Compute exact match score."""
    return 1.0 if _normalize_answer(prediction) == _normalize_answer(ground_truth) else 0.0


def _mock_quality_metric(
    output: str | None = None,
    expected: str | None = None,
    config: Mapping[str, Any] | None = None,
    **kwargs: Any,
) -> float:
    """Return deterministic quality metrics for mock-mode evaluations.

    Note: Real HotpotQA quality scores with answer extraction are 48-71%.
    gpt-4o-mini leads (67-71%), haiku is competitive (54-67%), gpt-4o trails
    (48-50%) due to verbose outputs that hurt EM+F1 precision.
    """
    config = config or {}
    model = str(config.get("model", "gpt-4o-mini")).lower()
    temperature = float(config.get("temperature", 0.3))
    retriever_k = int(config.get("retriever_k", 4))
    prompt_style = str(config.get("prompt_style", "vanilla"))

    # Base quality varies by model - calibrated to real HotpotQA results
    # with answer extraction (extracts concise answer from verbose output)
    if "gpt-4o" in model and "mini" not in model:
        base = 0.45  # Real ~48-50%, verbose but extraction helps
    elif "mini" in model:
        base = 0.68  # Real ~67-71%, concise answers score best
    elif "haiku" in model:
        base = 0.60  # Real ~54-67%, competitive with extraction
    else:
        base = 0.50

    # Temperature effect: lower is more accurate for factual (small effect)
    temp_effect = 0.02 - (temperature * 0.04)

    # Retriever k effect: diminishing returns after k=5
    k_effect = min(retriever_k * 0.005, 0.025)

    # Prompt style effect: CoT may HURT due to verbosity penalty in EM+F1 scoring
    # Real results showed verbose CoT answers get lower F1 scores
    style_effect = -0.02 if prompt_style == "cot" else 0.0

    quality = base + temp_effect + k_effect + style_effect
    return min(max(quality, 0.0), 1.0)


def _mock_latency_metric(
    output: str | None = None,
    expected: str | None = None,
    config: Mapping[str, Any] | None = None,
    **kwargs: Any,
) -> float:
    """Return deterministic latency metrics for mock-mode evaluations (in ms).

    Note: Updated to reflect real-world latencies observed in parallel execution.
    Real HotpotQA runs showed 2000-7000ms per call depending on model and config.
    """
    config = config or {}
    model = str(config.get("model", "gpt-4o-mini")).lower()
    retriever_k = int(config.get("retriever_k", 4))
    prompt_style = str(config.get("prompt_style", "vanilla"))

    # Base latency varies by model - updated based on real results
    # Real results showed 2000-7000ms range with parallel execution
    if "gpt-4o" in model and "mini" not in model:
        base = 3500.0  # Was 450, real ~3000-5000ms
    elif "mini" in model:
        base = 2000.0  # Was 250, real ~2000-3000ms
    elif "haiku" in model:
        base = 1800.0  # Was 200, Haiku is fast
    else:
        base = 2500.0

    # More context increases latency (larger effect in real mode)
    k_effect = retriever_k * 150.0  # Was 15.0

    # CoT takes longer due to longer outputs
    style_effect = 800.0 if prompt_style == "cot" else 0.0  # Was 100.0

    return base + k_effect + style_effect


def _mock_cost_metric(
    output: str | None = None,
    expected: str | None = None,
    config: Mapping[str, Any] | None = None,
    **kwargs: Any,
) -> float:
    """Return deterministic cost metrics for mock-mode evaluations (USD per 1K)."""
    config = config or {}
    model = str(config.get("model", "gpt-4o-mini")).lower()
    retriever_k = int(config.get("retriever_k", 4))

    # Model costs per 1K tokens (approximate)
    if "gpt-4o" in model and "mini" not in model:
        base = 0.005
    elif "mini" in model:
        base = 0.00015
    elif "haiku" in model:
        base = 0.00025
    else:
        base = 0.001

    # More context increases cost
    k_multiplier = 1.0 + (retriever_k * 0.1)

    return base * k_multiplier


def _extract_after_answer_marker(text: str) -> str:
    """Extract text after 'Answer:' marker, preserving case."""
    lower_text = text.lower()
    if "answer:" not in lower_text:
        return text
    idx = lower_text.rfind("answer:")
    return text[idx + 7:].strip()


def _extract_yes_no(text: str) -> str | None:
    """Extract yes/no from answers like 'Yes, because...' or 'No.'."""
    lower_first = text.lower()
    for prefix in ("yes", "no"):
        if not lower_first.startswith(prefix):
            continue
        rest = text[len(prefix):].lstrip()
        # Simple yes/no: empty, or followed by punctuation/comma
        if not rest or rest[0] in ".,;:!,":
            return prefix.capitalize()
    return None


def _extract_first_sentence(text: str) -> str:
    """Extract first sentence, careful not to split on abbreviations."""
    # Find period followed by space and uppercase (new sentence start)
    match = re.search(r"\.\s+[A-Z]", text)
    if match:
        return text[: match.start() + 1].strip()
    # If no new sentence, check for period at end
    if text.rstrip().endswith("."):
        return text.rstrip()
    return text


def _extract_final_answer(output: str) -> str:
    """Extract the concise final answer from model output.

    HotpotQA answers are typically very short (a word, name, or short phrase).
    This function extracts only the core answer, not explanations.
    """
    text = output.strip()

    # Step 1: Extract text after "Answer:" marker
    text = _extract_after_answer_marker(text)

    # Step 2: Take first line only
    if "\n" in text:
        text = text.split("\n")[0].strip()

    # Step 3: Handle yes/no questions (common in HotpotQA comparisons)
    yes_no = _extract_yes_no(text)
    if yes_no:
        return yes_no

    # Step 4: Take first sentence
    text = _extract_first_sentence(text)

    # Step 5: Clean trailing punctuation
    return text.rstrip(".,;:!?")


def _real_quality_metric(
    output: str | None = None,
    expected: str | None = None,
    config: Mapping[str, Any] | None = None,
    **kwargs: Any,
) -> float:
    """Compute quality score from actual model output.

    Uses HotpotQA's official metrics: EM (exact match) + F1 (token overlap).
    We weight EM higher (0.6) because exact match is the primary metric.
    """
    if output is None or expected is None:
        return 0.0

    # Extract the concise answer from potentially verbose output
    answer = _extract_final_answer(output)

    em = _compute_em(answer, expected)
    f1 = _compute_f1(answer, expected)

    # Weighted combination — favour F1 to tolerate minor phrasing differences
    # (e.g. "Salzach River" vs "Salzach") that are correct but not exact.
    return 0.4 * em + 0.6 * f1


def _real_latency_metric(
    output: str | None = None,
    expected: str | None = None,
    config: Mapping[str, Any] | None = None,
    llm_metrics: dict[str, Any] | None = None,
    **kwargs: Any,
) -> float:
    """Return actual latency from Traigent's automatic timing (in ms).

    Traigent tracks function execution time and passes it via llm_metrics["response_time_ms"].
    """
    if llm_metrics:
        return float(llm_metrics.get("response_time_ms", 0.0))
    return 0.0


# Model pricing per 1K tokens (input, output) in USD - as of early 2024
_MODEL_PRICING = {
    "gpt-4o": (0.005, 0.015),
    "gpt-4o-mini": (0.00015, 0.0006),
    "claude-3-haiku-20240307": (0.00025, 0.00125),
    "claude-3-5-sonnet-20241022": (0.003, 0.015),
}


def _real_cost_metric(
    output: str | None = None,
    expected: str | None = None,
    config: Mapping[str, Any] | None = None,
    llm_metrics: dict[str, Any] | None = None,
    **kwargs: Any,
) -> float:
    """Compute actual cost from Traigent's automatic token tracking (USD per 1K tokens).

    Traigent automatically captures token usage from LangChain responses and passes
    it via the llm_metrics parameter.
    """
    if not llm_metrics:
        return 0.0

    # Get token counts from Traigent's automatic tracking
    # LangChain provides these as prompt_tokens/completion_tokens or input_tokens/output_tokens
    input_tokens = llm_metrics.get("prompt_tokens", llm_metrics.get("input_tokens", 0))
    output_tokens = llm_metrics.get("completion_tokens", llm_metrics.get("output_tokens", 0))
    model = llm_metrics.get("model", config.get("model", "gpt-4o-mini") if config else "gpt-4o-mini")

    # Find pricing - try exact match first, then prefix match
    pricing = _MODEL_PRICING.get(model)
    if not pricing:
        for key, val in _MODEL_PRICING.items():
            if key in model.lower() or model.lower() in key:
                pricing = val
                break

    # Fallback to gpt-4o-mini pricing if model not found
    if not pricing:
        pricing = _MODEL_PRICING["gpt-4o-mini"]

    input_price, output_price = pricing

    # Calculate cost in USD per 1K tokens (normalized)
    total_tokens = input_tokens + output_tokens
    if total_tokens == 0:
        return 0.0

    input_cost = (input_tokens / 1000) * input_price
    output_cost = (output_tokens / 1000) * output_price
    total_cost = input_cost + output_cost

    # Return cost per 1K tokens (normalize to our metric)
    return (total_cost / total_tokens) * 1000


def build_hotpot_metric_functions(mock_mode: bool = False) -> dict[str, Any]:
    """Return metric functions wired up according to the execution mode."""
    if mock_mode:
        return {
            "quality": _mock_quality_metric,
            "latency_p95_ms": _mock_latency_metric,
            "cost_usd_per_1k": _mock_cost_metric,
        }

    return {
        "quality": _real_quality_metric,
        "latency_p95_ms": _real_latency_metric,
        "cost_usd_per_1k": _real_cost_metric,
    }
