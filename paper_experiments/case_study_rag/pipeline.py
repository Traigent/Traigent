"""Utility functions shared by the HotpotQA case study pipeline."""

from __future__ import annotations

import re
from typing import Any, Sequence

__all__ = [
    "build_mock_response",
    "format_context_block",
    "extract_answer_from_response",
]


def format_context_block(context: Sequence[str], top_k: int = 4) -> str:
    """Format context passages as a numbered list for prompting."""
    if not context:
        return "No context available."

    selected = context[:top_k] if top_k > 0 else context
    lines: list[str] = []
    for idx, passage in enumerate(selected, 1):
        lines.append(f"[{idx}] {passage}")
    return "\n".join(lines)


def extract_answer_from_response(response: str) -> str:
    """Extract the final answer from an LLM response.

    Looks for patterns like "Answer: X" or "The answer is X".
    Falls back to the last sentence if no pattern found.
    """
    text = (response or "").strip()

    # Try "Answer:" pattern
    answer_match = re.search(r"Answer:\s*(.+?)(?:\.|$)", text, re.IGNORECASE)
    if answer_match:
        return answer_match.group(1).strip()

    # Try "The answer is" pattern
    answer_is_match = re.search(
        r"The answer is\s*(.+?)(?:\.|$)", text, re.IGNORECASE
    )
    if answer_is_match:
        return answer_is_match.group(1).strip()

    # Fall back to last meaningful line
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    if lines:
        last_line = lines[-1]
        # Remove common prefixes
        for prefix in ["Therefore,", "Thus,", "So,", "In conclusion,"]:
            if last_line.lower().startswith(prefix.lower()):
                last_line = last_line[len(prefix) :].strip()
        return last_line

    return text


def build_mock_response(
    *,
    question: str,
    answer: str,
    context: Sequence[str],
    config: dict[str, Any],
) -> str:
    """Generate a mock LLM response for testing purposes."""
    model = str(config.get("model", "gpt-4o-mini"))
    prompt_style = str(config.get("prompt_style", "vanilla"))
    retriever_k = int(config.get("retriever_k", 4))

    selected_context = context[:retriever_k] if context else []

    if prompt_style == "cot":
        # Chain-of-thought style response
        if selected_context:
            reasoning_parts = []
            for idx, passage in enumerate(selected_context[:2], 1):
                # Extract key info from passage
                key_fact = (
                    passage.split(":")[-1].strip()[:50]
                    if ":" in passage
                    else passage[:50]
                )
                reasoning_parts.append(
                    f"From passage [{idx}], I note that {key_fact}..."
                )
            reasoning = " ".join(reasoning_parts)
        else:
            # Fallback reasoning when no context available
            reasoning = "Based on my knowledge of the question topic..."

        return (
            f"Let me analyze this step by step. {reasoning} "
            f"Combining these facts, I can conclude that the answer is {answer}.\n\n"
            f"Answer: {answer}"
        )
    else:
        # Direct answer style
        if selected_context:
            return (
                f"Based on the retrieved context from {len(selected_context)} "
                f"passages, the answer to the question is {answer}.\n\n"
                f"Answer: {answer}"
            )
        else:
            return f"The answer to the question is {answer}.\n\nAnswer: {answer}"
