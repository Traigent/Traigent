"""Helpers for parsing structured content out of LLM text responses."""

from __future__ import annotations


def extract_json_array_text(response: str) -> str:
    """Extract a JSON array payload from plain text or markdown-fenced text."""
    text = response.strip()
    if "```" not in text:
        return text

    for part in text.split("```"):
        stripped = part.strip()
        if stripped.startswith("json"):
            stripped = stripped[4:].strip()
        if stripped.startswith("["):
            return stripped
    return text
