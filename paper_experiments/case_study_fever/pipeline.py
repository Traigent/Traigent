"""Utility functions shared by the FEVER case study pipeline."""

from __future__ import annotations

import re
from typing import Any, Iterable, Sequence

__all__ = [
    "_extract_structured_response",
    "_format_numbered_evidence",
    "_map_citations_to_evidence",
    "_normalize_pipeline_output",
    "build_mock_response",
]

_CITED_PATTERN = re.compile(r"CITED_EVIDENCE:\s*\[?([0-9,\s]+)\]?", re.IGNORECASE)
_REASONING_PATTERN = re.compile(
    r"REASONING:\s*([^\n]+(?:\n(?!VERDICT:|CITED_EVIDENCE:)[^\n]+)*)",
    re.IGNORECASE,
)
_VERDICT_PATTERN = re.compile(r"VERDICT:\s*([^\n]+)", re.IGNORECASE)


def _format_numbered_evidence(evidence_items: Sequence[dict[str, Any]]) -> str:
    """Render retrieval results as numbered bullet points."""

    if not evidence_items:
        return "No evidence available to cite."

    lines: list[str] = []
    for idx, item in enumerate(evidence_items, 1):
        page = str(item.get("page", "Unknown")).strip() or "Unknown"
        line = item.get("line", 0)
        text = str(item.get("text", "(text unavailable)")).strip() or "(text unavailable)"
        lines.append(f"{idx}. [{page}, line {line}]: {text}")
    return "\n".join(lines)


def _extract_structured_response(response_text: str) -> tuple[str, str, list[int]]:
    """Parse the helper markers we add to LLM prompts."""

    reasoning = response_text.strip()
    verdict = ""

    reasoning_match = _REASONING_PATTERN.search(response_text)
    if reasoning_match:
        reasoning = reasoning_match.group(1).strip()

    verdict_match = _VERDICT_PATTERN.search(response_text)
    if verdict_match:
        verdict = verdict_match.group(1).strip()

    cited_numbers: list[int] = []
    cited_match = _CITED_PATTERN.search(response_text)
    if cited_match:
        raw = cited_match.group(1)
        cited_numbers = _parse_citation_numbers(raw)

    return verdict or reasoning, reasoning, cited_numbers


def _parse_citation_numbers(raw_numbers: str) -> list[int]:
    values: list[int] = []
    for chunk in raw_numbers.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if not chunk.lstrip("-").isdigit():
            continue
        try:
            number = int(chunk)
        except ValueError:
            continue
        values.append(number)
    return values


def _map_citations_to_evidence(
    cited_indices: Iterable[int],
    evidence_items: Sequence[dict[str, Any]],
) -> list[dict[str, int]]:
    """Convert 1-indexed citations into sanitized evidence entries."""

    mapped: list[dict[str, int]] = []
    seen: set[int] = set()
    total = len(evidence_items)

    for raw_index in cited_indices:
        if not isinstance(raw_index, int):
            continue
        if raw_index < 1 or raw_index > total:
            continue
        if raw_index in seen:
            continue

        seen.add(raw_index)
        source = evidence_items[raw_index - 1]
        page = str(source.get("page", "")).strip()
        if not page:
            continue
        try:
            line = int(source.get("line", 0))
        except (TypeError, ValueError):
            continue
        mapped.append({"page": page, "line": line})

    return mapped


def _normalize_pipeline_output(
    *,
    verdict: str,
    justification: str,
    evidence: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Ensure downstream consumers receive predictable output fields."""

    normalized_verdict = verdict.strip().upper() or "NOT ENOUGH INFO"
    normalized_justification = justification.strip() or "No justification provided."

    cleaned_evidence = _map_citations_to_evidence(range(1, len(evidence) + 1), evidence)

    return {
        "verdict": normalized_verdict,
        "justification": normalized_justification,
        "evidence": cleaned_evidence,
    }


def build_mock_response(
    *,
    verdict: str,
    reasoning: str,
    evidence_items: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Helper used by the mock simulator to emit normalized pipeline payloads."""

    evidence_block = _format_numbered_evidence(evidence_items)
    response_text = (
        f"CITED_EVIDENCE: [1]\n"
        f"{evidence_block}\n\n"
        f"REASONING: {reasoning}\n"
        f"VERDICT: {verdict}\n"
    )
    verdict_text, reasoning_text, cited = _extract_structured_response(response_text)
    mapped_evidence = _map_citations_to_evidence(cited, evidence_items)
    return _normalize_pipeline_output(
        verdict=verdict_text,
        justification=reasoning_text,
        evidence=mapped_evidence,
    )
