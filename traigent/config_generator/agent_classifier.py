"""Agent type classification from source code.

Classifies a function into an agent type (``"rag"``, ``"chat"``,
``"code_gen"``, etc.) using heuristic pattern matching on the source code
and optional LLM enrichment.

This is a **shared step** used by multiple subsystems (objectives,
safety constraints, recommendations) to avoid redundant LLM calls.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from traigent.config_generator.llm_backend import BudgetExhausted, ConfigGenLLM

logger = logging.getLogger(__name__)

# Agent types recognised by the system
AGENT_TYPES = frozenset(
    {
        "rag",
        "chat",
        "code_gen",
        "summarization",
        "classification",
        "router",
        "general_llm",
    }
)


@dataclass(frozen=True)
class ClassificationResult:
    """Result of agent type classification."""

    agent_type: str
    confidence: float  # 0.0–1.0
    source: str  # "heuristic" | "llm"
    reasoning: str


def classify_agent(
    source_code: str,
    *,
    llm: ConfigGenLLM | None = None,
) -> ClassificationResult:
    """Classify a function's agent type from its source code.

    Parameters
    ----------
    source_code:
        The full source code of the file or function.
    llm:
        Optional LLM for more accurate classification.  When ``None``,
        only heuristic pattern matching is used.
    """
    # Try heuristic first
    heuristic_result = _heuristic_classify(source_code)

    # If confidence is high or no LLM available, return heuristic
    if heuristic_result.confidence >= 0.8 or llm is None:
        return heuristic_result

    # Try LLM enrichment
    llm_result = _llm_classify(source_code, llm)
    if llm_result is not None and llm_result.confidence > heuristic_result.confidence:
        return llm_result

    return heuristic_result


# ---------------------------------------------------------------------------
# Heuristic classification
# ---------------------------------------------------------------------------

# Pattern groups: each maps to an agent type with a confidence boost per match
_PATTERNS: dict[str, list[re.Pattern[str]]] = {
    "rag": [
        re.compile(r"\bretriever\b", re.IGNORECASE),
        re.compile(r"\bsimilarity_search\b"),
        re.compile(r"\bvector[_\s]?store\b", re.IGNORECASE),
        re.compile(
            r"\bchromadb\b|\bpinecone\b|\bweaviate\b|\bqdrant\b|\bfaiss\b",
            re.IGNORECASE,
        ),
        re.compile(r"\bRAG\b"),
        re.compile(r"\bembedding\b", re.IGNORECASE),
        re.compile(r"\bchunk\b", re.IGNORECASE),
        re.compile(r"\bretriev\w+\b", re.IGNORECASE),
    ],
    "chat": [
        re.compile(r"\bChatOpenAI\b|\bChatAnthropic\b|\bChatModel\b"),
        re.compile(r"\bopenai\.chat\b"),
        re.compile(r"\bmessages\s*=\s*\["),
        re.compile(r"\brole.*user\b", re.IGNORECASE),
        re.compile(r"\bconversation\b|\bdialogue\b", re.IGNORECASE),
    ],
    "code_gen": [
        re.compile(r"\bexecute_code\b|\bexec\(|\beval\("),
        re.compile(r"\bsubprocess\b"),
        re.compile(r"\bcode_gen\b|\bgenerate_code\b", re.IGNORECASE),
        re.compile(r"\bHumanEval\b|\bMBPP\b", re.IGNORECASE),
        re.compile(r"\bcompile\b.*\brun\b", re.IGNORECASE),
    ],
    "summarization": [
        re.compile(r"\bsummariz\w+\b", re.IGNORECASE),
        re.compile(r"\bsummary\b", re.IGNORECASE),
        re.compile(r"\bcondense\b|\bextract\b.*\bkey\b", re.IGNORECASE),
        re.compile(r"\bTL;?DR\b", re.IGNORECASE),
    ],
    "classification": [
        re.compile(r"\bclassif\w+\b", re.IGNORECASE),
        re.compile(r"\bpredict\b.*\blabel\b", re.IGNORECASE),
        re.compile(r"\bsentiment\b", re.IGNORECASE),
        re.compile(r"\bcategor\w+\b", re.IGNORECASE),
    ],
    "router": [
        re.compile(r"\broute\b|\brouter\b", re.IGNORECASE),
        re.compile(r"\bdispatch\b|\bswitch\b", re.IGNORECASE),
        re.compile(r"\bagent_executor\b", re.IGNORECASE),
    ],
}


def _heuristic_classify(source_code: str) -> ClassificationResult:
    """Classify using regex pattern matching."""
    scores: dict[str, int] = dict.fromkeys(_PATTERNS, 0)

    for agent_type, patterns in _PATTERNS.items():
        for pattern in patterns:
            if pattern.search(source_code):
                scores[agent_type] += 1

    # Find the best match
    best_type = max(scores, key=scores.get)  # type: ignore[arg-type]
    best_score = scores[best_type]

    if best_score == 0:
        return ClassificationResult(
            agent_type="general_llm",
            confidence=0.3,
            source="heuristic",
            reasoning="No specific agent patterns detected; defaulting to general LLM",
        )

    # Confidence based on number of pattern matches
    # 1 match = 0.5, 2 = 0.65, 3+ = 0.8+
    confidence = min(0.3 + best_score * 0.2, 0.9)

    return ClassificationResult(
        agent_type=best_type,
        confidence=confidence,
        source="heuristic",
        reasoning=f"Matched {best_score} pattern(s) for '{best_type}'",
    )


# ---------------------------------------------------------------------------
# LLM classification
# ---------------------------------------------------------------------------

_LLM_CLASSIFY_PROMPT = """\
Classify the following Python function into exactly one agent type.

Agent types:
- rag: Retrieval-augmented generation (uses vector stores, document retrieval)
- chat: Conversational chatbot (uses chat models, message history)
- code_gen: Code generation or execution (generates/runs code)
- summarization: Document summarization
- classification: Text classification or labeling
- router: Agent routing or orchestration
- general_llm: General LLM usage (doesn't fit other categories)

Source code:
```python
{source_code}
```

Reply with ONLY a JSON object:
{{"agent_type": "<type>", "confidence": <0.0-1.0>, "reasoning": "<brief>"}}
"""


def _llm_classify(
    source_code: str,
    llm: ConfigGenLLM,
) -> ClassificationResult | None:
    """Classify using LLM."""
    import json

    prompt = _LLM_CLASSIFY_PROMPT.format(source_code=source_code[:3000])

    try:
        response = llm.complete(prompt, max_tokens=256)
    except BudgetExhausted:
        return None

    # Parse response
    text = response.strip()
    if "```" in text:
        for part in text.split("```"):
            stripped = part.strip()
            if stripped.startswith("json"):
                stripped = stripped[4:].strip()
            if stripped.startswith("{"):
                text = stripped
                break

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return None

    agent_type = data.get("agent_type", "")
    if agent_type not in AGENT_TYPES:
        return None

    try:
        confidence = float(data.get("confidence", 0.5))
    except (ValueError, TypeError):
        confidence = 0.5
    reasoning = data.get("reasoning", "LLM classification")

    return ClassificationResult(
        agent_type=agent_type,
        confidence=min(confidence, 1.0),
        source="llm",
        reasoning=reasoning,
    )
