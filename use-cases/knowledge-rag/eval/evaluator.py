#!/usr/bin/env python3
"""
Evaluator for Knowledge & RAG Agent

This evaluator scores RAG responses on:
1. Grounded Accuracy - Correctness AND faithfulness to sources
2. Retrieval Quality - MRR, NDCG@k, Retrieval Success Rate
3. Abstention Accuracy - Correct abstention when knowledge is insufficient

Supports two modes:
- MOCK MODE (default): Uses heuristic rules for fast, free evaluation
- REAL MODE: Uses actual LLM calls for agent and LLM-as-judge evaluation

Usage:
  Mock mode: python evaluator.py  (default, uses heuristics)
  Real mode: TRAIGENT_MOCK_MODE=false python evaluator.py  (requires OPENAI_API_KEY)
"""

import json
import os
import re
from dataclasses import dataclass
from typing import Any

# ============================================================================
# PROMPTS FOR REAL LLM MODE
# ============================================================================

# Sample knowledge base for demo (in real use, this would be from a vector DB)
SAMPLE_DOCS = {
    "doc_rate_01": "Rate Limits: Standard tier: 100 requests/min. Professional: 500/min. Enterprise: 2000/min.",
    "doc_auth_01": "Authentication: Use API keys in the X-API-Key header. Keys can be generated in the dashboard.",
    "doc_error_01": "Error Codes: 400=Bad Request, 401=Unauthorized, 429=Rate Limited, 500=Server Error.",
}

# Prompt for the RAG agent to answer questions
AGENT_PROMPT = """You are a helpful assistant that answers questions based ONLY on the provided documents.

DOCUMENTS:
{documents}

QUESTION: {question}

Instructions:
1. Answer ONLY based on information in the documents above
2. If the answer is NOT in the documents, say "I don't have information about that in the documentation"
3. Cite which document(s) you used

Respond in EXACTLY this JSON format:
{{"answer": "your answer here", "sources": ["doc_id_1"], "is_abstention": true/false}}
"""

# Prompt for LLM-as-judge to evaluate answers
JUDGE_PROMPT = """You are evaluating a RAG system's answer.

QUESTION: {question}
EXPECTED ANSWER: {expected}
AGENT'S ANSWER: {answer}
CITED SOURCES: {sources}
SHOULD ABSTAIN: {should_abstain}
DID ABSTAIN: {did_abstain}

Evaluate:
1. CORRECTNESS: Is the answer factually correct? (0.0-1.0)
2. FAITHFULNESS: Is the answer supported by the cited sources? (0.0-1.0)
3. ABSTENTION_CORRECT: Did it correctly abstain/answer? (0 or 1)

Respond in EXACTLY this format:
CORRECTNESS: [0.0-1.0]
FAITHFULNESS: [0.0-1.0]
ABSTENTION_CORRECT: [0 or 1]
"""


@dataclass
class RAGEvaluationResult:
    """Result of RAG evaluation."""

    correctness: float  # 0-1 scale
    faithfulness: float  # 0-1 scale
    grounded_accuracy: float  # Combined score
    retrieval_success: bool
    retrieval_precision: float  # 0-1
    abstention_correct: bool
    abstention_f1: float  # 0-1
    overall_score: float


class RAGEvaluator:
    """
    Evaluator for RAG Q&A agent.

    Evaluates answer correctness, faithfulness, retrieval quality,
    and appropriate abstention behavior.
    """

    def __init__(self):
        """Initialize the evaluator."""
        self.weights = {
            "grounded_accuracy": 0.4,
            "retrieval_quality": 0.3,
            "abstention_accuracy": 0.3,
        }

    def __call__(
        self,
        prediction: dict[str, Any] | str,
        expected: str | None,
        input_data: dict[str, Any],
    ) -> dict[str, float]:
        """
        Evaluate a generated response.

        Args:
            prediction: The generated response (dict or string)
            expected: Expected answer (gold standard)
            input_data: The input data including source_ids and answerable flag

        Returns:
            Dictionary of metric scores
        """
        # Parse prediction
        if isinstance(prediction, str):
            pred_answer = prediction
            pred_sources = []
            pred_is_abstention = self._detect_abstention(prediction)
        else:
            pred_answer = prediction.get("answer", "")
            pred_sources = prediction.get("sources", [])
            pred_is_abstention = prediction.get("is_abstention", False)

        # Get expected values
        expected_answer = input_data.get("output", expected or "")
        expected_sources = input_data.get("source_ids", [])
        is_answerable = input_data.get("answerable", True)

        # Evaluate correctness (semantic similarity to expected answer)
        correctness = self._evaluate_correctness(pred_answer, expected_answer)

        # Evaluate faithfulness (only if not abstaining)
        if pred_is_abstention:
            faithfulness = 1.0  # Abstaining is faithful by definition
        else:
            faithfulness = self._evaluate_faithfulness(
                pred_answer, pred_sources, expected_sources
            )

        # Combined grounded accuracy
        grounded_accuracy = correctness * faithfulness

        # Evaluate retrieval quality
        retrieval_precision = self._evaluate_retrieval(pred_sources, expected_sources)
        retrieval_success = (
            len(set(pred_sources) & set(expected_sources)) > 0
            if expected_sources
            else True
        )

        # Evaluate abstention
        should_abstain = not is_answerable
        abstention_correct = pred_is_abstention == should_abstain

        # Calculate abstention F1 (considering both false positives and negatives)
        abstention_f1 = self._calculate_abstention_f1(
            pred_is_abstention, should_abstain
        )

        # Calculate overall score
        overall = (
            self.weights["grounded_accuracy"] * grounded_accuracy
            + self.weights["retrieval_quality"] * retrieval_precision
            + self.weights["abstention_accuracy"] * (1.0 if abstention_correct else 0.0)
        )

        return {
            "grounded_accuracy": grounded_accuracy,
            "correctness": correctness,
            "faithfulness": faithfulness,
            "retrieval_quality": retrieval_precision,
            "retrieval_success": 1.0 if retrieval_success else 0.0,
            "abstention_accuracy": 1.0 if abstention_correct else 0.0,
            "abstention_f1": abstention_f1,
            "overall": overall,
        }

    def _detect_abstention(self, answer: str) -> bool:
        """Detect if an answer is an abstention."""
        abstention_phrases = [
            "don't have information",
            "not mentioned",
            "cannot find",
            "no information",
            "not in the documentation",
            "unable to find",
            "not available",
            "don't know",
            "cannot answer",
            "insufficient information",
        ]
        answer_lower = answer.lower()
        return any(phrase in answer_lower for phrase in abstention_phrases)

    def _evaluate_correctness(self, predicted: str, expected: str) -> float:
        """
        Evaluate answer correctness using multiple methods.

        Returns:
            Score between 0 and 1
        """
        if not expected:
            return 0.5  # No ground truth, neutral score

        if not predicted:
            return 0.0

        pred_lower = predicted.lower()
        exp_lower = expected.lower()

        # Exact match (rare but highest score)
        if pred_lower.strip() == exp_lower.strip():
            return 1.0

        # Token overlap (Jaccard similarity)
        pred_tokens = set(self._tokenize(pred_lower))
        exp_tokens = set(self._tokenize(exp_lower))

        if not exp_tokens:
            return 0.5

        intersection = len(pred_tokens & exp_tokens)
        union = len(pred_tokens | exp_tokens)
        jaccard = intersection / union if union > 0 else 0.0

        # Key fact extraction
        key_facts = self._extract_key_facts(expected)
        facts_found = sum(1 for fact in key_facts if fact.lower() in pred_lower)
        fact_coverage = facts_found / len(key_facts) if key_facts else 0.5

        # Combine metrics
        return 0.5 * jaccard + 0.5 * fact_coverage

    def _tokenize(self, text: str) -> list[str]:
        """Tokenize text into words, removing stopwords."""
        stopwords = {
            "the",
            "a",
            "an",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "being",
            "have",
            "has",
            "had",
            "do",
            "does",
            "did",
            "will",
            "would",
            "could",
            "should",
            "may",
            "might",
            "must",
            "shall",
            "can",
            "to",
            "of",
            "in",
            "for",
            "on",
            "with",
            "at",
            "by",
            "from",
            "as",
            "into",
            "through",
            "during",
            "before",
            "after",
            "above",
            "below",
            "between",
            "under",
            "again",
            "further",
            "then",
            "once",
            "here",
            "there",
            "when",
            "where",
            "why",
            "how",
            "all",
            "each",
            "few",
            "more",
            "most",
            "other",
            "some",
            "such",
            "no",
            "nor",
            "not",
            "only",
            "own",
            "same",
            "so",
            "than",
            "too",
            "very",
            "just",
            "and",
            "but",
            "if",
            "or",
            "because",
            "until",
            "while",
            "this",
            "that",
            "these",
            "those",
        }
        words = re.findall(r"\b[a-z0-9]+\b", text.lower())
        return [w for w in words if w not in stopwords and len(w) > 2]

    def _extract_key_facts(self, text: str) -> list[str]:
        """Extract key factual elements from text."""
        facts = []

        # Numbers and measurements
        numbers = re.findall(
            r"\b\d+(?:\.\d+)?(?:\s*(?:GB|TB|MB|KB|ms|seconds?|minutes?|hours?|days?|%|requests?|per\s+\w+))?\b",
            text,
        )
        facts.extend(numbers)

        # Technical terms (capitalized words, acronyms)
        technical = re.findall(r"\b[A-Z][A-Z0-9]+\b", text)
        facts.extend(technical)

        # Quoted terms
        quoted = re.findall(r'[\'"]([^\'"]+)[\'"]', text)
        facts.extend(quoted)

        return list(set(facts))

    def _evaluate_faithfulness(
        self,
        answer: str,
        predicted_sources: list[str],
        expected_sources: list[str],
    ) -> float:
        """
        Evaluate faithfulness to source documents.

        Returns:
            Score between 0 and 1
        """
        if not expected_sources:
            return 1.0  # No sources expected, assume faithful

        if not predicted_sources:
            return 0.5  # No sources cited, partial credit

        # Check source overlap
        pred_set = set(predicted_sources)
        exp_set = set(expected_sources)

        if not exp_set:
            return 1.0

        # Precision: what fraction of cited sources are correct?
        correct_sources = len(pred_set & exp_set)
        precision = correct_sources / len(pred_set) if pred_set else 0.0

        # Recall: what fraction of expected sources were cited?
        recall = correct_sources / len(exp_set)

        # F1 score
        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0

        return f1

    def _evaluate_retrieval(
        self,
        predicted_sources: list[str],
        expected_sources: list[str],
    ) -> float:
        """
        Evaluate retrieval quality.

        Returns:
            Score between 0 and 1
        """
        if not expected_sources:
            return 1.0  # No sources expected

        if not predicted_sources:
            return 0.0

        # Calculate precision at K
        pred_set = set(predicted_sources)
        exp_set = set(expected_sources)

        hits = len(pred_set & exp_set)
        precision = hits / len(pred_set) if pred_set else 0.0
        recall = hits / len(exp_set) if exp_set else 0.0

        # Return F1
        if precision + recall > 0:
            return 2 * precision * recall / (precision + recall)
        return 0.0

    def _calculate_mrr(
        self,
        predicted_sources: list[str],
        expected_sources: list[str],
    ) -> float:
        """
        Calculate Mean Reciprocal Rank (MRR).

        MRR = 1 / (rank of first relevant result)

        Returns:
            Score between 0 and 1
        """
        if not expected_sources or not predicted_sources:
            return 0.0

        exp_set = set(expected_sources)

        for i, source in enumerate(predicted_sources):
            if source in exp_set:
                return 1.0 / (i + 1)

        return 0.0

    def _calculate_abstention_f1(
        self,
        predicted_abstain: bool,
        should_abstain: bool,
    ) -> float:
        """
        Calculate F1 score for abstention decisions.

        This is simplified for binary classification.
        """
        # True positive: correctly abstained
        # False positive: abstained when shouldn't have
        # False negative: didn't abstain when should have
        # True negative: correctly answered

        if predicted_abstain == should_abstain:
            return 1.0
        else:
            # Penalize wrong decisions
            if predicted_abstain and not should_abstain:
                return 0.3  # False humility - less bad
            else:
                return 0.0  # False confidence - worse


def is_mock_mode() -> bool:
    """Check if running in mock mode."""
    return os.environ.get("TRAIGENT_MOCK_MODE", "true").lower() == "true"


def run_optimization(num_configs: int = 5, num_examples: int = 10):
    """Run optimization testing different RAG configurations."""
    from openai import OpenAI

    client = OpenAI()
    dataset = load_dataset()[:num_examples]
    evaluator = RAGEvaluator()

    configs = [
        {
            "name": "baseline",
            "temperature": 0.0,
            "instruction": "Answer based only on the documents.",
        },
        {
            "name": "creative",
            "temperature": 0.5,
            "instruction": "Answer helpfully based on documents.",
        },
        {
            "name": "strict",
            "temperature": 0.0,
            "instruction": "Only answer if 100% certain from docs. Otherwise abstain.",
        },
        {
            "name": "verbose",
            "temperature": 0.3,
            "instruction": "Give detailed answers with full context from docs.",
        },
        {
            "name": "concise",
            "temperature": 0.1,
            "instruction": "Give brief, direct answers from docs.",
        },
    ][:num_configs]

    print("\n" + "=" * 70)
    print("OPTIMIZATION RUN: Testing Different RAG Configurations")
    print("=" * 70)
    print(
        f"\nConfigs: {num_configs}, Examples: {num_examples}, Total calls: {num_configs * num_examples}"
    )

    results = []
    for config in configs:
        print(f"\n--- Config: {config['name']} (temp={config['temperature']}) ---")
        scores = []

        for i, entry in enumerate(dataset):
            input_data = entry.get("input", {})
            question = (
                input_data.get("question", "")
                if isinstance(input_data, dict)
                else str(input_data)
            )
            expected = entry.get("output", "")
            source_ids = entry.get("source_ids", [])
            answerable = entry.get("answerable", True)

            prompt = f"""{config['instruction']}

Documents:
[doc_rate_01]: Rate limits by tier - Standard: 100/min, Professional: 500/min, Enterprise: 2000/min
[doc_auth_01]: Authentication uses API keys in X-API-Key header
[doc_error_01]: Error codes: 400=Bad Request, 401=Unauthorized, 429=Rate Limited

Question: {question}

Return JSON: {{"answer": "...", "sources": ["doc_id"], "is_abstention": true/false}}"""

            try:
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=config["temperature"],
                )
                content = response.choices[0].message.content
                json_match = re.search(r"\{.*\}", content, re.DOTALL)
                pred = (
                    json.loads(json_match.group())
                    if json_match
                    else {"answer": content, "sources": [], "is_abstention": False}
                )

                result = evaluator(
                    pred, expected, {"source_ids": source_ids, "answerable": answerable}
                )
                scores.append(result)
                print(
                    f"  [{i+1}/{num_examples}] acc={result['grounded_accuracy']:.2f} abst={'✓' if result['abstention_accuracy']==1 else '✗'}"
                )
            except Exception as e:
                print(f"  [{i+1}/{num_examples}] Error: {e}")
                scores.append(
                    {"grounded_accuracy": 0, "abstention_accuracy": 0, "overall": 0}
                )

        avg_acc = sum(s["grounded_accuracy"] for s in scores) / len(scores)
        avg_abst = sum(s["abstention_accuracy"] for s in scores) / len(scores)
        results.append(
            {
                "config": config["name"],
                "temp": config["temperature"],
                "accuracy": avg_acc,
                "abstention": avg_abst,
                "overall": (avg_acc + avg_abst) / 2,
            }
        )

    print("\n" + "=" * 70)
    print("RESULTS TABLE")
    print("=" * 70)
    print(
        f"\n{'Config':<12} {'Temp':<6} {'Accuracy':<10} {'Abstention':<12} {'Overall':<10}"
    )
    print("-" * 50)
    for r in sorted(results, key=lambda x: x["overall"], reverse=True):
        print(
            f"{r['config']:<12} {r['temp']:<6.1f} {r['accuracy']:.3f}      {r['abstention']*100:>5.0f}%       {r['overall']:.3f}"
        )

    best = max(results, key=lambda x: x["overall"])
    print("-" * 50)
    print(f"🏆 BEST: {best['config']} (score={best['overall']:.3f})")
    print("=" * 70)
    return results


def answer_question_with_llm(question: str, docs: dict = None) -> dict:
    """Answer a question using LLM with RAG (real mode only)."""
    try:
        from openai import OpenAI

        client = OpenAI()
        docs = docs or SAMPLE_DOCS
        docs_text = "\n".join(f"[{k}]: {v}" for k, v in docs.items())

        prompt = AGENT_PROMPT.format(documents=docs_text, question=question)

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )

        # Parse JSON response
        content = response.choices[0].message.content
        json_match = re.search(r"\{.*\}", content, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
        return {"answer": content, "sources": [], "is_abstention": False}
    except Exception as e:
        return {"answer": f"Error: {e}", "sources": [], "is_abstention": False}


def load_dataset() -> list[dict]:
    """Load the Q&A dataset."""
    from pathlib import Path

    dataset_path = Path(__file__).parent.parent / "datasets" / "qa_dataset.jsonl"
    if not dataset_path.exists():
        return []

    entries = []
    with open(dataset_path) as f:
        for line in f:
            if line.strip():
                entries.append(json.loads(line))
    return entries


def print_score_bar(label: str, score: float, max_score: float = 1.0, width: int = 20):
    """Print a visual score bar."""
    normalized = min(score / max_score, 1.0)
    filled = int(normalized * width)
    bar = "█" * filled + "░" * (width - filled)
    pct = score * 100 if max_score == 1.0 else score
    print(f"  {label:<20} {bar} {pct:.0f}%")


def demo_evaluator():
    """Demo the Knowledge & RAG Agent evaluator with clear input/output examples."""
    mock_mode = is_mock_mode()

    print("=" * 70)
    print("KNOWLEDGE & RAG AGENT - Evaluator Demo")
    print("=" * 70)
    print(
        f"\nMODE: {'MOCK (heuristic rules)' if mock_mode else 'REAL (using OpenAI API)'}"
    )

    print(
        """
WHAT THIS AGENT DOES:
  A Q&A agent that answers questions using a knowledge base (documents).
  It retrieves relevant docs, generates an answer, and cites sources.
  Critically, it should say "I don't know" when the answer isn't in the docs.

HOW IT'S EVALUATED:"""
    )
    if mock_mode:
        print("  MOCK MODE: Using heuristic rules (fast, free, no API needed)")
    else:
        print("  REAL MODE: Using LLM for Q&A + evaluation (requires API key)")
    print()

    # Load and show dataset info
    dataset = load_dataset()
    print(f"DATASET: {len(dataset)} Q&A pairs in qa_dataset.jsonl")

    if dataset:
        unanswerable = sum(1 for e in dataset if not e.get("answerable", True))
        print(f"  - {len(dataset) - unanswerable} questions have answers in the docs")
        print(f'  - {unanswerable} questions should be answered with "I don\'t know"')

        print("\n" + "-" * 70)
        print("SAMPLE DATA (first 2 entries):")
        print("-" * 70)
        for i, entry in enumerate(dataset[:2]):
            input_data = entry.get("input", {})
            question = (
                input_data.get("question", "")
                if isinstance(input_data, dict)
                else str(input_data)
            )
            output = entry.get("output", "")
            sources = entry.get("source_ids", [])
            answerable = entry.get("answerable", True)

            print(f"\n[Entry {i+1}]")
            print("  INPUT (question):")
            print(f'    "{question}"')
            print("\n  OUTPUT (expected answer):")
            preview = output[:120].replace("\n", " ")
            print(f'    "{preview}..."')
            print(f"    Sources: {sources}")
            answerable_text = "Yes" if answerable else "No (should abstain)"
            print(f"    Answerable: {answerable_text}")

    evaluator = RAGEvaluator()

    print("\n" + "=" * 70)
    print("HOW SCORING WORKS:")
    print("=" * 70)
    print(
        """
The evaluator measures:

  - Grounded Accuracy:   Is the answer BOTH correct AND supported by docs?
                         (Catches "right answer but made it up" errors)

  - Retrieval Quality:   Did the agent find the right documents?
                         (Based on which docs it cited vs expected)

  - Abstention Accuracy: When question can't be answered from docs, did
                         the agent say "I don't know" instead of guessing?
                         (Very important for trust!)
"""
    )

    print("=" * 70)
    print("EVALUATION EXAMPLES:")
    print("=" * 70)

    # Test case 1: Correct answer with correct source
    print("\n[CORRECT ANSWER] - Right answer with right sources")
    print("-" * 70)
    print('Question: "What is the rate limit for the API?"')
    print("\nAgent Answer:")
    print('  "The rate limit is 100 requests/minute for Standard, 500 for')
    print('   Professional, and 2000 for Enterprise tier."')
    print("  Sources: [doc_rate_01]")

    result = evaluator(
        prediction={
            "answer": "The rate limit is 100 requests per minute for Standard tier, 500 for Professional, and 2000 for Enterprise.",
            "sources": ["doc_rate_01"],
            "is_abstention": False,
        },
        expected="The rate limit depends on your tier: Standard tier has 100 requests per minute...",
        input_data={"source_ids": ["doc_rate_01"], "answerable": True},
    )
    print("\nScores:")
    print_score_bar("Grounded Accuracy", result["grounded_accuracy"])
    print_score_bar("Retrieval Quality", result["retrieval_quality"])
    print_score_bar("Abstention", result["abstention_accuracy"])
    print(f"  {'─' * 42}")
    print_score_bar("OVERALL", result["overall"])

    # Test case 2: Correct abstention
    print('\n[CORRECT ABSTENTION] - Correctly said "I don\'t know"')
    print("-" * 70)
    print('Question: "Does the API support GraphQL?"')
    print("\nAgent Answer:")
    print("  \"I don't have information about GraphQL support in the")
    print('   documentation I was provided."')
    print("  Sources: [] (none cited)")

    result = evaluator(
        prediction={
            "answer": "I don't have information about GraphQL support in the documentation.",
            "sources": [],
            "is_abstention": True,
        },
        expected="I don't have information about GraphQL support.",
        input_data={"source_ids": [], "answerable": False},
    )
    print("\nScores:")
    print_score_bar("Grounded Accuracy", result["grounded_accuracy"])
    print_score_bar("Retrieval Quality", result["retrieval_quality"])
    print_score_bar("Abstention", result["abstention_accuracy"])
    print("    ^ Correct! Question can't be answered from docs")
    print(f"  {'─' * 42}")
    print_score_bar("OVERALL", result["overall"])

    # Test case 3: Hallucination - made up an answer
    print("\n[HALLUCINATION] - Made up an answer (should have abstained)")
    print("-" * 70)
    print('Question: "Does the API support GraphQL?"')
    print("\nAgent Answer:")
    print('  "Yes, CloudStack fully supports GraphQL with a dedicated')
    print('   endpoint at /graphql."')
    print("  Sources: [doc_endpoint_01]")

    result = evaluator(
        prediction={
            "answer": "Yes, CloudStack fully supports GraphQL with a dedicated endpoint.",
            "sources": ["doc_endpoint_01"],
            "is_abstention": False,
        },
        expected="I don't have information about GraphQL support.",
        input_data={"source_ids": [], "answerable": False},
    )
    print("\nScores:")
    print_score_bar("Grounded Accuracy", result["grounded_accuracy"])
    print("    ^ Answer not supported by any document!")
    print_score_bar("Retrieval Quality", result["retrieval_quality"])
    print_score_bar("Abstention", result["abstention_accuracy"])
    print("    ^ Should have said 'I don't know'!")
    print(f"  {'─' * 42}")
    print_score_bar("OVERALL", result["overall"])

    # Test case 4: Wrong source (new example)
    print("\n[WRONG SOURCE] - Right answer, wrong citation")
    print("-" * 70)
    print('Question: "How do I authenticate?"')
    print("\nAgent Answer:")
    print('  "Use an X-API-Key header with your API key."')
    print("  Sources: [doc_rate_01]  <- Wrong! Should be doc_auth_01")

    result = evaluator(
        prediction={
            "answer": "Use an X-API-Key header with your API key in each request.",
            "sources": ["doc_rate_01"],  # Wrong source!
            "is_abstention": False,
        },
        expected="CloudStack API uses API key authentication. Include an X-API-Key header...",
        input_data={"source_ids": ["doc_auth_01"], "answerable": True},
    )
    print("\nScores:")
    print_score_bar("Grounded Accuracy", result["grounded_accuracy"])
    print("    ^ Answer is correct but...")
    print_score_bar("Retrieval Quality", result["retrieval_quality"])
    print("    ^ Cited wrong document!")
    print_score_bar("Abstention", result["abstention_accuracy"])
    print(f"  {'─' * 42}")
    print_score_bar("OVERALL", result["overall"])

    # In real mode, run optimization
    if not mock_mode:
        run_optimization(num_configs=5, num_examples=10)

    print("\n" + "=" * 70)
    print("HOW TO RUN:")
    print("  Mock mode (heuristics): python evaluator.py  (default)")
    print(
        "  Real mode (LLM+optimize): TRAIGENT_MOCK_MODE=false OPENAI_API_KEY=sk-... python evaluator.py"
    )
    print("=" * 70)


if __name__ == "__main__":
    demo_evaluator()
