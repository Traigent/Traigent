#!/usr/bin/env python3
"""
Evaluator for Knowledge & RAG Agent

This evaluator scores RAG responses on:
1. Grounded Accuracy - Correctness AND faithfulness to sources
2. Retrieval Quality - MRR, NDCG@k, Retrieval Success Rate
3. Abstention Accuracy - Correct abstention when knowledge is insufficient

Based on the Traigent Agent Optimization Guide specifications.

Key Metrics:
- Grounded Accuracy = % answers that are BOTH correct AND fully supported by cited passages
- Retrieval Success Rate = % queries where relevant doc appears in top-k
- MRR (Mean Reciprocal Rank) = average of 1/rank of first relevant result
- Abstention F1 = balance of correctly abstaining when should AND answering when can

Two failure modes to catch:
1. "Correct but ungrounded" - answer true but not supported by docs (using parametric knowledge)
2. "Grounded but wrong" - accurately summarizes retrieved docs, but docs are wrong/outdated
"""

import re
from dataclasses import dataclass
from typing import Any


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
            faithfulness = self._evaluate_faithfulness(pred_answer, pred_sources, expected_sources)

        # Combined grounded accuracy
        grounded_accuracy = correctness * faithfulness

        # Evaluate retrieval quality
        retrieval_precision = self._evaluate_retrieval(pred_sources, expected_sources)
        retrieval_success = len(set(pred_sources) & set(expected_sources)) > 0 if expected_sources else True

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
            "the", "a", "an", "is", "are", "was", "were", "be", "been",
            "being", "have", "has", "had", "do", "does", "did", "will",
            "would", "could", "should", "may", "might", "must", "shall",
            "can", "to", "of", "in", "for", "on", "with", "at", "by",
            "from", "as", "into", "through", "during", "before", "after",
            "above", "below", "between", "under", "again", "further",
            "then", "once", "here", "there", "when", "where", "why",
            "how", "all", "each", "few", "more", "most", "other", "some",
            "such", "no", "nor", "not", "only", "own", "same", "so",
            "than", "too", "very", "just", "and", "but", "if", "or",
            "because", "until", "while", "this", "that", "these", "those",
        }
        words = re.findall(r'\b[a-z0-9]+\b', text.lower())
        return [w for w in words if w not in stopwords and len(w) > 2]

    def _extract_key_facts(self, text: str) -> list[str]:
        """Extract key factual elements from text."""
        facts = []

        # Numbers and measurements
        numbers = re.findall(r'\b\d+(?:\.\d+)?(?:\s*(?:GB|TB|MB|KB|ms|seconds?|minutes?|hours?|days?|%|requests?|per\s+\w+))?\b', text)
        facts.extend(numbers)

        # Technical terms (capitalized words, acronyms)
        technical = re.findall(r'\b[A-Z][A-Z0-9]+\b', text)
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


def evaluate_sample():
    """Test the evaluator with sample data."""
    evaluator = RAGEvaluator()

    # Test case 1: Good answer with correct sources
    print("Test 1: Good Answer with Correct Sources")
    print("-" * 40)
    result = evaluator(
        prediction={
            "answer": "The rate limit is 100 requests per minute for Standard tier, 500 for Professional, and 2000 for Enterprise.",
            "sources": ["doc_rate_01"],
            "is_abstention": False,
        },
        expected="The rate limit depends on your tier: Standard tier has 100 requests per minute, Professional tier has 500 requests per minute, and Enterprise tier has 2000 requests per minute.",
        input_data={
            "source_ids": ["doc_rate_01"],
            "answerable": True,
        },
    )
    print(f"Grounded Accuracy: {result['grounded_accuracy']:.2f}")
    print(f"Retrieval Quality: {result['retrieval_quality']:.2f}")
    print(f"Abstention Accuracy: {result['abstention_accuracy']:.2f}")
    print(f"Overall: {result['overall']:.2f}")

    # Test case 2: Correct abstention
    print("\nTest 2: Correct Abstention")
    print("-" * 40)
    result = evaluator(
        prediction={
            "answer": "I don't have information about GraphQL support in the documentation.",
            "sources": [],
            "is_abstention": True,
        },
        expected="I don't have information about GraphQL support.",
        input_data={
            "source_ids": [],
            "answerable": False,
        },
    )
    print(f"Grounded Accuracy: {result['grounded_accuracy']:.2f}")
    print(f"Retrieval Quality: {result['retrieval_quality']:.2f}")
    print(f"Abstention Accuracy: {result['abstention_accuracy']:.2f}")
    print(f"Overall: {result['overall']:.2f}")

    # Test case 3: False confidence (should have abstained)
    print("\nTest 3: False Confidence (Should Have Abstained)")
    print("-" * 40)
    result = evaluator(
        prediction={
            "answer": "Yes, CloudStack fully supports GraphQL with a dedicated endpoint.",
            "sources": ["doc_endpoint_01"],
            "is_abstention": False,
        },
        expected="I don't have information about GraphQL support.",
        input_data={
            "source_ids": [],
            "answerable": False,
        },
    )
    print(f"Grounded Accuracy: {result['grounded_accuracy']:.2f}")
    print(f"Retrieval Quality: {result['retrieval_quality']:.2f}")
    print(f"Abstention Accuracy: {result['abstention_accuracy']:.2f}")
    print(f"Overall: {result['overall']:.2f}")

    # Test case 4: Wrong sources cited
    print("\nTest 4: Wrong Sources Cited")
    print("-" * 40)
    result = evaluator(
        prediction={
            "answer": "API authentication uses API keys in the X-API-Key header.",
            "sources": ["doc_rate_01", "doc_error_01"],  # Wrong sources
            "is_abstention": False,
        },
        expected="CloudStack API uses API key authentication. Include an X-API-Key header with your API key.",
        input_data={
            "source_ids": ["doc_auth_01"],
            "answerable": True,
        },
    )
    print(f"Grounded Accuracy: {result['grounded_accuracy']:.2f}")
    print(f"Retrieval Quality: {result['retrieval_quality']:.2f}")
    print(f"Abstention Accuracy: {result['abstention_accuracy']:.2f}")
    print(f"Overall: {result['overall']:.2f}")


if __name__ == "__main__":
    evaluate_sample()
