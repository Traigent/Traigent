"""
Diversity Analyzer for Detecting Similarity and Ensuring Example Variety.

This module provides sophisticated analysis to prevent repetition and ensure
maximum diversity across generated examples.
"""

import hashlib
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Dict, List, Set, Tuple


@dataclass
class DiversityMetrics:
    """Metrics for measuring diversity of a problem set."""

    pattern_entropy: float  # Shannon entropy of pattern distribution
    difficulty_balance: float  # How well balanced across difficulties
    topic_coverage: float  # Unique topics / total examples
    lexical_diversity: float  # Unique words / total words
    structural_diversity: float  # Variety in input/output structures
    similarity_score: float  # Average pairwise similarity (lower is better)

    @property
    def overall_diversity_score(self) -> float:
        """Compute overall diversity score (0-100)."""
        weights = {
            "pattern_entropy": 0.20,
            "difficulty_balance": 0.15,
            "topic_coverage": 0.20,
            "lexical_diversity": 0.15,
            "structural_diversity": 0.15,
            "similarity_score": 0.15,
        }

        # Invert similarity score (lower similarity = higher diversity)
        inverted_similarity = 1.0 - self.similarity_score

        score = (
            weights["pattern_entropy"] * self.pattern_entropy
            + weights["difficulty_balance"] * self.difficulty_balance
            + weights["topic_coverage"] * self.topic_coverage
            + weights["lexical_diversity"] * self.lexical_diversity
            + weights["structural_diversity"] * self.structural_diversity
            + weights["similarity_score"] * inverted_similarity
        )

        return round(score * 100, 2)


class DiversityAnalyzer:
    """
    Analyzes diversity of generated examples to prevent repetition.

    Uses multiple techniques including:
    - Semantic fingerprinting
    - Pattern distribution analysis
    - Lexical diversity measurement
    - Structural variety assessment
    """

    def __init__(self, similarity_threshold: float = 0.85):
        """
        Initialize diversity analyzer.

        Args:
            similarity_threshold: Threshold above which examples are too similar
        """
        self.similarity_threshold = similarity_threshold
        self.fingerprint_cache: Dict[int, str] = {}
        self.word_frequency: Counter = Counter()
        self.pattern_counts: Counter = Counter()
        self.structure_signatures: Set[str] = set()

    def analyze_diversity(self, examples: List[Dict]) -> DiversityMetrics:
        """
        Analyze diversity of a set of examples.

        Args:
            examples: List of generated examples

        Returns:
            DiversityMetrics object with comprehensive analysis
        """
        if not examples:
            return DiversityMetrics(0, 0, 0, 0, 0, 0)

        # Calculate individual metrics
        pattern_entropy = self._calculate_pattern_entropy(examples)
        difficulty_balance = self._calculate_difficulty_balance(examples)
        topic_coverage = self._calculate_topic_coverage(examples)
        lexical_diversity = self._calculate_lexical_diversity(examples)
        structural_diversity = self._calculate_structural_diversity(examples)
        similarity_score = self._calculate_average_similarity(examples)

        return DiversityMetrics(
            pattern_entropy=pattern_entropy,
            difficulty_balance=difficulty_balance,
            topic_coverage=topic_coverage,
            lexical_diversity=lexical_diversity,
            structural_diversity=structural_diversity,
            similarity_score=similarity_score,
        )

    def check_similarity(
        self, new_example: Dict, existing_examples: List[Dict]
    ) -> Tuple[bool, float]:
        """
        Check if a new example is too similar to existing ones.

        Args:
            new_example: The new example to check
            existing_examples: List of existing examples

        Returns:
            Tuple of (is_too_similar, max_similarity_score)
        """
        new_fingerprint = self._generate_fingerprint(new_example)
        max_similarity = 0.0

        for existing in existing_examples:
            existing_fingerprint = self._generate_fingerprint(existing)
            similarity = self._calculate_similarity(
                new_fingerprint, existing_fingerprint
            )
            max_similarity = max(max_similarity, similarity)

            if similarity > self.similarity_threshold:
                return True, similarity

        return False, max_similarity

    def suggest_diversity_improvements(self, examples: List[Dict]) -> List[str]:
        """
        Suggest improvements to increase diversity.

        Args:
            examples: Current set of examples

        Returns:
            List of suggestions
        """
        suggestions = []
        metrics = self.analyze_diversity(examples)

        if metrics.pattern_entropy < 0.7:
            suggestions.append(
                "Increase variety in example patterns (questions, statements, requests)"
            )

        if metrics.difficulty_balance < 0.8:
            distribution = self._get_difficulty_distribution(examples)
            under_represented = [
                d for d, c in distribution.items() if c < len(examples) * 0.15
            ]
            if under_represented:
                suggestions.append(
                    f"Add more examples for difficulties: {', '.join(under_represented)}"
                )

        if metrics.topic_coverage < 0.6:
            suggestions.append(
                "Expand topic variety - current examples may be too focused on specific areas"
            )

        if metrics.lexical_diversity < 0.5:
            suggestions.append("Use more varied vocabulary and terminology")

        if metrics.structural_diversity < 0.7:
            suggestions.append(
                "Vary input/output structures (e.g., different field combinations)"
            )

        if metrics.similarity_score > 0.3:
            suggestions.append(
                "Examples are too similar - introduce more unique scenarios"
            )

        return suggestions

    def _generate_fingerprint(self, example: Dict) -> str:
        """Generate a semantic fingerprint for an example."""
        # Extract all text content
        text_parts = []

        if isinstance(example.get("input_data"), dict):
            for value in example["input_data"].values():
                if isinstance(value, str):
                    text_parts.append(value)
        elif isinstance(example.get("input_data"), str):
            text_parts.append(example["input_data"])

        if isinstance(example.get("expected_output"), str):
            text_parts.append(example["expected_output"])

        # Combine and normalize
        combined_text = " ".join(text_parts).lower()

        # Extract features for fingerprint
        features = []

        # Word-level features
        words = combined_text.split()
        features.append(f"len:{len(words)//10}")  # Length bucket
        features.append(f"first:{words[0] if words else 'empty'}")
        features.append(f"last:{words[-1] if words else 'empty'}")

        # Pattern features
        if "?" in combined_text:
            features.append("question")
        if re.search(r"\b(need|want|require)\b", combined_text):
            features.append("request")
        if re.search(r"\b(issue|problem|error)\b", combined_text):
            features.append("problem")

        # Domain indicators
        domains = ["technical", "financial", "medical", "legal", "customer"]
        for domain in domains:
            if domain in combined_text:
                features.append(f"domain:{domain}")

        # Structure signature
        if isinstance(example.get("input_data"), dict):
            keys = sorted(example["input_data"].keys())
            features.append(f"struct:{'-'.join(keys)}")

        # Create fingerprint
        fingerprint = "|".join(sorted(features))
        return hashlib.sha256(fingerprint.encode()).hexdigest()

    def _calculate_similarity(self, fingerprint1: str, fingerprint2: str) -> float:
        """Calculate similarity between two fingerprints."""
        # Simple character-based similarity for fingerprints
        return SequenceMatcher(None, fingerprint1, fingerprint2).ratio()

    def _calculate_pattern_entropy(self, examples: List[Dict]) -> float:
        """Calculate Shannon entropy of pattern distribution."""
        patterns = []

        for ex in examples:
            text = self._extract_text(ex)
            if "?" in text:
                patterns.append("question")
            elif re.search(r"\b(need|want|please)\b", text):
                patterns.append("request")
            elif re.search(r"\b(is|are|was|were)\b", text):
                patterns.append("statement")
            else:
                patterns.append("other")

        # Calculate entropy
        pattern_counts = Counter(patterns)
        total = len(patterns)
        entropy = 0.0

        for count in pattern_counts.values():
            if count > 0:
                probability = count / total
                entropy -= probability * math.log2(probability)

        # Normalize to 0-1 range (max entropy for 4 patterns is 2)
        return min(entropy / 2.0, 1.0)

    def _calculate_difficulty_balance(self, examples: List[Dict]) -> float:
        """Calculate how well balanced the difficulty distribution is."""
        distribution = self._get_difficulty_distribution(examples)

        if not distribution:
            return 0.0

        # Ideal distribution percentages
        ideal = {
            "easy": 0.20,
            "medium": 0.30,
            "hard": 0.30,
            "very_hard": 0.15,
            "expert": 0.05,
        }

        # Calculate deviation from ideal
        total_examples = len(examples)
        total_deviation = 0.0

        for difficulty, ideal_ratio in ideal.items():
            actual_count = distribution.get(difficulty, 0)
            actual_ratio = actual_count / total_examples
            deviation = abs(actual_ratio - ideal_ratio)
            total_deviation += deviation

        # Convert to 0-1 score (lower deviation = higher score)
        max_possible_deviation = 2.0  # Sum of all ratios could be 2
        balance_score = 1.0 - (total_deviation / max_possible_deviation)

        return max(0.0, balance_score)

    def _calculate_topic_coverage(self, examples: List[Dict]) -> float:
        """Calculate topic diversity coverage."""
        unique_topics = set()

        for ex in examples:
            text = self._extract_text(ex)
            # Extract key terms as topics
            words = text.lower().split()
            key_terms = [w for w in words if len(w) > 4 and w.isalpha()]
            unique_topics.update(key_terms[:3])  # Top 3 terms per example

        # Coverage is ratio of unique topics to total examples
        coverage = len(unique_topics) / max(len(examples), 1)
        return min(coverage, 1.0)  # Cap at 1.0

    def _calculate_lexical_diversity(self, examples: List[Dict]) -> float:
        """Calculate lexical diversity (unique words / total words)."""
        all_words = []

        for ex in examples:
            text = self._extract_text(ex)
            words = text.lower().split()
            all_words.extend(words)

        if not all_words:
            return 0.0

        unique_words = set(all_words)
        diversity = len(unique_words) / len(all_words)

        return diversity

    def _calculate_structural_diversity(self, examples: List[Dict]) -> float:
        """Calculate diversity in input/output structures."""
        structures = set()

        for ex in examples:
            # Get input structure signature
            if isinstance(ex.get("input_data"), dict):
                input_sig = tuple(sorted(ex["input_data"].keys()))
            else:
                input_sig = ("single_value",)

            # Get output type signature
            output = ex.get("expected_output")
            if isinstance(output, dict):
                output_sig = tuple(sorted(output.keys()))
            elif isinstance(output, list):
                output_sig = ("list",)
            else:
                output_sig = (type(output).__name__,)

            structures.add((input_sig, output_sig))

        # Diversity is ratio of unique structures to total examples
        diversity = len(structures) / max(len(examples), 1)
        return min(diversity, 1.0)

    def _calculate_average_similarity(self, examples: List[Dict]) -> float:
        """Calculate average pairwise similarity (sampled for efficiency)."""
        if len(examples) < 2:
            return 0.0

        # Sample pairs for large sets
        max_pairs = 100
        total_similarity = 0.0
        pairs_checked = 0

        # Generate fingerprints
        fingerprints = [self._generate_fingerprint(ex) for ex in examples]

        # Sample pairs
        import random

        indices = list(range(len(examples)))

        for _ in range(min(max_pairs, len(examples) * 2)):
            i, j = random.sample(indices, 2)
            similarity = self._calculate_similarity(fingerprints[i], fingerprints[j])
            total_similarity += similarity
            pairs_checked += 1

        return total_similarity / max(pairs_checked, 1)

    def _extract_text(self, example: Dict) -> str:
        """Extract text content from example."""
        text_parts = []

        if isinstance(example.get("input_data"), dict):
            for value in example["input_data"].values():
                if isinstance(value, str):
                    text_parts.append(value)
        elif isinstance(example.get("input_data"), str):
            text_parts.append(example["input_data"])

        if isinstance(example.get("expected_output"), str):
            text_parts.append(example["expected_output"])

        return " ".join(text_parts)

    def _get_difficulty_distribution(self, examples: List[Dict]) -> Dict[str, int]:
        """Get difficulty distribution from examples."""
        distribution = defaultdict(int)

        for ex in examples:
            difficulty = ex.get("difficulty", "medium")
            distribution[difficulty] += 1

        return dict(distribution)
