"""
Example Memory System for Compact Storage and Retrieval of Generated Examples.

This module provides efficient storage of example patterns to maintain diversity
across large-scale generation while minimizing token usage.
"""

import hashlib
import json
import re
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set

from traigent.utils.secure_path import validate_path

@dataclass
class ExampleSummary:
    """Compact representation of an example for diversity tracking."""

    id: int
    difficulty: str
    pattern_type: str
    domain_markers: List[str]
    length_category: str  # short, medium, long
    complexity_features: List[str]
    topic_hash: str  # Hash of main topic for quick comparison

    def to_compact_dict(self) -> Dict[str, Any]:
        """Convert to compact dictionary for API inclusion."""
        return {
            "d": self.difficulty[0],  # e/m/h/v/x for easy/medium/hard/very_hard/expert
            "p": self.pattern_type[:3],  # First 3 chars of pattern
            "t": self.topic_hash[:6],  # First 6 chars of hash
            "c": len(self.complexity_features),  # Complexity count
            "l": self.length_category[0],  # s/m/l
        }

    def to_readable_summary(self) -> str:
        """Create human-readable summary for diversity guidance."""
        features = (
            f"+{','.join(self.complexity_features[:2])}"
            if self.complexity_features
            else ""
        )
        return f"{self.difficulty}:{self.pattern_type}:{self.domain_markers[0] if self.domain_markers else 'general'}{features}"


class ExampleMemory:
    """
    Manages memory of generated examples for diversity optimization.

    Stores compact representations to guide future generation while
    minimizing token usage in API calls.
    """

    def __init__(self, max_summaries_per_batch: int = 30):
        """
        Initialize example memory.

        Args:
            max_summaries_per_batch: Maximum summaries to include in each generation batch
        """
        self.max_summaries_per_batch = max_summaries_per_batch
        self.examples_by_difficulty: Dict[str, List[ExampleSummary]] = defaultdict(list)
        self.examples_by_pattern: Dict[str, List[ExampleSummary]] = defaultdict(list)
        self.topic_hashes: Set[str] = set()
        self.total_examples = 0

        # Pattern detection rules
        self.pattern_rules = {
            "single_issue": r"^[^,;]+$",
            "multi_issue": r"[,;]|and|also|plus",
            "question_form": r"\?|how|what|when|where|why",
            "statement_form": r"^[A-Z].*\.$",
            "complaint": r"issue|problem|broken|failed|error",
            "request": r"need|want|please|could|would",
            "technical": r"API|database|server|code|function|algorithm",
            "business": r"revenue|profit|customer|market|sales",
            "edge_case": r"unusual|rare|special|exception|corner",
        }

        # Complexity indicators
        self.complexity_indicators = {
            "ambiguity": r"maybe|possibly|might|could be|unclear",
            "technical_jargon": r"API|SDK|HTTP|JSON|async|backend",
            "multi_stakeholder": r"team|department|company|vendor|client",
            "temporal": r"yesterday|today|tomorrow|last week|deadline",
            "conditional": r"if|when|unless|except|but only",
            "negation": r"not|never|no|without|unable",
        }

    def add_example(self, example: Dict[str, Any], example_id: int) -> ExampleSummary:
        """
        Add an example to memory and return its summary.

        Args:
            example: The generated example dictionary
            example_id: Unique identifier for the example

        Returns:
            ExampleSummary object
        """
        # Extract text content for analysis
        text_content = self._extract_text_content(example)

        # Analyze example properties
        difficulty = example.get("difficulty", "medium")
        pattern_type = self._detect_pattern_type(text_content)
        domain_markers = self._extract_domain_markers(text_content)
        length_category = self._categorize_length(text_content)
        complexity_features = self._detect_complexity_features(text_content)
        topic_hash = self._generate_topic_hash(text_content)

        # Create summary
        summary = ExampleSummary(
            id=example_id,
            difficulty=difficulty,
            pattern_type=pattern_type,
            domain_markers=domain_markers,
            length_category=length_category,
            complexity_features=complexity_features,
            topic_hash=topic_hash,
        )

        # Store in memory
        self.examples_by_difficulty[difficulty].append(summary)
        self.examples_by_pattern[pattern_type].append(summary)
        self.topic_hashes.add(topic_hash)
        self.total_examples += 1

        return summary

    def get_diverse_summaries(
        self, target_difficulty: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get diverse set of example summaries for inclusion in generation.

        Args:
            target_difficulty: Optional difficulty to focus on

        Returns:
            List of compact example summaries
        """
        summaries = []

        # If targeting specific difficulty, weight selection
        if target_difficulty:
            # 50% from target difficulty, 50% from others
            target_count = self.max_summaries_per_batch // 2
            other_count = self.max_summaries_per_batch - target_count

            # Get from target difficulty
            if target_difficulty in self.examples_by_difficulty:
                target_examples = self.examples_by_difficulty[target_difficulty]
                selected = self._select_diverse_subset(target_examples, target_count)
                summaries.extend([ex.to_compact_dict() for ex in selected])

            # Get from other difficulties
            other_examples = []
            for diff, examples in self.examples_by_difficulty.items():
                if diff != target_difficulty:
                    other_examples.extend(examples)

            if other_examples:
                selected = self._select_diverse_subset(other_examples, other_count)
                summaries.extend([ex.to_compact_dict() for ex in selected])
        else:
            # Get diverse selection across all difficulties
            all_examples = []
            for examples in self.examples_by_difficulty.values():
                all_examples.extend(examples)

            selected = self._select_diverse_subset(
                all_examples, self.max_summaries_per_batch
            )
            summaries = [ex.to_compact_dict() for ex in selected]

        return summaries

    def get_pattern_distribution(self) -> Dict[str, int]:
        """Get distribution of pattern types."""
        return {
            pattern: len(examples)
            for pattern, examples in self.examples_by_pattern.items()
        }

    def get_difficulty_distribution(self) -> Dict[str, int]:
        """Get distribution of difficulty levels."""
        return {
            diff: len(examples)
            for diff, examples in self.examples_by_difficulty.items()
        }

    def _extract_text_content(self, example: Dict[str, Any]) -> str:
        """Extract all text content from example for analysis."""
        text_parts = []

        # Extract from input_data
        if isinstance(example.get("input_data"), dict):
            for value in example["input_data"].values():
                if isinstance(value, str):
                    text_parts.append(value)

        # Extract from expected_output if string
        if isinstance(example.get("expected_output"), str):
            text_parts.append(example["expected_output"])

        return " ".join(text_parts).lower()

    def _detect_pattern_type(self, text: str) -> str:
        """Detect the pattern type of the example."""
        for pattern_name, pattern_regex in self.pattern_rules.items():
            if re.search(pattern_regex, text, re.IGNORECASE):
                return pattern_name
        return "general"

    def _extract_domain_markers(self, text: str) -> List[str]:
        """Extract domain-specific markers from text."""
        markers = []
        words = set(text.split())

        # Common domain keywords
        domain_keywords = {
            "technical": {"api", "code", "bug", "server", "database", "function"},
            "financial": {"payment", "invoice", "budget", "cost", "revenue", "profit"},
            "customer": {
                "order",
                "refund",
                "shipping",
                "product",
                "service",
                "support",
            },
            "medical": {"patient", "symptom", "diagnosis", "treatment", "medicine"},
            "legal": {"contract", "agreement", "compliance", "law", "regulation"},
            "educational": {"student", "course", "lesson", "assignment", "grade"},
        }

        for domain, keywords in domain_keywords.items():
            if words & keywords:
                markers.append(domain)

        # Extract specific terms as markers
        important_words = [w for w in words if len(w) > 4 and w.isalpha()]
        markers.extend(important_words[:3])  # Top 3 important words

        return markers[:5]  # Limit to 5 markers

    def _categorize_length(self, text: str) -> str:
        """Categorize text length."""
        word_count = len(text.split())
        if word_count < 10:
            return "short"
        elif word_count < 50:
            return "medium"
        else:
            return "long"

    def _detect_complexity_features(self, text: str) -> List[str]:
        """Detect complexity features in the text."""
        features = []

        for feature_name, pattern in self.complexity_indicators.items():
            if re.search(pattern, text, re.IGNORECASE):
                features.append(feature_name)

        return features

    def _generate_topic_hash(self, text: str) -> str:
        """Generate a hash representing the main topic."""
        # Extract key terms (nouns and important words)
        words = text.split()
        key_terms = [w for w in words if len(w) > 4 and w.isalpha()]
        key_terms.sort()  # Sort for consistency

        # Create hash from key terms
        topic_string = " ".join(key_terms[:5])  # Top 5 key terms
        return hashlib.sha256(topic_string.encode()).hexdigest()

    def _select_diverse_subset(
        self, examples: List[ExampleSummary], count: int
    ) -> List[ExampleSummary]:
        """Select diverse subset of examples using clustering-like approach."""
        if len(examples) <= count:
            return examples

        selected = []
        remaining = examples.copy()

        # Start with one from each pattern type
        patterns_seen = set()
        for ex in remaining[:]:
            if ex.pattern_type not in patterns_seen and len(selected) < count:
                selected.append(ex)
                remaining.remove(ex)
                patterns_seen.add(ex.pattern_type)

        # Add examples with unique topic hashes
        topics_seen = {ex.topic_hash for ex in selected}
        for ex in remaining[:]:
            if ex.topic_hash not in topics_seen and len(selected) < count:
                selected.append(ex)
                remaining.remove(ex)
                topics_seen.add(ex.topic_hash)

        # Fill remaining slots with diverse difficulties
        while len(selected) < count and remaining:
            # Pick the example most different from current selection
            best_ex = None
            best_score = -1

            for ex in remaining:
                # Simple diversity score based on unique attributes
                score = 0
                if ex.difficulty not in [s.difficulty for s in selected[-5:]]:
                    score += 2
                if ex.length_category not in [s.length_category for s in selected[-3:]]:
                    score += 1
                if (
                    len(ex.complexity_features) != len(selected[-1].complexity_features)
                    if selected
                    else 0
                ):
                    score += 1

                if score > best_score:
                    best_score = score
                    best_ex = ex

            if best_ex:
                selected.append(best_ex)
                remaining.remove(best_ex)

        return selected

    def save_to_file(self, filepath: str):
        """Save memory state to file for persistence."""
        state = {
            "total_examples": self.total_examples,
            "topic_hashes": list(self.topic_hashes),
            "examples_by_difficulty": {
                diff: [
                    {
                        "id": ex.id,
                        "difficulty": ex.difficulty,
                        "pattern_type": ex.pattern_type,
                        "domain_markers": ex.domain_markers,
                        "length_category": ex.length_category,
                        "complexity_features": ex.complexity_features,
                        "topic_hash": ex.topic_hash,
                    }
                    for ex in examples
                ]
                for diff, examples in self.examples_by_difficulty.items()
            },
        }

        output_path = validate_path(filepath, Path.cwd())
        with open(output_path, "w") as f:
            json.dump(state, f, indent=2)

    def load_from_file(self, filepath: str):
        """Load memory state from file."""
        input_path = validate_path(filepath, Path.cwd(), must_exist=True)
        with open(input_path) as f:
            state = json.load(f)

        self.total_examples = state["total_examples"]
        self.topic_hashes = set(state["topic_hashes"])

        # Reconstruct example summaries
        self.examples_by_difficulty.clear()
        self.examples_by_pattern.clear()

        for diff, examples_data in state["examples_by_difficulty"].items():
            for ex_data in examples_data:
                summary = ExampleSummary(**ex_data)
                self.examples_by_difficulty[diff].append(summary)
                self.examples_by_pattern[summary.pattern_type].append(summary)
