"""
Problem Intelligence Engine using Claude Code SDK.

This module provides intelligent analysis and generation capabilities for
LangChain optimization problems, leveraging Claude's understanding of
natural language descriptions and code patterns.
"""

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple


@dataclass
class ProblemInsights:
    """Insights generated from analyzing a problem description."""

    suggested_type: str  # classification, generation, analysis, structured
    complexity_level: str  # Beginner, Advanced, Expert
    suggested_domain: str  # inferred domain
    input_structure: Dict[str, Any]  # suggested input format
    output_structure: Dict[str, Any]  # suggested output format
    suggested_categories: List[str]  # for classification problems
    suggested_metrics: List[str]  # appropriate evaluation metrics
    difficulty_tiers: List[str]  # suggested difficulty progression
    estimated_examples_needed: int  # recommended number of examples
    key_challenges: List[str]  # main challenges this problem tests
    reasoning: str  # explanation of analysis


@dataclass
class ExampleSpecification:
    """Specification for generating examples."""

    count: int
    difficulty_distribution: Dict[str, int]  # tier -> count
    focus_areas: List[str]  # specific areas to emphasize
    edge_cases: bool  # include edge cases
    domain_specific: bool  # use domain-specific patterns


class ProblemIntelligence:
    """
    Core intelligence engine for problem analysis and generation.

    Uses Claude Code SDK to understand natural language descriptions and
    generate appropriate problem structures and examples.
    """

    def __init__(self):
        """Initialize the intelligence engine."""
        # Load domain patterns
        self.domain_patterns = self._load_domain_patterns()

        # Problem type indicators
        self.type_indicators = {
            "classification": [
                "classify",
                "categorize",
                "label",
                "tag",
                "identify type",
                "determine category",
                "sort into",
                "recognize as",
            ],
            "generation": [
                "generate",
                "create",
                "write",
                "compose",
                "produce",
                "draft",
                "synthesize",
                "formulate",
                "construct",
            ],
            "analysis": [
                "analyze",
                "evaluate",
                "assess",
                "review",
                "examine",
                "investigate",
                "study",
                "inspect",
                "critique",
            ],
            "structured": [
                "extract",
                "parse",
                "structure",
                "format",
                "organize",
                "identify key points",
                "summarize with structure",
            ],
        }

        # Domain indicators
        self.domain_indicators = {
            "legal": [
                "legal",
                "law",
                "contract",
                "compliance",
                "regulation",
                "attorney",
                "court",
                "lawsuit",
                "statute",
                "liability",
            ],
            "medical": [
                "medical",
                "health",
                "diagnosis",
                "patient",
                "clinical",
                "doctor",
                "treatment",
                "symptom",
                "disease",
                "therapy",
            ],
            "financial": [
                "financial",
                "finance",
                "banking",
                "investment",
                "trading",
                "credit",
                "loan",
                "insurance",
                "accounting",
                "audit",
            ],
            "technical": [
                "technical",
                "code",
                "software",
                "programming",
                "system",
                "algorithm",
                "database",
                "network",
                "security",
                "bug",
            ],
            "customer_service": [
                "customer",
                "support",
                "service",
                "ticket",
                "complaint",
                "inquiry",
                "help",
                "assistance",
                "issue",
                "problem",
            ],
            "educational": [
                "educational",
                "learning",
                "teaching",
                "student",
                "course",
                "curriculum",
                "academic",
                "study",
                "lesson",
                "training",
            ],
        }

    def _load_domain_patterns(self) -> Dict[str, Any]:
        """Load domain-specific patterns and knowledge."""
        patterns = {}
        domain_dir = Path(__file__).parent / "domain_knowledge"

        # Load existing domain files if they exist
        for domain_file in domain_dir.glob("*.json"):
            try:
                with open(domain_file) as f:
                    domain_name = domain_file.stem
                    patterns[domain_name] = json.load(f)
            except Exception:
                continue  # Skip invalid files

        return patterns

    async def analyze_description(self, description: str) -> ProblemInsights:
        """
        Analyze a natural language description to infer problem structure.

        This method uses pattern matching and heuristics to understand the
        problem type, domain, and appropriate structure.
        """
        description_lower = description.lower()

        # Infer problem type
        suggested_type = self._infer_problem_type(description_lower)

        # Infer domain
        suggested_domain = self._infer_domain(description_lower)

        # Determine complexity
        complexity_level = self._infer_complexity(description_lower)

        # Generate input/output structure
        input_structure, output_structure = self._infer_io_structure(
            description, suggested_type, suggested_domain
        )

        # Suggest categories for classification problems
        suggested_categories = self._suggest_categories(
            description, suggested_type, suggested_domain
        )

        # Suggest appropriate metrics
        suggested_metrics = self._suggest_metrics(suggested_type, suggested_domain)

        # Determine difficulty progression
        difficulty_tiers = self._suggest_difficulty_tiers(complexity_level)

        # Estimate examples needed
        estimated_examples = self._estimate_examples_needed(
            suggested_type, complexity_level
        )

        # Identify key challenges
        key_challenges = self._identify_key_challenges(
            description, suggested_type, suggested_domain
        )

        # Generate reasoning
        reasoning = self._generate_reasoning(
            description, suggested_type, suggested_domain, complexity_level
        )

        return ProblemInsights(
            suggested_type=suggested_type,
            complexity_level=complexity_level,
            suggested_domain=suggested_domain,
            input_structure=input_structure,
            output_structure=output_structure,
            suggested_categories=suggested_categories,
            suggested_metrics=suggested_metrics,
            difficulty_tiers=difficulty_tiers,
            estimated_examples_needed=estimated_examples,
            key_challenges=key_challenges,
            reasoning=reasoning,
        )

    def _infer_problem_type(self, description: str) -> str:
        """Infer the problem type from description."""
        type_scores = {}

        for problem_type, indicators in self.type_indicators.items():
            score = sum(1 for indicator in indicators if indicator in description)
            type_scores[problem_type] = score

        # Return type with highest score, default to classification
        if not type_scores or max(type_scores.values()) == 0:
            return "classification"

        return max(type_scores, key=type_scores.get)

    def _infer_domain(self, description: str) -> str:
        """Infer the domain from description."""
        domain_scores = {}

        for domain, indicators in self.domain_indicators.items():
            score = sum(1 for indicator in indicators if indicator in description)
            domain_scores[domain] = score

        # Return domain with highest score, default to general
        if not domain_scores or max(domain_scores.values()) == 0:
            return "general"

        return max(domain_scores, key=domain_scores.get)

    def _infer_complexity(self, description: str) -> str:
        """Infer complexity level from description."""
        complexity_indicators = {
            "Expert": [
                "complex",
                "sophisticated",
                "advanced",
                "nuanced",
                "subtle",
                "edge cases",
                "ambiguous",
                "multi-faceted",
                "comprehensive",
            ],
            "Advanced": [
                "challenging",
                "difficult",
                "requires reasoning",
                "context",
                "understanding",
                "analysis",
                "multiple factors",
            ],
            "Beginner": [
                "simple",
                "basic",
                "straightforward",
                "clear",
                "obvious",
                "easy",
                "fundamental",
                "elementary",
            ],
        }

        scores = {}
        for level, indicators in complexity_indicators.items():
            score = sum(1 for indicator in indicators if indicator in description)
            scores[level] = score

        # Default to Advanced if no clear indicators
        if not scores or max(scores.values()) == 0:
            return "Advanced"

        return max(scores, key=scores.get)

    def _infer_io_structure(
        self, description: str, problem_type: str, domain: str
    ) -> Tuple[Dict, Dict]:
        """Infer appropriate input/output structure."""
        # Default structures by problem type
        input_structures = {
            "classification": {"text": "str"},
            "generation": {"prompt": "str", "context": "str"},
            "analysis": {"document": "str", "criteria": "List[str]"},
            "structured": {"text": "str", "format": "str"},
        }

        output_structures = {
            "classification": {"category": "str"},
            "generation": {"generated_text": "str"},
            "analysis": {"analysis_result": "Dict[str, Any]"},
            "structured": {"extracted_data": "Dict[str, Any]"},
        }

        # Customize based on domain
        input_structure = input_structures.get(problem_type, {"input": "str"})
        output_structure = output_structures.get(problem_type, {"output": "str"})

        # Domain-specific customizations
        if domain == "legal" and problem_type == "analysis":
            input_structure = {"contract_text": "str", "analysis_type": "str"}
            output_structure = {"risks": "List[str]", "compliance_issues": "List[str]"}
        elif domain == "medical" and problem_type == "classification":
            input_structure = {"symptoms": "str", "patient_history": "str"}
            output_structure = {"diagnosis_category": "str", "confidence": "str"}
        elif domain == "customer_service" and problem_type == "classification":
            input_structure = {"query": "str"}
            output_structure = {"category": "str"}

        return input_structure, output_structure

    def _suggest_categories(
        self, description: str, problem_type: str, domain: str
    ) -> List[str]:
        """Suggest categories for classification problems."""
        if problem_type != "classification":
            return []

        # Domain-specific category suggestions
        domain_categories = {
            "customer_service": [
                "billing_issue",
                "technical_support",
                "product_inquiry",
                "shipping_inquiry",
                "return_request",
                "account_support",
            ],
            "legal": [
                "contract_review",
                "compliance_check",
                "risk_assessment",
                "liability_analysis",
                "regulatory_review",
            ],
            "medical": [
                "emergency",
                "routine_checkup",
                "specialist_referral",
                "prescription_request",
                "test_results",
            ],
            "technical": [
                "bug_report",
                "feature_request",
                "performance_issue",
                "security_concern",
                "documentation_update",
            ],
            "financial": [
                "loan_application",
                "investment_advice",
                "fraud_detection",
                "credit_assessment",
                "insurance_claim",
            ],
        }

        return domain_categories.get(
            domain, ["category_a", "category_b", "category_c", "category_d"]
        )

    def _suggest_metrics(self, problem_type: str, domain: str) -> List[str]:
        """Suggest appropriate evaluation metrics."""
        base_metrics = ["accuracy", "success_rate"]

        type_metrics = {
            "classification": [
                "accuracy",
                "f1_score",
                "precision",
                "recall",
                "category_balance",
            ],
            "generation": ["coherence", "relevance", "creativity", "fluency"],
            "analysis": ["completeness", "depth", "accuracy", "insight_quality"],
            "structured": ["extraction_accuracy", "format_compliance", "completeness"],
        }

        domain_metrics = {
            "legal": ["compliance_accuracy", "risk_detection_rate"],
            "medical": ["diagnostic_accuracy", "safety_score"],
            "customer_service": ["resolution_accuracy", "satisfaction_prediction"],
            "technical": ["issue_classification_accuracy", "severity_assessment"],
            "financial": ["fraud_detection_rate", "risk_assessment_accuracy"],
        }

        suggested = type_metrics.get(problem_type, base_metrics).copy()
        suggested.extend(domain_metrics.get(domain, []))

        return list(set(suggested))  # Remove duplicates

    def _suggest_difficulty_tiers(self, complexity_level: str) -> List[str]:
        """Suggest difficulty tier progression."""

        if complexity_level == "Expert":
            return ["easy", "medium", "hard", "very_hard", "expert"]
        elif complexity_level == "Advanced":
            return ["easy", "medium", "hard", "very_hard"]
        else:  # Beginner
            return ["easy", "medium", "hard"]

    def _estimate_examples_needed(
        self, problem_type: str, complexity_level: str
    ) -> int:
        """Estimate appropriate number of examples."""
        base_counts = {"Beginner": 15, "Advanced": 25, "Expert": 35}

        type_multipliers = {
            "classification": 1.0,
            "generation": 0.8,  # Usually need fewer for generation
            "analysis": 1.2,  # Need more for complex analysis
            "structured": 1.1,
        }

        base_count = base_counts.get(complexity_level, 25)
        multiplier = type_multipliers.get(problem_type, 1.0)

        return int(base_count * multiplier)

    def _identify_key_challenges(
        self, description: str, problem_type: str, domain: str
    ) -> List[str]:
        """Identify key challenges this problem tests."""
        challenges = []

        # Type-specific challenges
        if problem_type == "classification":
            challenges.extend(
                [
                    "Category disambiguation",
                    "Context understanding",
                    "Edge case handling",
                ]
            )
        elif problem_type == "generation":
            challenges.extend(
                ["Coherence and fluency", "Relevance to prompt", "Creative variation"]
            )
        elif problem_type == "analysis":
            challenges.extend(
                [
                    "Comprehensive analysis",
                    "Key insight identification",
                    "Evidence-based reasoning",
                ]
            )

        # Domain-specific challenges
        domain_challenges = {
            "legal": [
                "Legal terminology understanding",
                "Regulatory compliance",
                "Risk assessment",
            ],
            "medical": [
                "Medical accuracy",
                "Safety considerations",
                "Clinical reasoning",
            ],
            "customer_service": [
                "Intent recognition",
                "Empathy requirements",
                "Solution orientation",
            ],
            "technical": [
                "Technical accuracy",
                "Problem diagnosis",
                "Solution feasibility",
            ],
            "financial": [
                "Risk assessment",
                "Regulatory compliance",
                "Quantitative analysis",
            ],
        }

        challenges.extend(domain_challenges.get(domain, []))

        return challenges

    def _generate_reasoning(
        self, description: str, problem_type: str, domain: str, complexity: str
    ) -> str:
        """Generate explanation of the analysis."""
        return f"""
Based on the description "{description[:100]}...", this appears to be a {problem_type} problem
in the {domain} domain with {complexity} complexity level.

Key indicators:
- Problem type: Inferred from keywords and task structure
- Domain: Identified through domain-specific terminology
- Complexity: Determined by challenge level and nuance requirements

This problem will test the model's ability to handle {domain}-specific tasks requiring
{complexity.lower()}-level reasoning and understanding.
        """.strip()

    def generate_problem_name(self, description: str) -> str:
        """Generate a suitable problem name from description."""
        # Extract key words and create a name
        words = re.findall(r"\b\w+\b", description.lower())

        # Remove common stop words
        stop_words = {
            "the",
            "a",
            "an",
            "and",
            "or",
            "but",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "with",
            "by",
            "from",
            "as",
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
            "can",
            "that",
            "this",
            "these",
            "those",
            "i",
            "you",
            "he",
            "she",
            "it",
            "we",
            "they",
        }

        key_words = [word for word in words if word not in stop_words and len(word) > 2]

        # Take first 2-3 key words and create name
        name_words = key_words[:3] if len(key_words) >= 3 else key_words

        if not name_words:
            return "custom_problem"

        return "_".join(name_words)

    async def analyze_existing_examples(self, examples: List[Dict]) -> Dict[str, Any]:
        """Analyze existing examples to understand patterns."""
        if not examples:
            return {}

        analysis = {
            "total_count": len(examples),
            "difficulty_distribution": {},
            "input_patterns": {},
            "output_patterns": {},
            "common_themes": [],
            "complexity_indicators": [],
        }

        # Analyze difficulty distribution
        for example in examples:
            difficulty = example.get("metadata", {}).get("difficulty", "unknown")
            analysis["difficulty_distribution"][difficulty] = (
                analysis["difficulty_distribution"].get(difficulty, 0) + 1
            )

        # Analyze input patterns
        input_keys = set()
        for example in examples:
            if "input_data" in example:
                input_keys.update(example["input_data"].keys())
        analysis["input_patterns"] = list(input_keys)

        # Extract common themes (simplified analysis)
        all_text = []
        for example in examples:
            input_data = example.get("input_data", {})
            for value in input_data.values():
                if isinstance(value, str):
                    all_text.append(value.lower())

        # Simple word frequency analysis for themes
        word_counts = {}
        for text in all_text:
            words = re.findall(r"\b\w+\b", text)
            for word in words:
                if len(word) > 4:  # Only consider longer words
                    word_counts[word] = word_counts.get(word, 0) + 1

        # Get top themes
        if word_counts:
            top_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[
                :10
            ]
            analysis["common_themes"] = [word for word, count in top_words if count > 1]

        return analysis
