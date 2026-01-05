"""
Problem Analyzer for LangChain Optimization Problems.

This module provides analysis and validation capabilities for existing problems,
including quality assessment, bias detection, and improvement suggestions.
"""

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from .domain_knowledge import DomainKnowledge
from .intelligence import ProblemIntelligence


@dataclass
class ProblemAnalysisReport:
    """Report from analyzing a problem."""

    problem_name: str
    current_count: int
    difficulty_distribution: Dict[str, int]
    quality_score: float
    issues: List[str]
    suggestions: List[str]
    bias_analysis: Optional[Dict[str, Any]] = None
    completeness_score: float = 0.0

    def format_report(self) -> str:
        """Format the analysis report for display."""
        report = f"""
📊 Problem Analysis Report: {self.problem_name}
{'=' * 60}

📈 Basic Statistics:
   Total Examples: {self.current_count}
   Quality Score: {self.quality_score:.2f}/10.0
   Completeness: {self.completeness_score:.2f}/10.0

📋 Difficulty Distribution:"""

        for difficulty, count in sorted(self.difficulty_distribution.items()):
            percentage = (
                (count / self.current_count * 100) if self.current_count > 0 else 0
            )
            report += f"\n   {difficulty.title()}: {count} ({percentage:.1f}%)"

        if self.issues:
            report += f"\n\n❌ Issues Found ({len(self.issues)}):"
            for issue in self.issues:
                report += f"\n   • {issue}"

        if self.suggestions:
            report += f"\n\n💡 Suggestions ({len(self.suggestions)}):"
            for suggestion in self.suggestions:
                report += f"\n   • {suggestion}"

        if self.bias_analysis:
            report += "\n\n🔍 Bias Analysis:"
            bias = self.bias_analysis
            if bias.get("potential_biases"):
                report += (
                    f"\n   Potential Biases Found: {len(bias['potential_biases'])}"
                )
                for bias_type, details in bias["potential_biases"].items():
                    report += f"\n   • {bias_type}: {details}"
            else:
                report += "\n   No significant biases detected"

        report += "\n"
        return report


@dataclass
class ValidationResult:
    """Result from validating a problem."""

    problem_name: str
    is_valid: bool
    issues: List[str]
    warnings: List[str]
    suggestions: List[str]


class ProblemAnalyzer:
    """
    Analyzes and validates LangChain optimization problems.

    Provides comprehensive analysis including quality assessment,
    bias detection, structural validation, and improvement suggestions.
    """

    def __init__(self):
        """Initialize the problem analyzer."""
        self.intelligence = ProblemIntelligence()
        self.domain_knowledge = DomainKnowledge()

        # Quality assessment criteria
        self.quality_criteria = {
            "difficulty_balance": 2.0,  # Balanced across difficulty tiers
            "example_diversity": 2.0,  # Diverse example patterns
            "domain_appropriateness": 2.0,  # Domain-specific patterns
            "clarity": 1.5,  # Clear inputs/outputs
            "completeness": 1.5,  # Complete metadata
            "edge_case_coverage": 1.0,  # Edge cases included
        }

    async def analyze_problem(
        self,
        problem_name: str,
        detailed: bool = False,
        check_bias: bool = False,
        suggest_improvements: bool = False,
    ) -> ProblemAnalysisReport:
        """
        Analyze a problem comprehensively.

        Args:
            problem_name: Name of problem to analyze
            detailed: Include detailed analysis
            check_bias: Perform bias detection
            suggest_improvements: Generate improvement suggestions

        Returns:
            Comprehensive analysis report
        """
        # Load and analyze the problem
        problem_data = await self._load_problem_data(problem_name)
        raw_examples = problem_data.get("examples", [])

        # IMPORTANT: Always normalize examples to dict format to prevent AttributeError
        examples = self._normalize_examples(raw_examples)

        # Basic statistics
        current_count = len(examples)
        # Use raw_examples for difficulty distribution since it handles both formats
        difficulty_distribution = self._analyze_difficulty_distribution(raw_examples)

        # Quality assessment
        quality_score = await self._assess_quality(problem_name, examples, detailed)
        completeness_score = self._assess_completeness(problem_data)

        # Issue detection (pass raw_examples - method handles normalization)
        issues = await self._detect_issues(problem_name, raw_examples)

        # Generate suggestions (pass raw_examples - method handles both formats)
        suggestions = []
        if suggest_improvements:
            suggestions = await self._generate_suggestions(
                problem_name, raw_examples, issues
            )

        # Bias analysis (pass raw_examples - method handles normalization)
        bias_analysis = None
        if check_bias:
            bias_analysis = await self._analyze_bias(raw_examples)

        return ProblemAnalysisReport(
            problem_name=problem_name,
            current_count=current_count,
            difficulty_distribution=difficulty_distribution,
            quality_score=quality_score,
            issues=issues,
            suggestions=suggestions,
            bias_analysis=bias_analysis,
            completeness_score=completeness_score,
        )

    async def analyze_existing_problem(
        self, problem_name: str
    ) -> ProblemAnalysisReport:
        """Quick analysis for existing problem (used by add-examples mode)."""
        return await self.analyze_problem(problem_name, detailed=False)

    async def validate_problems(
        self, problem_names: List[str], fix_issues: bool = False
    ) -> Dict[str, ValidationResult]:
        """
        Validate multiple problems.

        Args:
            problem_names: List of problem names to validate
            fix_issues: Attempt to fix common issues

        Returns:
            Validation results for each problem
        """
        results = {}

        for problem_name in problem_names:
            try:
                result = await self._validate_single_problem(problem_name, fix_issues)
                results[problem_name] = result
            except Exception as e:
                results[problem_name] = ValidationResult(
                    problem_name=problem_name,
                    is_valid=False,
                    issues=[f"Validation failed: {str(e)}"],
                    warnings=[],
                    suggestions=[],
                )

        return results

    async def _load_problem_data(self, problem_name: str) -> Dict[str, Any]:
        """Load problem data from module using the problem registry."""
        try:
            # Import the langchain_problems package and load all problems
            sys.path.insert(0, str(Path(__file__).parent.parent.parent))
            from examples.langchain_problems import (
                get_available_problems,
                get_problem_class,
                load_all_problems,
            )

            # Load all problems to populate the registry
            load_all_problems()

            # Check if the problem exists
            available_problems = get_available_problems()
            if problem_name not in available_problems:
                raise FileNotFoundError(
                    f"Problem '{problem_name}' not found. Available problems: {available_problems}"
                )

            # Get the problem class
            problem_class = get_problem_class(problem_name)

            # Instantiate the problem
            problem_instance = problem_class()

            # Get examples data
            examples = []
            if hasattr(problem_instance, "_load_examples"):
                examples = problem_instance._load_examples()
            elif hasattr(problem_instance, "get_dataset"):
                dataset = problem_instance.get_dataset()
                if hasattr(dataset, "examples"):
                    examples = dataset.examples
                elif isinstance(dataset, list):
                    examples = dataset

            module_path = Path(f"examples/langchain_problems/{problem_name}.py")

            return {
                "examples": examples,
                "problem_class": problem_class,
                "problem_instance": problem_instance,
                "module_path": module_path,
            }

        except ImportError as e:
            raise ImportError(f"Failed to import problem '{problem_name}': {e}") from e
        except Exception as e:
            raise ValueError(f"Failed to load problem '{problem_name}': {e}") from e

    def _analyze_difficulty_distribution(self, examples: List[Any]) -> Dict[str, int]:
        """Analyze difficulty distribution in examples."""
        distribution = {}

        for example in examples:
            # Handle both dict and EvaluationExample formats
            if hasattr(example, "metadata"):  # EvaluationExample
                metadata = example.metadata or {}
            else:  # Dict format
                metadata = example.get("metadata", {})

            if isinstance(metadata, dict):
                difficulty = metadata.get("difficulty", "unknown")
            else:
                difficulty = "unknown"

            distribution[difficulty] = distribution.get(difficulty, 0) + 1

        return distribution

    def _normalize_example(self, example: Any) -> Dict[str, Any]:
        """Normalize example to dictionary format."""
        if hasattr(example, "input_data"):  # EvaluationExample
            return {
                "input_data": example.input_data,
                "expected_output": example.expected_output,
                "metadata": example.metadata or {},
            }
        else:  # Already dict format
            return example

    def _normalize_examples(self, examples: List[Any]) -> List[Dict[str, Any]]:
        """Normalize all examples to dictionary format."""
        return [self._normalize_example(ex) for ex in examples]

    async def _assess_quality(
        self, problem_name: str, examples: List[Any], detailed: bool
    ) -> float:
        """Assess overall quality score."""
        if not examples:
            return 0.0

        # Normalize examples to consistent format
        normalized_examples = self._normalize_examples(examples)

        total_score = 0.0
        max_score = sum(self.quality_criteria.values())

        # Difficulty balance assessment
        difficulty_score = self._assess_difficulty_balance(normalized_examples)
        total_score += difficulty_score * self.quality_criteria["difficulty_balance"]

        # Example diversity assessment
        diversity_score = self._assess_example_diversity(normalized_examples)
        total_score += diversity_score * self.quality_criteria["example_diversity"]

        # Domain appropriateness assessment
        domain_score = await self._assess_domain_appropriateness(normalized_examples)
        total_score += domain_score * self.quality_criteria["domain_appropriateness"]

        # Clarity assessment
        clarity_score = self._assess_clarity(normalized_examples)
        total_score += clarity_score * self.quality_criteria["clarity"]

        # Completeness assessment
        completeness_score = self._assess_example_completeness(normalized_examples)
        total_score += completeness_score * self.quality_criteria["completeness"]

        # Edge case coverage assessment
        edge_case_score = self._assess_edge_case_coverage(normalized_examples)
        total_score += edge_case_score * self.quality_criteria["edge_case_coverage"]

        # Convert to 0-10 scale
        return (total_score / max_score) * 10.0

    def _assess_difficulty_balance(self, examples: List[Dict]) -> float:
        """Assess how well-balanced the difficulty distribution is."""
        distribution = self._analyze_difficulty_distribution(examples)

        if not distribution:
            return 0.0

        # Ideal distribution: more examples in middle tiers
        ideal_ratios = {
            "easy": 0.2,
            "medium": 0.3,
            "hard": 0.3,
            "very_hard": 0.15,
            "expert": 0.05,
        }

        total_examples = sum(distribution.values())
        balance_score = 0.0

        for difficulty, ideal_ratio in ideal_ratios.items():
            actual_ratio = distribution.get(difficulty, 0) / total_examples
            # Score based on how close to ideal ratio
            diff = abs(actual_ratio - ideal_ratio)
            balance_score += max(0, 1.0 - (diff * 2))  # Penalty for deviation

        return balance_score / len(ideal_ratios)

    def _assess_example_diversity(self, examples: List[Dict]) -> float:
        """Assess diversity of examples."""
        if not examples:
            return 0.0

        # Normalize examples first if needed
        normalized_examples = (
            self._normalize_examples(examples)
            if not all(isinstance(ex, dict) for ex in examples)
            else examples
        )

        # Check input diversity
        input_patterns = set()
        output_patterns = set()

        for example in normalized_examples:
            input_data = example.get("input_data", {})
            expected_output = example.get("expected_output")

            # Hash input structure
            input_signature = tuple(sorted(input_data.keys()))
            input_patterns.add(input_signature)

            # Track output types
            output_type = type(expected_output).__name__
            if isinstance(expected_output, dict):
                output_signature = tuple(sorted(expected_output.keys()))
                output_patterns.add(output_signature)
            else:
                output_patterns.add(output_type)

        # Score based on pattern diversity
        input_diversity = min(1.0, len(input_patterns) / max(1, len(examples) // 5))
        output_diversity = min(1.0, len(output_patterns) / max(1, len(examples) // 10))

        return (input_diversity + output_diversity) / 2.0

    async def _assess_domain_appropriateness(self, examples: List[Dict]) -> float:
        """Assess how well examples fit the domain."""
        if not examples:
            return 0.0

        # Normalize examples first if needed
        normalized_examples = (
            self._normalize_examples(examples)
            if not all(isinstance(ex, dict) for ex in examples)
            else examples
        )

        # Extract domain indicators from examples
        all_text = []
        for example in normalized_examples:
            input_data = example.get("input_data", {})
            for value in input_data.values():
                if isinstance(value, str):
                    all_text.append(value.lower())

        if not all_text:
            return 0.5  # Neutral score if no text to analyze

        # Simple domain appropriateness check
        combined_text = " ".join(all_text)

        # Count domain-specific terms
        domain_scores = {}
        for domain, indicators in self.intelligence.domain_indicators.items():
            score = sum(1 for indicator in indicators if indicator in combined_text)
            domain_scores[domain] = score

        # If examples clearly belong to a domain, score higher
        max_score = max(domain_scores.values()) if domain_scores else 0
        total_words = len(combined_text.split())

        # Score based on domain specificity
        if max_score > 0:
            domain_ratio = max_score / max(
                1, total_words // 50
            )  # Normalize by text length
            return min(1.0, domain_ratio * 5)  # Scale appropriately

        return 0.5  # Neutral score for general content

    def _assess_clarity(self, examples: List[Dict]) -> float:
        """Assess clarity of examples."""
        if not examples:
            return 0.0

        # Normalize examples first if needed
        normalized_examples = (
            self._normalize_examples(examples)
            if not all(isinstance(ex, dict) for ex in examples)
            else examples
        )

        clarity_scores = []

        for example in normalized_examples:
            score = 1.0

            # Check if input_data exists and is not empty
            input_data = example.get("input_data", {})
            if not input_data:
                score -= 0.5

            # Check if expected_output exists
            expected_output = example.get("expected_output")
            if expected_output is None:
                score -= 0.5

            # Check for metadata
            metadata = example.get("metadata", {})
            if not metadata:
                score -= 0.2

            # Check for reasoning
            if not metadata.get("reasoning"):
                score -= 0.2

            clarity_scores.append(max(0.0, score))

        return sum(clarity_scores) / len(clarity_scores)

    def _assess_example_completeness(self, examples: List[Dict]) -> float:
        """Assess completeness of example metadata."""
        if not examples:
            return 0.0

        # Normalize examples first if needed
        normalized_examples = (
            self._normalize_examples(examples)
            if not all(isinstance(ex, dict) for ex in examples)
            else examples
        )

        required_fields = ["input_data", "expected_output", "metadata"]
        recommended_metadata = ["difficulty", "reasoning", "domain"]

        completeness_scores = []

        for example in normalized_examples:
            score = 0.0

            # Check required fields
            for field in required_fields:
                if field in example and example[field] is not None:
                    score += 1.0 / len(required_fields)

            # Check recommended metadata
            metadata = example.get("metadata", {})
            for meta_field in recommended_metadata:
                if meta_field in metadata and metadata[meta_field]:
                    score += 0.5 / len(recommended_metadata)

            completeness_scores.append(min(1.0, score))

        return sum(completeness_scores) / len(completeness_scores)

    def _assess_edge_case_coverage(self, examples: List[Dict]) -> float:
        """Assess coverage of edge cases."""
        distribution = self._analyze_difficulty_distribution(examples)

        # Edge cases are typically in 'very_hard' and 'expert' tiers
        edge_case_count = distribution.get("very_hard", 0) + distribution.get(
            "expert", 0
        )
        total_count = sum(distribution.values())

        if total_count == 0:
            return 0.0

        # Good edge case coverage is around 15-25% of examples
        edge_case_ratio = edge_case_count / total_count

        if 0.15 <= edge_case_ratio <= 0.25:
            return 1.0
        elif edge_case_ratio < 0.15:
            return edge_case_ratio / 0.15  # Partial credit
        else:
            return max(
                0.0, 1.0 - ((edge_case_ratio - 0.25) * 2)
            )  # Penalty for too many

    def _assess_completeness(self, problem_data: Dict[str, Any]) -> float:
        """Assess overall completeness of the problem."""
        score = 0.0

        # Check if examples exist
        examples = problem_data.get("examples", [])
        if examples:
            score += 3.0

        # Check example count
        example_count = len(examples)
        if example_count >= 20:
            score += 2.0
        elif example_count >= 10:
            score += 1.0

        # Check difficulty distribution
        distribution = self._analyze_difficulty_distribution(examples)
        if len(distribution) >= 3:
            score += 2.0
        elif len(distribution) >= 2:
            score += 1.0

        # Check for problem class
        if problem_data.get("problem_class"):
            score += 2.0

        # Check for proper methods
        problem_instance = problem_data.get("problem_instance")
        if problem_instance:
            required_methods = [
                "create_dataset",
                "create_function",
                "create_optimized_function",
            ]
            for method in required_methods:
                if hasattr(problem_instance, method):
                    score += 1.0 / len(required_methods)

        return score  # Max score is 10.0

    async def _detect_issues(self, problem_name: str, examples: List[Any]) -> List[str]:
        """Detect issues in the problem."""
        issues = []

        # Normalize examples to consistent format
        normalized_examples = self._normalize_examples(examples)

        # Check example count
        if len(normalized_examples) < 5:
            issues.append(
                f"Too few examples ({len(normalized_examples)}). Recommend at least 15-20."
            )

        # Check difficulty distribution
        distribution = self._analyze_difficulty_distribution(examples)

        if len(distribution) < 2:
            issues.append(
                "Insufficient difficulty variety. Need at least 2 difficulty levels."
            )

        if "unknown" in distribution:
            issues.append(
                f"{distribution['unknown']} examples have unknown difficulty."
            )

        # Check for missing metadata
        missing_metadata_count = 0
        for example in normalized_examples:
            metadata = example.get("metadata", {})
            if not metadata or not metadata.get("difficulty"):
                missing_metadata_count += 1

        if missing_metadata_count > 0:
            issues.append(f"{missing_metadata_count} examples missing metadata.")

        # Check for empty inputs/outputs
        empty_inputs = sum(1 for ex in normalized_examples if not ex.get("input_data"))
        empty_outputs = sum(
            1 for ex in normalized_examples if ex.get("expected_output") is None
        )

        if empty_inputs > 0:
            issues.append(f"{empty_inputs} examples have empty input data.")

        if empty_outputs > 0:
            issues.append(f"{empty_outputs} examples have empty expected output.")

        return issues

    async def _generate_suggestions(
        self, problem_name: str, examples: List[Dict], issues: List[str]
    ) -> List[str]:
        """Generate improvement suggestions."""
        suggestions = []

        # Suggestions based on issues
        if "Too few examples" in str(issues):
            suggestions.append(
                "Add more examples using: python problem_manager.py add-examples"
            )

        if "difficulty variety" in str(issues):
            suggestions.append(
                "Add examples with different difficulty levels (--difficulty parameter)"
            )

        if "missing metadata" in str(issues):
            suggestions.append("Review and fix example metadata fields")

        # Additional suggestions based on analysis
        distribution = self._analyze_difficulty_distribution(examples)

        # Suggest balancing if needed
        total_count = sum(distribution.values())
        if total_count > 0:
            easy_ratio = distribution.get("easy", 0) / total_count
            hard_ratio = (
                distribution.get("very_hard", 0) + distribution.get("expert", 0)
            ) / total_count

            if easy_ratio < 0.1:
                suggestions.append("Add more easy examples to improve accessibility")

            if hard_ratio < 0.1:
                suggestions.append("Add more challenging examples to test model limits")

        # Domain-specific suggestions
        if len(examples) > 0:
            # Normalize first example if needed
            if hasattr(examples[0], "metadata"):
                metadata = examples[0].metadata or {}
            else:
                metadata = examples[0].get("metadata", {})
            domain = (
                metadata.get("domain", "general")
                if isinstance(metadata, dict)
                else "general"
            )

            if domain != "general":
                suggestions.append(
                    f"Consider adding domain-specific metrics for {domain}"
                )

        return suggestions

    async def _analyze_bias(self, examples: List[Any]) -> Dict[str, Any]:
        """Analyze potential biases in examples."""
        bias_analysis = {
            "potential_biases": {},
            "diversity_scores": {},
            "recommendations": [],
        }

        # Normalize examples to consistent format
        normalized_examples = self._normalize_examples(examples)

        # Simple bias detection (can be expanded)

        # Check for length bias
        input_lengths = []
        for example in normalized_examples:
            input_data = example.get("input_data", {})
            for value in input_data.values():
                if isinstance(value, str):
                    input_lengths.append(len(value))

        if input_lengths:
            avg_length = sum(input_lengths) / len(input_lengths)
            std_length = (
                sum((x - avg_length) ** 2 for x in input_lengths) / len(input_lengths)
            ) ** 0.5

            if std_length / avg_length > 0.5:  # High variability
                bias_analysis["diversity_scores"]["length_diversity"] = "High"
            else:
                bias_analysis["potential_biases"][
                    "length_bias"
                ] = "Examples have similar lengths"

        # Check for difficulty bias
        distribution = self._analyze_difficulty_distribution(examples)
        if distribution:
            max_difficulty = max(distribution.values())
            min_difficulty = min(distribution.values())

            if max_difficulty > min_difficulty * 3:
                bias_analysis["potential_biases"][
                    "difficulty_bias"
                ] = "Uneven difficulty distribution"

        return bias_analysis

    async def _validate_single_problem(
        self, problem_name: str, fix_issues: bool
    ) -> ValidationResult:
        """Validate a single problem."""
        issues = []
        warnings = []
        suggestions = []

        try:
            # Load problem data
            problem_data = await self._load_problem_data(problem_name)
            examples = problem_data.get("examples", [])
            problem_instance = problem_data.get("problem_instance")

            # Structural validation
            if not examples:
                issues.append("No examples found")
            elif len(examples) < 5:
                warnings.append(f"Only {len(examples)} examples (recommend 15+)")

            # Method validation
            if problem_instance:
                required_methods = [
                    "create_dataset",
                    "create_function",
                    "create_optimized_function",
                ]
                for method in required_methods:
                    if not hasattr(problem_instance, method):
                        issues.append(f"Missing required method: {method}")

            # Example validation
            normalized_examples = self._normalize_examples(examples)
            for i, example in enumerate(normalized_examples):
                if not example.get("input_data"):
                    issues.append(f"Example {i+1}: Missing input_data")

                if example.get("expected_output") is None:
                    issues.append(f"Example {i+1}: Missing expected_output")

                metadata = example.get("metadata", {})
                if not metadata.get("difficulty"):
                    warnings.append(f"Example {i+1}: Missing difficulty metadata")

            # Fix issues if requested
            if fix_issues and issues:
                suggestions.append("Auto-fix not implemented yet")

        except Exception as e:
            issues.append(f"Failed to load problem: {str(e)}")

        is_valid = len(issues) == 0

        return ValidationResult(
            problem_name=problem_name,
            is_valid=is_valid,
            issues=issues,
            warnings=warnings,
            suggestions=suggestions,
        )
