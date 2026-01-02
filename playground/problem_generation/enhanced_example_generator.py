"""
Enhanced Example Generator with Memory and Diversity Optimization.

This module extends the basic example generator with memory-aware generation
and diversity tracking to create large-scale, diverse problem sets.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from playground.problem_management.example_generator import ExampleGenerator, GeneratedExample
from playground.problem_management.intelligence import ProblemInsights, ProblemIntelligence
from traigent.utils.secure_path import validate_path

from .diversity_analyzer import DiversityAnalyzer, DiversityMetrics
from .example_memory import ExampleMemory, ExampleSummary


@dataclass
class GenerationBatch:
    """Represents a batch of generated examples."""

    batch_id: int
    examples: list[GeneratedExample]
    diversity_metrics: DiversityMetrics
    memory_summaries: list[ExampleSummary]
    generation_time: float


@dataclass
class GenerationConfig:
    """Configuration for enhanced generation."""

    batch_size: int = 50  # Examples per API batch
    diversity_threshold: float = 0.7  # Minimum diversity score
    max_retries: int = 3  # Retries for low diversity
    include_memory_context: bool = True  # Include past examples
    memory_summaries_per_batch: int = 30  # Compact summaries to include
    adaptive_difficulty: bool = True  # Adjust difficulty based on gaps


class EnhancedExampleGenerator:
    """
    Enhanced example generator with memory and diversity optimization.

    Features:
    - Memory-aware generation to avoid repetition
    - Diversity tracking and enforcement
    - Batch generation with compact context
    - Adaptive difficulty distribution
    """

    def __init__(self, config: GenerationConfig | None = None):
        """
        Initialize enhanced generator.

        Args:
            config: Generation configuration
        """
        self.config = config or GenerationConfig()
        self.base_generator = ExampleGenerator()
        self.intelligence = ProblemIntelligence()
        self.memory = ExampleMemory(max_summaries_per_batch=config.memory_summaries_per_batch)
        self.diversity_analyzer = DiversityAnalyzer()
        self.total_generated = 0

    async def generate_problem_examples(
        self,
        problem_name: str,
        description: str,
        domain: str,
        problem_type: str,
        target_count: int = 1000,
        insights: ProblemInsights | None = None,
    ) -> list[GenerationBatch]:
        """
        Generate all examples for a problem with diversity optimization.

        Args:
            problem_name: Name of the problem
            description: Problem description
            domain: Problem domain
            problem_type: Type of problem
            target_count: Total examples to generate
            insights: Optional problem insights

        Returns:
            List of generation batches
        """
        batches = []
        generated_count = 0

        # Calculate number of batches
        num_batches = (target_count + self.config.batch_size - 1) // self.config.batch_size

        print(f"🎯 Generating {target_count} examples in {num_batches} batches")

        for batch_id in range(num_batches):
            # Calculate batch size (handle last batch)
            batch_size = min(self.config.batch_size, target_count - generated_count)

            # Generate batch with diversity optimization
            batch = await self._generate_diverse_batch(
                batch_id=batch_id,
                batch_size=batch_size,
                problem_name=problem_name,
                description=description,
                domain=domain,
                problem_type=problem_type,
                insights=insights,
            )

            batches.append(batch)
            generated_count += len(batch.examples)

            # Update progress
            progress = (generated_count / target_count) * 100
            print(f"📊 Progress: {generated_count}/{target_count} ({progress:.1f}%)")

            # Check diversity trends
            if batch_id > 0 and batch_id % 5 == 0:
                overall_metrics = self._calculate_overall_metrics(batches)
                print(f"📈 Overall diversity score: {overall_metrics.overall_diversity_score:.1f}")

                # Suggest improvements if needed
                if overall_metrics.overall_diversity_score < 70:
                    suggestions = self.diversity_analyzer.suggest_diversity_improvements(
                        self._flatten_examples(batches)
                    )
                    print("⚠️  Diversity suggestions:")
                    for suggestion in suggestions:
                        print(f"   - {suggestion}")

        return batches

    async def _generate_diverse_batch(
        self,
        batch_id: int,
        batch_size: int,
        problem_name: str,
        description: str,
        domain: str,
        problem_type: str,
        insights: ProblemInsights | None,
    ) -> GenerationBatch:
        """Generate a single batch with diversity optimization."""
        import time

        start_time = time.time()

        # Get memory context for diversity
        memory_context = []
        if self.config.include_memory_context and self.memory.total_examples > 0:
            # Get diverse summaries from memory
            target_difficulty = self._determine_target_difficulty(batch_id)
            memory_context = self.memory.get_diverse_summaries(target_difficulty)

        # Prepare generation prompt with memory context
        enhanced_prompt = self._create_enhanced_prompt(
            description=description,
            domain=domain,
            problem_type=problem_type,
            batch_size=batch_size,
            memory_context=memory_context,
            insights=insights,
        )

        # Generate examples with retries for diversity
        best_examples = []
        best_diversity = 0.0

        for retry in range(self.config.max_retries):
            # Generate batch
            examples = await self._generate_batch_with_context(
                problem_type=problem_type,
                domain=domain,
                enhanced_prompt=enhanced_prompt,
                batch_size=batch_size,
                insights=insights,
            )

            # Analyze diversity
            diversity_metrics = self.diversity_analyzer.analyze_diversity(
                [self._example_to_dict(ex) for ex in examples]
            )

            # Check if this batch is better
            if diversity_metrics.overall_diversity_score > best_diversity:
                best_examples = examples
                best_diversity = diversity_metrics.overall_diversity_score

            # Check if diversity is acceptable
            if diversity_metrics.overall_diversity_score >= self.config.diversity_threshold * 100:
                break
            else:
                print(
                    f"   Retry {retry + 1}: Diversity {diversity_metrics.overall_diversity_score:.1f} < threshold"
                )

        # Add examples to memory
        memory_summaries = []
        for i, example in enumerate(best_examples):
            example_id = self.total_generated + i
            summary = self.memory.add_example(self._example_to_dict(example), example_id)
            memory_summaries.append(summary)

        self.total_generated += len(best_examples)

        # Recalculate final metrics
        final_metrics = self.diversity_analyzer.analyze_diversity(
            [self._example_to_dict(ex) for ex in best_examples]
        )

        generation_time = time.time() - start_time

        return GenerationBatch(
            batch_id=batch_id,
            examples=best_examples,
            diversity_metrics=final_metrics,
            memory_summaries=memory_summaries,
            generation_time=generation_time,
        )

    def _create_enhanced_prompt(
        self,
        description: str,
        domain: str,
        problem_type: str,
        batch_size: int,
        memory_context: list[dict[str, Any]],
        insights: ProblemInsights | None,
    ) -> str:
        """Create enhanced prompt with memory context."""
        prompt_parts = [
            f"Generate {batch_size} diverse examples for a {problem_type} problem.",
            f"Domain: {domain}",
            f"Description: {description}",
            "",
            "Requirements:",
            "- Ensure maximum diversity in patterns, topics, and complexity",
            "- Cover different edge cases and scenarios",
            "- Maintain realistic and practical examples",
            "- Follow domain-specific conventions and terminology",
        ]

        if insights:
            if insights.suggested_categories:
                prompt_parts.append(
                    f"- Categories to cover: {', '.join(insights.suggested_categories)}"
                )
            if insights.complexity_indicators:
                prompt_parts.append(
                    f"- Include complexity: {', '.join(insights.complexity_indicators)}"
                )

        if memory_context:
            prompt_parts.extend(
                [
                    "",
                    "Existing Example Patterns (avoid repetition):",
                    json.dumps(memory_context, indent=2),
                    "",
                    "Generate NEW examples that are different from the above patterns.",
                ]
            )

        # Add distribution guidance
        difficulty_dist = self.memory.get_difficulty_distribution()
        pattern_dist = self.memory.get_pattern_distribution()

        if difficulty_dist:
            prompt_parts.extend(
                [
                    "",
                    f"Current difficulty distribution: {difficulty_dist}",
                    "Balance the generation to fill gaps in underrepresented difficulties.",
                ]
            )

        if pattern_dist:
            prompt_parts.extend(
                [
                    f"Current pattern distribution: {pattern_dist}",
                    "Prioritize patterns that are underrepresented.",
                ]
            )

        return "\n".join(prompt_parts)

    async def _generate_batch_with_context(
        self,
        problem_type: str,
        domain: str,
        enhanced_prompt: str,
        batch_size: int,
        insights: ProblemInsights | None,
    ) -> list[GeneratedExample]:
        """Generate batch using enhanced prompt."""
        # This is where we would normally call Claude API
        # For now, use the base generator with the enhanced context

        # Parse difficulty distribution from prompt
        self._extract_difficulty_guidance(enhanced_prompt)

        # Generate using base generator
        examples = await self.base_generator.generate_examples(
            problem_type=problem_type,
            domain=domain,
            description=enhanced_prompt,  # Pass enhanced prompt as description
            count=batch_size,
            difficulty="Advanced",  # Default to advanced
            insights=insights,
        )

        return examples

    def _determine_target_difficulty(self, batch_id: int) -> str | None:
        """Determine target difficulty for current batch."""
        if not self.config.adaptive_difficulty:
            return None

        # Get current distribution
        distribution = self.memory.get_difficulty_distribution()
        if not distribution:
            return None

        # Find most underrepresented difficulty
        total = sum(distribution.values())
        if total == 0:
            return None

        # Calculate ideal percentages
        ideal = {
            "easy": 0.20,
            "medium": 0.30,
            "hard": 0.30,
            "very_hard": 0.15,
            "expert": 0.05,
        }

        # Find biggest gap
        biggest_gap = 0.0
        target_difficulty = None

        for difficulty, ideal_ratio in ideal.items():
            current_ratio = distribution.get(difficulty, 0) / total
            gap = ideal_ratio - current_ratio

            if gap > biggest_gap:
                biggest_gap = gap
                target_difficulty = difficulty

        return target_difficulty

    def _extract_difficulty_guidance(self, enhanced_prompt: str) -> list[str]:
        """Extract difficulty guidance from enhanced prompt."""
        # Simple extraction - in real implementation would be more sophisticated
        difficulties = ["easy", "medium", "hard", "very_hard", "expert"]
        mentioned = []

        for difficulty in difficulties:
            if difficulty in enhanced_prompt.lower():
                mentioned.append(difficulty)

        return mentioned if mentioned else ["easy", "medium", "hard"]

    def _example_to_dict(self, example: GeneratedExample) -> dict[str, Any]:
        """Convert GeneratedExample to dictionary for analysis."""
        return {
            "input_data": example.input_data,
            "expected_output": example.expected_output,
            "difficulty": example.difficulty,
            "metadata": example.metadata,
        }

    def _calculate_overall_metrics(self, batches: list[GenerationBatch]) -> DiversityMetrics:
        """Calculate overall diversity metrics across all batches."""
        all_examples = self._flatten_examples(batches)
        return self.diversity_analyzer.analyze_diversity(all_examples)

    def _flatten_examples(self, batches: list[GenerationBatch]) -> list[dict[str, Any]]:
        """Flatten all examples from batches."""
        all_examples = []
        for batch in batches:
            all_examples.extend([self._example_to_dict(ex) for ex in batch.examples])
        return all_examples

    def save_generation_report(self, batches: list[GenerationBatch], output_file: str):
        """Save detailed generation report."""
        report = {
            "summary": {
                "total_batches": len(batches),
                "total_examples": sum(len(b.examples) for b in batches),
                "total_generation_time": sum(b.generation_time for b in batches),
                "average_diversity_score": (
                    sum(b.diversity_metrics.overall_diversity_score for b in batches) / len(batches)
                    if batches
                    else 0
                ),
            },
            "batches": [],
        }

        for batch in batches:
            batch_data = {
                "batch_id": batch.batch_id,
                "example_count": len(batch.examples),
                "generation_time": batch.generation_time,
                "diversity_metrics": {
                    "overall_score": batch.diversity_metrics.overall_diversity_score,
                    "pattern_entropy": batch.diversity_metrics.pattern_entropy,
                    "difficulty_balance": batch.diversity_metrics.difficulty_balance,
                    "topic_coverage": batch.diversity_metrics.topic_coverage,
                    "lexical_diversity": batch.diversity_metrics.lexical_diversity,
                    "structural_diversity": batch.diversity_metrics.structural_diversity,
                    "similarity_score": batch.diversity_metrics.similarity_score,
                },
            }
            report["batches"].append(batch_data)

        # Add distribution analysis
        all_examples = self._flatten_examples(batches)
        difficulty_dist = {}
        pattern_dist = {}

        for ex in all_examples:
            # Count difficulties
            diff = ex.get("difficulty", "unknown")
            difficulty_dist[diff] = difficulty_dist.get(diff, 0) + 1

            # Count patterns (simplified)
            if "metadata" in ex and "category" in ex["metadata"]:
                pattern = ex["metadata"]["category"]
                pattern_dist[pattern] = pattern_dist.get(pattern, 0) + 1

        report["distributions"] = {
            "difficulty": difficulty_dist,
            "patterns": pattern_dist,
        }

        # Add improvement suggestions
        final_metrics = self._calculate_overall_metrics(batches)
        suggestions = self.diversity_analyzer.suggest_diversity_improvements(all_examples)
        report["final_analysis"] = {
            "overall_diversity_score": final_metrics.overall_diversity_score,
            "suggestions": suggestions,
        }

        output_path = validate_path(output_file, Path.cwd())
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)

        print(f"📝 Generation report saved to: {output_path}")
