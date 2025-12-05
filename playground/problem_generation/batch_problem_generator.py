"""
Batch Problem Generator - Orchestrates Large-Scale Problem Generation.

This module manages the generation of multiple problems with thousands of examples
each, ensuring diversity both within and across problems.
"""

import asyncio
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from ..problem_management.code_generator import CodeGenerator
from ..problem_management.intelligence import ProblemIntelligence
from .enhanced_example_generator import (
    EnhancedExampleGenerator,
    GenerationBatch,
    GenerationConfig,
)
from .problem_diversity_manager import ProblemDiversityManager, ProblemSpecification


@dataclass
class ProblemGenerationResult:
    """Result of generating a single problem."""

    problem_name: str
    problem_spec: ProblemSpecification
    generation_batches: List[GenerationBatch]
    code_file_path: str
    total_examples: int
    generation_time: float
    diversity_score: float
    success: bool
    error: Optional[str] = None


@dataclass
class BatchGenerationReport:
    """Complete report for batch problem generation."""

    total_problems: int
    successful_problems: int
    failed_problems: int
    total_examples_generated: int
    total_generation_time: float
    average_diversity_score: float
    problem_results: List[ProblemGenerationResult]
    coverage_analysis: Dict[str, Dict[str, float]]


class BatchProblemGenerator:
    """
    Orchestrates generation of multiple problems with diversity optimization.

    Features:
    - Generates problems based on gap analysis
    - Ensures diversity within and across problems
    - Creates complete problem modules with code
    - Tracks and reports on generation progress
    """

    def __init__(
        self,
        output_dir: str = "examples/langchain_problems",
        generation_config: Optional[GenerationConfig] = None,
    ):
        """
        Initialize batch generator.

        Args:
            output_dir: Directory to save generated problems
            generation_config: Configuration for example generation
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.diversity_manager = ProblemDiversityManager(str(self.output_dir))
        self.example_generator = EnhancedExampleGenerator(
            generation_config or GenerationConfig()
        )
        self.code_generator = CodeGenerator()
        self.intelligence = ProblemIntelligence()

        # Create generation subdirectories
        self.reports_dir = self.output_dir / "generation_reports"
        self.reports_dir.mkdir(exist_ok=True)

    async def generate_problem_suite(
        self,
        num_problems: int = 30,
        examples_per_problem: int = 1000,
        use_gap_analysis: bool = True,
        parallel_generation: bool = True,
        max_concurrent: int = 3,
    ) -> BatchGenerationReport:
        """
        Generate a complete suite of problems.

        Args:
            num_problems: Number of problems to generate
            examples_per_problem: Examples per problem
            use_gap_analysis: Whether to use gap analysis for problem selection
            parallel_generation: Whether to generate problems in parallel
            max_concurrent: Maximum concurrent problem generations

        Returns:
            Complete generation report
        """
        start_time = time.time()

        print(f"🚀 Starting batch generation of {num_problems} problems")
        print(f"📊 Target: {examples_per_problem} examples per problem")

        # Get problem specifications
        if use_gap_analysis:
            print("🔍 Analyzing existing problems and identifying gaps...")
            problem_specs = self.diversity_manager.suggest_new_problems(num_problems)
        else:
            print("📝 Generating problem specifications...")
            problem_specs = self._generate_default_specs(num_problems)

        # Save problem plan
        self._save_problem_plan(problem_specs)

        # Generate problems
        problem_results = []

        if parallel_generation:
            # Generate in parallel with concurrency limit
            problem_results = await self._generate_problems_parallel(
                problem_specs, examples_per_problem, max_concurrent
            )
        else:
            # Generate sequentially
            for i, spec in enumerate(problem_specs):
                print(f"\n{'=' * 60}")
                print(f"Problem {i+1}/{num_problems}: {spec.name}")
                print(f"{'=' * 60}")

                result = await self._generate_single_problem(spec, examples_per_problem)
                problem_results.append(result)

                # Progress update
                self._print_progress(i + 1, num_problems, problem_results)

        # Calculate final statistics
        total_time = time.time() - start_time
        report = self._create_final_report(problem_results, total_time, num_problems)

        # Save report
        self._save_generation_report(report)

        # Print summary
        self._print_summary(report)

        return report

    async def _generate_single_problem(
        self, spec: ProblemSpecification, target_examples: int
    ) -> ProblemGenerationResult:
        """Generate a single problem with examples."""
        start_time = time.time()

        try:
            print(f"\n📝 Generating problem: {spec.name}")
            print(f"   Domain: {spec.domain}")
            print(f"   Type: {spec.problem_type}")
            print(f"   Difficulty: {spec.difficulty_level}")

            # Analyze problem description for insights
            insights = await self.intelligence.analyze_description(spec.description)

            # Generate examples with diversity optimization
            batches = await self.example_generator.generate_problem_examples(
                problem_name=spec.name,
                description=spec.description,
                domain=spec.domain,
                problem_type=spec.problem_type,
                target_count=target_examples,
                insights=insights,
            )

            # Calculate overall diversity
            all_examples = []
            for batch in batches:
                all_examples.extend(batch.examples)

            diversity_metrics = (
                self.example_generator.diversity_analyzer.analyze_diversity(
                    [self.example_generator._example_to_dict(ex) for ex in all_examples]
                )
            )

            # Generate code module
            print("💻 Generating code module...")
            code_path = await self._generate_problem_code(
                spec, all_examples, diversity_metrics
            )

            # Save example data
            self._save_example_data(spec.name, batches)

            generation_time = time.time() - start_time

            return ProblemGenerationResult(
                problem_name=spec.name,
                problem_spec=spec,
                generation_batches=batches,
                code_file_path=str(code_path),
                total_examples=len(all_examples),
                generation_time=generation_time,
                diversity_score=diversity_metrics.overall_diversity_score,
                success=True,
            )

        except Exception as e:
            print(f"❌ Error generating problem {spec.name}: {str(e)}")
            generation_time = time.time() - start_time

            return ProblemGenerationResult(
                problem_name=spec.name,
                problem_spec=spec,
                generation_batches=[],
                code_file_path="",
                total_examples=0,
                generation_time=generation_time,
                diversity_score=0.0,
                success=False,
                error=str(e),
            )

    async def _generate_problems_parallel(
        self,
        problem_specs: List[ProblemSpecification],
        examples_per_problem: int,
        max_concurrent: int,
    ) -> List[ProblemGenerationResult]:
        """Generate multiple problems in parallel."""
        results = []

        # Create semaphore for concurrency control
        semaphore = asyncio.Semaphore(max_concurrent)

        async def generate_with_semaphore(spec: ProblemSpecification, index: int):
            async with semaphore:
                print(
                    f"\n🔄 Starting problem {index + 1}/{len(problem_specs)}: {spec.name}"
                )
                result = await self._generate_single_problem(spec, examples_per_problem)
                print(f"✅ Completed problem {index + 1}: {spec.name}")
                return result

        # Create all tasks
        tasks = [
            generate_with_semaphore(spec, i) for i, spec in enumerate(problem_specs)
        ]

        # Run tasks and collect results
        results = await asyncio.gather(*tasks)

        return results

    async def _generate_problem_code(
        self, spec: ProblemSpecification, examples: List, diversity_metrics
    ) -> Path:
        """Generate Python code for the problem."""
        # Convert examples to code format
        code_examples = []
        for i, ex in enumerate(examples):
            code_examples.append(
                {
                    "id": i + 1,
                    "input_data": ex.input_data,
                    "expected_output": ex.expected_output,
                    "metadata": ex.metadata,
                }
            )

        # Generate code
        code = await self.code_generator.generate_problem_code(
            problem_name=spec.name,
            description=spec.description,
            problem_type=spec.problem_type,
            domain=spec.domain,
            difficulty=spec.difficulty_level,
            examples=code_examples,
            metrics=spec.evaluation_metrics,
            include_imports=True,
            include_metadata=True,
        )

        # Save code
        safe_filename = self._make_safe_filename(spec.name)
        code_path = self.output_dir / f"{safe_filename}.py"

        with open(code_path, "w") as f:
            f.write(code)

        return code_path

    def _save_example_data(self, problem_name: str, batches: List[GenerationBatch]):
        """Save detailed example data for analysis."""
        data_dir = self.output_dir / "example_data"
        data_dir.mkdir(exist_ok=True)

        safe_name = self._make_safe_filename(problem_name)
        data_file = data_dir / f"{safe_name}_examples.json"

        # Convert batches to serializable format
        batch_data = []
        for batch in batches:
            batch_dict = {
                "batch_id": batch.batch_id,
                "examples": [
                    {
                        "input_data": ex.input_data,
                        "expected_output": ex.expected_output,
                        "difficulty": ex.difficulty,
                        "reasoning": ex.reasoning,
                        "metadata": ex.metadata,
                    }
                    for ex in batch.examples
                ],
                "diversity_metrics": {
                    "overall_score": batch.diversity_metrics.overall_diversity_score,
                    "pattern_entropy": batch.diversity_metrics.pattern_entropy,
                    "difficulty_balance": batch.diversity_metrics.difficulty_balance,
                    "topic_coverage": batch.diversity_metrics.topic_coverage,
                },
                "generation_time": batch.generation_time,
            }
            batch_data.append(batch_dict)

        with open(data_file, "w") as f:
            json.dump(batch_data, f, indent=2)

    def _generate_default_specs(self, count: int) -> List[ProblemSpecification]:
        """Generate default problem specifications."""
        # This would be implemented with default templates
        # For now, return empty list
        return []

    def _save_problem_plan(self, specs: List[ProblemSpecification]):
        """Save the problem generation plan."""
        plan_file = self.reports_dir / f"problem_plan_{int(time.time())}.json"

        plan_data = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_problems": len(specs),
            "problems": [
                {
                    "name": spec.name,
                    "description": spec.description,
                    "domain": spec.domain,
                    "problem_type": spec.problem_type,
                    "difficulty_level": spec.difficulty_level,
                    "evaluation_metrics": spec.evaluation_metrics,
                    "technical_requirements": spec.technical_requirements,
                }
                for spec in specs
            ],
        }

        with open(plan_file, "w") as f:
            json.dump(plan_data, f, indent=2)

        print(f"📋 Problem plan saved to: {plan_file}")

    def _create_final_report(
        self,
        results: List[ProblemGenerationResult],
        total_time: float,
        num_problems: int,
    ) -> BatchGenerationReport:
        """Create final generation report."""
        successful = [r for r in results if r.success]
        failed = [r for r in results if not r.success]

        total_examples = sum(r.total_examples for r in successful)
        avg_diversity = (
            sum(r.diversity_score for r in successful) / len(successful)
            if successful
            else 0.0
        )

        # Analyze coverage
        self.diversity_manager.analyze_existing_problems()
        coverage = {
            "domains": self.diversity_manager._calculate_domain_coverage(),
            "types": self.diversity_manager._calculate_type_coverage(),
            "difficulties": self.diversity_manager._calculate_difficulty_coverage(),
        }

        return BatchGenerationReport(
            total_problems=num_problems,
            successful_problems=len(successful),
            failed_problems=len(failed),
            total_examples_generated=total_examples,
            total_generation_time=total_time,
            average_diversity_score=avg_diversity,
            problem_results=results,
            coverage_analysis=coverage,
        )

    def _save_generation_report(self, report: BatchGenerationReport):
        """Save detailed generation report."""
        report_file = self.reports_dir / f"generation_report_{int(time.time())}.json"

        report_data = {
            "summary": {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "total_problems": report.total_problems,
                "successful_problems": report.successful_problems,
                "failed_problems": report.failed_problems,
                "total_examples_generated": report.total_examples_generated,
                "total_generation_time": report.total_generation_time,
                "average_diversity_score": report.average_diversity_score,
            },
            "problems": [
                {
                    "name": r.problem_name,
                    "success": r.success,
                    "examples_generated": r.total_examples,
                    "diversity_score": r.diversity_score,
                    "generation_time": r.generation_time,
                    "code_file": r.code_file_path,
                    "error": r.error,
                }
                for r in report.problem_results
            ],
            "coverage_analysis": report.coverage_analysis,
        }

        with open(report_file, "w") as f:
            json.dump(report_data, f, indent=2)

        print(f"\n📊 Full report saved to: {report_file}")

    def _print_progress(
        self, current: int, total: int, results: List[ProblemGenerationResult]
    ):
        """Print progress update."""
        successful = len([r for r in results if r.success])
        failed = len([r for r in results if not r.success])

        print(f"\n📈 Progress: {current}/{total} problems")
        print(f"   ✅ Successful: {successful}")
        print(f"   ❌ Failed: {failed}")

        if results:
            recent = results[-1]
            if recent.success:
                print(
                    f"   📊 Last problem: {recent.total_examples} examples, "
                    f"diversity: {recent.diversity_score:.1f}"
                )

    def _print_summary(self, report: BatchGenerationReport):
        """Print final summary."""
        print("\n" + "=" * 60)
        print("🎉 BATCH GENERATION COMPLETE")
        print("=" * 60)
        print(f"Total problems attempted: {report.total_problems}")
        print(f"Successful: {report.successful_problems}")
        print(f"Failed: {report.failed_problems}")
        print(f"Total examples generated: {report.total_examples_generated:,}")
        print(f"Average diversity score: {report.average_diversity_score:.1f}")
        print(f"Total time: {report.total_generation_time:.1f} seconds")
        print(
            f"Average time per problem: {report.total_generation_time / report.total_problems:.1f} seconds"
        )

        if report.failed_problems > 0:
            print("\n⚠️  Failed problems:")
            for r in report.problem_results:
                if not r.success:
                    print(f"   - {r.problem_name}: {r.error}")

    def _make_safe_filename(self, name: str) -> str:
        """Convert problem name to safe filename."""
        import re

        # Remove special characters and replace spaces
        safe_name = re.sub(r"[^\w\s-]", "", name.lower())
        safe_name = re.sub(r"[-\s]+", "_", safe_name)
        return safe_name
