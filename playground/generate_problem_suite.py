#!/usr/bin/env python3
"""
Generate Problem Suite - Main Script for Large-Scale Problem Generation.

This script uses Claude to generate a diverse set of optimization problems
with thousands of examples each.

Usage:
    python generate_problem_suite.py [options]

Options:
    --problems: Number of problems to generate (default: 30)
    --examples: Examples per problem (default: 1000)
    --batch-size: Examples per API batch (default: 50)
    --output-dir: Output directory (default: examples/langchain_problems)
    --test-mode: Run in test mode with smaller numbers
    --parallel: Enable parallel generation
    --max-concurrent: Maximum concurrent generations (default: 3)
"""

import argparse
import asyncio
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from problem_generation import (
    BatchProblemGenerator,
    GenerationConfig,
    ProblemDiversityManager,
)


async def generate_suite(
    num_problems: int,
    examples_per_problem: int,
    batch_size: int,
    output_dir: str,
    parallel: bool,
    max_concurrent: int,
):
    """Generate complete problem suite."""
    print("🚀 Traigent Problem Suite Generator")
    print("=" * 60)
    print(f"Target: {num_problems} problems with {examples_per_problem} examples each")
    print(f"Total examples: {num_problems * examples_per_problem:,}")
    print(f"Batch size: {batch_size} examples per API call")
    print(f"Output directory: {output_dir}")
    print(f"Parallel generation: {'Yes' if parallel else 'No'}")
    print("=" * 60)

    # Create generation config
    config = GenerationConfig(
        batch_size=batch_size,
        diversity_threshold=0.7,
        max_retries=3,
        include_memory_context=True,
        memory_summaries_per_batch=30,
        adaptive_difficulty=True,
    )

    # Initialize generator
    generator = BatchProblemGenerator(output_dir=output_dir, generation_config=config)

    # Analyze existing problems first
    print("\n📊 Analyzing existing problems...")
    diversity_manager = ProblemDiversityManager(output_dir)
    existing_problems = diversity_manager.analyze_existing_problems()

    if existing_problems:
        print(f"Found {len(existing_problems)} existing problems")

        # Show coverage summary
        domain_coverage = diversity_manager._calculate_domain_coverage()
        type_coverage = diversity_manager._calculate_type_coverage()

        print("\nDomain coverage:")
        for domain, coverage in sorted(
            domain_coverage.items(), key=lambda x: x[1], reverse=True
        ):
            if coverage > 0:
                print(f"  - {domain}: {coverage * 100:.1f}%")

        print("\nProblem type coverage:")
        for ptype, coverage in sorted(
            type_coverage.items(), key=lambda x: x[1], reverse=True
        ):
            if coverage > 0:
                print(f"  - {ptype}: {coverage * 100:.1f}%")
    else:
        print("No existing problems found - starting fresh")

    # Export gap analysis
    print("\n📝 Exporting gap analysis...")
    gap_file = Path(output_dir) / "generation_reports" / "gap_analysis.json"
    gap_file.parent.mkdir(parents=True, exist_ok=True)
    diversity_manager.export_gap_analysis(str(gap_file))
    print(f"Gap analysis saved to: {gap_file}")

    # Confirm before proceeding
    print(f"\n⚡ Ready to generate {num_problems} problems")
    print("This will:")
    print(f"  1. Create {num_problems} new problem modules")
    print(f"  2. Generate {examples_per_problem} examples for each")
    print("  3. Use gap analysis to ensure diverse coverage")
    print(f"  4. Save all data and reports to {output_dir}")

    response = input("\nProceed with generation? (y/n): ")
    if response.lower() != "y":
        print("Generation cancelled.")
        return

    # Run generation
    print("\n🎯 Starting generation...")
    report = await generator.generate_problem_suite(
        num_problems=num_problems,
        examples_per_problem=examples_per_problem,
        use_gap_analysis=True,
        parallel_generation=parallel,
        max_concurrent=max_concurrent,
    )

    # Show final coverage
    print("\n📊 Final Coverage Analysis:")
    print("Domains:")
    for domain, coverage in sorted(
        report.coverage_analysis["domains"].items(), key=lambda x: x[1], reverse=True
    ):
        if coverage > 0:
            print(f"  - {domain}: {coverage * 100:.1f}%")

    print("\nProblem types:")
    for ptype, coverage in sorted(
        report.coverage_analysis["types"].items(), key=lambda x: x[1], reverse=True
    ):
        if coverage > 0:
            print(f"  - {ptype}: {coverage * 100:.1f}%")

    print("\n✅ Generation complete!")


async def run_test_generation():
    """Run a small test generation."""
    print("🧪 Running test generation (2 problems, 100 examples each)")

    await generate_suite(
        num_problems=2,
        examples_per_problem=100,
        batch_size=20,
        output_dir="examples/langchain_problems_test",
        parallel=False,
        max_concurrent=1,
    )


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate diverse LangChain optimization problems at scale",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate default suite (30 problems, 1000 examples each)
  python generate_problem_suite.py

  # Test mode (2 problems, 100 examples)
  python generate_problem_suite.py --test-mode

  # Custom configuration
  python generate_problem_suite.py --problems 20 --examples 500 --parallel

  # Generate with custom output directory
  python generate_problem_suite.py --output-dir my_problems --problems 10
        """,
    )

    parser.add_argument(
        "--problems",
        type=int,
        default=30,
        help="Number of problems to generate (default: 30)",
    )

    parser.add_argument(
        "--examples",
        type=int,
        default=1000,
        help="Examples per problem (default: 1000)",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Examples per API batch (default: 50)",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="examples/langchain_problems",
        help="Output directory (default: examples/langchain_problems)",
    )

    parser.add_argument(
        "--test-mode", action="store_true", help="Run in test mode with smaller numbers"
    )

    parser.add_argument(
        "--parallel", action="store_true", help="Enable parallel generation"
    )

    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=3,
        help="Maximum concurrent generations (default: 3)",
    )

    args = parser.parse_args()

    # Run appropriate mode
    if args.test_mode:
        asyncio.run(run_test_generation())
    else:
        asyncio.run(
            generate_suite(
                num_problems=args.problems,
                examples_per_problem=args.examples,
                batch_size=args.batch_size,
                output_dir=args.output_dir,
                parallel=args.parallel,
                max_concurrent=args.max_concurrent,
            )
        )


if __name__ == "__main__":
    main()
