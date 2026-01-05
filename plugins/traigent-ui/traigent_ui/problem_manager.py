#!/usr/bin/env python3
"""
LangChain Problem Management Tool
===============================

A powerful CLI tool that uses Claude Code SDK to intelligently generate, modify,
and manage LangChain optimization problems for the Traigent framework.

This tool provides multiple modes for comprehensive problem management:
- CREATE: Generate new problems from natural language descriptions
- ADD-EXAMPLES: Extend existing problems with additional examples
- ANALYZE: Analyze problem quality and suggest improvements
- VALIDATE: Validate problem structure and implementation
- CONVERT: Convert datasets to full problem modules
- CLONE: Create variants of existing problems

Usage Examples:
--------------
# Create a new problem
python problem_manager.py create \
  --description "Analyze legal contracts for risks and compliance" \
  --examples 25 --domain legal --difficulty Expert

# Add examples to existing problem
python problem_manager.py add-examples customer_support --count 10 --balance

# Analyze problem quality
python problem_manager.py analyze customer_support

# Validate all problems
python problem_manager.py validate --all

Requirements:
------------
- Claude Code SDK access
- LangChain and dependencies
- Traigent SDK

Author: Traigent SDK / Claude Code
"""

import argparse
import asyncio
import sys
from pathlib import Path

from traigent.utils.secure_path import validate_path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import problem management modules
try:
    from langchain_problems import (
        get_available_problems,
        load_all_problems,
    )
    from problem_management import (
        CodeGenerator,
        ExampleGenerator,
        ProblemAnalyzer,
        ProblemIntelligence,
    )
except ImportError as e:
    print(f"❌ Error importing problem management modules: {e}")
    print(
        "   Please ensure all dependencies are installed and the project structure is correct."
    )
    sys.exit(1)


def create_parser() -> argparse.ArgumentParser:
    """Create the main argument parser with all modes and options."""
    parser = argparse.ArgumentParser(
        description="LangChain Problem Management Tool - Intelligent problem generation and management",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create new legal contract analysis problem
  python problem_manager.py create \\
    --description "Analyze legal contracts for risks and compliance issues" \\
    --examples 25 --domain legal --difficulty Expert --name contract_analysis

  # Add balanced examples across difficulty levels
  python problem_manager.py add-examples customer_support --count 15 --balance

  # Analyze problem quality and get suggestions
  python problem_manager.py analyze customer_support --detailed

  # Validate all problems
  python problem_manager.py validate --all

  # Convert JSONL dataset to full problem
  python problem_manager.py convert dataset.jsonl --type classification --domain medical

  # Clone existing problem with modifications
  python problem_manager.py clone customer_support --name tech_support --domain technical

For more information, see the Traigent SDK documentation.
        """,
    )

    # Create subparsers for different modes
    subparsers = parser.add_subparsers(dest="mode", help="Available modes")

    # CREATE mode
    create_parser = subparsers.add_parser(
        "create",
        help="Generate new problem from description",
        description="Create a new LangChain optimization problem from a natural language description",
    )
    create_parser.add_argument(
        "--description",
        "-d",
        required=True,
        help="Natural language description of the problem to create",
    )
    create_parser.add_argument(
        "--examples",
        "-n",
        type=int,
        default=20,
        help="Number of examples to generate (default: 20)",
    )
    create_parser.add_argument(
        "--name", help="Problem name (auto-generated from description if not provided)"
    )
    create_parser.add_argument(
        "--domain",
        choices=[
            "customer_service",
            "legal",
            "medical",
            "technical",
            "financial",
            "educational",
            "general",
        ],
        default="general",
        help="Problem domain for specialized generation (default: general)",
    )
    create_parser.add_argument(
        "--difficulty",
        choices=["Beginner", "Advanced", "Expert"],
        default="Advanced",
        help="Overall difficulty level (default: Advanced)",
    )
    create_parser.add_argument(
        "--output-type",
        choices=["classification", "generation", "analysis", "structured"],
        help="Force specific output type (auto-detected if not specified)",
    )
    create_parser.add_argument(
        "--metrics",
        help='Custom metrics (comma-separated, e.g., "accuracy,f1_score,precision")',
    )
    create_parser.add_argument(
        "--models",
        default="gpt-3.5-turbo,gpt-4o-mini,gpt-4o",
        help="Model configuration space (default: gpt-3.5-turbo,gpt-4o-mini,gpt-4o)",
    )
    create_parser.add_argument(
        "--temperature-range",
        default="0.1,0.7",
        help="Temperature range (default: 0.1,0.7)",
    )
    create_parser.add_argument(
        "--max-tokens",
        type=int,
        help="Max tokens for model responses (auto-determined if not specified)",
    )
    create_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be generated without creating files",
    )

    # ADD-EXAMPLES mode
    add_parser = subparsers.add_parser(
        "add-examples",
        help="Add examples to existing problem",
        description="Extend an existing problem with additional examples",
    )
    add_parser.add_argument("problem_name", help="Name of existing problem to extend")
    add_parser.add_argument(
        "--count", "-n", type=int, required=True, help="Number of examples to add"
    )
    add_parser.add_argument(
        "--difficulty",
        choices=["easy", "medium", "hard", "very_hard", "expert"],
        help="Focus on specific difficulty tier",
    )
    add_parser.add_argument(
        "--balance",
        action="store_true",
        help="Distribute examples evenly across all difficulty tiers",
    )
    add_parser.add_argument(
        "--edge-cases", action="store_true", help="Focus on challenging edge cases"
    )
    add_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what examples would be added without modifying files",
    )

    # ANALYZE mode
    analyze_parser = subparsers.add_parser(
        "analyze",
        help="Analyze problem quality",
        description="Analyze existing problem for quality, bias, and improvement opportunities",
    )
    analyze_parser.add_argument("problem_name", help="Name of problem to analyze")
    analyze_parser.add_argument(
        "--detailed", action="store_true", help="Show detailed analysis report"
    )
    analyze_parser.add_argument(
        "--suggest-improvements",
        action="store_true",
        help="Generate specific improvement suggestions",
    )
    analyze_parser.add_argument(
        "--bias-check",
        action="store_true",
        help="Check for potential biases in examples",
    )

    # VALIDATE mode
    validate_parser = subparsers.add_parser(
        "validate",
        help="Validate problem structure",
        description="Validate problem module structure and implementation",
    )
    validate_parser.add_argument(
        "problem_name",
        nargs="?",
        help="Name of problem to validate (validates all if not specified)",
    )
    validate_parser.add_argument(
        "--all", action="store_true", help="Validate all problems"
    )
    validate_parser.add_argument(
        "--fix", action="store_true", help="Attempt to automatically fix common issues"
    )

    # CONVERT mode
    convert_parser = subparsers.add_parser(
        "convert",
        help="Convert dataset to problem module",
        description="Convert simple dataset files to full problem modules",
    )
    convert_parser.add_argument(
        "dataset_file", help="Path to dataset file (JSONL format)"
    )
    convert_parser.add_argument(
        "--type",
        choices=["classification", "generation", "analysis"],
        required=True,
        help="Problem type for conversion",
    )
    convert_parser.add_argument(
        "--domain", default="general", help="Problem domain (default: general)"
    )
    convert_parser.add_argument(
        "--name", help="Problem name (derived from filename if not provided)"
    )

    # CLONE mode
    clone_parser = subparsers.add_parser(
        "clone",
        help="Clone and modify existing problem",
        description="Create new problem based on existing one with modifications",
    )
    clone_parser.add_argument(
        "source_problem", help="Name of existing problem to clone"
    )
    clone_parser.add_argument("--name", required=True, help="Name for the new problem")
    clone_parser.add_argument("--domain", help="Change domain for the cloned problem")
    clone_parser.add_argument(
        "--modify-categories",
        help="Modify categories (comma-separated for classification problems)",
    )
    clone_parser.add_argument(
        "--difficulty",
        choices=["Beginner", "Advanced", "Expert"],
        help="Change difficulty level",
    )

    # TEST mode
    test_parser = subparsers.add_parser(
        "test",
        help="Generate test cases for problem",
        description="Generate test cases and edge cases for problem validation",
    )
    test_parser.add_argument("problem_name", help="Name of problem to test")
    test_parser.add_argument(
        "--count",
        type=int,
        default=10,
        help="Number of test cases to generate (default: 10)",
    )
    test_parser.add_argument(
        "--adversarial", action="store_true", help="Generate adversarial test cases"
    )

    return parser


async def handle_create_mode(args) -> bool:
    """Handle CREATE mode - generate new problem from description."""
    print(f"🎯 Creating new problem: {args.description}")
    print(
        f"📊 Parameters: {args.examples} examples, {args.domain} domain, {args.difficulty} difficulty"
    )

    try:
        # Initialize problem intelligence
        intelligence = ProblemIntelligence()

        # Analyze description and infer problem structure
        print("🔍 Analyzing problem description...")
        problem_insights = await intelligence.analyze_description(args.description)

        # Generate problem name if not provided
        problem_name = args.name or intelligence.generate_problem_name(args.description)
        print(f"📝 Problem name: {problem_name}")

        # Determine output type if not specified
        output_type = args.output_type or problem_insights.suggested_type
        print(f"🎯 Problem type: {output_type}")

        if args.dry_run:
            print("\n🔍 DRY RUN - Would generate:")
            print(f"   Problem name: {problem_name}")
            print(f"   Type: {output_type}")
            print(f"   Domain: {args.domain}")
            print(f"   Examples: {args.examples}")
            print(f"   Difficulty: {args.difficulty}")
            return True

        # Generate examples
        print(f"📚 Generating {args.examples} examples...")
        example_generator = ExampleGenerator()
        examples = await example_generator.generate_examples(
            problem_type=output_type,
            domain=args.domain,
            description=args.description,
            count=args.examples,
            difficulty=args.difficulty,
        )

        # Generate code
        print("🏗️ Generating problem module...")
        code_generator = CodeGenerator()
        problem_module = await code_generator.generate_problem_module(
            name=problem_name,
            description=args.description,
            domain=args.domain,
            difficulty=args.difficulty,
            examples=examples,
            output_type=output_type,
            models=args.models.split(","),
            temperature_range=[float(x) for x in args.temperature_range.split(",")],
            max_tokens=args.max_tokens,
            custom_metrics=args.metrics.split(",") if args.metrics else None,
        )

        # Save generated module
        base_dir = Path.cwd()
        problems_dir = validate_path(Path("examples/langchain_problems"), base_dir)
        problems_dir.mkdir(parents=True, exist_ok=True)
        problem_file = validate_path(
            problems_dir / f"{problem_name}.py",
            problems_dir,
        )
        problem_file.write_text(problem_module)

        print(f"✅ Successfully created problem: {problem_file}")
        print(
            f"🎉 Use: python examples/core/tool-use-calculator/run.py --problem {problem_name}"
        )

        return True

    except Exception as e:
        print(f"❌ Error creating problem: {e}")
        return False


async def handle_add_examples_mode(args) -> bool:
    """Handle ADD-EXAMPLES mode - extend existing problem."""
    print(f"➕ Adding {args.count} examples to problem: {args.problem_name}")

    try:
        # Load existing problems
        load_all_problems()
        available_problems = get_available_problems()

        if args.problem_name not in available_problems:
            print(f"❌ Problem '{args.problem_name}' not found.")
            print(f"Available problems: {', '.join(available_problems)}")
            return False

        # Analyze existing problem
        analyzer = ProblemAnalyzer()
        existing_analysis = await analyzer.analyze_existing_problem(args.problem_name)

        if args.dry_run:
            print("\n🔍 DRY RUN - Would add examples:")
            print(f"   Problem: {args.problem_name}")
            print(f"   Count: {args.count}")
            print(f"   Current examples: {existing_analysis.current_count}")
            if args.difficulty:
                print(f"   Focus difficulty: {args.difficulty}")
            if args.balance:
                print("   Mode: Balanced across all tiers")
            if args.edge_cases:
                print("   Mode: Edge cases focus")
            return True

        # Generate additional examples
        example_generator = ExampleGenerator()
        new_examples = await example_generator.generate_additional_examples(
            existing_problem=args.problem_name,
            count=args.count,
            difficulty=args.difficulty,
            balance=args.balance,
            edge_cases=args.edge_cases,
        )

        # Update problem module
        code_generator = CodeGenerator()
        updated_module = await code_generator.add_examples_to_module(
            problem_name=args.problem_name, new_examples=new_examples
        )

        # Save updated module
        base_dir = Path.cwd()
        problems_dir = validate_path(Path("examples/langchain_problems"), base_dir)
        problem_file = validate_path(
            problems_dir / f"{args.problem_name}.py",
            problems_dir,
        )
        problem_file.write_text(updated_module)

        print(
            f"✅ Successfully added {len(new_examples)} examples to {args.problem_name}"
        )

        return True

    except Exception as e:
        print(f"❌ Error adding examples: {e}")
        return False


async def handle_analyze_mode(args) -> bool:
    """Handle ANALYZE mode - analyze problem quality."""
    print(f"🔍 Analyzing problem: {args.problem_name}")

    try:
        analyzer = ProblemAnalyzer()
        analysis_report = await analyzer.analyze_problem(
            problem_name=args.problem_name,
            detailed=args.detailed,
            check_bias=args.bias_check,
            suggest_improvements=args.suggest_improvements,
        )

        print(analysis_report.format_report())

        return True

    except Exception as e:
        print(f"❌ Error analyzing problem: {e}")
        return False


async def handle_validate_mode(args) -> bool:
    """Handle VALIDATE mode - validate problem structure."""
    if args.all or not args.problem_name:
        print("✅ Validating all problems...")
        problems_to_validate = get_available_problems()
    else:
        print(f"✅ Validating problem: {args.problem_name}")
        problems_to_validate = [args.problem_name]

    try:
        analyzer = ProblemAnalyzer()
        validation_results = await analyzer.validate_problems(
            problems_to_validate, fix_issues=args.fix
        )

        for problem, result in validation_results.items():
            if result.is_valid:
                print(f"✅ {problem}: Valid")
            else:
                print(f"❌ {problem}: Issues found")
                for issue in result.issues:
                    print(f"   - {issue}")

        return all(r.is_valid for r in validation_results.values())

    except Exception as e:
        print(f"❌ Error validating problems: {e}")
        return False


def print_header():
    """Print the tool header."""
    print("🚀 LangChain Problem Management Tool")
    print("=" * 50)
    print()


def main():
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()

    # Show help if no mode specified
    if not args.mode:
        parser.print_help()
        return

    print_header()

    # Route to appropriate handler
    success = False
    try:
        if args.mode == "create":
            success = asyncio.run(handle_create_mode(args))
        elif args.mode == "add-examples":
            success = asyncio.run(handle_add_examples_mode(args))
        elif args.mode == "analyze":
            success = asyncio.run(handle_analyze_mode(args))
        elif args.mode == "validate":
            success = asyncio.run(handle_validate_mode(args))
        elif args.mode == "convert":
            print("🔄 CONVERT mode - Coming soon!")
            print("   This mode will convert JSONL datasets to full problem modules.")
        elif args.mode == "clone":
            print("🧬 CLONE mode - Coming soon!")
            print("   This mode will clone and modify existing problems.")
        elif args.mode == "test":
            print("🧪 TEST mode - Coming soon!")
            print("   This mode will generate test cases for problems.")
        else:
            print(f"❌ Unknown mode: {args.mode}")
            parser.print_help()

    except KeyboardInterrupt:
        print("\n⏹️ Operation cancelled by user")
        success = False
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        success = False

    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
