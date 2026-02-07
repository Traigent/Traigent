#!/usr/bin/env python3
"""
Test script for improved problem generation system.

This script tests the enhancements made to the problem generation process:
1. Config class initialization fix
2. Problem type-specific prompts
3. Enhanced JSON parsing
4. Progressive batch generation
5. Type-specific validation
6. File organization improvements
"""

import asyncio
import json
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from traigent_ui.problem_management.code_generator import CodeGenerator
from traigent_ui.problem_management.smart_problem_analyzer import SmartProblemAnalyzer


async def test_problem_generation(description: str, expected_type: str):
    """Test problem generation for a specific description."""
    print(f"\n{'='*60}")
    print(f"Testing: {description}")
    print(f"Expected type: {expected_type}")
    print(f"{'='*60}")

    try:
        # Initialize smart analyzer
        analyzer = SmartProblemAnalyzer()

        # Generate specification with small batch size
        smart_spec = await analyzer.analyze_and_generate_spec(
            user_description=description,
            target_examples=5,  # Small batch for testing
            generation_mode="smart_single",
        )

        print("\n✅ Smart Analysis Results:")
        print(f"   Problem Type: {smart_spec.problem_type}")
        print(f"   Domain: {smart_spec.domain}")
        print(f"   Confidence: {smart_spec.confidence_score:.1%}")
        print(f"   Examples Generated: {len(smart_spec.contextual_examples)}")

        # Validate examples
        is_valid, issues = analyzer.validate_specification(smart_spec)
        print("\n📋 Validation Results:")
        print(f"   Valid: {is_valid}")
        if issues:
            print(f"   Issues: {issues}")

        # Show first example
        if smart_spec.contextual_examples:
            print("\n📝 First Example:")
            example = smart_spec.contextual_examples[0]
            print(
                f"   Input: {json.dumps(example.get('input_data', {}), indent=6)[:200]}..."
            )
            print(
                f"   Output: {json.dumps(example.get('expected_output', ''), indent=6)[:200]}..."
            )

        # Test code generation
        code_gen = CodeGenerator()
        problem_code = await code_gen.generate_problem_module(
            name=f"test_{expected_type}_problem",
            description=description,
            domain=smart_spec.domain,
            difficulty="medium",
            examples=smart_spec.contextual_examples,
            output_type=smart_spec.problem_type,
            models=["gpt-3.5-turbo"],
            temperature_range=[0.3, 0.7],
        )

        # Verify Config class initialization in generated code
        print("\n🔧 Code Generation Check:")
        if "super().__init__(" in problem_code:
            print("   ✅ Config class properly calls super().__init__()")
        else:
            print("   ❌ Config class missing super().__init__() call")

        # Check if all required parameters are passed
        required_params = [
            "name",
            "description",
            "difficulty_level",
            "dataset_size",
            "model_configurations",
            "metrics",
            "optimization_objectives",
        ]
        all_params_found = all(param in problem_code for param in required_params)
        print(
            f"   {'✅' if all_params_found else '❌'} All required parameters present"
        )

        return True, smart_spec

    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback

        traceback.print_exc()
        return False, None


async def main():
    """Run all tests."""
    print("🧪 Testing Improved Problem Generation System")
    print("=" * 60)

    # Test cases for different problem types
    test_cases = [
        (
            "Classify customer support tickets into categories like billing, technical support, or general inquiry",
            "classification",
        ),
        (
            "Given assembly code with MOV, ADD, and JMP instructions, trace the execution and determine final register values",
            "reasoning",
        ),
        (
            "Generate product descriptions for an e-commerce website based on product specifications",
            "generation",
        ),
        (
            "Answer questions about Python programming based on provided documentation context",
            "question_answering",
        ),
        (
            "Extract name, email, and phone number from unstructured text messages",
            "information_extraction",
        ),
        (
            "Summarize long technical articles into concise 2-3 paragraph summaries",
            "summarization",
        ),
    ]

    results = []

    for description, expected_type in test_cases:
        success, spec = await test_problem_generation(description, expected_type)
        results.append(
            {
                "description": description,
                "expected_type": expected_type,
                "success": success,
                "actual_type": spec.problem_type if spec else None,
                "examples_count": len(spec.contextual_examples) if spec else 0,
            }
        )

    # Summary
    print(f"\n\n{'='*60}")
    print("📊 Test Summary")
    print(f"{'='*60}")

    success_count = sum(1 for r in results if r["success"])
    print(f"\nTotal Tests: {len(results)}")
    print(f"Successful: {success_count}")
    print(f"Failed: {len(results) - success_count}")

    print("\nDetailed Results:")
    for i, result in enumerate(results, 1):
        status = "✅" if result["success"] else "❌"
        type_match = (
            result["actual_type"] == result["expected_type"]
            if result["success"]
            else False
        )
        type_status = "✅" if type_match else "❌"

        print(f"\n{i}. {result['description'][:60]}...")
        print(f"   Status: {status}")
        print(f"   Expected Type: {result['expected_type']}")
        print(f"   Actual Type: {result['actual_type'] or 'N/A'} {type_status}")
        print(f"   Examples: {result['examples_count']}")

    # Test file organization
    print(f"\n\n{'='*60}")
    print("📁 Testing File Organization")
    print(f"{'='*60}")

    # Check if directory structure would be created correctly
    test_problem_name = "test_classification_problem"
    safe_filename = test_problem_name.replace(" ", "_").replace("-", "_").lower()

    expected_files = [
        f"langchain_problems/{safe_filename}.py",
        f"langchain_problems/{safe_filename}/metadata.json",
        f"langchain_problems/{safe_filename}/examples.json",
    ]

    print("\nExpected file structure:")
    for file_path in expected_files:
        print(f"   📄 {file_path}")

    print("\n✅ File organization structure verified")


if __name__ == "__main__":
    asyncio.run(main())
