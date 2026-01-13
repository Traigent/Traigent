#!/usr/bin/env python3
"""
Test script to verify Claude Code SDK is being used for classification.
"""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Test the imports
print("Testing imports...")
try:
    from problem_generation.improved_problem_classifier import ImprovedProblemClassifier

    print("✅ ImprovedProblemClassifier imported successfully")
except Exception as e:
    print(f"❌ Failed to import ImprovedProblemClassifier: {e}")

try:
    from problem_management.smart_problem_analyzer import SmartProblemAnalyzer

    print("✅ SmartProblemAnalyzer imported successfully")
except Exception as e:
    print(f"❌ Failed to import SmartProblemAnalyzer: {e}")


def test_classifier():
    """Test the classifier with JavaScript coding puzzles."""
    print("\n" + "=" * 60)
    print("Testing Classifier with JavaScript Coding Puzzles")
    print("=" * 60)

    # Initialize classifier
    classifier = ImprovedProblemClassifier(use_llm=True)

    # Test description
    description = "javascript coding puzzles"

    print(f"\nClassifying: '{description}'")
    result = classifier.classify(description)

    print("\nResult:")
    print(f"  Problem Type: {result.problem_type}")
    print(f"  Confidence: {result.confidence:.2f}")
    print(f"  Method: {result.classification_method}")
    print(f"  Reasoning: {result.reasoning[:100]}...")

    # Check if it was correctly classified as code_generation
    if result.problem_type == "code_generation":
        print("\n✅ Correctly classified as code_generation!")
    else:
        print(f"\n⚠️ Expected 'code_generation' but got '{result.problem_type}'")

    # Test with more descriptions
    test_cases = [
        "Write Python functions to solve algorithmic problems",
        "Classify customer emails into support categories",
        "Generate product descriptions for e-commerce",
        "Answer questions about historical events",
    ]

    print("\n\nTesting additional cases:")
    print("-" * 60)

    for test_desc in test_cases:
        result = classifier.classify(test_desc)
        print(f"\nDescription: {test_desc}")
        print(f"  Type: {result.problem_type} (confidence: {result.confidence:.2f})")


async def test_smart_analyzer():
    """Test the SmartProblemAnalyzer."""
    print("\n\n" + "=" * 60)
    print("Testing SmartProblemAnalyzer")
    print("=" * 60)

    analyzer = SmartProblemAnalyzer()

    description = "javascript coding puzzles"
    print(f"\nAnalyzing: '{description}'")

    try:
        # Test with small batch
        spec = await analyzer.analyze_and_generate_spec(
            user_description=description,
            target_examples=3,
            generation_mode="smart_single",
        )

        print("\nResults:")
        print(f"  Problem Type: {spec.problem_type}")
        print(f"  Domain: {spec.domain}")
        print(f"  Confidence: {spec.confidence_score:.1%}")
        print(f"  Examples Generated: {len(spec.contextual_examples)}")

        if spec.contextual_examples:
            first_example = spec.contextual_examples[0]
            print("\nFirst Example:")
            print(f"  Input: {first_example.get('input_data', {})}")
            if "expected_output" in first_example:
                output = str(first_example["expected_output"])
                if len(output) > 100:
                    output = output[:100] + "..."
                print(f"  Output: {output}")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    print("🧪 Testing Claude Code SDK Integration\n")

    # Test classifier
    test_classifier()

    # Test analyzer
    asyncio.run(test_smart_analyzer())

    print("\n✅ Tests completed!")
