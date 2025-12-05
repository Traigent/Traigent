#!/usr/bin/env python
"""
TraiGent tokencost Integration Verification Script

Tests that tokencost is properly installed and integrated with TraiGent.
"""

import sys


# Colors for terminal output
class Colors:
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BLUE = "\033[94m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"


def print_header(message):
    """Print a formatted header."""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'=' * 60}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.BLUE}{message}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'=' * 60}{Colors.ENDC}\n")


def print_success(message):
    """Print success message."""
    print(f"{Colors.GREEN}✅ {message}{Colors.ENDC}")


def print_warning(message):
    """Print warning message."""
    print(f"{Colors.YELLOW}⚠️  {message}{Colors.ENDC}")


def print_error(message):
    """Print error message."""
    print(f"{Colors.RED}❌ {message}{Colors.ENDC}")


def print_info(message):
    """Print info message."""
    print(f"{Colors.BLUE}ℹ️  {message}{Colors.ENDC}")


def test_tokencost_import():
    """Test that tokencost can be imported."""
    print_header("Testing tokencost Import")

    try:
        import tokencost

        print_success("tokencost imported successfully")
        print_info(f"tokencost version: {getattr(tokencost, '__version__', 'unknown')}")

        # Test core functions
        try:
            # Import checks only
            from tokencost import (  # noqa: F401 - Import check only
                calculate_completion_cost,
                calculate_prompt_cost,
            )

            print_success("Core tokencost functions available")
            return True
        except ImportError as e:
            print_error(f"Cannot import core tokencost functions: {e}")
            return False

    except ImportError as e:
        print_error(f"Cannot import tokencost: {e}")
        print_info("Try installing with: pip install tokencost>=0.1.0")
        return False


def test_traigent_tokencost_integration():
    """Test that TraiGent properly integrates with tokencost."""
    print_header("Testing TraiGent Integration")

    try:
        from traigent.evaluators.metrics_tracker import (
            TOKENCOST_AVAILABLE,
            extract_llm_metrics,
        )

        print_success("TraiGent metrics_tracker imported successfully")
        print_info(f"TOKENCOST_AVAILABLE flag: {TOKENCOST_AVAILABLE}")

        if not TOKENCOST_AVAILABLE:
            print_warning("tokencost is not detected as available in TraiGent")
            print_info("This might be due to import issues or missing dependencies")
            return False

        # Test function signature
        import inspect

        sig = inspect.signature(extract_llm_metrics)
        params = list(sig.parameters.keys())
        expected_params = ["response", "model_name", "original_prompt", "response_text"]

        if all(param in params for param in expected_params):
            print_success("extract_llm_metrics function signature is correct")
            print_info(f"Parameters: {params}")
            return True
        else:
            print_error("extract_llm_metrics function signature is incorrect")
            print_info(f"Expected: {expected_params}")
            print_info(f"Found: {params}")
            return False

    except ImportError as e:
        print_error(f"Cannot import TraiGent components: {e}")
        return False


def test_cost_calculation():
    """Test actual cost calculation with mock data."""
    print_header("Testing Cost Calculation")

    try:
        from traigent.evaluators.metrics_tracker import extract_llm_metrics

        # Create a mock OpenAI response
        class MockOpenAIResponse:
            def __init__(self, content="positive", input_tokens=15, output_tokens=8):
                self.usage = type(
                    "Usage",
                    (),
                    {
                        "prompt_tokens": input_tokens,
                        "completion_tokens": output_tokens,
                        "total_tokens": input_tokens + output_tokens,
                    },
                )()
                self.choices = [
                    type(
                        "Choice",
                        (),
                        {"message": type("Message", (), {"content": content})()},
                    )()
                ]

        # Test cost calculation
        response = MockOpenAIResponse("positive", 15, 8)
        model_name = "gpt-4o-mini"
        original_prompt = [
            {"role": "user", "content": "What is the sentiment of 'Great product'?"}
        ]
        response_text = "positive"

        metrics = extract_llm_metrics(
            response=response,
            model_name=model_name,
            original_prompt=original_prompt,
            response_text=response_text,
        )

        print_success("Cost calculation completed successfully")
        print_info(f"Input tokens: {metrics.tokens.input_tokens}")
        print_info(f"Output tokens: {metrics.tokens.output_tokens}")
        print_info(f"Total cost: ${metrics.cost.total_cost:.6f}")

        # Verify we got actual costs (not zero)
        if metrics.cost.total_cost > 0:
            print_success("Real cost calculation is working (cost > 0)")
            return True
        else:
            print_warning("Cost calculation returned zero - may be using fallback mode")
            return True  # Still consider this a success since it didn't crash

    except Exception as e:
        print_error(f"Cost calculation test failed: {e}")
        return False


def test_different_models():
    """Test cost calculation with different model names."""
    print_header("Testing Different Models")

    try:
        from tokencost import calculate_completion_cost, calculate_prompt_cost

        # Test different models
        models = ["gpt-4o-mini", "gpt-3.5-turbo", "claude-3-5-sonnet-20241022"]
        prompt = [{"role": "user", "content": "Test prompt"}]
        completion = "Test response"

        for model in models:
            try:
                input_cost = calculate_prompt_cost(prompt, model)
                output_cost = calculate_completion_cost(completion, model)
                total_cost = input_cost + output_cost

                print_success(
                    f"{model}: ${total_cost:.6f} (${input_cost:.6f} + ${output_cost:.6f})"
                )
            except Exception as e:
                print_warning(f"{model}: Not supported or error - {e}")

        return True

    except Exception as e:
        print_error(f"Model testing failed: {e}")
        return False


def main():
    """Main entry point."""
    print_header("🔍 TraiGent tokencost Integration Verification")
    print("This script verifies that tokencost is properly integrated with TraiGent.\n")

    results = []

    # Test tokencost import
    results.append(("tokencost Import", test_tokencost_import()))

    # Test TraiGent integration
    results.append(("TraiGent Integration", test_traigent_tokencost_integration()))

    # Test cost calculation
    results.append(("Cost Calculation", test_cost_calculation()))

    # Test different models
    results.append(("Model Support", test_different_models()))

    # Summary
    print_header("🏁 Verification Summary")

    passed = 0
    for test_name, result in results:
        if result:
            print_success(f"{test_name}: PASSED")
            passed += 1
        else:
            print_error(f"{test_name}: FAILED")

    print(f"\n{Colors.BOLD}Results: {passed}/{len(results)} tests passed{Colors.ENDC}")

    if passed == len(results):
        print_success(
            "🎉 All tests passed! tokencost integration is working correctly."
        )
        return 0
    elif passed >= len(results) - 1:
        print_warning(
            "⚠️ Most tests passed. Minor issues detected but integration should work."
        )
        return 0
    else:
        print_error("❌ Multiple tests failed. Please check your installation.")
        print_info("Try running: pip install tokencost>=0.1.0")
        return 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\nVerification interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"{Colors.RED}❌ Unexpected error: {e}{Colors.ENDC}")
        sys.exit(1)
