"""Test script for the new constraint DSL syntax."""

from traigent.api.constraints import implies, require, when
from traigent.api.parameter_ranges import Choices, Range


def test_dsl_syntax():
    """Test all DSL syntax styles."""
    model = Choices(["gpt-4", "gpt-3.5"], name="model")
    temp = Range(0.0, 2.0, name="temperature")
    max_tokens = Range(100, 4096, name="max_tokens")

    # Style 1: Functional (canonical) - always works
    c1 = implies(model.equals("gpt-4"), temp.lte(0.7))
    print("1. Functional style (implies):", type(c1).__name__)

    # Style 2: Operator-based (>>) - concise
    c2 = model.equals("gpt-4") >> temp.lte(0.7)
    print("2. Operator style (>>):", type(c2).__name__)

    # Style 3: Fluent (when/then) - readable
    c3 = when(model.equals("gpt-4")).then(temp.lte(0.7))
    print("3. Fluent style (when/then):", type(c3).__name__)

    # Style 4: Method-based (.implies) - also fluent
    c4 = model.equals("gpt-4").implies(temp.lte(0.7))
    print("4. Method style (.implies):", type(c4).__name__)

    # Compound expressions with & | ~
    c5 = model.equals("gpt-4") & temp.lte(0.7)
    print("5. Conjunction (&):", type(c5).__name__)

    c6 = model.equals("gpt-4") | model.equals("gpt-3.5")
    print("6. Disjunction (|):", type(c6).__name__)

    c7 = ~model.equals("gpt-4")
    print("7. Negation (~):", type(c7).__name__)

    # Complex constraint: (A & B) >> C
    c8 = (model.equals("gpt-4") & temp.lte(0.5)) >> max_tokens.gte(1000)
    print("8. Complex ((A & B) >> C):", type(c8).__name__)

    # Standalone requirement
    c9 = require(temp.lte(1.5))
    print("9. Standalone (require):", type(c9).__name__)

    # Test evaluation
    config_valid = {"model": "gpt-4", "temperature": 0.5, "max_tokens": 2000}
    config_invalid = {"model": "gpt-4", "temperature": 0.9, "max_tokens": 500}

    var_names = {
        id(model): "model",
        id(temp): "temperature",
        id(max_tokens): "max_tokens",
    }

    print("\n--- Evaluation ---")
    print(f"c1 on valid config: {c1.evaluate(config_valid, var_names)}")
    print(f"c1 on invalid config: {c1.evaluate(config_invalid, var_names)}")

    # Test to_expression
    print("\n--- Expression Export ---")
    print(f"c1 expression: {c1.to_expression(var_names)}")
    print(f"c8 expression: {c8.to_expression(var_names)}")

    print("\n✓ All DSL syntax tests passed!")


if __name__ == "__main__":
    test_dsl_syntax()
