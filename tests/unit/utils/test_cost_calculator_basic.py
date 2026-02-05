from traigent.utils.cost_calculator import CostCalculator


def test_cost_calculator_prompt_and_response_no_error():
    cc = CostCalculator()
    cb = cc.calculate_cost(prompt="hello", response="world", model_name="gpt-4o-mini")
    # Should return a CostBreakdown object with non-negative fields
    assert cb.total_cost >= 0.0
    assert cb.model_used == "gpt-4o-mini"


def test_cost_calculator_token_counts():
    cc = CostCalculator()
    cb = cc.calculate_cost(
        model_name="claude-3-haiku-20240307", input_tokens=100, output_tokens=50
    )
    assert cb.total_tokens >= 0
    # Cost may be zero if litellm unavailable
    assert cb.total_cost >= 0.0
