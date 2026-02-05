"""Unit tests for with_usage() helper function."""

import pytest

import traigent
from traigent.config.context import trial_context


class TestWithUsage:
    """Test with_usage() helper function."""

    def test_with_usage_in_optimization_mode(self):
        """Should return dict with __traigent_meta__ when optimizing."""
        token = trial_context.set({"trial_id": 1})
        try:
            result = traigent.with_usage(
                text="hello",
                total_cost=0.0023,
                input_tokens=100,
                output_tokens=50,
            )
            assert result == {
                "text": "hello",
                "__traigent_meta__": {
                    "total_cost": 0.0023,
                    "usage": {"input_tokens": 100, "output_tokens": 50},
                },
            }
        finally:
            trial_context.reset(token)

    def test_with_usage_in_production_mode(self):
        """Should return text directly when not optimizing."""
        # Ensure no trial context
        assert traigent.get_trial_context() is None

        result = traigent.with_usage(
            text="hello", total_cost=0.0023, input_tokens=100, output_tokens=50
        )
        assert result == "hello"  # No wrapper in production

    def test_with_usage_rejects_non_string(self):
        """Should raise TypeError if text is not a string."""
        token = trial_context.set({"trial_id": 1})
        try:
            with pytest.raises(TypeError, match="requires text to be a string"):
                traigent.with_usage(
                    text={"answer": "hello"},  # Dict not allowed
                    total_cost=0.0023,
                )
        finally:
            trial_context.reset(token)

    def test_with_usage_only_total_cost(self):
        """Should work with only total_cost (no token counts)."""
        token = trial_context.set({"trial_id": 1})
        try:
            result = traigent.with_usage(text="hello", total_cost=0.0023)
            assert result == {
                "text": "hello",
                "__traigent_meta__": {
                    "total_cost": 0.0023,
                    # No usage dict
                },
            }
        finally:
            trial_context.reset(token)

    def test_with_usage_only_input_tokens(self):
        """Should include usage dict when only input_tokens provided."""
        token = trial_context.set({"trial_id": 1})
        try:
            result = traigent.with_usage(
                text="hello", total_cost=0.001, input_tokens=100
            )
            assert result == {
                "text": "hello",
                "__traigent_meta__": {
                    "total_cost": 0.001,
                    "usage": {"input_tokens": 100},
                },
            }
        finally:
            trial_context.reset(token)

    def test_with_usage_only_output_tokens(self):
        """Should include usage dict when only output_tokens provided."""
        token = trial_context.set({"trial_id": 1})
        try:
            result = traigent.with_usage(
                text="hello", total_cost=0.001, output_tokens=50
            )
            assert result == {
                "text": "hello",
                "__traigent_meta__": {
                    "total_cost": 0.001,
                    "usage": {"output_tokens": 50},
                },
            }
        finally:
            trial_context.reset(token)

    def test_with_usage_explicit_zero_tokens(self):
        """Should allow explicit zero tokens (not treat as None)."""
        token = trial_context.set({"trial_id": 1})
        try:
            result = traigent.with_usage(
                text="hello", total_cost=0.001, input_tokens=0, output_tokens=10
            )
            assert result == {
                "text": "hello",
                "__traigent_meta__": {
                    "total_cost": 0.001,
                    "usage": {"input_tokens": 0, "output_tokens": 10},
                },
            }
        finally:
            trial_context.reset(token)

    def test_with_usage_cost_conversion(self):
        """Should convert cost to float."""
        token = trial_context.set({"trial_id": 1})
        try:
            result = traigent.with_usage(
                text="hello",
                total_cost=1,  # int should be converted to float
            )
            assert result["__traigent_meta__"]["total_cost"] == 1.0
            assert isinstance(result["__traigent_meta__"]["total_cost"], float)
        finally:
            trial_context.reset(token)

    def test_with_usage_token_conversion(self):
        """Should convert tokens to int."""
        token = trial_context.set({"trial_id": 1})
        try:
            result = traigent.with_usage(
                text="hello",
                total_cost=0.001,
                input_tokens=100.5,  # float should be converted to int
                output_tokens=50.8,
            )
            assert result["__traigent_meta__"]["usage"]["input_tokens"] == 100
            assert result["__traigent_meta__"]["usage"]["output_tokens"] == 50
            assert isinstance(result["__traigent_meta__"]["usage"]["input_tokens"], int)
        finally:
            trial_context.reset(token)
