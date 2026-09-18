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


class TestWithUsageModelCosts:
    """Per-call/per-model cost breakdown for multi-model agents (Traigent#1598)."""

    def test_model_costs_included_as_calls(self):
        """model_costs is threaded into __traigent_meta__['calls']."""
        ctx_handle = trial_context.set({"trial_id": 1})
        try:
            result = traigent.with_usage(
                text="answer",
                total_cost=0.01,
                model_costs=[
                    {
                        "model": "gpt-4o-mini",
                        "input_tokens": 200,
                        "output_tokens": 40,
                        "cost": 0.002,
                    },
                    {
                        "model": "gpt-4o",
                        "input_tokens": 500,
                        "output_tokens": 300,
                        "cost": 0.008,
                    },
                ],
            )
            assert result["__traigent_meta__"]["calls"] == [
                {
                    "model": "gpt-4o-mini",
                    "input_tokens": 200,
                    "output_tokens": 40,
                    "cost": 0.002,
                },
                {
                    "model": "gpt-4o",
                    "input_tokens": 500,
                    "output_tokens": 300,
                    "cost": 0.008,
                },
            ]
            # The blended total_cost stays the required, authoritative value.
            assert result["__traigent_meta__"]["total_cost"] == 0.01
        finally:
            trial_context.reset(ctx_handle)

    def test_model_costs_defaults_missing_tokens_to_zero(self):
        ctx_handle = trial_context.set({"trial_id": 1})
        try:
            result = traigent.with_usage(
                text="answer",
                total_cost=0.01,
                model_costs=[{"model": "gpt-4o-mini", "cost": 0.01}],
            )
            assert result["__traigent_meta__"]["calls"] == [
                {
                    "model": "gpt-4o-mini",
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "cost": 0.01,
                }
            ]
        finally:
            trial_context.reset(ctx_handle)

    def test_model_costs_none_omits_calls_key(self):
        ctx_handle = trial_context.set({"trial_id": 1})
        try:
            result = traigent.with_usage(text="answer", total_cost=0.01)
            assert "calls" not in result["__traigent_meta__"]
        finally:
            trial_context.reset(ctx_handle)

    def test_model_costs_not_a_list_raises(self):
        ctx_handle = trial_context.set({"trial_id": 1})
        try:
            with pytest.raises(TypeError, match="model_costs to be a list"):
                traigent.with_usage(
                    text="answer",
                    total_cost=0.01,
                    model_costs={"model": "gpt-4o-mini", "cost": 0.01},
                )
        finally:
            trial_context.reset(ctx_handle)

    def test_model_costs_entry_missing_model_raises(self):
        ctx_handle = trial_context.set({"trial_id": 1})
        try:
            with pytest.raises(TypeError, match="non-empty string 'model' key"):
                traigent.with_usage(
                    text="answer",
                    total_cost=0.01,
                    model_costs=[{"cost": 0.01}],
                )
        finally:
            trial_context.reset(ctx_handle)

    def test_model_costs_entry_missing_cost_raises(self):
        ctx_handle = trial_context.set({"trial_id": 1})
        try:
            with pytest.raises(TypeError, match="numeric 'cost' key"):
                traigent.with_usage(
                    text="answer",
                    total_cost=0.01,
                    model_costs=[{"model": "gpt-4o-mini"}],
                )
        finally:
            trial_context.reset(ctx_handle)

    def test_model_costs_production_mode_returns_plain_text(self):
        """Not in a trial: text is returned unwrapped, same as without model_costs."""
        assert traigent.get_trial_context() is None
        result = traigent.with_usage(
            text="answer",
            total_cost=0.01,
            model_costs=[{"model": "gpt-4o-mini", "cost": 0.01}],
        )
        assert result == "answer"

    def test_model_costs_bool_cost_raises(self):
        """``bool`` is a subclass of ``int``; ``float(True)`` is a $1.00 charge.

        ``meta_types`` rejects bools, but it never sees one from this path --
        ``with_usage`` normalises to ``float`` first, so the bool is already gone
        by the time that validator runs. Measured before this guard:
        ``cost=True`` produced ``{'cost': 1.0}`` with no error raised.
        """
        ctx_handle = trial_context.set({"trial_id": 1})
        try:
            with pytest.raises(TypeError, match="numeric 'cost' key"):
                traigent.with_usage(
                    text="answer",
                    total_cost=0.01,
                    model_costs=[{"model": "gpt-4o-mini", "cost": True}],
                )
        finally:
            trial_context.reset(ctx_handle)

    @pytest.mark.parametrize("field", ["input_tokens", "output_tokens"])
    def test_model_costs_bool_token_counts_raise(self, field):
        """Same subclass trap on the token counts: ``int(True)`` is 1."""
        ctx_handle = trial_context.set({"trial_id": 1})
        entry = {"model": "gpt-4o-mini", "cost": 0.01, field: True}
        try:
            with pytest.raises(TypeError, match=f"numeric '{field}' key"):
                traigent.with_usage(text="answer", total_cost=0.01, model_costs=[entry])
        finally:
            trial_context.reset(ctx_handle)

    def test_model_costs_non_numeric_token_counts_raise(self):
        ctx_handle = trial_context.set({"trial_id": 1})
        try:
            with pytest.raises(TypeError, match="numeric 'input_tokens' key"):
                traigent.with_usage(
                    text="answer",
                    total_cost=0.01,
                    model_costs=[
                        {"model": "gpt-4o-mini", "cost": 0.01, "input_tokens": "12"}
                    ],
                )
        finally:
            trial_context.reset(ctx_handle)
