"""Unit tests for __traigent_meta__ extraction in LocalEvaluator."""

from traigent.evaluators.local import LocalEvaluator
from traigent.evaluators.metrics_tracker import ExampleMetrics


class TestMetaExtraction:
    """Test __traigent_meta__ extraction and injection."""

    def test_meta_extraction_with_full_meta(self):
        """Should extract __traigent_meta__ and inject into metrics."""
        evaluator = LocalEvaluator()
        metrics = ExampleMetrics()

        output = {
            "text": "answer",
            "__traigent_meta__": {
                "usage": {"input_tokens": 100, "output_tokens": 50},
                "total_cost": 0.0023,
            },
        }
        meta = evaluator._extract_and_inject_traigent_meta(output, metrics)

        assert meta is not None
        assert metrics.tokens.input_tokens == 100
        assert metrics.tokens.output_tokens == 50
        assert metrics.tokens.total_tokens == 150
        assert metrics.cost.total_cost == 0.0023

    def test_meta_extraction_plain_string(self):
        """Should return None for plain strings (not modify metrics)."""
        evaluator = LocalEvaluator()
        metrics = ExampleMetrics()
        # Set non-zero value to verify it's unchanged
        metrics.tokens.input_tokens = 500

        meta = evaluator._extract_and_inject_traigent_meta("plain answer", metrics)

        assert meta is None
        assert metrics.tokens.input_tokens == 500  # Unchanged

    def test_meta_extraction_dict_without_meta(self):
        """Should return None for dict without __traigent_meta__."""
        evaluator = LocalEvaluator()
        metrics = ExampleMetrics()
        metrics.tokens.input_tokens = 500

        output = {"text": "answer", "other_field": "value"}
        meta = evaluator._extract_and_inject_traigent_meta(output, metrics)

        assert meta is None
        assert metrics.tokens.input_tokens == 500  # Unchanged

    def test_meta_extraction_explicit_zero_tokens(self):
        """Should allow explicit zero tokens to override existing values."""
        evaluator = LocalEvaluator()
        metrics = ExampleMetrics()
        metrics.tokens.input_tokens = 500  # Pre-existing value

        output = {
            "text": "answer",
            "__traigent_meta__": {
                "usage": {"input_tokens": 0, "output_tokens": 10},  # Explicit zero
                "total_cost": 0.001,
            },
        }
        meta = evaluator._extract_and_inject_traigent_meta(output, metrics)

        assert meta is not None
        assert metrics.tokens.input_tokens == 0  # Zero overrides 500
        assert metrics.tokens.output_tokens == 10

    def test_meta_extraction_only_input_tokens(self):
        """Should inject only input_tokens when output_tokens not provided."""
        evaluator = LocalEvaluator()
        metrics = ExampleMetrics()
        metrics.tokens.output_tokens = 999  # Pre-existing value

        output = {
            "text": "answer",
            "__traigent_meta__": {
                "usage": {"input_tokens": 100},  # Only input_tokens
                "total_cost": 0.001,
            },
        }
        meta = evaluator._extract_and_inject_traigent_meta(output, metrics)

        assert meta is not None
        assert metrics.tokens.input_tokens == 100
        assert metrics.tokens.output_tokens == 999  # Unchanged
        assert metrics.tokens.total_tokens == 1099  # Recomputed

    def test_meta_extraction_only_output_tokens(self):
        """Should inject only output_tokens when input_tokens not provided."""
        evaluator = LocalEvaluator()
        metrics = ExampleMetrics()
        metrics.tokens.input_tokens = 888  # Pre-existing value

        output = {
            "text": "answer",
            "__traigent_meta__": {
                "usage": {"output_tokens": 50},  # Only output_tokens
                "total_cost": 0.001,
            },
        }
        meta = evaluator._extract_and_inject_traigent_meta(output, metrics)

        assert meta is not None
        assert metrics.tokens.input_tokens == 888  # Unchanged
        assert metrics.tokens.output_tokens == 50
        assert metrics.tokens.total_tokens == 938  # Recomputed

    def test_meta_extraction_only_cost(self):
        """Should inject only cost when usage not provided."""
        evaluator = LocalEvaluator()
        metrics = ExampleMetrics()
        metrics.tokens.input_tokens = 100
        metrics.cost.total_cost = 0.0  # Initial value

        output = {
            "text": "answer",
            "__traigent_meta__": {
                "total_cost": 0.0023,
                # No usage dict
            },
        }
        meta = evaluator._extract_and_inject_traigent_meta(output, metrics)

        assert meta is not None
        assert metrics.cost.total_cost == 0.0023
        assert metrics.tokens.input_tokens == 100  # Unchanged

    def test_meta_extraction_malformed_usage_graceful(self):
        """Malformed usage fails type guard - returns None."""
        evaluator = LocalEvaluator()
        metrics = ExampleMetrics()
        metrics.tokens.input_tokens = 100  # Initial value

        output = {
            "text": "answer",
            "__traigent_meta__": {
                "usage": "not a dict",  # Malformed - fails type guard
                "total_cost": 0.0023,
            },
        }
        # Type guard rejects entire structure when usage is malformed
        meta = evaluator._extract_and_inject_traigent_meta(output, metrics)

        assert meta is None  # Type guard fails
        assert metrics.tokens.input_tokens == 100  # Unchanged

    def test_meta_extraction_malformed_cost_graceful(self):
        """Malformed cost fails type guard - returns None."""
        evaluator = LocalEvaluator()
        metrics = ExampleMetrics()
        metrics.tokens.input_tokens = 0  # Initial value

        output = {
            "text": "answer",
            "__traigent_meta__": {
                "usage": {"input_tokens": 100, "output_tokens": 50},
                "total_cost": "not a number",  # Malformed - fails type guard
            },
        }
        # Type guard rejects entire structure when total_cost is malformed
        meta = evaluator._extract_and_inject_traigent_meta(output, metrics)

        assert meta is None  # Type guard fails
        assert metrics.tokens.input_tokens == 0  # Unchanged

    def test_meta_extraction_negative_cost_clamped(self):
        """Should clamp negative cost to 0."""
        evaluator = LocalEvaluator()
        metrics = ExampleMetrics()

        output = {
            "text": "answer",
            "__traigent_meta__": {
                "total_cost": -0.5,  # Negative cost
            },
        }
        meta = evaluator._extract_and_inject_traigent_meta(output, metrics)

        assert meta is not None
        assert metrics.cost.total_cost == 0.0  # Clamped to 0

    def test_meta_extraction_negative_tokens_clamped(self):
        """Should clamp negative tokens to 0."""
        evaluator = LocalEvaluator()
        metrics = ExampleMetrics()

        output = {
            "text": "answer",
            "__traigent_meta__": {
                "usage": {"input_tokens": -100, "output_tokens": -50},
                "total_cost": 0.001,
            },
        }
        meta = evaluator._extract_and_inject_traigent_meta(output, metrics)

        assert meta is not None
        assert metrics.tokens.input_tokens == 0  # Clamped
        assert metrics.tokens.output_tokens == 0  # Clamped
        assert metrics.tokens.total_tokens == 0

    def test_meta_extraction_preserves_other_fields(self):
        """Should not affect other fields in the output dict."""
        evaluator = LocalEvaluator()
        metrics = ExampleMetrics()

        output = {
            "text": "answer",
            "citations": ["source1", "source2"],
            "raw_response": "full response",
            "__traigent_meta__": {
                "total_cost": 0.001,
            },
        }
        meta = evaluator._extract_and_inject_traigent_meta(output, metrics)

        assert meta is not None
        # Output dict unchanged (meta extraction doesn't modify output)
        assert output["citations"] == ["source1", "source2"]
        assert output["raw_response"] == "full response"
        assert output["text"] == "answer"
