"""Unit tests for __traigent_meta__ extraction in LocalEvaluator."""

import pytest

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

    def test_meta_extraction_clamps_implausible_reported_cost(self, caplog):
        """with_usage-style metadata cannot under-report known model token cost."""
        from traigent.utils.cost_calculator import cost_from_tokens

        evaluator = LocalEvaluator()
        metrics = ExampleMetrics()
        output = {
            "text": "answer",
            "__traigent_meta__": {
                "usage": {"input_tokens": 1000, "output_tokens": 500},
                "total_cost": 0.00000001,
            },
        }
        expected = sum(cost_from_tokens(1000, 500, "gpt-4o-mini", strict=False))

        with caplog.at_level("WARNING"):
            meta = evaluator._extract_and_inject_traigent_meta(
                output, metrics, model_name="gpt-4o-mini"
            )

        assert meta is not None
        assert metrics.cost.total_cost == expected
        assert "implausibly below token-derived estimate" in caplog.text

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


class TestCallBreakdownExtraction:
    """Per-call/per-model cost breakdown extraction (Traigent#1598)."""

    def test_calls_breakdown_injected_into_call_breakdown(self):
        """A valid 'calls' list is normalized onto metrics.call_breakdown."""
        evaluator = LocalEvaluator()
        metrics = ExampleMetrics()

        output = {
            "text": "answer",
            "__traigent_meta__": {
                "total_cost": 0.01,
                "calls": [
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
            },
        }
        meta = evaluator._extract_and_inject_traigent_meta(output, metrics)

        assert meta is not None
        assert metrics.call_breakdown == [
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
        # The blended total_cost is unaffected -- it stays authoritative.
        assert metrics.cost.total_cost == 0.01

    def test_no_calls_key_leaves_call_breakdown_empty(self):
        """Absence of 'calls' means no per-model attribution was reported."""
        evaluator = LocalEvaluator()
        metrics = ExampleMetrics()

        output = {"text": "answer", "__traigent_meta__": {"total_cost": 0.01}}
        meta = evaluator._extract_and_inject_traigent_meta(output, metrics)

        assert meta is not None
        assert metrics.call_breakdown == []

    def test_negative_values_clamped_to_zero(self):
        evaluator = LocalEvaluator()
        metrics = ExampleMetrics()

        output = {
            "text": "answer",
            "__traigent_meta__": {
                "total_cost": 0.01,
                "calls": [
                    {
                        "model": "gpt-4o-mini",
                        "input_tokens": -50,
                        "output_tokens": -10,
                        "cost": -0.001,
                    }
                ],
            },
        }
        evaluator._extract_and_inject_traigent_meta(output, metrics)

        assert metrics.call_breakdown == [
            {
                "model": "gpt-4o-mini",
                "input_tokens": 0,
                "output_tokens": 0,
                "cost": 0.0,
            }
        ]

    @pytest.mark.parametrize(
        "bad_calls",
        [
            pytest.param([{"cost": 0.001}], id="missing-model"),
            pytest.param([{"model": "m", "cost": float("nan")}], id="nan-cost"),
            pytest.param([{"model": "m", "cost": float("inf")}], id="inf-cost"),
            pytest.param([{"model": "m", "cost": True}], id="bool-cost"),
            pytest.param("not-a-list", id="not-a-list"),
        ],
    )
    def test_malformed_calls_drops_attribution_but_keeps_the_reported_cost(
        self, bad_calls
    ):
        """A bad attribution entry must not throw away a valid total_cost.

        `calls` says WHICH model spent the money; `total_cost` says HOW MUCH was
        spent. An earlier version failed the type guard for the whole
        ``__traigent_meta__`` and called that fail-closed -- but for money it is
        fail-OPEN: a known $0.01 charge was recorded as $0.00, the run
        under-reported spend, and that figure reaches ``format_for_backend`` and
        budget accounting. Drop the attribution; keep the money.
        """
        evaluator = LocalEvaluator()
        metrics = ExampleMetrics()

        output = {
            "text": "answer",
            "__traigent_meta__": {"total_cost": 0.01, "calls": bad_calls},
        }
        evaluator._extract_and_inject_traigent_meta(output, metrics)

        assert metrics.cost.total_cost == 0.01, (
            "a malformed attribution discarded an authoritative reported cost"
        )
        assert metrics.call_breakdown == []

    def test_valid_calls_keep_both_the_breakdown_and_the_cost(self):
        """The severing must not cost us the feature it guards."""
        evaluator = LocalEvaluator()
        metrics = ExampleMetrics()

        output = {
            "text": "answer",
            "__traigent_meta__": {
                "total_cost": 0.01,
                "calls": [
                    {"model": "gpt-4o-mini", "cost": 0.002},
                    {"model": "gpt-4o", "cost": 0.008},
                ],
            },
        }
        evaluator._extract_and_inject_traigent_meta(output, metrics)

        assert metrics.cost.total_cost == 0.01
        assert [entry["model"] for entry in metrics.call_breakdown] == [
            "gpt-4o-mini",
            "gpt-4o",
        ]

    def test_attribution_sum_never_moves_the_injected_total_cost(self):
        """A huge per-model sum must leave the billed total untouched.

        Both of this repo's recent cost regressions were an attribution path
        feeding the billed total. The invariant held when this was written, but
        no test pinned it -- a mutation adding sum(call costs) to
        ``metrics.cost.total_cost`` passed the entire suite.
        """
        evaluator = LocalEvaluator()
        metrics = ExampleMetrics()

        output = {
            "text": "answer",
            "__traigent_meta__": {
                "total_cost": 0.25,
                "calls": [
                    {"model": "gpt-4o", "cost": 5.0},
                    {"model": "gpt-4o-mini", "cost": 7.0},
                ],
            },
        }
        evaluator._extract_and_inject_traigent_meta(output, metrics)

        assert metrics.cost.total_cost == 0.25, (
            "the per-model attribution sum leaked into the billed total: "
            f"{metrics.cost.total_cost!r}"
        )
        assert sum(entry["cost"] for entry in metrics.call_breakdown) == 12.0
