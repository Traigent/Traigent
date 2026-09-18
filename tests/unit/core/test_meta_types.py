"""Unit tests for the __traigent_meta__ type guard (traigent.core.meta_types).

Covers the ``calls`` per-call/per-model breakdown field added for
multi-model/multi-step agents (Traigent#1598).
"""

from traigent.core.meta_types import is_traigent_metadata


class TestIsTraigentMetadataCalls:
    """Validation of the optional ``calls`` per-call cost breakdown."""

    def test_valid_calls_breakdown(self):
        """A well-formed calls list validates."""
        meta = {
            "total_cost": 0.01,
            "calls": [
                {
                    "model": "gpt-4o-mini",
                    "input_tokens": 100,
                    "output_tokens": 20,
                    "cost": 0.002,
                },
                {
                    "model": "gpt-4o",
                    "input_tokens": 500,
                    "output_tokens": 300,
                    "cost": 0.008,
                },
            ],
        }
        assert is_traigent_metadata(meta) is True

    def test_calls_without_token_counts_is_valid(self):
        """input_tokens/output_tokens are optional per call."""
        meta = {
            "total_cost": 0.01,
            "calls": [{"model": "gpt-4o-mini", "cost": 0.01}],
        }
        assert is_traigent_metadata(meta) is True

    def test_calls_not_a_list_is_invalid(self):
        meta = {"total_cost": 0.01, "calls": {"model": "gpt-4o-mini", "cost": 0.01}}
        assert is_traigent_metadata(meta) is False

    def test_calls_entry_not_a_dict_is_invalid(self):
        meta = {"total_cost": 0.01, "calls": ["gpt-4o-mini"]}
        assert is_traigent_metadata(meta) is False

    def test_calls_entry_missing_model_is_invalid(self):
        meta = {"total_cost": 0.01, "calls": [{"cost": 0.01}]}
        assert is_traigent_metadata(meta) is False

    def test_calls_entry_non_string_model_is_invalid(self):
        meta = {"total_cost": 0.01, "calls": [{"model": 123, "cost": 0.01}]}
        assert is_traigent_metadata(meta) is False

    def test_calls_entry_missing_cost_is_invalid(self):
        meta = {"total_cost": 0.01, "calls": [{"model": "gpt-4o-mini"}]}
        assert is_traigent_metadata(meta) is False

    def test_calls_entry_bool_cost_is_invalid(self):
        """bool is a subclass of int in Python; must be explicitly rejected."""
        meta = {"total_cost": 0.01, "calls": [{"model": "gpt-4o-mini", "cost": True}]}
        assert is_traigent_metadata(meta) is False

    def test_calls_entry_non_int_tokens_is_invalid(self):
        meta = {
            "total_cost": 0.01,
            "calls": [{"model": "gpt-4o-mini", "cost": 0.01, "input_tokens": "100"}],
        }
        assert is_traigent_metadata(meta) is False

    def test_calls_entry_bool_tokens_is_invalid(self):
        meta = {
            "total_cost": 0.01,
            "calls": [{"model": "gpt-4o-mini", "cost": 0.01, "input_tokens": True}],
        }
        assert is_traigent_metadata(meta) is False

    def test_empty_calls_list_is_valid(self):
        meta = {"total_cost": 0.01, "calls": []}
        assert is_traigent_metadata(meta) is True

    def test_missing_calls_key_still_valid(self):
        """calls is optional -- absence must not break existing callers."""
        meta = {"total_cost": 0.01, "usage": {"input_tokens": 10, "output_tokens": 5}}
        assert is_traigent_metadata(meta) is True
