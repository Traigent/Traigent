"""Characterization of ContentPolicy._name (span-name egress policy)."""

from __future__ import annotations

import pytest

from traigent.observability.otel.policy import ContentPolicy

SECRET_NAME = "sk-abcdefghijklmnopqrstuvwxyz0123456789"  # pragma: allowlist secret


def _name(policy, raw, attrs=None, mode=None):
    return policy._name(raw, attrs or {}, "scope", mode or policy.mode)


class TestRecordMode:
    def test_nonempty_string_returned_scrubbed(self):
        policy = ContentPolicy("record")
        assert _name(policy, f"{SECRET_NAME} x") == "[REDACTED:api_key] x"

    def test_record_wins_over_allowlist(self):
        policy = ContentPolicy("record", allowed_span_names={"only.this"})
        assert _name(policy, "other name") == "other name"

    @pytest.mark.parametrize("length,expected", [(255, 255), (256, 256), (257, 256)])
    def test_record_name_capped_at_256(self, length, expected):
        policy = ContentPolicy("record")
        assert len(_name(policy, "a" * length)) == expected

    @pytest.mark.parametrize("raw", ["", None, 5, b"x"])
    def test_empty_or_non_string_falls_back_to_derived(self, raw):
        policy = ContentPolicy("record")
        assert _name(policy, raw, {"gen_ai.operation.name": "chat"}) == "chat"
        assert _name(policy, raw) == "span"

    def test_mode_argument_not_instance_mode_selects_record(self):
        policy = ContentPolicy("metadata")
        assert _name(policy, "free text here", mode="record") == "free text here"
        assert _name(policy, "free text here", mode="metadata") == "span"


class TestMetadataNames:
    def test_safe_token_returned(self):
        assert _name(ContentPolicy("metadata"), "my.operation") == "my.operation"

    @pytest.mark.parametrize(
        "raw", ["order_12345", "has space", "a" * 81, "", "user:42", SECRET_NAME]
    )
    def test_unsafe_name_not_returned(self, raw):
        assert _name(ContentPolicy("metadata"), raw) == "span"

    def test_safe_name_wins_over_derived(self):
        attrs = {"gen_ai.operation.name": "chat", "openinference.span.kind": "AGENT"}
        assert _name(ContentPolicy("metadata"), "my.operation", attrs) == "my.operation"

    def test_non_string_raw_ignored(self):
        assert _name(ContentPolicy("metadata"), 5) == "span"


class TestAllowlist:
    def test_listed_name_returned_even_when_identifier_shaped(self):
        policy = ContentPolicy("metadata", allowed_span_names={"order_12345"})
        assert _name(policy, "order_12345") == "order_12345"

    def test_unlisted_safe_token_denied_when_allowlist_set(self):
        policy = ContentPolicy("metadata", allowed_span_names={"listed"})
        assert _name(policy, "my.operation") == "span"
        assert (
            _name(policy, "my.operation", {"gen_ai.operation.name": "chat"}) == "chat"
        )

    def test_empty_allowlist_denies_everything(self):
        policy = ContentPolicy("metadata", allowed_span_names=set())
        assert _name(policy, "my.operation") == "span"


class TestDerivedName:
    P = ContentPolicy("metadata")

    def test_operation_and_model(self):
        attrs = {"gen_ai.operation.name": "chat", "gen_ai.request.model": "m-1"}
        assert _name(self.P, "bad name", attrs) == "chat m-1"

    def test_operation_only_when_model_not_string(self):
        attrs = {"gen_ai.operation.name": "chat", "gen_ai.request.model": 7}
        assert _name(self.P, "bad name", attrs) == "chat"

    def test_derived_length_boundary(self):
        # 80 + 128 = 208 characters is the longest accepted derived name.
        ok = {"gen_ai.operation.name": "o" * 208}
        too_long = {"gen_ai.operation.name": "o" * 209}
        assert _name(self.P, "bad name", ok) == "o" * 208
        assert _name(self.P, "bad name", too_long) == "span"

    def test_overlong_derivation_falls_through_to_kind_then_type(self):
        attrs = {
            "gen_ai.operation.name": "o" * 209,
            "openinference.span.kind": "LLM",
        }
        assert _name(self.P, "bad name", attrs) == "llm"
        attrs = {
            "gen_ai.operation.name": "o" * 209,
            "traigent.observation_type": "generation",
        }
        assert _name(self.P, "bad name", attrs) == "generation"

    def test_non_string_operation_falls_to_kind(self):
        attrs = {"gen_ai.operation.name": 5, "openinference.span.kind": "AGENT"}
        assert _name(self.P, "bad name", attrs) == "agent"

    def test_operation_wins_over_kind_and_type(self):
        attrs = {
            "gen_ai.operation.name": "chat",
            "openinference.span.kind": "AGENT",
            "traigent.observation_type": "generation",
        }
        assert _name(self.P, "bad name", attrs) == "chat"

    def test_kind_wins_over_type(self):
        attrs = {
            "openinference.span.kind": "TOOL",
            "traigent.observation_type": "generation",
        }
        assert _name(self.P, "bad name", attrs) == "tool"

    def test_non_string_kind_falls_to_type(self):
        attrs = {
            "openinference.span.kind": 3,
            "traigent.observation_type": "generation",
        }
        assert _name(self.P, "bad name", attrs) == "generation"

    @pytest.mark.parametrize("otype", ["unknown-type", 5, None, ""])
    def test_unknown_observation_type_yields_span(self, otype):
        assert _name(self.P, "bad name", {"traigent.observation_type": otype}) == "span"

    def test_empty_attrs_yield_span(self):
        assert _name(self.P, "bad name", {}) == "span"
