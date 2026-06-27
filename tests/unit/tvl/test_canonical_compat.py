"""Canonical TVL ↔ SDK compatibility (TVL-first program, SDK packet 6-A).

The drift this closes (verified against the canonical tvl repo's grammar):
canonical structural constraints use `=` equality, BARE tvar names, and
DOTTED tvar names that are flat keys (`retriever.k` is ONE name) — while the
SDK constraint compiler parsed Python expressions (`==`, `params.`-prefixed,
nested attributes). Vendored fixtures under tests/fixtures/tvl_canonical pin
the exact canonical surface; grammar drift in the tvl repo becomes a
deliberate fixture update, not a silent break.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from traigent.tvl.spec_loader import compile_constraint_expression, load_tvl_spec

FIXTURES = Path(__file__).resolve().parents[3] / "tests" / "fixtures" / "tvl_canonical"


def _constraints(artifact):
    return list(artifact.constraints or [])


class TestCanonicalEqualityLoads:
    def test_rag_fixture_loads(self):
        artifact = load_tvl_spec(spec_path=FIXTURES / "rag-canonical.tvl.yml")
        assert artifact.configuration_space
        assert _constraints(artifact), "structural constraints must compile"

    def test_router_fixture_loads(self):
        artifact = load_tvl_spec(spec_path=FIXTURES / "router-canonical.tvl.yml")
        assert _constraints(artifact)


class TestCanonicalImplicationSemantics:
    def test_bool_guard_and_dotted_consequent(self):
        """when zero_shot = true / then retriever.k = 0 — the dotted name is a
        FLAT config key."""
        artifact = load_tvl_spec(spec_path=FIXTURES / "rag-canonical.tvl.yml")
        (rule,) = _constraints(artifact)
        # guard true, consequent violated
        assert rule({"zero_shot": True, "retriever.k": 3}, None) is False
        # guard true, consequent satisfied
        assert rule({"zero_shot": True, "retriever.k": 0}, None) is True
        # guard false -> implication holds regardless
        assert rule({"zero_shot": False, "retriever.k": 7}, None) is True

    def test_string_guard_with_mixed_operator(self):
        artifact = load_tvl_spec(spec_path=FIXTURES / "router-canonical.tvl.yml")
        rules = _constraints(artifact)
        evaluate = lambda cfg: all(rule(cfg, None) for rule in rules)  # noqa: E731
        assert evaluate({"routing_model": "rules", "fallback_threshold": 0.95})
        assert not evaluate({"routing_model": "rules", "fallback_threshold": 0.85})
        assert evaluate({"routing_model": "ml", "fallback_threshold": 0.1})

    def test_string_literal_containing_equals_survives_translation(self):
        """The '=' -> '==' rewrite must not corrupt quoted values like
        'k=v-style'."""
        artifact = load_tvl_spec(spec_path=FIXTURES / "router-canonical.tvl.yml")
        rules = _constraints(artifact)
        evaluate = lambda cfg: all(rule(cfg, None) for rule in rules)  # noqa: E731
        assert evaluate({"routing_model": "k=v-style", "fallback_threshold": 0.6})
        assert not evaluate({"routing_model": "k=v-style", "fallback_threshold": 0.3})


class TestPythonExpressionCompatibilityPreserved:
    """The SDK's own `params.x ==` dialect must keep working unchanged."""

    def test_params_prefixed_double_equals(self):
        rule = compile_constraint_expression('params.model == "gpt-4o"', label="compat")
        assert rule({"model": "gpt-4o"}, None) is True
        assert rule({"model": "other"}, None) is False

    def test_comparison_and_logic_dialect(self):
        rule = compile_constraint_expression(
            "params.k >= 2 and params.k <= 8", label="compat"
        )
        assert rule({"k": 4}, None) is True
        assert rule({"k": 9}, None) is False

    def test_double_equals_not_mangled_by_translation(self):
        """`==` must not become `====` or similar under the new rewrite."""
        rule = compile_constraint_expression(
            "params.a == 1 and params.b != 2 and params.c <= 3 and params.d >= 4",
            label="compat",
        )
        assert rule({"a": 1, "b": 0, "c": 3, "d": 4}, None) is True


class TestBareAndDottedResolution:
    def test_bare_name_binding(self):
        rule = compile_constraint_expression("zero_shot = true", label="bare")
        assert rule({"zero_shot": True}, None) is True
        assert rule({"zero_shot": False}, None) is False

    def test_dotted_flat_key_resolution(self):
        rule = compile_constraint_expression("retriever.k = 0", label="dotted")
        assert rule({"retriever.k": 0}, None) is True
        assert rule({"retriever.k": 5}, None) is False

    def test_params_prefixed_dotted_flat_key(self):
        """params.retriever.k must also resolve the FLAT 'retriever.k' key."""
        rule = compile_constraint_expression("params.retriever.k == 0", label="pdotted")
        assert rule({"retriever.k": 0}, None) is True
        assert rule({"retriever.k": 5}, None) is False

    def test_nested_mapping_still_works(self):
        """Genuinely nested configs keep attribute-chain semantics."""
        rule = compile_constraint_expression("params.retriever.k == 0", label="nested")
        assert rule({"retriever": {"k": 0}}, None) is True
        assert rule({"retriever": {"k": 2}}, None) is False

    def test_reserved_names_not_shadowed_by_config(self):
        """A config key named 'params'/'metrics'/'math' must not shadow the
        evaluation context."""
        rule = compile_constraint_expression("params.x == 1", label="reserved")
        assert rule({"x": 1, "params": "evil", "math": "evil"}, None) is True


class TestConfigSpaceCanonicalRoundTrip:
    def test_to_tvl_spec_emits_canonical_domains(self):
        from traigent.api.config_space import ConfigSpace
        from traigent.api.parameter_ranges import Choices, IntRange, LogRange, Range

        space = ConfigSpace.from_decorator_args(
            configuration_space={
                "model": Choices(values=["a", "b"]),
                "temp": Range(low=0.0, high=1.0, step=0.1),
                "k": IntRange(low=0, high=8),
                "lr": LogRange(low=1e-5, high=1e-1),
            }
        )
        spec = space.to_tvl_spec()
        by_name = {t["name"]: t for t in spec["tvars"]}
        # canonical literal-array enum
        assert by_name["model"]["domain"] == ["a", "b"]
        # canonical {range, resolution}
        assert by_name["temp"]["domain"] == {"range": [0.0, 1.0], "resolution": 0.1}
        assert by_name["k"]["domain"] == {"range": [0, 8]}
        assert by_name["lr"]["domain"] == {"range": [1e-5, 1e-1], "log": True}
        # subclass marker preserved as a sibling extension key
        assert by_name["lr"]["x_traigent_parameter_range"] == "LogRange"

    def test_round_trip_reconstructs_ranges(self):
        from traigent.api.config_space import ConfigSpace
        from traigent.api.parameter_ranges import Choices, IntRange, LogRange, Range

        space = ConfigSpace.from_decorator_args(
            configuration_space={
                "model": Choices(values=["a", "b"]),
                "temp": Range(low=0.0, high=1.0, step=0.1),
                "k": IntRange(low=0, high=8),
                "lr": LogRange(low=1e-5, high=1e-1),
            }
        )
        rebuilt = ConfigSpace.from_tvl_spec(space.to_tvl_spec())
        assert set(rebuilt.tvars) == set(space.tvars)
        for name in space.tvars:
            assert type(rebuilt.tvars[name]) is type(space.tvars[name]), name

    def test_from_tvl_spec_accepts_canonical_and_legacy_forms(self):
        from traigent.api.config_space import ConfigSpace
        from traigent.api.parameter_ranges import Choices, IntRange, Range

        spec = {
            "tvars": [
                # canonical literal-array enum (no marker -> type inference)
                {"name": "model", "type": "enum[str]", "domain": ["a", "b"]},
                # canonical kind-less range mapping
                {
                    "name": "temp",
                    "type": "float",
                    "domain": {"range": [0.0, 1.0], "resolution": 0.05},
                },
                # canonical set form
                {"name": "k", "type": "int", "domain": {"set": [1, 2, 4]}},
                # legacy SDK form with explicit kind (back-compat)
                {
                    "name": "legacy",
                    "type": "float",
                    "domain": {"kind": "range", "range": [0.0, 2.0]},
                },
            ]
        }
        space = ConfigSpace.from_tvl_spec(spec)
        assert isinstance(space.tvars["model"], Choices)
        assert isinstance(space.tvars["temp"], Range)
        assert isinstance(space.tvars["k"], Choices)
        assert space.tvars["temp"].step == pytest.approx(0.05)
        assert isinstance(space.tvars["legacy"], (Range, IntRange))


class TestReviewRegressionCoverage:
    """Round-1 codex review regressions (6-A convergence)."""

    def test_escaped_backslash_before_closing_quote(self):
        """BLOCKING fix: a quote preceded by an ESCAPED backslash (even
        parity) closes the string; '=' after it must still rewrite."""
        rule = compile_constraint_expression(
            'params.x = "a\\\\" and params.y = 1', label="bs"
        )
        assert rule({"x": "a\\", "y": 1}, None) is True
        assert rule({"x": "a\\", "y": 2}, None) is False

    def test_quoted_keyword_literals_not_rewritten(self):
        """'true'/'&&'/'null' INSIDE quotes pass through verbatim."""
        rule = compile_constraint_expression('params.mode = "true"', label="qk")
        assert rule({"mode": "true"}, None) is True
        rule = compile_constraint_expression('params.op = "a && b"', label="qa")
        assert rule({"op": "a && b"}, None) is True

    def test_unquoted_keywords_still_rewritten(self):
        rule = compile_constraint_expression(
            "params.flag = true && params.other = null", label="uk"
        )
        assert rule({"flag": True, "other": None}, None) is True

    def test_reserved_root_collision_documented_behavior(self):
        """A config key whose root collides with a reserved name cannot be
        referenced bare — reserved names always win (documented). The
        params.<name> form remains available."""
        rule = compile_constraint_expression("params.len == 3", label="rsv")
        assert rule({"len": 3}, None) is True
        bare = compile_constraint_expression("len = 3", label="rsv2")
        assert bare({"len": 3}, None) is False  # compares builtin len to 3

    def test_nested_wins_over_flat_dotted_key(self):
        """Deterministic precedence: a genuine nested mapping shadows a flat
        dotted key for the same path."""
        rule = compile_constraint_expression("params.retriever.k == 1", label="prec")
        assert rule({"retriever": {"k": 2}, "retriever.k": 1}, None) is False
        assert rule({"retriever": {"k": 1}, "retriever.k": 2}, None) is True

    def test_invalid_canonical_domain_payloads_rejected(self):
        from traigent.api.config_space import ConfigSpace

        for bad_domain in ({"set": "abc"}, {"values": 1}, {"set": 1}):
            with pytest.raises(ValueError):
                ConfigSpace.from_tvl_spec(
                    {"tvars": [{"name": "x", "type": "int", "domain": bad_domain}]}
                )
