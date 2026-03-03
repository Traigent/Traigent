"""Tests for config_generator.presets.constraint_templates."""

from __future__ import annotations

import pytest

from traigent.config_generator.presets.constraint_templates import (
    ConstraintTemplate,
    get_matching_constraints,
)


class TestConstraintTemplate:
    def test_matches_all_required_present(self) -> None:
        tmpl = ConstraintTemplate(
            name="test",
            description="desc",
            requires_tvars=frozenset({"a", "b"}),
        )
        assert tmpl.matches({"a", "b", "c"}) is True

    def test_no_match_missing_tvar(self) -> None:
        tmpl = ConstraintTemplate(
            name="test",
            description="desc",
            requires_tvars=frozenset({"a", "b"}),
        )
        assert tmpl.matches({"a", "c"}) is False

    def test_frozen(self) -> None:
        tmpl = ConstraintTemplate(
            name="test",
            description="desc",
            requires_tvars=frozenset({"x"}),
        )
        with pytest.raises(AttributeError):
            tmpl.name = "other"  # type: ignore[misc]


class TestGetMatchingConstraints:
    def test_temperature_top_p(self) -> None:
        constraints = get_matching_constraints({"temperature", "top_p"})
        names = {c.description for c in constraints}
        assert any("temperature" in d.lower() and "top_p" in d.lower() for d in names)

    def test_model_temperature(self) -> None:
        constraints = get_matching_constraints({"model", "temperature"})
        assert len(constraints) >= 1
        assert all(c.source == "template" for c in constraints)

    def test_model_max_tokens(self) -> None:
        constraints = get_matching_constraints({"model", "max_tokens"})
        assert len(constraints) >= 1
        assert any("max_tokens" in c.constraint_code for c in constraints)

    def test_chunk_size_overlap(self) -> None:
        constraints = get_matching_constraints({"chunk_size", "chunk_overlap"})
        assert len(constraints) >= 1
        assert any("chunk_overlap" in c.constraint_code for c in constraints)

    def test_no_matches_for_unrelated_tvars(self) -> None:
        constraints = get_matching_constraints({"foo", "bar"})
        assert constraints == []

    def test_empty_tvars(self) -> None:
        constraints = get_matching_constraints(set())
        assert constraints == []

    def test_multiple_templates_match(self) -> None:
        # temperature + top_p + model triggers 2 templates
        constraints = get_matching_constraints({"temperature", "top_p", "model"})
        assert len(constraints) >= 2

    def test_all_constraints_have_requires_tvars(self) -> None:
        constraints = get_matching_constraints(
            {
                "temperature",
                "top_p",
                "model",
                "max_tokens",
                "chunk_size",
                "chunk_overlap",
            }
        )
        for c in constraints:
            assert len(c.requires_tvars) >= 2

    def test_all_constraints_have_reasoning(self) -> None:
        constraints = get_matching_constraints({"temperature", "top_p"})
        for c in constraints:
            assert c.reasoning != ""
