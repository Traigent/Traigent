"""Tests for config_generator.subsystems.structural_constraints."""

from __future__ import annotations

from unittest.mock import MagicMock

from traigent.config_generator.llm_backend import BudgetExhausted
from traigent.config_generator.subsystems.structural_constraints import (
    generate_structural_constraints,
)
from traigent.config_generator.types import TVarSpec


def _make_tvar(name: str) -> TVarSpec:
    return TVarSpec(name=name, range_type="Range", range_kwargs={"low": 0, "high": 1})


class TestGenerateStructuralConstraints:
    def test_template_matching_temp_top_p(self) -> None:
        tvars = [_make_tvar("temperature"), _make_tvar("top_p")]
        constraints = generate_structural_constraints(tvars)
        assert len(constraints) >= 1
        assert all(c.source == "template" for c in constraints)

    def test_no_tvars_returns_empty(self) -> None:
        constraints = generate_structural_constraints([])
        assert constraints == []

    def test_single_tvar_no_match(self) -> None:
        constraints = generate_structural_constraints([_make_tvar("temperature")])
        assert constraints == []

    def test_unrelated_tvars_no_match(self) -> None:
        tvars = [_make_tvar("foo"), _make_tvar("bar")]
        constraints = generate_structural_constraints(tvars)
        assert constraints == []

    def test_multiple_templates_match(self) -> None:
        tvars = [_make_tvar("temperature"), _make_tvar("top_p"), _make_tvar("model")]
        constraints = generate_structural_constraints(tvars)
        # Should match temp+top_p and model+temperature
        assert len(constraints) >= 2

    def test_llm_enrichment_adds_constraints(self) -> None:
        import json

        llm = MagicMock()
        llm.complete.return_value = json.dumps(
            [
                {
                    "description": "LLM constraint",
                    "constraint_code": "require(x.gte(0))",
                    "reasoning": "LLM said so",
                }
            ]
        )
        tvars = [_make_tvar("x"), _make_tvar("y")]
        constraints = generate_structural_constraints(
            tvars, llm=llm, source_code="def f(): pass"
        )
        llm_constraints = [c for c in constraints if c.source == "llm"]
        assert len(llm_constraints) == 1
        assert llm_constraints[0].description == "LLM constraint"

    def test_llm_budget_exhausted_returns_templates_only(self) -> None:
        llm = MagicMock()
        llm.complete.side_effect = BudgetExhausted(0.10)
        tvars = [_make_tvar("temperature"), _make_tvar("top_p")]
        constraints = generate_structural_constraints(tvars, llm=llm)
        # Should still return template matches
        assert len(constraints) >= 1
        assert all(c.source == "template" for c in constraints)

    def test_llm_invalid_json_returns_templates_only(self) -> None:
        llm = MagicMock()
        llm.complete.return_value = "not valid json"
        tvars = [_make_tvar("temperature"), _make_tvar("top_p")]
        constraints = generate_structural_constraints(tvars, llm=llm)
        assert len(constraints) >= 1
        assert all(c.source == "template" for c in constraints)

    def test_llm_not_called_with_single_tvar(self) -> None:
        llm = MagicMock()
        tvars = [_make_tvar("temperature")]
        generate_structural_constraints(tvars, llm=llm)
        llm.complete.assert_not_called()

    def test_llm_markdown_code_fence(self) -> None:
        import json

        llm = MagicMock()
        llm.complete.return_value = (
            "```json\n"
            + json.dumps(
                [
                    {
                        "description": "fenced",
                        "constraint_code": "x > 0",
                        "reasoning": "r",
                    }
                ]
            )
            + "\n```"
        )
        tvars = [_make_tvar("a"), _make_tvar("b")]
        constraints = generate_structural_constraints(tvars, llm=llm)
        llm_constraints = [c for c in constraints if c.source == "llm"]
        assert len(llm_constraints) == 1
        assert llm_constraints[0].description == "fenced"
