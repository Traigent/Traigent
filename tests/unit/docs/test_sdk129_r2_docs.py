from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]


def _t(rel: str) -> str:
    return re.sub(r"\s+", " ", (REPO / rel).read_text(encoding="utf-8"))


def test_empty_result_weighted_score_fallback_documented() -> None:
    from traigent.api.types import OptimizationResult

    src = OptimizationResult.calculate_weighted_scores
    assert "0.0" in (REPO / "traigent/api/types.py").read_text()
    t = _t("docs/api-reference/complete-function-specification.md")
    assert "returns `best_weighted_score: None`" not in t
    assert (
        "`best_weighted_score: 0.0`" in t and "`best_weighted_config` is present" in t
    )
    assert src is not None


def test_best_score_score_objective_exception_documented() -> None:
    t = _t("docs/api-reference/complete-function-specification.md")
    assert "`results.best_score` remains the winner's primary-objective value" not in t
    assert "named `score`" in t and "`best_score` is the weighted aggregate" in t


def test_session_privacy_distinguishes_entry_points() -> None:
    for rel in ("docs/hybrid-mode-client-guide.md", "docs/hybrid-mode-api-contract.md"):
        t = _t(rel)
        assert "`run_description`" in t and "`create_hybrid_session()`" in t, rel


def test_output_null_guidance() -> None:
    t = _t("docs/hybrid-mode-api-contract.md")
    assert "a non-null `output` is sent and takes precedence" not in t
    assert "omit the `output` key entirely" in t


def test_cost_lower_bound_includes_estimates() -> None:
    t = _t("docs/hybrid-mode-api-contract.md")
    assert "equal-share allocated estimates" in t
    assert "no matching output stays unmeasured" in t
