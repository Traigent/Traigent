"""The plan-limit messages pinned here link ONE pricing URL.

Pinned: the licence error, the session-quota error and remedy block, and the
billing limit suggestions, plus a scan that the URL literal is defined once.
Not pinned: every other error the SDK can raise about plans or quotas.

The URL lives in one constant (``traigent.config.backend_config.PRICING_URL``).
The marketing site's sitemap and canonical tag use ``/pricing`` (no trailing
slash). The live site answers that form with a redirect to ``/pricing/``; both
forms work.
"""

from __future__ import annotations

import ast
import asyncio
from pathlib import Path

import pytest

from traigent.cloud.billing import BillingManager, UsageTracker
from traigent.cloud.client import SessionQuotaRefusedError
from traigent.config.backend_config import PRICING_URL
from traigent.core.backend_session_manager import _format_untracked_warning_block
from traigent.core.license import FeatureRequiresLicenseError
from traigent.core.session_types import (
    SessionCreationFailureClassification,
    SessionCreationFailureDetail,
    SessionCreationFailureReason,
    SessionCreationResult,
)
from traigent.utils.exceptions import ConfigurationError

PACKAGE_ROOT = Path(__file__).resolve().parents[3] / "traigent"


def test_constant_is_canonical_pricing_url() -> None:
    assert PRICING_URL == "https://traigent.ai/pricing"


def test_license_error_links_pricing() -> None:
    assert PRICING_URL in str(FeatureRequiresLicenseError("parallel_execution"))
    assert PRICING_URL in str(FeatureRequiresLicenseError("x", "Enterprise"))


def test_session_quota_refused_error_links_pricing() -> None:
    assert PRICING_URL in str(SessionQuotaRefusedError(429, "quota_exceeded"))


def test_quota_remedy_block_links_pricing() -> None:
    detail = SessionCreationFailureDetail.from_http_response(
        429, '{"error_code": "quota_exceeded"}'
    )
    result = SessionCreationResult.fallback(
        session_id="local_x",
        reason=SessionCreationFailureReason.SESSION_FAILED,
        detail="d",
        failure_response=detail,
        typed_legacy_session_create_400=True,
    )
    block = _format_untracked_warning_block(
        result, SessionCreationFailureClassification.QUOTA_EXCEEDED, aborting=True
    )
    assert PRICING_URL in block


@pytest.mark.parametrize(
    "trials, dataset, prior_records",
    [(21, 1, 0), (1, 101, 0), (1, 1, 250)],
    ids=["trials", "dataset", "monthly-credits"],
)
def test_billing_limit_suggestions_link_pricing(
    trials: int, dataset: int, prior_records: int, tmp_path: Path
) -> None:
    async def run() -> dict:
        tracker = UsageTracker(storage_path=str(tmp_path / "usage.json"))
        manager = BillingManager(tracker)
        manager.current_plan = "free"
        for i in range(prior_records):
            await tracker.record_optimization(f"f{i}", 20, 100, 60.0, "free")
        return await manager.check_usage_limits(trials, dataset)

    result = asyncio.run(run())
    assert result["allowed"] is False
    assert PRICING_URL in result["suggested_action"]


def test_local_plan_upgrade_refusal_links_pricing(tmp_path: Path) -> None:
    manager = BillingManager(UsageTracker(storage_path=str(tmp_path / "usage.json")))
    manager.current_plan = "free"
    with pytest.raises(ConfigurationError) as exc:
        manager.upgrade_plan("enterprise")
    assert PRICING_URL in str(exc.value)


def _string_constants_outside_docstrings(tree: ast.AST):
    docstrings: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(
            node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
        ):
            body = node.body
            if (
                body
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)
            ):
                docstrings.add(id(body[0].value))
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and id(node) not in docstrings
        ):
            yield node


def test_pricing_url_literal_is_defined_once() -> None:
    """No module re-spells the pricing URL in code; they import the constant.

    Only non-docstring string constants count, so prose in comments and
    docstrings never trips this.
    """
    hits = [
        f"{p.relative_to(PACKAGE_ROOT.parent)}:{node.lineno}"
        for p in sorted(PACKAGE_ROOT.rglob("*.py"))
        for node in _string_constants_outside_docstrings(
            ast.parse(p.read_text(encoding="utf-8"), filename=str(p))
        )
        if "traigent.ai/pricing" in node.value
    ]
    assert len(hits) == 1, hits
    assert hits[0].startswith("traigent/config/backend_config.py:"), hits
