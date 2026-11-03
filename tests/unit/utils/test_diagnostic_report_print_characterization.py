"""Golden characterization for DiagnosticReport.print_report."""

from __future__ import annotations

import sys

from traigent.utils import diagnostics
from traigent.utils.diagnostics import DiagnosticReport

BAR = "=" * 60
PY = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"


def _lines(report, monkeypatch):
    out: list[str] = []
    monkeypatch.setattr(diagnostics, "_safe_print", lambda s="": out.append(s))
    report.print_report()
    return out


def test_empty_report(monkeypatch):
    r = DiagnosticReport()
    assert _lines(r, monkeypatch) == [
        "\n" + BAR,
        "🔍 Traigent Diagnostic Report",
        BAR,
        "\n📊 System Info:",
        f"  Python: {PY}",
        f"  Platform: {r.platform}",
        "\n" + BAR,
        "✅ No critical issues found!",
        BAR + "\n",
    ]


def test_full_report_section_order(monkeypatch):
    r = DiagnosticReport()
    r.add_issue("c1", "broken", "do x")
    r.add_issue("c2", "also broken")
    r.add_warning("w", "careful")
    r.add_success("s", "fine")
    r.add_recommendation("try y")
    out = _lines(r, monkeypatch)
    assert out[6:] == [
        "\n✅ Successes (1):",
        "  [s] fine",
        "\n⚠️  Warnings (1):",
        "  [w] careful",
        "\n❌ Issues (2):",
        "  [c1] broken",
        "     💡 Fix: do x",
        "  [c2] also broken",
        "\n💡 Recommendations:",
        "  • try y",
        "\n" + BAR,
        "❌ Found 2 issue(s) that need attention.",
        BAR + "\n",
    ]


def test_warnings_only_still_reports_no_critical_issues(monkeypatch):
    r = DiagnosticReport()
    r.add_warning("w", "careful")
    out = _lines(r, monkeypatch)
    assert "✅ No critical issues found!" in out
    assert "\n⚠️  Warnings (1):" in out
