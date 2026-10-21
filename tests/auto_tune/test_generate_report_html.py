"""Regression tests for scripts/auto_tune/generate_report.py HTML rendering.

The HTML template carries literal CSS braces; rendering it with str.format used
to raise KeyError before any report was written.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _load_generate_report():
    spec = importlib.util.spec_from_file_location(
        "auto_tune_generate_report_html",
        ROOT / "scripts" / "auto_tune" / "generate_report.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_html_report_renders_css_braces_and_escapes_values():
    gen = _load_generate_report()
    markdown = "# Traigent Auto-Tuning Report\n\n**Note:** <script>x</script> & done\n- item one"

    html = gen.generate_html_report(markdown)

    assert "body {" in html  # literal CSS braces survive the template
    assert ".metric-good { color: #28a745; font-weight: bold; }" in html
    assert "<h1>Traigent Auto-Tuning Report" in html
    assert "<li>item one</li>" in html
    assert "<script>x</script>" not in html  # result-derived text is escaped
    assert "&lt;script&gt;x&lt;/script&gt; &amp; done" in html
