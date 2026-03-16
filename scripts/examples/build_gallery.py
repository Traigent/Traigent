#!/usr/bin/env python3
"""Generate the Examples gallery pages from catalog metadata.

This script reads ``examples/catalog.yaml`` and renders the landing page
``examples/gallery/index.html`` plus the generated core example detail pages.
All example links, run commands, and dataset references flow from the catalog
so the site stays consistent with the directory structure.
"""

from __future__ import annotations

import argparse
import html
import re
import sys
import textwrap
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml

# Ensure sibling modules are importable regardless of cwd.
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from example_detail_pages import _RAW_FILE_RE
from example_detail_pages import _sort_key as catalog_sort_key  # noqa: E402
from example_detail_pages import generate_detail_pages

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLES_DIR = REPO_ROOT / "examples"
CATALOG_PATH = REPO_ROOT / "examples" / "catalog.yaml"
OUTPUT_PATH = REPO_ROOT / "examples" / "gallery" / "index.html"

# Order and labels used for navigation
CATEGORY_CONFIG = {
    "core": {"label": "Core Examples", "description": "Hands-on tutorials that introduce key Traigent concepts."},
    "advanced": {"label": "Advanced Playbooks", "description": "Scenario-focused guides, deep dives, and experimental workflows."},
    "integrations": {"label": "Integrations", "description": "Examples that wire Traigent into CI/CD, monitoring, or partner tooling."},
}

TVL_SECTION_HTML = textwrap.dedent(
    """
    <section id="tvl" class="category-section coming-soon" aria-labelledby="tvl-heading">
        <div class="section-header">
            <h2 id="tvl-heading">Traigent Value Library (Coming Soon)</h2>
            <p>The Traigent Value Library (TVL) adds a typed configuration language for tuning pipelines, with explicit constraints, objectives, and promotion gates. Here's what to expect when it lands.</p>
        </div>
        <div class="tvl-grid">
            <article class="tvl-card">
                <header>
                    <div class="coming-soon-badge">Spec Highlights</div>
                    <h3 class="example-title">Structured for Safe Launches</h3>
                </header>
                <ul>
                    <li>Typed TVAR domains bound to environment snapshots and workloads.</li>
                    <li>Structural constraints expressed in DNF and compiled to SAT/SMT solvers.</li>
                    <li>Objectives with quality/latency bands and chance constraints.</li>
                    <li>Epsilon-Pareto promotion gate with Benjamini–Hochberg control.</li>
                </ul>
            </article>
            <article class="tvl-card">
                <header>
                    <div class="coming-soon-badge">Example Specs</div>
                    <h3 class="example-title">Preview Scenarios</h3>
                </header>
                <p class="example-description">TVL ships with curated specs in <code>spec/examples/</code>:</p>
                <ul>
                    <li><strong>rag-support-bot.tvl.yml</strong> — NSGA-II exploration with latency guardrails.</li>
                    <li><strong>text-to-sql.yml</strong> — semantic correctness bands for SQL generation.</li>
                    <li><strong>tool-use.yml</strong> — deterministic tool orchestration with safety gates.</li>
                </ul>
                <p class="example-description">Validation phase configs demonstrate staged rollout policies.</p>
            </article>
            <article class="tvl-card">
                <header>
                    <div class="coming-soon-badge">Spec Excerpt</div>
                    <h3 class="example-title">RAG Support Bot</h3>
                </header>
                <pre class="tvl-snippet"><code>tvl:
  module: corp.support.rag_bot
tvars:
  - name: model
    type: enum[str]
    domain: ["gpt-4o-mini", "gpt-4o", "llama3.1"]
  - name: temperature
    type: float
    domain:
      range: [0.0, 1.0]
      resolution: 0.05
constraints:
  structural:
    - when: zero_shot = true
      then: retriever.k = 0
objectives:
  - { name: quality, direction: maximize }
  - { name: latency_p95_ms, direction: minimize }
promotion_policy:
  dominance: epsilon_pareto
  alpha: 0.05
  min_effect:
    quality: 0.5
    latency_p95_ms: 50</code></pre>
            </article>
            <article class="tvl-card">
                <header>
                    <div class="coming-soon-badge">Tooling Roadmap</div>
                    <h3 class="example-title">CLI Surface</h3>
                </header>
                <p class="example-description">Lightweight tooling keeps specs auditable from CI:</p>
                <pre class="tvl-snippet"><code>tvl-parse spec/examples/rag-support-bot.tvl.yml
tvl-lint  spec/examples/rag-support-bot.tvl.yml
tvl-validate spec/examples/rag-support-bot.tvl.yml
tvl-config-validate
  spec/examples/rag-support-bot.tvl.yml
  spec/configurations/rag-support-bot.config.yml
tvl-measure-validate
  spec/examples/rag-support-bot.tvl.yml
  spec/configurations/rag-support-bot.config.yml
  spec/measurements/rag-support-bot.measure.yml</code></pre>
                <p class="example-description">Optional extras: <code>z3-solver</code> for SMT checks, <code>mkdocs-material</code> for docs builds.</p>
            </article>
        </div>
        <div class="tvl-footer">
            <p>TVL is under active development. Watch the release notes for availability updates.</p>
        </div>
    </section>
    """
).strip()


def load_catalog(path: Path) -> list[dict[str, Any]]:
    """Parse catalog entries."""
    if not path.exists():
        raise FileNotFoundError(f"Catalog file not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or []
    if not isinstance(data, list):
        raise ValueError("Catalog must be a list of entries")
    entries: list[dict[str, Any]] = []
    for index, raw in enumerate(data):
        if not isinstance(raw, dict):
            raise ValueError("Catalog entries must be dictionaries")
        entry = raw.copy()
        entry["_catalog_index"] = index
        entry.setdefault("slug", "")
        entry.setdefault("category", "core")
        entry.setdefault("title", entry["slug"].replace("-", " ").title())
        entry.setdefault("summary", "")
        entry.setdefault("run", None)
        entry.setdefault("docs", None)
        entry.setdefault("datasets", [])
        entry.setdefault("difficulty", None)
        entry.setdefault("est_time", None)
        entry.setdefault("tags", [])
        entries.append(entry)
    return entries


def format_run_command(run_path: str | None) -> str:
    if not run_path:
        return ""
    cmd = f"python examples/{run_path}"
    if not run_path.startswith("integrations/"):
        # Highlight mock flag for quickstarts
        cmd = f"TRAIGENT_MOCK_LLM=true {cmd}"
    return cmd


def render_card(entry: dict[str, Any]) -> str:
    title = html.escape(entry["title"])
    summary = html.escape(entry.get("summary") or "")
    slug = entry["slug"]
    docs_ref = entry.get("docs")
    run_cmd = format_run_command(entry.get("run"))
    datasets = entry.get("datasets") or []
    difficulty = entry.get("difficulty")
    est_time = entry.get("est_time")
    tags = entry.get("tags") or []

    doc_sections = entry.get("docs_sections") or []

    primary_link = None
    if docs_ref and not _RAW_FILE_RE.search(docs_ref):
        primary_link = docs_ref if docs_ref.startswith("http") else f"{docs_ref}"
    elif entry.get("run"):
        # Prefer an example index page when it exists; otherwise link to the
        # runnable source file so the gallery never emits a dead detail link.
        parts = entry["run"].split("/")
        if len(parts) >= 3:
            index_path = EXAMPLES_DIR.joinpath(*parts[:2], "index.html")
            if index_path.exists():
                # e.g., core/rag-optimization/run.py -> ../core/rag-optimization/index.html
                primary_link = "/".join([".."] + parts[:2] + ["index.html"])
            else:
                primary_link = "/".join([".."] + parts)
    button_html = ""
    if primary_link:
        button_html = (
            f'<a class="example-link" href="{html.escape(primary_link)}" '
            f'aria-label="Open primary documentation for {title}">View Example</a>'
        )

    difficulty_html = (
        f'<span class="difficulty-badge difficulty-{html.escape(difficulty.lower())}">'
        f'{html.escape(difficulty.title())}</span>'
        if difficulty
        else ""
    )
    time_html = (
        f'<span class="time-badge" aria-label="Estimated time">{html.escape(str(est_time))}</span>'
        if est_time
        else ""
    )
    tags_html = ""
    if tags:
        chips = "".join(
            f'<span class="tag-chip">{html.escape(tag)}</span>' for tag in tags
        )
        tags_html = f'<div class="tag-row" aria-label="Tags">{chips}</div>'

    datasets_html = ""
    if datasets:
        dataset_items = "".join(
            f"<li><code>{html.escape(ds)}</code></li>" for ds in datasets
        )
        datasets_html = textwrap.dedent(
            f"""
            <div class="dataset-block" aria-label="Datasets">
                <span class="dataset-heading">Datasets</span>
                <ul>{dataset_items}</ul>
            </div>
            """
        ).strip()

    run_html = (
        f'<code class="run-command">{html.escape(run_cmd)}</code>'
        if run_cmd
        else '<span class="run-command disabled" aria-disabled="true">No direct run command</span>'
    )

    docs_sections_html = ""
    if doc_sections:
        items = []
        for ref in doc_sections:
            if isinstance(ref, dict):
                label = html.escape(ref.get("label") or "Documentation")
                raw_href = ref.get("href") or "#"
            else:
                label = html.escape(str(ref))
                raw_href = str(ref)
            if _RAW_FILE_RE.search(raw_href):
                items.append(f'<li><span class="repo-ref">{label}</span></li>')
            else:
                items.append(f'<li><a href="{html.escape(raw_href)}">{label}</a></li>')
        docs_sections_html = textwrap.dedent(
            f"""
            <div class="docs-block" aria-label="Related docs">
                <span class="docs-heading">Documentation</span>
                <ul>
                    {' '.join(items)}
                </ul>
            </div>
            """
        ).strip()

    return textwrap.dedent(
        f"""
        <article class="example-card" id="{html.escape(slug)}">
            <header>
                <h3 class="example-title">{title}</h3>
                <p class="example-description">{summary}</p>
            </header>
            <div class="example-meta">
                {difficulty_html}
                {time_html}
            </div>
            {tags_html}
            {datasets_html}
            {docs_sections_html}
            <div class="run-command-row">
                <span class="run-label">Run:</span>{run_html}
            </div>
            {button_html}
        </article>
        """
    ).strip()


def build_html(entries: list[dict[str, Any]]) -> str:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for entry in entries:
        grouped[entry["category"]].append(entry)

    # Ensure deterministic ordering while respecting catalog sequence or explicit order overrides.
    def sort_key(entry: dict[str, Any]) -> tuple[int, Any]:
        order_raw = entry.get("order")
        if order_raw is None:
            order_raw = entry.get("position")
        if order_raw is not None:
            try:
                # First element flags that an explicit order was provided.
                return (0, float(order_raw))
            except (TypeError, ValueError):
                # Fall back to original sequence if the override is unusable.
                pass
        # Second element ensures stable order based on catalog appearance.
        return (1, entry.get("_catalog_index", 0))

    for group in grouped.values():
        group.sort(key=sort_key)

    sections_html = []
    for category_key in CATEGORY_CONFIG:
        if category_key not in grouped:
            continue
        cfg = CATEGORY_CONFIG[category_key]
        cards_html = "\n".join(render_card(e) for e in grouped[category_key])
        sections_html.append(
            textwrap.dedent(
                f"""
                <section id="{category_key}" class="category-section" aria-labelledby="{category_key}-heading">
                    <div class="section-header">
                        <h2 id="{category_key}-heading">{html.escape(cfg['label'])}</h2>
                        <p>{html.escape(cfg['description'])}</p>
                    </div>
                    <div class="examples-grid">
                        {cards_html}
                    </div>
                </section>
                """
            ).strip()
        )

    nav_items = [
        f'<a href="#{key}" class="nav-link">{html.escape(cfg["label"])}</a>'
        for key, cfg in CATEGORY_CONFIG.items()
        if key in grouped
    ]
    nav_items.append('<a href="#tvl" class="nav-link">TVL (Coming Soon)</a>')
    nav_html = "\n".join(nav_items)

    return textwrap.dedent(
        f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Traigent Examples Gallery</title>
            <link rel="stylesheet" href="sections/shared-styles.css">
            <script defer src="sections/theme.js"></script>
            <link href="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/themes/prism-vsc-dark-plus.min.css" rel="stylesheet" />
            <style>
                .page-container {{ max-width: 1180px; margin: 0 auto; padding: 32px 24px 80px; }}
                .hero {{ text-align: center; margin-bottom: 48px; }}
                .hero h1 {{ font-size: 2.75rem; margin-bottom: 16px; }}
                .hero p {{ font-size: 1.1rem; color: var(--text-secondary); margin-bottom: 24px; }}
                .category-nav {{ display: flex; justify-content: center; gap: 16px; flex-wrap: wrap; margin-bottom: 48px; }}
                .category-nav .nav-link {{ padding: 8px 16px; border-radius: 999px; border: 1px solid var(--border-color); text-decoration: none; color: var(--text-secondary); }}
                .category-nav .nav-link:hover, .category-nav .nav-link:focus-visible {{ border-color: var(--primary); color: var(--text-primary); outline: none; }}
                .category-section {{ margin-bottom: 72px; }}
                .section-header h2 {{ font-size: 2rem; margin-bottom: 12px; }}
                .section-header p {{ color: var(--text-secondary); margin-bottom: 24px; }}
                .examples-grid {{ display: grid; gap: 20px; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); }}
                .example-card {{ background: var(--card-bg); border: 1px solid var(--card-border); border-radius: 12px; padding: 20px; display: flex; flex-direction: column; gap: 16px; }}
                .example-card:hover {{ border-color: var(--primary); box-shadow: 0 4px 16px rgba(0,0,0,0.12); transform: translateY(-2px); }}
                .example-title {{ font-size: 1.2rem; margin: 0; }}
                .example-description {{ color: var(--text-secondary); margin: 0; flex-grow: 1; }}
                .example-meta, .run-command-row {{ display: flex; align-items: center; gap: 12px; flex-wrap: wrap; }}
                .run-label {{ font-weight: 600; color: var(--text-secondary); }}
                .run-command {{ background: var(--code-bg); border-radius: 6px; padding: 6px 8px; font-size: 0.95rem; }}
                .run-command.disabled {{ opacity: 0.6; font-style: italic; }}
                .example-link {{ align-self: flex-start; padding: 8px 14px; border-radius: 6px; background: var(--primary); color: #fff; text-decoration: none; font-weight: 500; }}
                .example-link:hover {{ background: var(--primary-hover); }}
                .difficulty-badge {{ padding: 4px 10px; border-radius: 999px; font-size: 0.75rem; font-weight: 600; }}
                .difficulty-beginner {{ background: rgba(34, 197, 94, 0.15); color: #22c55e; }}
                .difficulty-intermediate {{ background: rgba(251, 191, 36, 0.15); color: #fbbf24; }}
                .difficulty-advanced {{ background: rgba(239, 68, 68, 0.15); color: #ef4444; }}
                .time-badge {{ background: rgba(59,130,246,0.15); color: var(--primary); padding: 4px 10px; border-radius: 999px; font-size: 0.75rem; font-weight: 600; }}
                .tag-row {{ display: flex; gap: 8px; flex-wrap: wrap; }}
                .tag-chip {{ background: var(--chip-bg); color: var(--text-secondary); padding: 4px 8px; border-radius: 6px; font-size: 0.75rem; }}
                .dataset-block ul {{ margin: 8px 0 0 16px; padding: 0; }}
                .dataset-heading {{ font-weight: 600; color: var(--text-secondary); }}
                .docs-block ul {{ margin: 8px 0 0 16px; padding: 0; }}
                .docs-block ul li {{ margin-bottom: 4px; }}
                .docs-block a {{ color: var(--primary); text-decoration: none; }}
                .docs-block a:hover {{ text-decoration: underline; }}
                .docs-heading {{ font-weight: 600; color: var(--text-secondary); }}
                .repo-ref {{ color: var(--text-secondary); font-style: italic; }}
.coming-soon .example-card {{ border-style: dashed; }}
.coming-soon-badge {{ display: inline-block; background: rgba(96,165,250,0.15); color: #60a5fa; padding: 4px 10px; border-radius: 999px; font-size: 0.75rem; font-weight: 600; }}
.tvl-grid {{ display: grid; gap: 20px; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); margin-top: 12px; }}
.tvl-card {{ background: var(--card-bg); border: 1px dashed var(--card-border); border-radius: 12px; padding: 20px; display: flex; flex-direction: column; gap: 12px; }}
.tvl-card ul {{ margin: 0 0 0 18px; padding: 0; }}
.tvl-card pre {{ background: var(--code-bg); border: 1px solid var(--border-color); border-radius: 8px; padding: 12px; font-size: 0.85rem; overflow-x: auto; }}
.tvl-snippet code {{ white-space: pre; }}
.tvl-footer {{ margin-top: 24px; color: var(--text-secondary); font-size: 0.95rem; }}
                @media (max-width: 600px) {{
                    .hero h1 {{ font-size: 2.1rem; }}
                }}
            </style>
        </head>
        <body>
            <main class="page-container">
                <section class="hero">
                    <h1>Traigent Examples Gallery</h1>
                    <p>Explore runnable examples, advanced playbooks, and integration guides. Every entry links directly to the source and includes the CLI command to get started.</p>
                    <nav class="category-nav" aria-label="Example categories">
                        {nav_html}
                    </nav>
                </section>
                {"".join(sections_html)}
                {TVL_SECTION_HTML}
            </main>
        </body>
        </html>
        """
    ).strip()


def validate_catalog(entries: list[dict[str, Any]]) -> list[str]:
    """Check catalog entries for missing targets. Returns a list of warnings/errors."""
    issues: list[str] = []
    examples_dir = REPO_ROOT / "examples"
    for entry in entries:
        slug = entry.get("slug") or ""
        category = entry.get("category") or "core"
        # Allow an explicit ``dir`` override for entries whose slug
        # differs from the on-disk directory name.
        dir_name = entry.get("dir") or slug
        # Check that the example directory exists
        example_dir = examples_dir / category / dir_name
        if not example_dir.is_dir():
            issues.append(f"ERROR: [{slug}] directory not found: {example_dir.relative_to(REPO_ROOT)}")
        # Check run target
        run_path = entry.get("run")
        if run_path and not (examples_dir / run_path).exists():
            issues.append(f"ERROR: [{slug}] run target missing: examples/{run_path}")
        # Check datasets
        for ds in entry.get("datasets") or []:
            if not (examples_dir / ds).exists():
                issues.append(f"WARN:  [{slug}] dataset not found: examples/{ds}")
        # Flag raw .md/.yml doc hrefs (informational)
        for ref in entry.get("docs_sections") or []:
            href = ref.get("href", "") if isinstance(ref, dict) else str(ref)
            if _RAW_FILE_RE.search(href):
                label = ref.get("label", href) if isinstance(ref, dict) else href
                issues.append(f"INFO:  [{slug}] docs_sections href is a raw repo file (rendered as label): {label}")
    return issues


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate examples gallery page.")
    parser.add_argument(
        "--catalog",
        type=Path,
        default=CATALOG_PATH,
        help="Path to catalog YAML file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_PATH,
        help="Destination HTML path.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        default=False,
        help="Fail on any validation errors (not just warnings).",
    )
    args = parser.parse_args()

    entries = load_catalog(args.catalog)

    # --- Build-time validation ---
    issues = validate_catalog(entries)
    for issue in issues:
        print(issue)
    errors = [i for i in issues if i.startswith("ERROR:")]
    if errors and args.strict:
        raise SystemExit(f"Validation failed with {len(errors)} error(s). Use --strict=false to continue.")
    elif errors:
        print(f"  ({len(errors)} error(s) found — pass --strict to abort on errors)")

    html_content = build_html(entries)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(f"{html_content}\n", encoding="utf-8")
    detail_pages = generate_detail_pages(REPO_ROOT, entries)
    print(f"Wrote gallery to {args.output}")
    print(f"Generated {len(detail_pages)} detail pages under examples/core/")


if __name__ == "__main__":
    main()
