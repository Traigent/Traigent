"""Generate richer core example detail pages from README content."""

from __future__ import annotations

import html
import os
import re
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
CORE_ROOT = REPO_ROOT / "examples" / "core"
EXAMPLES_ROOT = REPO_ROOT / "examples"
GALLERY_ROOT = EXAMPLES_ROOT / "gallery"

TRAILING_INDEXABLE_LINK_RE = re.compile(r"^(?![a-z]+:|#|/)(.+/)$", re.IGNORECASE)
_RAW_FILE_RE = re.compile(r"\.(?:md|yml|yaml)$", re.IGNORECASE)
LIST_ITEM_RE = re.compile(r"^([-*])\s+(.*)$")
ORDERED_ITEM_RE = re.compile(r"^(\d+)\.\s+(.*)$")

LEGACY_PAGE_OVERRIDES: dict[str, dict[str, Any]] = {
    "hello-world": {
        "title": "Hello World",
        "summary": (
            "A legacy starter page for local RAG optimization with retrieval toggles, "
            "deterministic mock runs, and a scenario matrix."
        ),
        "difficulty": "beginner",
        "est_time": "5 min",
        "tags": ["legacy", "RAG", "intro"],
        "run": "core/hello-world/run.py",
        "datasets": [
            "datasets/hello-world/evaluation_set.jsonl",
            "datasets/hello-world/context_documents.jsonl",
        ],
        "docs_sections": [
            {"label": "Simple Prompt First", "href": "../core/simple-prompt/index.html"},
            {"label": "RAG Optimization", "href": "../core/rag-optimization/index.html"},
            {"label": "Examples Guide", "href": "../../gallery/index.html"},
        ],
        "legacy": True,
    }
}


@dataclass(slots=True)
class ParsedReadme:
    title: str
    intro: str
    sections: list[tuple[str, str]]


def _sort_key(entry: dict[str, Any]) -> tuple[int, Any]:
    order_raw = entry.get("order")
    if order_raw is None:
        order_raw = entry.get("position")
    if order_raw is not None:
        try:
            return (0, float(order_raw))
        except (TypeError, ValueError):
            pass
    return (1, entry.get("_catalog_index", 0))


def parse_readme(readme_path: Path) -> ParsedReadme:
    lines = readme_path.read_text(encoding="utf-8").splitlines()
    title = readme_path.parent.name.replace("-", " ").title()
    intro_lines: list[str] = []
    sections: list[tuple[str, str]] = []
    current_heading: str | None = None
    current_lines: list[str] = []
    saw_title = False

    for line in lines:
        if line.startswith("# ") and not saw_title:
            title = line[2:].strip()
            saw_title = True
            continue
        if line.startswith("## "):
            if current_heading is None:
                intro = "\n".join(intro_lines).strip()
            else:
                sections.append((current_heading, "\n".join(current_lines).strip()))
            current_heading = line[3:].strip()
            current_lines = []
            continue

        if current_heading is None:
            intro_lines.append(line)
        else:
            current_lines.append(line)

    if current_heading is None:
        intro = "\n".join(intro_lines).strip()
    else:
        intro = "\n".join(intro_lines).strip()
        sections.append((current_heading, "\n".join(current_lines).strip()))

    return ParsedReadme(title=title, intro=intro, sections=sections)


def _normalize_readme_href(href: str) -> str:
    href = href.strip()
    if not href:
        return "#"
    match = TRAILING_INDEXABLE_LINK_RE.match(href)
    if match:
        return f"{match.group(1)}index.html"
    return href


def _format_inline(text: str) -> str:
    placeholders: dict[str, str] = {}
    counter = 0

    def stash(fragment: str) -> str:
        nonlocal counter
        token = f"__HTML_TOKEN_{counter}__"
        placeholders[token] = fragment
        counter += 1
        return token

    def link_sub(match: re.Match[str]) -> str:
        label = html.escape(match.group(1).strip())
        raw_href = match.group(2).strip()
        if _RAW_FILE_RE.search(raw_href):
            return stash(f'<span class="repo-ref">{label}</span>')
        href = html.escape(_normalize_readme_href(raw_href))
        return stash(f'<a href="{href}">{label}</a>')

    def code_sub(match: re.Match[str]) -> str:
        return stash(f"<code>{html.escape(match.group(1))}</code>")

    text = re.sub(r"`([^`]+)`", code_sub, text)
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", link_sub, text)
    text = html.escape(text)
    text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"\*(.+?)\*", r"<em>\1</em>", text)

    for token, fragment in placeholders.items():
        text = text.replace(token, fragment)
    return text


def _render_table(lines: list[str]) -> str:
    rows = []
    for line in lines:
        stripped = line.strip().strip("|")
        rows.append([cell.strip() for cell in stripped.split("|")])
    headers = rows[0]
    body = rows[2:]
    thead = "".join(f"<th>{_format_inline(cell)}</th>" for cell in headers)
    tbody = "".join(
        "<tr>" + "".join(f"<td>{_format_inline(cell)}</td>" for cell in row) + "</tr>"
        for row in body
    )
    return (
        '<div class="detail-table-wrap"><table class="detail-table"><thead><tr>'
        f"{thead}</tr></thead><tbody>{tbody}</tbody></table></div>"
    )


def _render_markdown(markdown_text: str) -> str:
    if not markdown_text.strip():
        return ""

    lines = markdown_text.strip().splitlines()
    blocks: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if not stripped:
            i += 1
            continue

        if stripped.startswith("```"):
            language = stripped[3:].strip()
            code_lines: list[str] = []
            i += 1
            while i < len(lines) and not lines[i].strip().startswith("```"):
                code_lines.append(lines[i])
                i += 1
            blocks.append(
                '<pre class="detail-code"><code'
                + (f' class="language-{html.escape(language)}"' if language else "")
                + f">{html.escape(chr(10).join(code_lines))}</code></pre>"
            )
            i += 1
            continue

        if stripped.startswith("|") and i + 1 < len(lines):
            separator = lines[i + 1].strip()
            if separator.startswith("|") and set(separator.replace("|", "").replace(":", "").replace("-", "").strip()) == set():
                table_lines = [lines[i], lines[i + 1]]
                i += 2
                while i < len(lines) and lines[i].strip().startswith("|"):
                    table_lines.append(lines[i])
                    i += 1
                blocks.append(_render_table(table_lines))
                continue

        if stripped.startswith("### "):
            blocks.append(f"<h3>{_format_inline(stripped[4:])}</h3>")
            i += 1
            continue

        if stripped.startswith(">"):
            quote_lines = []
            while i < len(lines) and lines[i].strip().startswith(">"):
                quote_lines.append(lines[i].strip()[1:].strip())
                i += 1
            blocks.append(
                f'<aside class="detail-callout">{_render_markdown(chr(10).join(quote_lines))}</aside>'
            )
            continue

        bullet_match = LIST_ITEM_RE.match(stripped)
        ordered_match = ORDERED_ITEM_RE.match(stripped)
        if bullet_match or ordered_match:
            is_ordered = ordered_match is not None
            items = []
            while i < len(lines):
                current = lines[i].strip()
                if not current:
                    break
                match = ORDERED_ITEM_RE.match(current) if is_ordered else LIST_ITEM_RE.match(current)
                if not match:
                    break
                items.append(f"<li>{_format_inline(match.group(2))}</li>")
                i += 1
            tag = "ol" if is_ordered else "ul"
            blocks.append(f"<{tag} class=\"detail-list\">{''.join(items)}</{tag}>")
            continue

        paragraph_lines = [line.strip()]
        i += 1
        while i < len(lines):
            current = lines[i]
            current_stripped = current.strip()
            if not current_stripped:
                break
            if current_stripped.startswith(("```", "### ", ">", "|")):
                break
            if LIST_ITEM_RE.match(current_stripped) or ORDERED_ITEM_RE.match(current_stripped):
                break
            paragraph_lines.append(current_stripped)
            i += 1
        blocks.append(f"<p>{_format_inline(' '.join(paragraph_lines))}</p>")

    return "\n".join(blocks)


def _strip_markdown(text: str) -> str:
    text = re.sub(r"```.*?```", " ", text, flags=re.DOTALL)
    text = re.sub(r"`([^`]+)`", r"\1", text)
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1", text)
    text = re.sub(r"[*_>#-]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _first_sentence(text: str) -> str:
    # Use only the first paragraph so list items don't leak into the summary.
    first_para = text.split("\n\n")[0]
    cleaned = _strip_markdown(first_para)
    if not cleaned:
        return ""
    sentence_match = re.match(r"(.+?[.!?])(?:\s|$)", cleaned)
    if sentence_match:
        return sentence_match.group(1).strip()
    return cleaned


def _extract_code_block(markdown_text: str) -> str | None:
    match = re.search(r"```(?:[\w+-]+)?\n(.*?)```", markdown_text, flags=re.DOTALL)
    if not match:
        return None
    return match.group(1).strip()


def _slugify_fragment(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    return text.strip("-") or "section"


def _extract_source_file(command_block: str | None, entry: dict[str, Any], example_dir: Path) -> str | None:
    if command_block:
        candidates = re.findall(r"(?:python|uv run python)\s+([^\s]+\.py)", command_block)
        for candidate in reversed(candidates):
            candidate_path = candidate.strip()
            if candidate_path.startswith("examples/core/"):
                return Path(candidate_path).name
            if (example_dir / candidate_path).exists():
                return candidate_path
    run_path = entry.get("run")
    if isinstance(run_path, str) and run_path.startswith(f"core/{example_dir.name}/"):
        return Path(run_path).name
    default = example_dir / "run.py"
    if default.exists():
        return default.name
    return None


def _rebase_gallery_href(href: str, output_dir: Path) -> str:
    if not href or href.startswith(("http://", "https://", "mailto:", "#")):
        return href or "#"
    absolute_target = (GALLERY_ROOT / href).resolve()
    return Path(os.path.relpath(absolute_target, output_dir)).as_posix()


def _rel_from_page(target: Path, output_dir: Path) -> str:
    return Path(os.path.relpath(target, output_dir)).as_posix()


def _collect_support_files(example_dir: Path, primary_source: str | None) -> list[str]:
    files: list[str] = []
    if primary_source:
        files.append(primary_source)
    for pattern in ("run*.py", "prompt*.txt", "matrix.html"):
        for path in sorted(example_dir.glob(pattern)):
            if path.name not in files:
                files.append(path.name)
    return files


def _build_fallback_entry(slug: str, parsed: ParsedReadme) -> dict[str, Any]:
    datasets_dir = EXAMPLES_ROOT / "datasets" / slug
    datasets = []
    if datasets_dir.exists():
        datasets = [
            f"datasets/{slug}/{path.name}" for path in sorted(datasets_dir.iterdir()) if path.is_file()
        ]
    entry = {
        "slug": slug,
        "category": "core",
        "title": parsed.title,
        "summary": _first_sentence(parsed.intro),
        "run": f"core/{slug}/run.py" if (CORE_ROOT / slug / "run.py").exists() else None,
        "difficulty": None,
        "est_time": None,
        "tags": [],
        "datasets": datasets,
        "docs_sections": [],
        "_catalog_index": 10_000,
    }
    entry.update(LEGACY_PAGE_OVERRIDES.get(slug, {}))
    return entry


def _build_breadcrumb(entry: dict[str, Any]) -> str:
    trail = [
        '<a href="../../gallery/index.html">Examples</a>',
        '<a href="../../gallery/index.html#core">Core</a>',
    ]
    if entry.get("legacy"):
        trail.append("<span>Legacy</span>")
    trail.append(f'<span aria-current="page">{html.escape(entry["title"])}</span>')
    return " <span class=\"breadcrumb-sep\">/</span> ".join(trail)


def _build_meta(entry: dict[str, Any]) -> str:
    chips = ['<span class="meta-chip meta-chip-category">Core example</span>']
    difficulty = entry.get("difficulty")
    if difficulty:
        chips.append(
            f'<span class="meta-chip meta-chip-difficulty">{html.escape(str(difficulty).title())}</span>'
        )
    est_time = entry.get("est_time")
    if est_time:
        chips.append(f'<span class="meta-chip">{html.escape(str(est_time))}</span>')
    if entry.get("legacy"):
        chips.append('<span class="meta-chip meta-chip-legacy">Legacy route</span>')
    for tag in entry.get("tags") or []:
        chips.append(f'<span class="meta-chip meta-chip-tag">{html.escape(str(tag))}</span>')
    return "".join(chips)


def _build_related_docs(entry: dict[str, Any], output_dir: Path) -> str:
    items = []
    for ref in entry.get("docs_sections") or []:
        if isinstance(ref, dict):
            label = html.escape(str(ref.get("label") or "Documentation"))
            raw_href = str(ref.get("href") or "#")
        else:
            label = html.escape(str(ref))
            raw_href = "#"
        if _RAW_FILE_RE.search(raw_href):
            # Repo-only file — render as a non-linked reference label
            items.append(f'<li><span class="repo-ref">{label}</span></li>')
        else:
            href = html.escape(_rebase_gallery_href(raw_href, output_dir))
            items.append(f'<li><a href="{href}">{label}</a></li>')
    if not items:
        items.append('<li><span class="repo-ref">Examples Guide</span></li>')
    return "".join(items)


def _build_dataset_list(entry: dict[str, Any]) -> str:
    items = []
    for dataset in entry.get("datasets") or []:
        dataset_path = str(dataset)
        href = dataset_path
        if dataset_path.startswith("datasets/"):
            href = f"../../{dataset_path}"
        items.append(
            f'<li><a href="{html.escape(href)}"><code>{html.escape(dataset_path)}</code></a></li>'
        )
    if not items:
        items.append("<li>No bundled dataset</li>")
    return "".join(items)


def _build_support_files(example_dir: Path, primary_source: str | None) -> str:
    items = []
    for filename in _collect_support_files(example_dir, primary_source):
        items.append(f'<li><a href="{html.escape(filename)}">{html.escape(filename)}</a></li>')
    return "".join(items)


def _build_nav(entry: dict[str, Any], tour_entries: list[dict[str, Any]]) -> str:
    slug = entry["slug"]
    slug_to_index = {item["slug"]: idx for idx, item in enumerate(tour_entries)}
    if slug not in slug_to_index:
        return textwrap.dedent(
            """
            <section class="tour-card">
                <h2>Example Tour</h2>
                <p>This page remains available for direct links, but the active tour starts from the gallery.</p>
                <a class="tour-link" href="../../gallery/index.html#core">Open Core Gallery</a>
            </section>
            """
        ).strip()

    position = slug_to_index[slug]
    prev_entry = tour_entries[position - 1] if position > 0 else None
    next_entry = tour_entries[position + 1] if position + 1 < len(tour_entries) else None
    prev_html = (
        f'<a class="tour-link secondary" href="../{prev_entry["slug"]}/index.html">Previous: {html.escape(prev_entry["title"])}</a>'
        if prev_entry
        else '<span class="tour-link secondary disabled">Start of tour</span>'
    )
    next_html = (
        f'<a class="tour-link" href="../{next_entry["slug"]}/index.html">Next: {html.escape(next_entry["title"])}</a>'
        if next_entry
        else '<span class="tour-link disabled">End of tour</span>'
    )
    return textwrap.dedent(
        f"""
        <section class="tour-card">
            <h2>Example Tour</h2>
            <p><strong>{position + 1}</strong> of <strong>{len(tour_entries)}</strong> in the guided core sequence.</p>
            <div class="tour-actions">
                {prev_html}
                {next_html}
            </div>
        </section>
        """
    ).strip()


def build_detail_page(
    entry: dict[str, Any],
    parsed: ParsedReadme,
    output_dir: Path,
    tour_entries: list[dict[str, Any]],
) -> str:
    _RUN_HEADINGS = {"quick start", "run", "running the example", "usage", "getting started"}
    quick_start_body = next(
        (body for heading, body in parsed.sections if heading.lower() in _RUN_HEADINGS), ""
    )
    quick_start_code = _extract_code_block(quick_start_body)
    primary_source = _extract_source_file(quick_start_code, entry, output_dir)

    intro_html = _render_markdown(parsed.intro)
    summary = entry.get("summary") or _first_sentence(parsed.intro)
    if not summary:
        summary = _first_sentence(next((body for _, body in parsed.sections), ""))

    rendered_sections = []
    for heading, body in parsed.sections:
        if not body.strip():
            continue
        rendered_sections.append(
            textwrap.dedent(
                f"""
                <section class="detail-section" id="{html.escape(entry["slug"])}-{_slugify_fragment(heading)}">
                    <header class="detail-section-header">
                        <h2>{html.escape(heading)}</h2>
                    </header>
                    <div class="detail-section-body">
                        {_render_markdown(body)}
                    </div>
                </section>
                """
            ).strip()
        )

    related_docs_html = _build_related_docs(entry, output_dir)
    source_viewer_html = ""
    if primary_source:
        source_viewer_html = textwrap.dedent(
            f"""
            <section class="detail-section detail-section-source">
                <header class="detail-section-header">
                    <h2>Optimization Target</h2>
                    <p>Preview the decorated function that Traigent tunes when this example runs.</p>
                </header>
                <div class="detail-section-body">
                    <div data-optimize-source="{html.escape(primary_source)}" data-optimize-height="360"></div>
                </div>
            </section>
            """
        ).strip()

    primary_button = (
        f'<a class="hero-button" href="{html.escape(primary_source)}">Open Source</a>'
        if primary_source
        else ""
    )
    slug_in_tour = any(e["slug"] == entry["slug"] for e in tour_entries)
    gallery_anchor = entry["slug"] if slug_in_tour and not entry.get("legacy") else "core"
    gallery_button = (
        f'<a class="hero-button secondary" href="../../gallery/index.html#{html.escape(gallery_anchor)}">Back to Gallery</a>'
    )
    docs_button = ""
    first_doc = next(iter(entry.get("docs_sections") or []), None)
    if isinstance(first_doc, dict):
        first_href = str(first_doc.get("href") or "#")
        if not _RAW_FILE_RE.search(first_href):
            docs_button = (
                f'<a class="hero-button ghost" href="{html.escape(_rebase_gallery_href(first_href, output_dir))}">'
                f'{html.escape(str(first_doc.get("label") or "Related guide"))}</a>'
            )

    quick_start_panel = (
        f'<pre class="hero-code"><code>{html.escape(quick_start_code)}</code></pre>'
        if quick_start_code
        else '<p class="hero-panel-empty">Check the source file for the runnable entrypoint.</p>'
    )

    return textwrap.dedent(
        f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>{html.escape(entry["title"])} - Traigent Examples</title>
            <link rel="stylesheet" href="../../gallery/sections/shared-styles.css">
            <link rel="stylesheet" href="../../gallery/sections/example-detail.css">
            <script defer src="../../gallery/sections/theme.js"></script>
        </head>
        <body>
            <main class="detail-shell">
                <nav class="detail-breadcrumb" aria-label="Breadcrumb">
                    {_build_breadcrumb(entry)}
                </nav>

                <section class="detail-hero">
                    <div class="hero-copy">
                        <div class="hero-kicker">Traigent example</div>
                        <h1>{html.escape(entry["title"])}</h1>
                        <p class="hero-summary">{html.escape(summary)}</p>
                        <div class="hero-meta">
                            {_build_meta(entry)}
                        </div>
                        <div class="hero-actions">
                            {primary_button}
                            {gallery_button}
                            {docs_button}
                        </div>
                    </div>
                    <aside class="hero-panel">
                        <div class="hero-panel-label">Quick Start</div>
                        {quick_start_panel}
                    </aside>
                </section>

                <div class="detail-layout">
                    <section class="detail-main">
                        {f'<section class="detail-section detail-section-intro"><div class="detail-section-body">{intro_html}</div></section>' if intro_html else ''}
                        {''.join(rendered_sections)}
                        {source_viewer_html}
                    </section>

                    <aside class="detail-rail">
                        <section class="rail-card">
                            <h2>Datasets</h2>
                            <ul class="rail-list">
                                {_build_dataset_list(entry)}
                            </ul>
                        </section>
                        <section class="rail-card">
                            <h2>Project Files</h2>
                            <ul class="rail-list">
                                {_build_support_files(output_dir, primary_source)}
                            </ul>
                        </section>
                        <section class="rail-card">
                            <h2>Related Docs</h2>
                            <ul class="rail-list">
                                {related_docs_html}
                            </ul>
                        </section>
                        {_build_nav(entry, tour_entries)}
                    </aside>
                </div>

                <footer class="detail-footer">
                    <p>Built from <code>README.md</code> and <code>examples/catalog.yaml</code> so the gallery stays in sync with runnable code.</p>
                </footer>
            </main>
            <script type="module" src="../../gallery/static/js/example-viewer.js"></script>
        </body>
        </html>
        """
    ).strip()


def generate_detail_pages(repo_root: Path, catalog_entries: list[dict[str, Any]]) -> list[Path]:
    entry_map = {entry["slug"]: entry for entry in catalog_entries if entry.get("category") == "core"}
    core_entries = sorted(
        [entry for entry in catalog_entries if entry.get("category") == "core"],
        key=_sort_key,
    )
    generated_paths: list[Path] = []
    detail_entries: list[dict[str, Any]] = []

    for example_dir in sorted((repo_root / "examples" / "core").iterdir()):
        if not example_dir.is_dir():
            continue
        readme_path = example_dir / "README.md"
        output_path = example_dir / "index.html"
        if not readme_path.exists():
            continue
        parsed = parse_readme(readme_path)
        entry = entry_map.get(example_dir.name)
        if entry is None:
            entry = _build_fallback_entry(example_dir.name, parsed)
        detail_entries.append(entry)

    detail_entries.sort(
        key=lambda entry: (
            0 if entry["slug"] in entry_map else 1,
            _sort_key(entry),
            entry["slug"],
        )
    )

    tour_entries = [entry for entry in core_entries if (CORE_ROOT / entry["slug"] / "README.md").exists()]

    for entry in detail_entries:
        output_dir = repo_root / "examples" / "core" / entry["slug"]
        parsed = parse_readme(output_dir / "README.md")
        html_content = build_detail_page(entry, parsed, output_dir, tour_entries)
        output_file = output_dir / "index.html"
        output_file.write_text(html_content, encoding="utf-8")
        generated_paths.append(output_file)

    return generated_paths
