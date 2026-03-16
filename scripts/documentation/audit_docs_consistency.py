#!/usr/bin/env python3
"""
Inventory documentation files, diagram assets, and runnable commands.

This script produces machine-readable audit artifacts so a documentation review
can be reproduced and compared over time.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shlex
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DOC_ROOTS = ("docs", "examples", "walkthrough")
ROOT_DOCS = ("README.md", "CONTRIBUTING.md", "CHANGELOG.md")
SKIP_PARTS = {
    ".git",
    ".venv",
    "__pycache__",
    "node_modules",
    ".traigent_local",
    ".traigent_results",
    "local",
    ".archive",
    ".release_review",
    ".post_release_recommendation_fixes",
}
ASSET_SUFFIXES = {
    ".png",
    ".jpg",
    ".jpeg",
    ".svg",
    ".gif",
    ".webp",
    ".drawio",
    ".mmd",
    ".mermaid",
}
DOC_SUFFIXES = {".md", ".mdx", ".rst", ".txt", ".html", ".htm"}


@dataclass
class CommandRecord:
    doc_path: str
    command: str
    block_lang: str
    cwd_context: str
    local_ref: str | None
    resolved_ref: str | None
    ref_status: str


@dataclass
class DocRecord:
    path: str
    exists: bool
    images: int
    broken_images: int
    links: int
    broken_links: int
    mermaid_blocks: int
    command_blocks: int
    runnable_commands: int


def should_skip(path: Path) -> bool:
    return any(part in SKIP_PARTS for part in path.parts)


def iter_doc_files() -> list[Path]:
    docs: list[Path] = []
    for root_doc in ROOT_DOCS:
        path = PROJECT_ROOT / root_doc
        if path.exists():
            docs.append(path)
    for root_name in DOC_ROOTS:
        root = PROJECT_ROOT / root_name
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if path.suffix.lower() not in DOC_SUFFIXES:
                continue
            if should_skip(path):
                continue
            docs.append(path)
    return sorted(set(docs))


def iter_asset_files() -> list[Path]:
    assets: list[Path] = []
    for root_name in DOC_ROOTS:
        root = PROJECT_ROOT / root_name
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if path.suffix.lower() not in ASSET_SUFFIXES:
                continue
            if should_skip(path):
                continue
            assets.append(path)
    return sorted(set(assets))


def resolve_link(doc_path: Path, ref: str) -> tuple[str, str | None]:
    if not ref or ref.startswith(("#", "mailto:", "tel:")):
        return "anchor", None
    if ref.startswith(("http://", "https://")):
        return "external", None
    if ref.startswith("/"):
        resolved = PROJECT_ROOT / ref.lstrip("/")
        if resolved.exists():
            return "exists", str(resolved.relative_to(PROJECT_ROOT))
        if "examples" in doc_path.parts:
            examples_resolved = PROJECT_ROOT / "examples" / ref.lstrip("/")
            if examples_resolved.exists():
                return "exists", str(examples_resolved.relative_to(PROJECT_ROOT))
        return "site-absolute-missing", str(resolved.relative_to(PROJECT_ROOT))

    relative = (doc_path.parent / ref).resolve()
    repo_relative = (PROJECT_ROOT / ref).resolve()
    examples_relative = (PROJECT_ROOT / "examples" / ref).resolve()
    if relative.exists():
        return "exists", str(relative.relative_to(PROJECT_ROOT))
    if "examples" in doc_path.parts and examples_relative.exists():
        return "exists", str(examples_relative.relative_to(PROJECT_ROOT))
    if repo_relative.exists():
        return "root-style", str(repo_relative.relative_to(PROJECT_ROOT))
    return "missing", None


def extract_refs(text: str, pattern: str) -> list[str]:
    refs: list[str] = []
    for ref in re.findall(pattern, text, flags=re.IGNORECASE):
        refs.append(ref.split("#", 1)[0].strip())
    return refs


def strip_fenced_code_blocks(text: str) -> str:
    """Remove fenced code block contents before link/image extraction."""
    lines: list[str] = []
    in_block = False
    for line in text.splitlines():
        if re.match(r"^```([A-Za-z0-9_-]+)?\s*$", line):
            in_block = not in_block
            lines.append("")
            continue
        lines.append("" if in_block else line)
    return "\n".join(lines)


def iter_code_blocks(text: str) -> list[tuple[str, str]]:
    blocks: list[tuple[str, str]] = []
    in_block = False
    lang = ""
    lines: list[str] = []
    for line in text.splitlines():
        match = re.match(r"^```([A-Za-z0-9_-]+)?\s*$", line)
        if match:
            if not in_block:
                in_block = True
                lang = (match.group(1) or "").lower()
                lines = []
            else:
                blocks.append((lang, "\n".join(lines)))
                in_block = False
                lang = ""
                lines = []
            continue
        if in_block:
            lines.append(line)
    return blocks


def join_shell_lines(block: str) -> list[str]:
    logical_lines: list[str] = []
    current = ""
    for raw_line in block.splitlines():
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        candidate = stripped[2:].strip() if stripped.startswith(("$ ", "> ")) else stripped
        if current:
            current = f"{current} {candidate}"
        else:
            current = candidate
        if current.endswith("\\"):
            current = current[:-1].rstrip()
            continue
        logical_lines.append(current)
        current = ""
    if current:
        logical_lines.append(current)
    return logical_lines


def peel_env_assignments(parts: list[str]) -> list[str]:
    index = 0
    while index < len(parts):
        token = parts[index]
        if "=" not in token:
            break
        key = token.split("=", 1)[0]
        if not key or not re.match(r"^[A-Z0-9_]+$", key):
            break
        index += 1
    return parts[index:]


def classify_ref(ref: str | None) -> str:
    if not ref:
        return "no-local-ref"
    if any(marker in ref for marker in ("<", ">", "${", "your_", "YOUR_")):
        return "template"
    if ref in {"-m", "-c", "-"}:
        return "module-or-inline"
    return "candidate"


def discover_generated_refs(text: str) -> set[str]:
    generated: set[str] = set()
    for block_lang, block in iter_code_blocks(text):
        if block_lang in {"python", "py"}:
            lines = [line.strip() for line in block.splitlines() if line.strip()]
            if lines:
                match = re.match(r"^#\s+([A-Za-z0-9_./-]+\.[A-Za-z0-9_]+)\s*$", lines[0])
                if match:
                    generated.add(match.group(1))
        if block_lang not in {"", "bash", "sh", "shell", "console", "zsh"}:
            continue
        for command in join_shell_lines(block):
            cat_match = re.search(r"\bcat\s*>\s*([^\s]+)", command)
            if cat_match:
                generated.add(cat_match.group(1))
            try:
                parts = shlex.split(command)
            except ValueError:
                continue
            if parts[:2] == ["traigent", "generate"] and "-o" in parts:
                output_index = parts.index("-o")
                if output_index + 1 < len(parts):
                    generated.add(parts[output_index + 1])
    return generated


def resolve_command_ref(
    doc_path: Path,
    cwd_context: Path,
    ref: str | None,
    generated_refs: set[str],
) -> tuple[str, str | None]:
    status = classify_ref(ref)
    if status != "candidate":
        return status, None
    assert ref is not None
    if ref in generated_refs or Path(ref).name in {Path(item).name for item in generated_refs}:
        return "generated-in-doc", relativize((cwd_context / ref).resolve())
    candidates = [
        (cwd_context / ref).resolve(),
        (doc_path.parent / ref).resolve(),
        (PROJECT_ROOT / ref).resolve(),
    ]
    for candidate in candidates:
        if candidate.exists():
            return "exists", str(candidate.relative_to(PROJECT_ROOT))
    return "missing", None


def extract_commands(doc_path: Path, text: str) -> list[CommandRecord]:
    commands: list[CommandRecord] = []
    generated_refs = discover_generated_refs(text)
    for block_lang, block in iter_code_blocks(text):
        if block_lang not in {"", "bash", "sh", "shell", "console", "zsh"}:
            continue
        cwd_context = doc_path.parent
        for command in join_shell_lines(block):
            if command.startswith(("cd ", "pushd ")):
                parts = shlex.split(command)
                if len(parts) > 1:
                    for candidate in (
                        (cwd_context / parts[1]).resolve(),
                        (PROJECT_ROOT / parts[1]).resolve(),
                    ):
                        if candidate.exists():
                            cwd_context = candidate
                            break
                continue
            try:
                parts = peel_env_assignments(shlex.split(command))
            except ValueError:
                parts = []
            if not parts:
                continue
            local_ref: str | None = None
            head = parts[0]
            if head in {"python", "python3"}:
                if len(parts) > 1:
                    local_ref = parts[1]
            elif head in {"bash", "sh"}:
                if len(parts) > 1:
                    local_ref = parts[1]
            elif head == "pytest":
                for token in parts[1:]:
                    if token.startswith("-"):
                        continue
                    if token.endswith(".py") or token.startswith("tests/"):
                        local_ref = token
                        break
            elif head.startswith("./"):
                local_ref = head
            else:
                continue
            ref_status, resolved_ref = resolve_command_ref(
                doc_path, cwd_context, local_ref, generated_refs
            )
            commands.append(
                CommandRecord(
                    doc_path=str(doc_path.relative_to(PROJECT_ROOT)),
                    command=command,
                    block_lang=block_lang or "plain",
                    cwd_context=relativize(cwd_context) or str(cwd_context),
                    local_ref=local_ref,
                    resolved_ref=resolved_ref,
                    ref_status=ref_status,
                )
            )
    return commands


def build_records() -> tuple[list[DocRecord], list[dict], list[CommandRecord]]:
    asset_inventory = []
    commands: list[CommandRecord] = []
    doc_records: list[DocRecord] = []

    for asset in iter_asset_files():
        asset_inventory.append(
            {
                "path": str(asset.relative_to(PROJECT_ROOT)),
                "suffix": asset.suffix.lower(),
                "size_bytes": asset.stat().st_size,
            }
        )

    for doc_path in iter_doc_files():
        text = doc_path.read_text(encoding="utf-8", errors="ignore")
        rendered_text = strip_fenced_code_blocks(text)
        image_refs = extract_refs(rendered_text, r"!\[[^\]]*\]\(([^)]+)\)")
        image_refs.extend(
            extract_refs(rendered_text, r'<img[^>]+src=["\']([^"\']+)["\']')
        )
        link_refs = extract_refs(rendered_text, r"(?<!!)\[[^\]]+\]\(([^)]+)\)")
        link_refs.extend(
            extract_refs(rendered_text, r'<a[^>]+href=["\']([^"\']+)["\']')
        )
        mermaid_blocks = len(re.findall(r"^```mermaid\s*$", text, flags=re.MULTILINE))
        code_blocks = iter_code_blocks(text)
        block_commands = extract_commands(doc_path, text)
        commands.extend(block_commands)

        broken_images = 0
        for ref in image_refs:
            status, _ = resolve_link(doc_path, ref)
            if status in {"missing", "root-style", "site-absolute-missing"}:
                broken_images += 1

        broken_links = 0
        for ref in link_refs:
            status, _ = resolve_link(doc_path, ref)
            if status in {"missing", "root-style", "site-absolute-missing"}:
                broken_links += 1

        doc_records.append(
            DocRecord(
                path=str(doc_path.relative_to(PROJECT_ROOT)),
                exists=True,
                images=len(image_refs),
                broken_images=broken_images,
                links=len(link_refs),
                broken_links=broken_links,
                mermaid_blocks=mermaid_blocks,
                command_blocks=sum(
                    1
                    for lang, _ in code_blocks
                    if lang in {"", "bash", "sh", "shell", "console", "zsh"}
                ),
                runnable_commands=len(block_commands),
            )
        )

    return doc_records, asset_inventory, commands


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def relativize(path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def write_csv(path: Path, rows: list[DocRecord]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def write_summary(
    path: Path,
    docs: list[DocRecord],
    assets: list[dict],
    commands: list[CommandRecord],
) -> None:
    ref_status_counts = Counter(command.ref_status for command in commands)
    broken_docs = [doc for doc in docs if doc.broken_links or doc.broken_images]
    lines = [
        "# Docs Audit Inventory",
        "",
        f"- Docs scanned: {len(docs)}",
        f"- Image/diagram assets scanned: {len(assets)}",
        f"- Runnable commands found: {len(commands)}",
        f"- Docs with broken links/images: {len(broken_docs)}",
        "",
        "## Command Ref Status",
        "",
    ]
    for key, value in sorted(ref_status_counts.items()):
        lines.append(f"- `{key}`: {value}")
    lines.extend(
        [
            "",
            "## Docs With Broken References",
            "",
        ]
    )
    if broken_docs:
        for doc in broken_docs:
            lines.append(
                f"- `{doc.path}`: {doc.broken_links} broken links, {doc.broken_images} broken images"
            )
    else:
        lines.append("- None")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_tracking(
    path: Path,
    docs: list[DocRecord],
    assets: list[dict],
    commands: list[CommandRecord],
) -> None:
    command_counts = Counter(command.doc_path for command in commands)
    lines = [
        "# Docs Scan Tracking",
        "",
        "## Documentation Files",
        "",
        "| Path | Status | Images | Links | Mermaid | Commands |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for doc in docs:
        if doc.broken_links or doc.broken_images:
            status = "attention"
        elif command_counts.get(doc.path):
            status = "verified-surface"
        else:
            status = "scanned"
        lines.append(
            f"| `{doc.path}` | `{status}` | {doc.images} | {doc.links} | {doc.mermaid_blocks} | {command_counts.get(doc.path, 0)} |"
        )

    lines.extend(
        [
            "",
            "## Image And Diagram Assets",
            "",
            "| Path | Type | Size (bytes) |",
            "| --- | --- | ---: |",
        ]
    )
    for asset in assets:
        lines.append(
            f"| `{asset['path']}` | `{asset['suffix']}` | {asset['size_bytes']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    docs, assets, commands = build_records()

    write_json(output_dir / "doc_inventory.json", [asdict(doc) for doc in docs])
    write_csv(output_dir / "doc_inventory.csv", docs)
    write_json(output_dir / "image_diagram_inventory.json", assets)
    write_json(output_dir / "command_inventory.json", [asdict(command) for command in commands])
    write_summary(output_dir / "inventory_summary.md", docs, assets, commands)
    write_tracking(output_dir / "scan_tracking.md", docs, assets, commands)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
