#!/usr/bin/env python3
"""
Extract inline code examples from an HTML documentation page and write them
into a structured folder hierarchy with optional eval dataset stubs.

Output structure (per example):
examples/docs/page-inline/<page-id>/<section-id>/<example-id>/
  - <example-id>.py (code)
  - eval/<dataset_name>.jsonl (if referenced)
  - README.txt (short metadata)

Usage:
  python scripts/extract_inline_examples.py \
    examples/archive/docs/sections/configuration-management.html \
    configuration-management
"""
from __future__ import annotations

import json
import os
import re
import shutil
import sys
from html.parser import HTMLParser
from pathlib import Path
from typing import Dict, List, Optional

from traigent.utils.secure_path import PathTraversalError, validate_path

class Node:
    def __init__(
        self, tag: Optional[str], attrs: Dict[str, str], parent: Optional[Node]
    ) -> None:
        self.tag = tag
        self.attrs = attrs
        self.parent = parent
        self.children: List[Node] = []
        self.text_chunks: List[str] = []

    @property
    def text(self) -> str:
        return "".join(self.text_chunks)

    def add_child(self, node: Node) -> None:
        self.children.append(node)

    def find_all(
        self, tag: Optional[str] = None, cls: Optional[str] = None
    ) -> List[Node]:
        out: List[Node] = []
        stack = [self]
        while stack:
            node = stack.pop()
            stack.extend(node.children)
            if tag is not None and node.tag != tag:
                continue
            if cls is not None:
                classes = node.attrs.get("class", "").split()
                if cls not in classes:
                    continue
            if tag is None or node.tag == tag:
                if cls is None or cls in node.attrs.get("class", "").split():
                    out.append(node)
        return out


class SimpleHTMLParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.root = Node(tag=None, attrs={}, parent=None)
        self.stack: List[Node] = [self.root]

    def handle_starttag(self, tag, attrs):
        # Convert attrs list to dict; handle duplicate class attr by join
        attr_dict: Dict[str, str] = {}
        for k, v in attrs:
            if k == "class":
                attr_dict[k] = v or ""
            else:
                attr_dict[k] = v or ""
        node = Node(tag=tag, attrs=attr_dict, parent=self.stack[-1])
        self.stack[-1].add_child(node)
        self.stack.append(node)

    def handle_endtag(self, tag):
        # Pop until we match tag (robustness)
        while len(self.stack) > 1:
            node = self.stack.pop()
            if node.tag == tag:
                break

    def handle_data(self, data):
        if data:
            self.stack[-1].text_chunks.append(data)


def get_attr(node: Node, key: str, default: str = "") -> str:
    return node.attrs.get(key, default)


def extract_examples(html: str) -> List[Dict[str, str]]:
    parser = SimpleHTMLParser()
    parser.feed(html)

    examples: List[Dict[str, str]] = []

    # A section is a div.doc-section (id identifies section)
    for section in parser.root.find_all(tag="div", cls="doc-section"):
        section_id = get_attr(section, "id") or "no-section-id"

        # Within a section, examples are under .code-tabs .tab-content
        for tab_content in section.find_all(tag="div", cls="tab-content"):
            example_id = get_attr(tab_content, "id") or "example"
            # Find the first <code> element under this tab
            code_nodes = [n for n in tab_content.find_all(tag="code")]
            if not code_nodes:
                continue
            code_text = code_nodes[0].text

            # language from class="language-xyz" if available
            lang = ""
            code_class = code_nodes[0].attrs.get("class", "")
            for part in code_class.split():
                if part.startswith("language-"):
                    lang = part.split("-", 1)[1]
                    break

            examples.append(
                {
                    "section_id": section_id,
                    "example_id": example_id,
                    "language": lang or "text",
                    "code": code_text,
                }
            )

    return examples


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def maybe_add_python_imports(code: str) -> str:
    lines = code.splitlines()

    def has_import(pattern: str) -> bool:
        return any(re.search(pattern, ln) for ln in lines)

    preamble: List[str] = []

    # If code references traigent but no import, add it
    uses_traigent = re.search(r"@\s*traigent\.|\btraigent\.\w", code) is not None
    has_traigent_import = has_import(r"^\s*import\s+traigent\b") or has_import(
        r"^\s*from\s+traigent\b"
    )
    if uses_traigent and not has_traigent_import:
        preamble.append("import traigent")

    # If code references ChatOpenAI and no import exists, add it
    uses_chatopenai = re.search(r"\bChatOpenAI\s*\(", code) is not None
    has_chatopenai_import = has_import(
        r"^\s*from\s+langchain_openai\s+import\s+ChatOpenAI\b"
    ) or has_import(r"^\s*import\s+langchain_openai\b")
    if uses_chatopenai and not has_chatopenai_import:
        preamble.append("from langchain_openai import ChatOpenAI")

    if preamble:
        # Put imports at top, but keep shebang/encoding if present
        insert_at = 0
        if lines and lines[0].startswith("#!/"):
            insert_at = 1
        if len(lines) > insert_at and "coding:" in lines[insert_at]:
            insert_at += 1
        return "\n".join(lines[:insert_at] + preamble + [""] + lines[insert_at:])
    return code


def append_python_main(code: str) -> str:
    # Determine a callable to run and append a minimal __main__ that prints output
    # Try to find preferred functions; else last defined function; else a known class pattern
    preferred = [
        "optimized_summary",
        "intelligent_content_system",
        "adaptive_chat_bot",
        "smart_document_processor",
        "customer_support_agent",
        "production_support_agent",
        "debug_example",
        "my_function",
    ]
    # Find functions
    func_defs = re.findall(
        r"^def\s+(\w+)\s*\(([^)]*)\)\s*(?:->[^:]+)?\s*:", code, flags=re.MULTILINE
    )
    func_name = None
    func_params = ""
    candidates = {name: params for name, params in func_defs}
    for name in preferred:
        if name in candidates:
            func_name = name
            func_params = candidates[name]
            break
    if func_name is None and func_defs:
        func_name, func_params = func_defs[-1]

    # Simple required-args count
    def count_required(params: str) -> int:
        params = params.strip()
        if not params:
            return 0
        # split by comma, ignore *args/**kwargs and defaults
        req = 0
        for p in [x.strip() for x in params.split(",") if x.strip()]:
            if p.startswith("*"):
                continue
            # remove annotations and defaults
            name = p.split(":", 1)[0].split("=", 1)[0].strip()
            if name not in ("self",):
                # consider it required if no default present
                if "=" not in p:
                    req += 1
        return req

    required_args = count_required(func_params) if func_name else 0

    # Class-based fallback
    uses_bot = re.search(r"class\s+OptimizedChatBot\b", code) is not None

    main_lines = [
        "",
        'if __name__ == "__main__":',
        "    try:",
    ]
    if func_name:
        if required_args <= 0:
            call = f"{func_name}()"
        else:
            # Provide 'test input' for required positional params
            call = f"{func_name}('test input')"
        main_lines += [
            f"        _res = {call}",
            "        print(getattr(_res, 'content', _res))",
        ]
    elif uses_bot:
        main_lines += [
            "        bot = OptimizedChatBot()",
            "        _res = bot.generate_response('test input') if hasattr(bot, 'generate_response') else 'ok'",
            "        print(getattr(_res, 'content', _res))",
        ]
    else:
        main_lines += [
            "        pass",
        ]
    main_lines += [
        "    except Exception as e:",
        "        print(e)",
        "",
    ]
    return code.rstrip() + "\n" + "\n".join(main_lines)


def clean_snippet(code: str) -> str:
    """Remove verbose debug/helper lines from docs (keep example concise)."""
    lines = code.splitlines()
    out: List[str] = []
    skip_returns_block = False
    for ln in lines:
        # Drop noisy comment headers
        if ln.strip().startswith("# Enable global debug mode"):
            continue
        if ln.strip().startswith("# View injection logs"):
            continue
        # Drop global debug toggles and injection logs lines
        if ln.strip().startswith("traigent.set_debug_mode("):
            continue
        if ln.strip().startswith("traigent.get_injection_logs("):
            skip_returns_block = True
            continue
        # Skip following comment dump after get_injection_logs()
        if skip_returns_block:
            if ln.strip().startswith("#") or ln.strip() == "":
                continue
            else:
                skip_returns_block = False
        out.append(ln)
    return "\n".join(out)


def write_example(base_out: Path, page_id: str, ex: Dict[str, str]) -> str:
    section = ex["section_id"] or "no-section"
    example = ex["example_id"] or "example"
    lang = ex["language"] or "text"
    code = ex["code"]

    # Build folder: examples/docs/page-inline/<page-id>/<section>
    folder = validate_path(base_out / page_id / section, base_out)
    ensure_dir(folder)

    # Choose filename extension based on language
    ext = {
        "python": ".py",
        "json": ".json",
        "yaml": ".yaml",
        "yml": ".yml",
        "bash": ".sh",
        "sh": ".sh",
        "text": ".txt",
    }.get(lang, ".txt")

    code_filename = f"{example}{ext}"
    code_path = validate_path(folder / code_filename, base_out)
    # Minimal cleanup + import fix-ups for Python examples and append a __main__ block
    if lang == "python":
        code = clean_snippet(code)
        code = maybe_add_python_imports(code)
        code = append_python_main(code)
    with open(code_path, "w", encoding="utf-8") as f:
        f.write(code.rstrip() + "\n")

    # Attempt to identify eval_dataset from code and create stub
    # Looks for patterns like eval_dataset="name.jsonl"
    dataset_name: Optional[str] = None
    for marker in ['eval_dataset="', "eval_dataset='"]:
        if marker in code:
            start = code.find(marker) + len(marker)
            end = code.find('"' if marker.endswith('"') else "'", start)
            if end > start:
                dataset_name = code[start:end]
                break

    eval_path_written: Optional[str] = None
    if dataset_name and dataset_name.endswith(".jsonl"):
        eval_dir = validate_path(folder / "eval", base_out)
        ensure_dir(eval_dir)
        eval_path = validate_path(eval_dir / dataset_name, base_out)
        # Minimal JSONL stub
        sample = {"input": "sample text", "expected": "sample expected output"}
        with open(eval_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(sample) + "\n")
        eval_path_written = eval_path
        # Remove any duplicate root-level dataset file if present
        root_eval_path = validate_path(folder / dataset_name, base_out)
        if root_eval_path.exists() and root_eval_path.resolve() != eval_path.resolve():
            try:
                root_eval_path.unlink()
            except OSError:
                pass

    # Minimal metadata README (created once)
    readme_path = validate_path(folder / "README.txt", base_out)
    if not readme_path.exists():
        with open(readme_path, "w", encoding="utf-8") as f:
            f.write(
                f"page: {page_id}\n"
                f"section: {section}\n"
                + (f"eval_dataset: {eval_path_written}\n" if eval_path_written else "")
            )

    # Remove legacy nested folder if exists (keep only one folder per example)
    legacy_dir = validate_path(folder / example, base_out)
    if legacy_dir.is_dir():
        shutil.rmtree(legacy_dir, ignore_errors=True)

    # Cleanup legacy helper files if they exist
    legacy_runner = validate_path(folder / "run_example.py", base_out)
    if legacy_runner.exists():
        try:
            legacy_runner.unlink()
        except OSError:
            pass
    legacy_results = validate_path(folder / "results.run.json", base_out)
    if legacy_results.exists():
        try:
            legacy_results.unlink()
        except OSError:
            pass

    return str(folder)


def main() -> int:
    if len(sys.argv) < 3:
        print(
            "Usage: python scripts/extract_inline_examples.py <html_path> <page_id> [<out_base>=examples/docs/page-inline]",
            file=sys.stderr,
        )
        return 2

    html_path = sys.argv[1]
    page_id = sys.argv[2]
    out_base = (
        sys.argv[3] if len(sys.argv) > 3 else os.path.join("examples", "docs/page-inline")
    )

    base_dir = Path.cwd()
    try:
        html_path = validate_path(Path(html_path), base_dir, must_exist=True)
        out_base = validate_path(Path(out_base), base_dir)
    except (PathTraversalError, FileNotFoundError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    out_base.mkdir(parents=True, exist_ok=True)

    with open(html_path, encoding="utf-8") as f:
        html = f.read()

    examples = extract_examples(html)
    if not examples:
        print("No examples found.")
        return 1

    written: List[str] = []
    for ex in examples:
        folder = write_example(out_base, page_id, ex)
        if folder not in written:
            written.append(folder)

    print(json.dumps({"count": len(written), "folders": written}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
