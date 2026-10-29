"""Guard: shipped docs, examples, plugins and scripts use real decorator kwargs.

Regression for #2836 / #2838. ``@traigent.optimize(...)`` rejects unknown
keyword arguments at decoration time, so a snippet that passes
``debug_evaluation=``, ``spec=``, ``config_class=`` ... raises ``TypeError`` on
its first line, and ``execution_mode="edge_analytics"`` raises
``ConfigurationError``.

The scan is textual (docs are Markdown, HTML, shell heredocs and Python), so it
is deliberately conservative: a keyword is flagged when it is not an accepted
decorator option AND either (a) it is a name known to have been taught
wrongly, or (b) its value is a literal (a bare list/string/number cannot be an
inline tuned-variable definition, which must be a ``Range``/``Choices`` call).
A keyword bound to a variable or call may be a legitimate inline tuned
variable (``temperature=temperature``) and is not flagged.
"""

from __future__ import annotations

import ast
import asyncio
import inspect
import re
import subprocess
import sys
from pathlib import Path

import pytest

from traigent.api import decorators

REPO_ROOT = Path(__file__).resolve().parents[3]
PLUGIN_UI = REPO_ROOT / "plugins" / "traigent-ui"

ACCEPTED = (
    (set(inspect.signature(decorators.optimize).parameters) - {"runtime_overrides"})
    | set(decorators._DIRECT_OPTION_KEYS)
    | set(decorators._ALLOWED_RUNTIME_OVERRIDE_KEYS)
) - set(decorators._REMOVED_PARAMETERS)

# Names taught in docs/examples that never existed (or were removed).
KNOWN_BAD = {
    "debug_evaluation",
    "timeout_per_trial",
    "direction",
    "config_space",
    "spec",
    "debug",
    "dataset",
    "config_class",
    "metrics",
    "callbacks",
    "random_seed",
    "mock_mode_config",
    "strategy",
    "budget_limit",
    "budget_metric",
    "budget_include_pruned",
}

# (file, keyword) pairs that deliberately show a removed/invalid call.
ALLOWED_REMOVED_API_NOTES = {
    ("docs/architecture/stop_conditions.md", "budget_limit"),
    ("docs/architecture/stop_conditions.md", "budget_metric"),
    ("traigent/api/decorators.py", "strategy"),
}

SCANNED_SUFFIXES = (".py", ".md", ".sh", ".html", ".txt", ".rst")
OPEN = re.compile(r"(?:@(?:traigent\.)?optimize|(?<![\w.])traigent\.optimize)\(")
ARG = re.compile(r"^\s*([A-Za-z_]\w*)\s*=(?!=)\s*(.*)$", re.S)
LITERAL = re.compile(
    r"""^(?:["'\[{(]|-?\d|True\b|False\b|None\b)""",
)


def _split_args(text: str, start: int) -> list[str]:
    depth, i, n = 1, start, len(text)
    cur: list[str] = []
    args: list[str] = []
    while i < n:
        c = text[i]
        if c in "\"'":
            q = text[i : i + 3] if text[i : i + 3] in ('"""', "'''") else c
            j = text.find(q, i + len(q))
            while j != -1 and q == c and text[j - 1] == "\\":
                j = text.find(q, j + 1)
            if j == -1:
                return args
            cur.append(text[i : j + len(q)])
            i = j + len(q)
            continue
        if c == "#":
            j = text.find("\n", i)
            i = n if j == -1 else j
            continue
        if c in "([{":
            depth += 1
        elif c in ")]}":
            depth -= 1
            if depth == 0:
                args.append("".join(cur))
                return args
        if c == "," and depth == 1:
            args.append("".join(cur))
            cur = []
        else:
            cur.append(c)
        i += 1
    return args


def _is_comment_line(text: str, pos: int) -> bool:
    line_start = text.rfind("\n", 0, pos) + 1
    return text[line_start:pos].lstrip().startswith(("#", "echo"))


def _tracked_files() -> list[str]:
    out = subprocess.run(
        ["git", "ls-files"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
    ).stdout
    return [
        f
        for f in out.splitlines()
        if f.endswith(SCANNED_SUFFIXES) and not f.startswith("tests/")
    ]


def find_bad_decorator_kwargs(files: list[str]) -> list[str]:
    offenders: list[str] = []
    for rel in files:
        try:
            text = (REPO_ROOT / rel).read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        for match in OPEN.finditer(text):
            if _is_comment_line(text, match.start()):
                continue
            for arg in _split_args(text, match.end()):
                parsed = ARG.match(arg)
                if not parsed:
                    continue
                name, value = parsed.group(1), parsed.group(2).strip()
                if name in ACCEPTED or (rel, name) in ALLOWED_REMOVED_API_NOTES:
                    continue
                if name in KNOWN_BAD or LITERAL.match(value):
                    line = text.count("\n", 0, match.start()) + 1
                    offenders.append(f"{rel}:{line}: {name}=")
    return offenders


def test_scanner_flags_the_reported_kwargs() -> None:
    """The scan itself must be red on the shapes the issue reported."""
    sample = '@traigent.optimize(debug_evaluation=True, eval_dataset="x")\n'
    path = REPO_ROOT / "docs" / "_scan_probe.md"
    try:
        path.write_text(sample, encoding="utf-8")
        assert find_bad_decorator_kwargs(["docs/_scan_probe.md"]) == [
            "docs/_scan_probe.md:1: debug_evaluation="
        ]
    finally:
        path.unlink(missing_ok=True)


def test_shipped_snippets_use_only_accepted_decorator_kwargs() -> None:
    assert find_bad_decorator_kwargs(_tracked_files()) == []


def test_no_edge_analytics_selector_in_plugins_or_scripts() -> None:
    out = subprocess.run(
        ["git", "grep", "-n", "-i", "edge.analytics", "--", "plugins", "scripts"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    ).stdout
    assert out == ""


def _optimize_decorator_keywords(source: str) -> list[tuple[str, list[ast.keyword]]]:
    found = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Call):
            fn = node.func
            dotted = (
                f"{fn.value.id}.{fn.attr}"
                if isinstance(fn, ast.Attribute) and isinstance(fn.value, ast.Name)
                else getattr(fn, "id", "")
            )
            if dotted in {"traigent.optimize", "optimize"}:
                found.append((dotted, node.keywords))
    return found


def _assert_kwargs_accepted(source: str, label: str) -> None:
    calls = _optimize_decorator_keywords(source)
    assert calls, f"{label}: no @traigent.optimize call found"
    for _, keywords in calls:
        for kw in keywords:
            assert kw.arg in ACCEPTED, f"{label}: {kw.arg}="
            if kw.arg == "execution_mode":
                assert not (
                    isinstance(kw.value, ast.Constant)
                    and kw.value.value == "edge_analytics"
                ), label


@pytest.mark.parametrize(
    "path",
    sorted((PLUGIN_UI / "traigent_ui" / "langchain_problems").glob("*.py")),
    ids=lambda p: p.name,
)
def test_plugin_problem_modules_decorate_with_accepted_kwargs(path: Path) -> None:
    source = path.read_text(encoding="utf-8")
    if not _optimize_decorator_keywords(source):
        pytest.skip("no decorator in this module")
    _assert_kwargs_accepted(source, path.name)


def test_code_generator_emits_only_accepted_decorator_kwargs(monkeypatch) -> None:
    monkeypatch.syspath_prepend(str(PLUGIN_UI))
    for name in [m for m in sys.modules if m.split(".")[0] == "traigent_ui"]:
        monkeypatch.delitem(sys.modules, name)
    from traigent_ui.problem_management.code_generator import CodeGenerator

    examples = [{"input_data": {"text": "a"}, "expected_output": "b", "metadata": {}}]
    source = asyncio.run(
        CodeGenerator().generate_problem_module(
            name="demo_problem",
            description="demo",
            domain="technical",
            difficulty="easy",
            examples=examples,
            output_type="classification",
            models=["gpt-4o-mini", "gpt-4o"],
            temperature_range=[0.1, 0.5],
        )
    )
    _assert_kwargs_accepted(source, "generated module")
