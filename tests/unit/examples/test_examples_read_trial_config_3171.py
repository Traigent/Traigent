"""Shipped examples must apply the per-trial config they tune (#3171).

In the default CONTEXT injection mode the per-trial config reaches the function
only through ``traigent.get_config()`` / ``get_trial_config()`` / ``current_config``.
A decorated function that declares a ``configuration_space`` (or ``tvl_spec``)
but reads none of them runs every trial identically, so the SDK warns about a
"phantom best_config" and the example prints a winner no tuning produced.

This is a static mirror of the decoration-time check in
``traigent.api.decorators._warn_context_mode_param_shadowing``.
"""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SCAN_DIRS = ("examples", "walkthrough")
ACCESSORS = ("get_config", "get_trial_config", "current_config")

# (relative path, function) -> why the site is exempt.
ALLOWED = {
    (
        "examples/advanced/surrogate_evaluator_coevolution_demo.py",
        "polymath_agent",
    ): "deterministic synthetic agent; the demo is about surrogate scoring, not tuning",
    (
        "walkthrough/mock/advanced/05_langgraph_multiagent_demo.py",
        "run_rag_workflow",
    ): "LangGraph adapter injection is documented as unavailable in this demo",
    (
        "walkthrough/mock/advanced/06_unified_platform_campaign_copilot.py",
        "run_campaign_copilot",
    ): "config is read in graph-node helpers (_config_value), not in the decorated fn",
}


def _is_context_mode(call: ast.Call) -> bool:
    for kw in call.keywords:
        if kw.arg == "injection_mode":
            return "context" in ast.unparse(kw.value).lower()
    return True


def find_phantom_sites(source: str) -> list[tuple[int, str]]:
    tree = ast.parse(source)
    found: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for dec in node.decorator_list:
            if not isinstance(dec, ast.Call):
                continue
            func = dec.func
            name = (
                func.attr
                if isinstance(func, ast.Attribute)
                else getattr(func, "id", "")
            )
            if name != "optimize":
                continue
            kwargs = {k.arg for k in dec.keywords if k.arg}
            if not kwargs & {"configuration_space", "tvl_spec"}:
                continue
            if not _is_context_mode(dec):
                continue
            body = ast.get_source_segment(source, node) or ""
            if not any(acc in body for acc in ACCESSORS):
                found.append((node.lineno, node.name))
    return found


def test_scanner_flags_a_phantom_site() -> None:
    bad = (
        "import traigent\n"
        "@traigent.optimize(configuration_space={'temperature': [0.0, 0.5]})\n"
        "def f(q):\n    return llm(temperature=0.0)\n"
    )
    good = bad.replace("llm(temperature=0.0)", "llm(**traigent.get_config())")
    assert find_phantom_sites(bad) == [(3, "f")]
    assert find_phantom_sites(good) == []


def test_shipped_examples_read_the_trial_config_3171() -> None:
    offenders: list[str] = []
    for top in SCAN_DIRS:
        for path in sorted((ROOT / top).rglob("*.py")):
            rel = path.relative_to(ROOT).as_posix()
            try:
                sites = find_phantom_sites(path.read_text(encoding="utf-8"))
            except SyntaxError:
                continue
            offenders.extend(
                f"{rel}:{line} {fn}" for line, fn in sites if (rel, fn) not in ALLOWED
            )
    assert not offenders, (
        "These @traigent.optimize sites never read the per-trial config, so every "
        "trial runs identically (phantom best_config):\n  " + "\n  ".join(offenders)
    )


def test_allowlist_entries_still_exist() -> None:
    """Stale exemptions must be removed once the site is fixed or deleted."""
    for rel, fn in ALLOWED:
        sites = find_phantom_sites((ROOT / rel).read_text(encoding="utf-8"))
        assert any(name == fn for _, name in sites), f"stale allowlist entry {rel}:{fn}"
