"""Shipped examples must apply the per-trial config they tune (#3171).

In the default CONTEXT injection mode the per-trial config reaches the function
only through ``traigent.get_config()`` / ``get_trial_config()`` / ``current_config``.
A decorated function that declares a ``configuration_space`` (or ``tvl_spec``)
but never *uses* that config runs every trial identically, so the SDK warns about
a "phantom best_config" and the example prints a winner no tuning produced.

This is a static mirror of the decoration-time check in
``traigent.api.decorators._warn_context_mode_param_shadowing``, but stricter than
that substring check: it needs a real call to an accessor, and the value must
flow into something observable (a call argument, a return value or a branch).
Comments, docstrings, and values that are assigned and then never used do not
count.
"""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SCAN_DIRS = ("examples", "walkthrough")
ACCESSOR_CALLS = frozenset({"get_config", "get_trial_config", "get_current_config"})
ACCESSOR_ATTRS = frozenset({"current_config"})
# Calls that only reshape a value: feeding the config to them is not a use.
PASS_THROUGH = frozenset(
    {
        "dict",
        "float",
        "int",
        "str",
        "bool",
        "list",
        "tuple",
        "set",
        "round",
        "max",
        "min",
        "abs",
        "getattr",
        "isinstance",
        "len",
        "sorted",
    }
)

# (relative path, function) -> why the site is exempt. Keep this list tiny:
# every entry still draws the SDK's decoration-time warning.
ALLOWED: dict[tuple[str, str], str] = {}


def _is_accessor(node: ast.AST) -> bool:
    if isinstance(node, ast.Call):
        func = node.func
        name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
        return name in ACCESSOR_CALLS
    if isinstance(node, ast.Attribute):
        return node.attr in ACCESSOR_ATTRS
    return False


def _names_loaded(node: ast.AST) -> set[str]:
    return {
        n.id
        for n in ast.walk(node)
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)
    }


def _contains_accessor(node: ast.AST) -> bool:
    return any(_is_accessor(n) for n in ast.walk(node))


def _tainted(node: ast.AST, tainted: set[str]) -> bool:
    """True when the expression carries (or reads) the per-trial config."""
    return _contains_accessor(node) or bool(_names_loaded(node) & tainted)


def _is_pass_through(call: ast.Call, tainted: set[str]) -> bool:
    """Accessor calls, ``cfg.get(...)``-style reads and reshaping builtins."""
    if _is_accessor(call):
        return True
    func = call.func
    if isinstance(func, ast.Name):
        return func.id in PASS_THROUGH
    if isinstance(func, ast.Attribute):
        return _tainted(func.value, tainted)
    return False


def reads_trial_config(func: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """True when the function body takes the trial config AND uses it."""
    nodes = [n for stmt in func.body for n in ast.walk(stmt)]
    tainted: set[str] = set()
    # Propagate "carries the config" through assignments until stable.
    changed = True
    while changed:
        changed = False
        for n in nodes:
            targets: list[ast.expr] = []
            value: ast.AST | None = None
            if isinstance(n, ast.Assign):
                targets, value = list(n.targets), n.value
            elif isinstance(n, (ast.AnnAssign, ast.AugAssign)) and n.value is not None:
                targets, value = [n.target], n.value
            elif isinstance(n, ast.NamedExpr):
                targets, value = [n.target], n.value
            if value is None or not _tainted(value, tainted):
                continue
            for target in targets:
                for t in ast.walk(target):
                    if isinstance(t, ast.Name) and t.id not in tainted:
                        tainted.add(t.id)
                        changed = True

    def carries(expr: ast.AST | None) -> bool:
        return expr is not None and _tainted(expr, tainted)

    for n in nodes:
        # Flows into a call (argument, keyword, ** unpacking), excluding the
        # accessor call itself.
        if isinstance(n, ast.Call) and not _is_pass_through(n, tainted):
            if any(carries(a) for a in n.args) or any(
                carries(k.value) for k in n.keywords
            ):
                return True
        if isinstance(n, ast.Return) and carries(n.value):
            return True
        if isinstance(n, (ast.If, ast.While, ast.IfExp)) and carries(n.test):
            return True
    return False


def _is_context_mode(call: ast.Call) -> bool:
    for kw in call.keywords:
        if kw.arg == "injection_mode":
            return "context" in ast.unparse(kw.value).lower()
    return True


def find_phantom_sites(source: str) -> list[tuple[int, str]]:
    """Return ``(line, function)`` for tuned functions that ignore the trial config.

    Raises ``SyntaxError`` for unparseable source; callers must not swallow it.
    """
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
            if not {k.arg for k in dec.keywords if k.arg} & {
                "configuration_space",
                "tvl_spec",
            }:
                continue
            if not _is_context_mode(dec):
                continue
            if not reads_trial_config(node):
                found.append((node.lineno, node.name))
    return found


_DECORATOR = "@traigent.optimize(configuration_space={'t': [0.0, 0.5]})\n"


def test_scanner_flags_phantom_shapes() -> None:
    cases = {
        "hardcoded": "def f(q):\n    return llm(temperature=0.0)\n",
        "comment_only": "def f(q):\n    # uses get_config() someday\n    return llm(0.0)\n",
        "docstring_only": 'def f(q):\n    """Call traigent.get_config()."""\n    return llm(0.0)\n',
        "unused_assignment": (
            "def f(q):\n    cfg = traigent.get_config()\n    t = cfg.get('t', 0.0)\n    pass\n"
        ),
        "pass_body_with_param_overrides": (
            "def f(t=0.0):\n    trial = traigent.get_config() or {}\n"
            "    t = trial.get('t', t)\n    pass\n"
        ),
    }
    for label, body in cases.items():
        assert find_phantom_sites(_DECORATOR + body) == [(2, "f")], label


def test_scanner_accepts_real_use() -> None:
    cases = {
        "direct_arg": "def f(q):\n    return llm(**traigent.get_config())\n",
        "via_variable": (
            "def f(q):\n    cfg = traigent.get_config() or {}\n"
            "    return llm(temperature=cfg.get('t', 0.0))\n"
        ),
        "branch": (
            "def f(q):\n    cfg = traigent.get_trial_config()\n"
            "    if cfg['t'] > 0.2:\n        return 'hot'\n    return 'cold'\n"
        ),
        "param_override_then_call": (
            "def f(q, t=0.0):\n    t = (traigent.get_config() or {}).get('t', t)\n"
            "    return llm(temperature=t)\n"
        ),
    }
    for label, body in cases.items():
        assert find_phantom_sites(_DECORATOR + body) == [], label


def test_unparseable_source_is_not_swallowed() -> None:
    try:
        find_phantom_sites("def broken(:\n")
    except SyntaxError:
        return
    raise AssertionError("a file that does not parse must fail, not be skipped")


def _shipped_files() -> list[Path]:
    return [p for top in SCAN_DIRS for p in sorted((ROOT / top).rglob("*.py"))]


def test_shipped_examples_read_the_trial_config_3171() -> None:
    offenders: list[str] = []
    for path in _shipped_files():
        rel = path.relative_to(ROOT).as_posix()
        try:
            sites = find_phantom_sites(path.read_text(encoding="utf-8"))
        except SyntaxError as exc:
            offenders.append(f"{rel}: does not parse ({exc.msg}, line {exc.lineno})")
            continue
        offenders.extend(
            f"{rel}:{line} {fn}" for line, fn in sites if (rel, fn) not in ALLOWED
        )
    assert not offenders, (
        "These @traigent.optimize sites never use the per-trial config, so every "
        "trial runs identically (phantom best_config):\n  " + "\n  ".join(offenders)
    )


def test_allowlist_entries_still_exist() -> None:
    """Stale exemptions must be removed once the site is fixed or deleted."""
    for rel, fn in ALLOWED:
        sites = find_phantom_sites((ROOT / rel).read_text(encoding="utf-8"))
        assert any(name == fn for _, name in sites), f"stale allowlist entry {rel}:{fn}"
