"""Every ``traigent <subcommand ...>`` the package prints or offers must exist.

The SDK tells users to run commands (hints, errors, onboarding offers). A
command that is printed but not registered on the real Click group sends the
user to a dead end (``traigent login`` instead of ``traigent auth login``,
``traigent mcp register`` that never existed). This test scans the package
source for command-shaped string literals and argv lists, resolves each against
``traigent.cli.main.cli`` and fails with the offending ``file:line``.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import click
import pytest

from traigent.cli.main import cli

PACKAGE_ROOT = Path(__file__).resolve().parents[3] / "traigent"

# A command mention is only counted when it is *shaped* like an instruction,
# not like prose ("Mapping from traigent params ...") or Python code
# ("from traigent import Range"): wrapped in backticks/quotes, closing rich
# markup (``[cyan]traigent ...``), an opening paren, after ``$ ``, after an
# imperative verb (optionally followed by a colon: ``Run: traigent ...``), or
# leading a line (``Next steps:\n  traigent ...``).
_COMMAND = re.compile(
    r"""(?:
        (?m:^)[ \t]*                                   # line-leading (indented) command
      | [`'"\](]                                       # quote, backtick, rich-markup close, paren
      | \$\s                                          # shell prompt
      | \b(?:[Rr]un|[Uu]se|[Tt]ry|[Ee]xecute|[Ss]ee)\s*:?\s+   # imperative verb, optional colon
    )
    traigent((?:[ \t]+[a-z][a-z0-9_-]*)+)""",
    re.VERBOSE,
)


def extract_commands(text: str) -> list[list[str]]:
    """Command word lists (text after ``traigent``) found in one string."""
    return [m.group(1).split() for m in _COMMAND.finditer(text)]


# Explicit allowlist. Only placeholders / non-CLI uses belong here. Never add a
# real printed command that does not exist -- fix the string or the CLI instead.
# Keyed by the full text after ``traigent``; value is (file the allowance is
# limited to, reason). An entry that is never needed fails the test (stale).
ALLOWED_TEXT: dict[str, tuple[str, str]] = {
    "mcp register": (
        "traigent/cli/onboard_commands.py",
        "offered only after probing `traigent mcp register --help` "
        "(_mcp_help_succeeds), so never shown while the command does not "
        "exist (#2628)",
    ),
    "api decorators": (
        "traigent/config_generator/apply.py",
        "attribute chain traigent.api.decorators",
    ),
    "optimization": (
        "traigent/integrations/observability/wandb.py",
        "a list of W&B tags",
    ),
}

# Commands printed today that do not resolve and are not yet fixed. Remove an
# entry when it is fixed (the test fails on stale entries).
KNOWN_BROKEN: dict[str, str] = {
    "validate-dataset": (
        "evaluators/base.py docstring names `traigent validate-dataset`, "
        "which is not registered. Tracked in "
        "https://github.com/Traigent/Traigent/issues/2719: change the docstring "
        "at evaluators/base.py to `traigent validate`, then delete this entry"
    ),
}


def _resolve(words: list[str]) -> bool:
    """True if ``words`` walk down real Click groups (leaf args are ignored)."""
    node: click.Command = cli
    for word in words:
        if not isinstance(node, click.Group):
            return True  # remaining words are arguments of a leaf command
        if word not in node.commands:
            return False
        node = node.commands[word]
    return True


def _argv_words(elts: list[ast.expr]) -> list[str] | None:
    """Command words of a literal argv list like ["traigent", "mcp", "x", "--a"]."""
    if not elts or not (
        isinstance(elts[0], ast.Constant) and elts[0].value == "traigent"
    ):
        return None
    words: list[str] = []
    for elt in elts[1:]:
        if (
            isinstance(elt, ast.Constant)
            and isinstance(elt.value, str)
            and re.fullmatch(r"[a-z][a-z0-9_-]*", elt.value)
        ):
            words.append(elt.value)
        else:
            break
    return words or None


def _collect() -> list[tuple[str, int, list[str]]]:
    found: list[tuple[str, int, list[str]]] = []
    for path in sorted(PACKAGE_ROOT.rglob("*.py")):
        rel = path.relative_to(PACKAGE_ROOT.parent)
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                for words in extract_commands(node.value):
                    found.append((str(rel), node.lineno, words))
            elif isinstance(node, (ast.List, ast.Tuple)):
                words = _argv_words(node.elts)
                if words:
                    found.append((str(rel), node.lineno, words))
    return found


def test_scanner_finds_commands() -> None:
    """Guard against the scan silently matching nothing."""
    found = _collect()
    assert len(found) > 50
    assert any(words[:2] == ["auth", "login"] for _, _, words in found)


def test_every_printed_traigent_command_exists() -> None:
    offenders: list[str] = []
    seen_known: set[str] = set()
    seen_allowed_text: set[str] = set()
    for rel, lineno, words in _collect():
        text = " ".join(words)
        if _resolve(words):
            continue
        if text in ALLOWED_TEXT and rel == ALLOWED_TEXT[text][0]:
            seen_allowed_text.add(text)
            continue
        if text in KNOWN_BROKEN:
            seen_known.add(text)
            continue
        offenders.append(f"{rel}:{lineno}: 'traigent {text}' is not a CLI command")
    assert not offenders, "Printed commands missing from the CLI:\n" + "\n".join(
        offenders
    )
    stale = set(KNOWN_BROKEN) - seen_known
    assert not stale, f"Stale KNOWN_BROKEN entries (fixed? remove): {sorted(stale)}"
    stale = set(ALLOWED_TEXT) - seen_allowed_text
    assert not stale, f"Stale ALLOWED_TEXT entries (unused? remove): {sorted(stale)}"


@pytest.mark.parametrize(
    "text, expected",
    [
        # existing shapes
        ("Run 'traigent auth login' first", [["auth", "login"]]),
        ("use `traigent doctor`", [["doctor"]]),
        ("[cyan]traigent auth status[/cyan]", [["auth", "status"]]),
        ("$ traigent hooks install", [["hooks", "install"]]),
        ("Run traigent auth login to rotate", [["auth", "login", "to", "rotate"]]),
        # new shapes
        ("Run: traigent login", [["login"]]),
        ("Next steps:\n  traigent login\n", [["login"]]),
        ("(traigent auth login)", [["auth", "login"]]),
        # negatives
        ("from traigent import x", []),
        (">>> from traigent import Range", []),
        ("Mapping from traigent params to framework params", []),
        ("Note that traigent doctor never issues an upgrade", []),
        ("Import traigent before litellm", []),
    ],
)
def test_extract_commands_shapes(text: str, expected: list[list[str]]) -> None:
    assert extract_commands(text) == expected


@pytest.mark.parametrize(
    "words, expected",
    [
        (["auth", "login"], True),
        (["login"], False),
        (["mcp", "register"], False),
        (["results", "show", "my_run"], True),
    ],
)
def test_resolver(words: list[str], expected: bool) -> None:
    assert _resolve(words) is expected
