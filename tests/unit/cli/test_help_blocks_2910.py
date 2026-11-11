"""CLI --help keeps 'Examples:'/'Args:' blocks one entry per line (#2910).

Click re-wraps help paragraphs unless they start with a ``\\b`` line, so an
unmarked multi-line block collapses into one run-on paragraph.
"""

from __future__ import annotations

import click
import pytest
from click.testing import CliRunner

from traigent.cli.main import cli

_BLOCK_HEADERS = ("Examples:", "Example:", "Args:")


def _walk(cmd: click.Command, path: tuple[str, ...] = ()):
    yield path, cmd
    if isinstance(cmd, click.Group):
        for name, sub in sorted(cmd.commands.items()):
            yield from _walk(sub, (*path, name))


def _all_commands() -> list[tuple[tuple[str, ...], click.Command]]:
    return list(_walk(cli))


def _unmarked_blocks(help_text: str) -> list[str]:
    lines = help_text.splitlines()
    bad: list[str] = []
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped not in _BLOCK_HEADERS:
            continue
        nxt = lines[i + 1].strip() if i + 1 < len(lines) else ""
        if not nxt:
            continue
        j = i - 1
        while j >= 0 and not lines[j].strip():
            j -= 1
        if j < 0 or lines[j].strip() != "\b":
            bad.append(stripped)
    return bad


@pytest.mark.parametrize(
    "path,cmd",
    _all_commands(),
    ids=lambda v: " ".join(v) if isinstance(v, tuple) else "",
)
def test_help_blocks_are_marked_no_rewrap(path, cmd) -> None:
    assert _unmarked_blocks(cmd.help or "") == [], (
        f"'traigent {' '.join(path)}' has an Examples/Args block without a \\b line"
    )


@pytest.mark.parametrize(
    "args",
    [[], ["auth", "login"], ["check"], ["optimize"]],
    ids=lambda a: " ".join(a) or "root",
)
def test_rendered_help_examples_one_per_line(args: list[str]) -> None:
    result = CliRunner().invoke(cli, [*args, "--help"], terminal_width=100)
    assert result.exit_code == 0, result.output
    out = result.output
    if "Examples:" not in out:
        pytest.skip("no Examples block")
    block = out.split("Examples:", 1)[1]
    entries = [ln for ln in block.splitlines() if ln.strip().startswith("traigent ")]
    assert entries
    for ln in entries:
        assert ln.count("traigent ") == 1, ln
