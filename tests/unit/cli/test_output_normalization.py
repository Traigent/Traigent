"""Pin the CLI output normalizer itself (Traigent#2052).

``plain_cli_text`` is what makes the substring assertions in this package
independent of ambient rendering, so it needs its own guard: without one, a
future edit could quietly weaken it back to raw-substring matching and the
tests it protects would go green on the author's machine anyway.

These tests render a known phrase through a real ``rich.Console`` that is
*forced* to be coloured and/or narrow, so they do not depend on the terminal
pytest was launched from. Each of the two halves of the normalizer is pinned
separately: delete the CSI strip and the colour case fails; delete the
whitespace collapse and the wrap case fails.
"""

from __future__ import annotations

import io
from collections.abc import Callable

from rich.console import Console

# The hardest real assertion in the suite: rich colours the retry count, which
# splits the phrase with SGR codes mid-sentence.
_PHRASE = "failed after 3 consecutive transport errors"
_MARKUP = (
    "[red]failed after [bold]3[/bold] consecutive transport errors. "
    "Please check your network connection and try again.[/red]"
)


def _render(markup: str, *, width: int, color: bool) -> str:
    """Render ``markup`` at a pinned width and colour setting.

    ``_environ={}`` hides the ambient environment from rich, so the sample is
    identical whether pytest was launched under ``FORCE_COLOR=1``,
    ``NO_COLOR=1 TERM=dumb``, or a narrow ``COLUMNS``. A pin test for
    rendering-independence must not itself depend on the rendering.
    """
    buffer = io.StringIO()
    Console(
        file=buffer,
        force_terminal=color,
        no_color=not color,
        width=width,
        legacy_windows=False,
        highlight=False,
        _environ={},
    ).print(markup)
    return buffer.getvalue()


def test_colour_defeats_raw_substring_matching(plain: Callable[[str], str]) -> None:
    """Colour alone splits the phrase — pins the ANSI-stripping half."""
    rendered = _render(_MARKUP, width=200, color=True)

    assert "\x1b[" in rendered, "sample must actually be coloured"
    assert _PHRASE not in rendered
    assert _PHRASE in plain(rendered)


def test_wrapping_defeats_raw_substring_matching(plain: Callable[[str], str]) -> None:
    """Soft-wrapping alone splits the phrase — pins the whitespace-collapse half."""
    rendered = _render(_MARKUP, width=40, color=False)

    assert "\x1b[" not in rendered, "sample must be uncoloured"
    assert "\n" in rendered.rstrip("\n"), "sample must actually be wrapped"
    assert _PHRASE not in rendered
    assert _PHRASE in plain(rendered)


def test_colour_and_wrapping_together_are_recovered(
    plain: Callable[[str], str],
) -> None:
    """Both at once, the shape real CLI output takes in a narrow terminal."""
    rendered = _render(_MARKUP, width=40, color=True)

    assert "\x1b[" in rendered
    assert "\n" in rendered.rstrip("\n")
    assert _PHRASE not in rendered
    assert _PHRASE in plain(rendered)


def test_plain_output_passes_through_unchanged(plain: Callable[[str], str]) -> None:
    """Identity on text that was never styled — the normalizer adds no meaning."""
    assert plain("HTTP status: 401") == "HTTP status: 401"
    assert plain("  Backend endpoint mismatch\n") == "Backend endpoint mismatch"


def test_osc8_hyperlinks_are_stripped(plain: Callable[[str], str]) -> None:
    """rich emits OSC-8 wrappers around URLs under ``force_terminal``."""
    linked = "\x1b]8;id=1;https://traigent.ai\x07Traigent docs\x1b]8;;\x07"
    assert plain(linked) == "Traigent docs"


def test_absent_phrase_stays_absent(plain: Callable[[str], str]) -> None:
    """Collapsing whitespace must not fabricate a match.

    The device-banner test asserts a *negative* against normalized text, so the
    normalizer must not join unrelated fragments into the forbidden phrase.
    """
    rendered = _render(
        "[cyan]Authenticating with:[/cyan] https://api.example.test/custom/v1",
        width=40,
        color=True,
    )
    assert "Authenticating with: http://localhost:5000" not in plain(rendered)
    assert "Authenticating with: https://api.example.test/custom/v1" in plain(rendered)
