"""Shared helpers for the Traigent CLI unit tests (Traigent#2052).

The CLI renders through ``rich``, which paints its output with ANSI colour and
soft-wraps it to the ambient terminal width.  A raw ``"phrase" in
result.output`` assertion therefore does not test what the CLI *said* — it
tests how the CLI happened to be *painted* on the machine that ran pytest.  The
same assertion flips to red when colour is forced on, or when ``COLUMNS`` is
narrow enough to wrap the phrase.

The two halves of that problem are fixed by two different mechanisms, and the
split is deliberate:

* **Width is pinned, never undone.**  ``_pin_cli_render_width`` exports
  ``COLUMNS=CLI_RENDER_WIDTH`` for every test in this package.  rich reads
  ``COLUMNS`` from the live environment on each render (``Console.size``), so
  the pin reaches the CLI's import-time ``Console()`` objects even though they
  were built long before the fixture ran.  At that width none of the messages
  asserted on here soft-wrap, so a newline in captured output is a real record
  boundary rather than a rendering artifact.
* **Styling is stripped.**  ``plain_cli_text`` removes SGR colour attributes
  and OSC-8 hyperlink wrappers.  Neither contributes a character the user
  reads, so removing them cannot change what the output *says*.

``plain_cli_text`` deliberately does **not** flatten output.  An earlier
revision collapsed every run of whitespace to a single space, and that
fabricates matches — it manufactures greens for phrases the CLI never printed:

* an erase-line sequence wipes a word off the screen, yet stripping the
  sequence puts the word back and a substring assertion passes on text the user
  never saw;
* two unrelated records printed on separate lines get joined across the newline
  into a phrase that was never emitted as one message;
* a carriage-return overwrite leaves the *replaced* text in the normalized
  string, so an assertion can match the value that was overwritten.

False greens are exactly the failure mode this package exists to prevent, so
``plain_cli_text`` preserves every logical boundary and *refuses* to normalize
output containing a display-altering control sequence.  Each of those three
cases is pinned by a test in ``test_output_normalization.py``.

What this does **not** cover: rich breaks words mid-token at narrow widths, and
no normalizer can undo that.  Measured against this package, assertions start
failing at ``COLUMNS=40`` — not the ``≈ 20`` an earlier revision of this
docstring claimed.  The width pin is what keeps that out of reach;
``plain_cli_text`` on its own is *not* rendering-independent and does not claim
to be.  Structured output (the onboarding ``PLAN_JSON_BEGIN``/``END`` frame) is
deliberately parsed from raw output so control-code corruption there stays
visible.
"""

from __future__ import annotations

import re
import sys
from collections.abc import Callable, Iterator

import pytest
from rich.console import Console

#: Terminal width pinned for every CLI unit test in this package.  Wide enough
#: that none of the asserted messages soft-wrap (the longest, once rich's panel
#: borders are counted, is well under this), and *fixed* so the same assertion
#: renders identically on CI, in a narrow tmux pane, and under ``COLUMNS=40``.
CLI_RENDER_WIDTH = 200

#: Pinned alongside the width so screen height cannot influence rendering
#: either (rich consults ``LINES`` from the same code path).
CLI_RENDER_HEIGHT = 100

# OSC-8 hyperlinks: ``ESC ] 8 ; params ; uri`` terminated by BEL or ST.  The
# wrapper carries no text; the label sitting between its two halves does, and
# is left alone.
_OSC8 = re.compile(r"\x1b\]8;[^\x07\x1b]*(?:\x07|\x1b\\)")

# SGR — colour, bold, underline, reset.  Purely presentational: removing these
# cannot change which characters the terminal ends up displaying.
_SGR = re.compile(r"\x1b\[[0-9;:]*m")

# Everything else introduced by ESC: CSI cursor movement/erase, OSC window
# titles, and the two-character Fe escapes.  These *change what is on screen*,
# so they are not strippable — see ``plain_cli_text``.
_DISPLAY_ALTERING_ESCAPE = re.compile(
    r"\x1b(?:\[[0-9;:?]*[ -/]*[@-~]|\][^\x07\x1b]*(?:\x07|\x1b\\)|[@-Z\\-_])"
)


def _imported_cli_consoles() -> Iterator[Console]:
    """Yield every module-level ``Console`` the CLI has already built.

    ``traigent.cli.*`` are the only modules in the package that hold a
    module-level console (verified by grep), and they build it at import time.
    """
    seen: set[int] = set()
    for name, module in list(sys.modules.items()):
        if module is None or not (
            name == "traigent.cli" or name.startswith("traigent.cli.")
        ):
            continue
        for value in vars(module).values():
            if isinstance(value, Console) and id(value) not in seen:
                seen.add(id(value))
                yield value


@pytest.fixture(autouse=True)
def _pin_cli_render_width(monkeypatch: pytest.MonkeyPatch) -> None:
    """Render every CLI test at :data:`CLI_RENDER_WIDTH`, whatever the terminal.

    This is the half of the fix that a normalizer cannot do.  Soft-wrapping
    destroys information — once rich has folded ``localhost:5000/a/b`` across a
    line break, no post-processing can tell that apart from two words that were
    always separate — so the only safe answer is to render at a width where the
    fold does not happen.  Autouse, because an assertion that forgets it is
    silently rendering-dependent again.

    Setting ``COLUMNS`` is necessary but not sufficient: rich reads it in
    ``Console.__init__`` and freezes the result on the instance, so the CLI's
    import-time consoles were already fixed to the ambient width before any
    fixture could run.  The existing instances are therefore re-pinned
    directly.  That deliberately also overrides the three production consoles
    built as ``Console(width=120)``: what the tests render at should be one
    number, not three.  ``monkeypatch`` restores every one of them afterwards.
    """
    monkeypatch.setenv("COLUMNS", str(CLI_RENDER_WIDTH))
    monkeypatch.setenv("LINES", str(CLI_RENDER_HEIGHT))
    for console in _imported_cli_consoles():
        monkeypatch.setattr(console, "_width", CLI_RENDER_WIDTH, raising=False)


def plain_cli_text(text: str) -> str:
    """Return a styling-free view of CLI output with its boundaries intact.

    Removes SGR colour attributes and OSC-8 hyperlink wrappers, then trims the
    trailing padding rich emits at the end of a line.  Newlines — and the
    separate records they delimit — are preserved: joining them would let an
    assertion match a phrase built from two messages the CLI printed
    independently.

    Args:
        text: Raw captured CLI output.

    Returns:
        The same text with colour and hyperlink wrappers removed.

    Raises:
        AssertionError: if ``text`` contains a control sequence that alters the
            display (cursor movement, line erase, carriage-return overwrite).
            Stripping those would make the result claim the user saw characters
            that were erased or overwritten, so the caller has to handle them
            explicitly — normally by asserting against the raw output, or by
            rendering without whatever emitted them.
    """
    stripped = _SGR.sub("", _OSC8.sub("", text))

    leftover = _DISPLAY_ALTERING_ESCAPE.search(stripped)
    if leftover is not None:
        raise AssertionError(
            f"refusing to normalize CLI output containing the display-altering "
            f"escape sequence {leftover.group()!r} at offset {leftover.start()}: "
            "it changes which characters stay on screen, so dropping it would "
            "let an assertion match text the user never saw. Assert against the "
            "raw output, or render without the sequence."
        )

    stripped = stripped.replace("\r\n", "\n")
    if "\r" in stripped:
        raise AssertionError(
            "refusing to normalize CLI output containing a carriage return: it "
            "overwrites the line, so the text before it was never displayed. "
            "Keeping it would let an assertion match the overwritten value. "
            "Assert against the raw output instead."
        )

    return "\n".join(line.rstrip() for line in stripped.split("\n")).strip()


@pytest.fixture
def plain() -> Callable[[str], str]:
    """Normalize captured CLI output before a substring assertion."""
    return plain_cli_text
