"""Pin the CLI output normalizer itself (Traigent#2052).

``plain_cli_text`` plus the ``_pin_cli_render_width`` fixture are what make the
substring assertions in this package independent of ambient rendering, so they
need their own guard: without one, a future edit could quietly weaken them back
to raw-substring matching and the tests they protect would go green on the
author's machine anyway.

The guards come in two directions, because a normalizer can fail both ways:

* **Fabrication** — normalizing text into a phrase the CLI never printed, so a
  positive assertion passes on output the user never saw.  Three concrete
  counterexamples are pinned below: a line-erase sequence, a newline between
  two independent records, and a carriage-return overwrite.  All three
  fabricated a match under the collapse-all-whitespace normalizer this file
  replaced.
* **Masking** — rendering that hides a phrase the CLI *did* print, so a
  negative assertion (``not in``) passes vacuously while the forbidden text was
  emitted.  Soft-wrapping does this, and no normalizer can undo it; the guard
  is that the render width is pinned wide enough for the fold never to happen.

Rendered samples go through a real ``rich.Console`` that is *forced* to be
coloured and/or narrow, so they do not depend on the terminal pytest was
launched from.
"""

from __future__ import annotations

import io
from collections.abc import Callable

import pytest
from rich.console import Console

from tests.unit.cli.conftest import CLI_RENDER_WIDTH

# The hardest real assertion in the suite: rich colours the retry count, which
# splits the phrase with SGR codes mid-sentence.
_PHRASE = "failed after 3 consecutive transport errors"
_MARKUP = (
    "[red]failed after [bold]3[/bold] consecutive transport errors. "
    "Please check your network connection and try again.[/red]"
)

# The device-auth banner, the one place this package asserts a *negative*: the
# banner must name the transport host and must not name the default one.  Long
# enough that a narrow terminal folds it mid-token, which is what makes the
# masking direction reachable.
_BANNER_HOST = "https://api.example.test/some/longer/custom/path/v1"
_BANNER = f"Authenticating with: {_BANNER_HOST}"


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


# --- what the normalizer legitimately removes -----------------------------------


def test_colour_defeats_raw_substring_matching(plain: Callable[[str], str]) -> None:
    """Colour alone splits the phrase — pins the SGR-stripping half."""
    rendered = _render(_MARKUP, width=CLI_RENDER_WIDTH, color=True)

    assert "\x1b[" in rendered, "sample must actually be coloured"
    assert _PHRASE not in rendered
    assert _PHRASE in plain(rendered)


def test_osc8_hyperlinks_are_stripped(plain: Callable[[str], str]) -> None:
    """rich emits OSC-8 wrappers around URLs under ``force_terminal``."""
    linked = "\x1b]8;id=1;https://traigent.ai\x07Traigent docs\x1b]8;;\x07"
    assert plain(linked) == "Traigent docs"


def test_plain_output_passes_through_unchanged(plain: Callable[[str], str]) -> None:
    """Identity on text that was never styled — the normalizer adds no meaning."""
    assert plain("HTTP status: 401") == "HTTP status: 401"
    assert plain("  Backend endpoint mismatch\n") == "Backend endpoint mismatch"
    assert plain("first record\nsecond record") == "first record\nsecond record"


# --- fabrication: the normalizer must not invent a match ------------------------


def test_erased_text_is_not_resurrected(plain: Callable[[str], str]) -> None:
    """``ESC[2K`` wipes the line, so the erased word was never displayed.

    Stripping the sequence as if it were styling yields ``Backend endpoint
    mismatch`` — a phrase the user never saw — and every ``Backend endpoint
    mismatch`` assertion in this package would accept it.  Refuse instead.
    """
    erased = "Backend \x1b[2Kendpoint mismatch"

    with pytest.raises(AssertionError, match="display-altering"):
        plain(erased)


def test_newline_does_not_join_independent_records(
    plain: Callable[[str], str],
) -> None:
    """Two ``Console.print`` calls are two records, not one phrase.

    Collapsing the newline to a space manufactures the exact banner the
    device-auth test forbids out of two messages that were printed
    independently, turning that ``not in`` into a false failure and every
    ``in`` assertion in this package into something two unrelated lines can
    satisfy.
    """
    two_records = "Authenticating with:\nhttp://localhost:5000"

    normalized = plain(two_records)

    assert "Authenticating with: http://localhost:5000" not in normalized
    assert "Authenticating with:" in normalized
    assert "http://localhost:5000" in normalized


def test_overwritten_text_is_not_preserved(plain: Callable[[str], str]) -> None:
    """A carriage return replaces the line; the old value was never displayed.

    ``HTTP status: 500`` followed by a carriage return and ``HTTP status: 200``
    shows the user only ``200``.  Treating the carriage return as a space keeps
    the 500 in the normalized text, so the status-classification assertions
    would accept a status the CLI overwrote.
    """
    overwritten = "HTTP status: 500\rHTTP status: 200"

    with pytest.raises(AssertionError, match="carriage return"):
        plain(overwritten)


def test_windows_line_endings_are_not_treated_as_an_overwrite(
    plain: Callable[[str], str],
) -> None:
    """CRLF is a line boundary, not an overwrite — normalize, do not refuse."""
    assert plain("first record\r\nsecond record") == "first record\nsecond record"


# --- masking: rendering must not hide a phrase the CLI did print ----------------


def test_soft_wrapping_is_not_undone(plain: Callable[[str], str]) -> None:
    """The normalizer does not pretend a wrapped phrase was contiguous.

    Undoing the fold is indistinguishable from the fabrication above — rich's
    soft-wrap newline looks exactly like a real record boundary — so
    ``plain_cli_text`` leaves it alone.  A wrapped phrase therefore fails its
    assertion loudly rather than passing on reconstructed text; the width pin,
    not the normalizer, is what keeps wrapping from happening.
    """
    rendered = _render(_MARKUP, width=40, color=False)

    assert "\n" in rendered.rstrip("\n"), "sample must actually be wrapped"
    assert _PHRASE not in plain(rendered)


def test_narrow_width_masks_a_forbidden_phrase(plain: Callable[[str], str]) -> None:
    """Why the width has to be pinned: a fold hides text that *was* printed.

    At 40 columns rich breaks the banner mid-token, so ``_BANNER not in
    plain(...)`` is vacuously true even though the CLI emitted the banner in
    full.  A negative assertion evaluated against this render is worthless —
    this test documents the hazard the pin removes.
    """
    rendered = _render(_BANNER, width=40, color=False)

    assert _BANNER not in plain(rendered), "sample must actually be masked"


def test_pinned_width_keeps_the_banner_detectable(plain: Callable[[str], str]) -> None:
    """At :data:`CLI_RENDER_WIDTH` the same banner survives whole.

    So a ``not in`` assertion over it means what it says.  Shrink
    ``CLI_RENDER_WIDTH`` below the longest asserted message and this fails.
    """
    rendered = _render(_BANNER, width=CLI_RENDER_WIDTH, color=True)

    assert "\n" not in rendered.rstrip("\n"), "sample must not be wrapped"
    assert _BANNER in plain(rendered)


def test_fixture_pins_the_width_of_an_ambient_console() -> None:
    """The pin has to reach consoles built at CLI import time, not just new ones.

    ``traigent.cli.*`` construct their ``Console()`` when the module is first
    imported — long before any fixture runs — so the pin only works because
    rich re-reads ``COLUMNS`` from the live environment on every render.  This
    builds a console the same way (no explicit width, no ``_environ``
    snapshot) and checks the autouse fixture actually governs it.
    """
    console = Console(file=io.StringIO(), force_terminal=True, legacy_windows=False)

    assert console.width == CLI_RENDER_WIDTH
