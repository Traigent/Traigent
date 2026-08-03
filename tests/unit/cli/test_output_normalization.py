"""Pin the CLI output normalizer itself (Traigent#2052).

``plain_cli_text`` plus the ``_pin_cli_render_width`` fixture are what make the
substring assertions in this package independent of ambient rendering, so they
need their own guard: without one, a future edit could quietly weaken them back
to raw-substring matching and the tests they protect would go green on the
author's machine anyway.

The guards come in two directions, because a normalizer can fail both ways:

* **Fabrication** — normalizing text into a phrase the CLI never printed, so a
  positive assertion passes on output the user never saw.  The counterexamples
  pinned below are a line-erase sequence, a newline between two independent
  records, a carriage-return overwrite, a backspace overwrite, and the SGR
  attributes that paint text invisibly (conceal, reverse video, a foreground
  equal to its background).  The first three fabricated a match under the
  collapse-all-whitespace normalizer this file replaced; the rest fabricated
  one under the strip-every-``ESC[...m`` normalizer that succeeded it.
* **Masking** — rendering that hides a phrase the CLI *did* print, so a
  negative assertion (``not in``) passes vacuously while the forbidden text was
  emitted.  Soft-wrapping does this, and no normalizer can undo it; the guard
  is that the render width is pinned wide enough for the fold never to happen.
  A retained zero-width control does it too — a BEL inside a banner leaves the
  banner legible while breaking the substring match — which is why controls the
  normalizer cannot account for are refused rather than kept.

A guard that refuses everything would satisfy both directions and protect
nothing, so ``test_ordinary_styling_still_passes_through`` and the 262 real
assertions in this package are the other half of the pin.

Rendered samples go through a real ``rich.Console`` that is *forced* to be
coloured and/or narrow, so they do not depend on the terminal pytest was
launched from.
"""

from __future__ import annotations

import io
from collections.abc import Callable

import pytest
from rich.console import Console

# Imported at module scope *on purpose*: this is the real CLI module whose
# ``Console(width=120)`` is built at import time, and the import has to happen
# during collection — before any fixture runs — for the width-repinning guard
# below to be testing what it claims to.
from traigent.cli import plan_command

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


def test_concealed_text_is_not_revealed(plain: Callable[[str], str]) -> None:
    """``ESC[8m`` is CONCEAL — an SGR sequence that displays nothing at all.

    This is why the normalizer parses SGR parameters instead of stripping
    every ``ESC[...m``.  "SGR" and "presentational" are not synonyms: strip
    this one as if it were colour and the status-classification assertions
    accept a ``500`` the terminal painted invisibly.
    """
    concealed = "HTTP status: \x1b[8m500\x1b[0m"

    with pytest.raises(AssertionError, match="conceal"):
        plain(concealed)


def test_reverse_video_is_refused(plain: Callable[[str], str]) -> None:
    """``ESC[7m`` swaps foreground and background, which can hide the text.

    Whether it does depends on the pair it swaps into — which the captured
    bytes do not record — so the honest answer is to refuse rather than to
    assume the text stayed legible.
    """
    reversed_video = "HTTP status: \x1b[7m500\x1b[0m"

    with pytest.raises(AssertionError, match="reverse video"):
        plain(reversed_video)


def test_text_painted_onto_its_own_background_is_refused(
    plain: Callable[[str], str],
) -> None:
    """Conceal is not the only way SGR hides a glyph — black on black does too.

    ``30`` and ``40`` select the same palette entry for the foreground and the
    background, so the digits occupy their columns and display nothing.  The
    check is on the resulting rendition, not on any one parameter, so it holds
    when the two halves arrive in separate sequences or as 256-colour indices.
    """
    for invisible in (
        "HTTP status: \x1b[30;40m500\x1b[0m",
        "HTTP status: \x1b[31m\x1b[41m500\x1b[0m",
        "HTTP status: \x1b[38;5;9m\x1b[48;5;9m500\x1b[0m",
    ):
        with pytest.raises(AssertionError, match="same colour"):
            plain(invisible)


def test_unrecognised_sgr_parameter_is_refused(plain: Callable[[str], str]) -> None:
    """The SGR allow-list is closed: an unknown attribute might be the next 8.

    ``ESC[8m`` was strippable for exactly as long as nobody asked what ``8``
    meant.  Parameters this normalizer cannot prove are paint-only are refused
    rather than assumed harmless.
    """
    with pytest.raises(AssertionError, match="unrecognised SGR parameter"):
        plain("HTTP status: \x1b[76m500\x1b[0m")


def test_backspace_overwrite_is_not_preserved(plain: Callable[[str], str]) -> None:
    """Backspacing rewrites the display without any ESC being involved.

    Three backspaces put the cursor back over ``500`` and ``200`` paints over
    it, so the user saw ``HTTP status: 200``.  Keeping the raw bytes leaves the
    ``500`` sitting in the normalized string where a substring assertion can
    still match it — the carriage-return fabrication, reached through a
    control the ESC-based checks never see.
    """
    overwritten = "HTTP status: 500\b\b\b200"

    with pytest.raises(AssertionError, match="control character"):
        plain(overwritten)


def test_zero_width_control_cannot_mask_a_forbidden_phrase(
    plain: Callable[[str], str],
) -> None:
    """BEL occupies no columns, so it hides a banner from a ``not in`` check.

    This is the masking direction rather than the fabrication one: the user
    reads ``Authenticating with: http://localhost:5000`` in full, but the
    retained ``\\x07`` sits inside the phrase and the device-auth test's
    negative assertion passes on output that contains exactly what it forbids.
    """
    belled = "Authenticating with:\x07 http://localhost:5000"

    with pytest.raises(AssertionError, match="control character"):
        plain(belled)


def test_eight_bit_csi_is_refused(plain: Callable[[str], str]) -> None:
    """``\\x9b`` is CSI as a single C1 byte — the same erase, no ESC prefix.

    The seven-bit form is already refused; matching on ``ESC[`` alone would let
    the identical sequence through in its eight-bit spelling.
    """
    erased = "Backend \x9b2Kendpoint mismatch"

    with pytest.raises(AssertionError, match="control character"):
        plain(erased)


def test_unterminated_escape_sequence_is_refused(plain: Callable[[str], str]) -> None:
    """A truncated sequence is not a harmless prefix — refuse the whole string.

    An OSC with no BEL or ST swallows everything after it until the terminal
    finds a terminator, so the text following it was never displayed as text.
    A bare trailing ESC is the same problem in miniature.
    """
    for truncated in (
        "Backend \x1b]0;window titleendpoint mismatch",
        "Backend endpoint mismatch\x1b",
        "Backend \x1b[endpoint mismatch",
    ):
        with pytest.raises(AssertionError):
            plain(truncated)


def test_ordinary_styling_still_passes_through(plain: Callable[[str], str]) -> None:
    """Refusing must not cost the normalizer its actual job.

    The tightened checks would be worthless if they also rejected the colour,
    bold and hyperlink wrappers that every real CLI render is full of — a
    normalizer that refuses everything protects nothing.
    """
    assert plain("\x1b[31mfailed after \x1b[1m3\x1b[0m errors\x1b[0m") == (
        "failed after 3 errors"
    )
    assert plain("\x1b[38;2;1;2;3m\x1b[48;2;9;9;9mcontrasting\x1b[0m") == "contrasting"
    assert plain("\x1b[4munderlined\x1b[24m") == "underlined"


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


def test_columns_governs_a_console_that_did_not_fix_its_width() -> None:
    """``COLUMNS`` is what pins the width-less consoles the CLI mostly uses.

    Most ``traigent.cli.*`` modules build a bare ``Console()``.  rich leaves
    such an instance's ``_width`` unset and re-reads ``COLUMNS`` from the live
    environment on every render, so exporting it in the fixture reaches them
    even though they were constructed at import time.  This builds one the same
    way (no explicit width, no ``_environ`` snapshot) and checks the export
    lands.

    It is *only* a guard on the export: an instance with no fixed width would
    report :data:`CLI_RENDER_WIDTH` whether or not the fixture re-pinned it.
    The three consoles that do fix their width are covered by the next test.
    """
    console = Console(file=io.StringIO(), force_terminal=True, legacy_windows=False)

    assert console.width == CLI_RENDER_WIDTH


def test_fixture_repins_a_console_built_at_cli_import_time() -> None:
    """``COLUMNS`` alone does not reach a console constructed with a width.

    ``traigent.cli.plan_command`` builds ``Console(width=120)`` when the module is
    first imported — during collection here, long before any fixture runs.
    rich stores that on the instance and never consults ``COLUMNS`` again, so
    the environment export is powerless over it and the fixture has to
    overwrite ``_width`` directly.  This asserts against the real module-level
    object, so deleting that loop from the fixture fails this test at 120.
    """
    console = plan_command.console

    assert console.width == CLI_RENDER_WIDTH, (
        "the CLI's import-time console is still rendering at its own width; "
        "assertions against its output are rendering-dependent again"
    )
