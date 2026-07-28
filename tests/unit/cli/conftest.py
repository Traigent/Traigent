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
* **Styling is stripped.**  ``plain_cli_text`` removes OSC-8 hyperlink
  wrappers and the SGR attributes that only decide how a character is
  *painted*.  Neither contributes a character the user reads, so removing them
  cannot change what the output *says*.  "SGR" is not a synonym for
  "presentational", though — ``ESC[8m`` is CONCEAL, and ``ESC[30;40m`` paints
  black on black — so the parameters are parsed rather than blanket-stripped,
  and a rendition that can hide its text is refused like any other
  display-altering control.

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

The same fabrication is reachable without an ESC at all — ``\b\b\b`` backspaces
over the digits of a status code, and the pre-backspace value survives
normalization — and the mirror-image failure, *masking*, is reachable through
controls that have no glyph at all: a stray ``BEL`` inside a forbidden banner
leaves the banner fully legible on screen while breaking the ``not in``
substring match that is supposed to catch it.

False greens are exactly the failure mode this package exists to prevent, so
``plain_cli_text`` preserves every logical boundary and *refuses* to normalize
output containing anything that is not either a literal character or one of the
explicitly allow-listed styling forms.  Refusal, not removal, is the answer for
every control it does not positively recognise as paint-only; an unrecognised
attribute might be the next ``ESC[8m``.  Each case is pinned by a test in
``test_output_normalization.py``.

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

# SGR — ``ESC [ params m``.  Matched as a unit so every one of them is
# *parsed*; which ones may then be dropped is decided by ``_strip_sgr``.
_SGR = re.compile(r"\x1b\[([0-9;:]*)m")

# SGR parameters that change how a character is painted but never whether it is
# painted: reset, the intensity/italic/underline/blink/strike/overline family
# and their off-switches, and the foreground/background colour selectors.
# Deliberately an allow-list — an SGR parameter that is not on it is refused
# rather than assumed harmless, because the interesting ones are not.
_PAINT_ONLY_SGR = frozenset(
    {0, 1, 2, 3, 4, 5, 6, 9, 21, 22, 23, 24, 25, 27, 28, 29, 53, 55, 58, 59}
    | set(range(30, 40))  # 8-colour foreground + default (39)
    | set(range(40, 50))  # 8-colour background + default (49)
    | set(range(90, 98))  # bright foreground
    | set(range(100, 108))  # bright background
)

# SGR parameters that remove the text from the screen.  ``8`` is CONCEAL: the
# terminal displays nothing at all, so stripping it fabricates every character
# it covered.  ``7`` swaps foreground and background, which hides the text
# whenever the pair it swaps into is the one the terminal is already painting.
_HIDING_SGR = {
    7: "reverse video (SGR 7)",
    8: "conceal (SGR 8)",
}

#: Extended-colour selectors: ``38``/``48``/``58`` followed by ``5;<index>`` or
#: ``2;<r>;<g>;<b>``.
_EXTENDED_COLOUR = frozenset({38, 48, 58})

# Everything else introduced by ESC: CSI cursor movement/erase, OSC window
# titles, and the two-character Fe escapes.  These *change what is on screen*,
# so they are not strippable — see ``plain_cli_text``.
_DISPLAY_ALTERING_ESCAPE = re.compile(
    r"\x1b(?:\[[0-9;:?]*[ -/]*[@-~]|\][^\x07\x1b]*(?:\x07|\x1b\\)|[@-Z\\-_])"
)

# C0 (``\x00``-``\x1f``) and C1 (``\x80``-``\x9f``) controls that survive the
# styling pass.  ``\n`` is a real record boundary and ``\t`` is horizontal
# whitespace the reader sees; every other one either rewrites the display
# (``\b`` backspaces over what was printed, ``\x9b`` is an eight-bit CSI) or
# occupies no columns at all (``\x07`` BEL), and both of those break the
# correspondence between the normalized string and the screen.  ``\r`` is
# excluded here only so the carriage-return check can report it by name.
_RESIDUAL_CONTROL = re.compile(r"[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f\x80-\x9f]")


def _sgr_colour(params: list[int], index: int) -> tuple[object, int]:
    """Read the colour selected at ``params[index]``.

    Returns the colour as a comparable token and the index just past it.
    ``30`` (black text) and ``40`` (black background) both yield palette index
    ``0``, so a foreground and a background can be compared for equality across
    the two parameter ranges.

    Raises:
        ValueError: if an extended-colour selector is truncated or malformed.
    """
    param = params[index]
    if param in _EXTENDED_COLOUR:
        rest = params[index + 1 :]
        if rest[:1] == [5] and len(rest) >= 2:
            return ("palette", rest[1]), index + 3
        if rest[:1] == [2] and len(rest) >= 4:
            return ("rgb", tuple(rest[1:4])), index + 5
        raise ValueError(f"malformed extended colour parameter {param}")
    if param in (39, 49):  # back to the terminal default
        return None, index + 1
    if 30 <= param <= 37:
        return ("palette", param - 30), index + 1
    if 40 <= param <= 47:
        return ("palette", param - 40), index + 1
    if 90 <= param <= 97:
        return ("palette", param - 82), index + 1
    return ("palette", param - 92), index + 1  # 100-107


def _strip_sgr(text: str) -> str:
    """Remove the paint-only SGR sequences from ``text``.

    Raises:
        AssertionError: if any SGR sequence selects a rendition under which the
            text it applies to would not be readable on screen — conceal,
            reverse video, or a foreground equal to the background — or uses a
            parameter this function cannot prove is paint-only.
    """
    out: list[str] = []
    end = 0
    foreground: object = None
    background: object = None

    for match in _SGR.finditer(text):
        out.append(text[end : match.start()])
        end = match.end()
        body = match.group(1)

        if ":" in body:
            raise AssertionError(
                f"refusing to normalize CLI output containing the colon-delimited "
                f"SGR sequence {match.group()!r} at offset {match.start()}: its "
                "sub-parameters are not parsed here, so whether it hides the text "
                "it applies to cannot be established. Assert against the raw "
                "output instead."
            )

        # ``ESC[m`` and an empty parameter both mean 0 (reset).
        params = [int(part) if part else 0 for part in body.split(";")]

        index = 0
        while index < len(params):
            param = params[index]
            if param in _HIDING_SGR:
                raise AssertionError(
                    f"refusing to normalize CLI output containing "
                    f"{_HIDING_SGR[param]} at offset {match.start()}: it hides the "
                    "text it applies to, so dropping it would let an assertion "
                    "match characters the user never saw. Assert against the raw "
                    "output, or render without the sequence."
                )
            if param not in _PAINT_ONLY_SGR:
                raise AssertionError(
                    f"refusing to normalize CLI output containing the unrecognised "
                    f"SGR parameter {param} at offset {match.start()}: it is not on "
                    "the allow-list of attributes that only change how characters "
                    "are painted, so it cannot be dropped without risking a match "
                    "on text the user never saw."
                )

            if param == 0:
                foreground = background = None
                index += 1
            elif param in _EXTENDED_COLOUR or 30 <= param <= 49 or param >= 90:
                try:
                    colour, index = _sgr_colour(params, index)
                except ValueError as exc:
                    raise AssertionError(
                        f"refusing to normalize CLI output containing "
                        f"{match.group()!r} at offset {match.start()}: {exc}, so the "
                        "rendition it selects cannot be established."
                    ) from exc
                if param == 58:  # underline colour — never paints the glyph
                    pass
                elif 30 <= param <= 39 or 90 <= param <= 97:
                    foreground = colour
                else:
                    background = colour
            else:
                index += 1

        if foreground is not None and foreground == background:
            raise AssertionError(
                f"refusing to normalize CLI output containing {match.group()!r} at "
                f"offset {match.start()}: it paints the foreground and the "
                "background the same colour, so the text it applies to is "
                "invisible and dropping it would let an assertion match characters "
                "the user never saw."
            )

    out.append(text[end:])
    return "".join(out)


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

    Removes OSC-8 hyperlink wrappers and the paint-only SGR attributes, then
    trims the trailing padding rich emits at the end of a line.  Newlines — and
    the separate records they delimit — are preserved: joining them would let
    an assertion match a phrase built from two messages the CLI printed
    independently.

    What is left after that pass must be text the user actually read, so
    anything still in it that is *not* a printable character is refused rather
    than dropped.  The allow-list is deliberately the whole of the policy: a
    control this function has no rule for is a control whose effect on the
    screen it cannot model, and silently deleting it is precisely how a
    normalizer manufactures a green.

    Args:
        text: Raw captured CLI output.

    Returns:
        The same text with hyperlink wrappers and paint-only styling removed.

    Raises:
        AssertionError: if ``text`` contains anything that alters or hides the
            display — an SGR rendition the text cannot be read under (conceal,
            reverse video, foreground painted onto the same background), cursor
            movement, a line erase, a carriage-return overwrite, or any other
            residual C0/C1 control such as a backspace or a BEL.  Stripping
            those would make the result claim the user saw characters that were
            erased, overwritten or hidden — or, for the zero-width ones, break
            a ``not in`` assertion over text that was displayed in full — so
            the caller has to handle them explicitly, normally by asserting
            against the raw output or by rendering without whatever emitted
            them.
    """
    stripped = _strip_sgr(_OSC8.sub("", text))

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

    residual = _RESIDUAL_CONTROL.search(stripped)
    if residual is not None:
        raise AssertionError(
            f"refusing to normalize CLI output containing the control character "
            f"{residual.group()!r} at offset {residual.start()}: it is neither a "
            "character the user reads nor styling this normalizer can account "
            "for. Controls like these either rewrite the display — a backspace "
            "erases the character before it, an eight-bit CSI drives the cursor "
            "— or occupy no columns at all, so keeping one breaks a substring "
            "match against text that *was* displayed and dropping one invents a "
            "match against text that was not. Assert against the raw output "
            "instead."
        )

    return "\n".join(line.rstrip() for line in stripped.split("\n")).strip()


@pytest.fixture
def plain() -> Callable[[str], str]:
    """Normalize captured CLI output before a substring assertion."""
    return plain_cli_text
