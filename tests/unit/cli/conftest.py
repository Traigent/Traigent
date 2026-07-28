"""Shared helpers for the Traigent CLI unit tests (Traigent#2052).

The CLI renders through ``rich``, which paints its output with ANSI colour and
soft-wraps it to the ambient terminal width.  A raw ``"phrase" in
result.output`` assertion therefore does not test what the CLI *said* — it
tests how the CLI happened to be *painted* on the machine that ran pytest.  The
same assertion flips to red when colour is forced on, or when ``COLUMNS`` is
narrow enough to wrap the phrase.

``plain_cli_text`` gives those assertions a rendering-independent view of the
same text so that the message stays the subject and the styling drops out.  It
is deliberately assertion-side: unlike ``NO_COLOR``/``TERM=dumb``, it holds
with colour both enabled and disabled, and it cannot be defeated by how pytest
was launched (the CLI modules build their ``Console`` at import time, so any
environment tweak made from inside a test arrives too late).

What it does **not** cover: below ``COLUMNS`` ≈ 20 rich breaks words mid-token,
which no normalizer can undo.  Structured output (the onboarding
``PLAN_JSON_BEGIN``/``END`` frame) is deliberately parsed from raw output so
control-code corruption there stays visible.
"""

from __future__ import annotations

import re
from collections.abc import Callable

import pytest

# OSC-8 hyperlinks: ``ESC ] 8 ; params ; uri`` terminated by BEL or ST.
_OSC8 = re.compile(r"\x1b\]8;[^\x07\x1b]*(?:\x07|\x1b\\)")
# CSI sequences — SGR colour/bold/underline and cursor control.
_CSI = re.compile(r"\x1b\[[0-9;:?]*[ -/]*[@-~]")


def plain_cli_text(text: str) -> str:
    """Return an ANSI- and wrap-independent view of CLI output.

    Strips OSC-8 hyperlink wrappers and CSI escape sequences, then collapses
    every run of whitespace (including the newlines rich inserts when it
    soft-wraps) to a single space.  Substring assertions over the result pin
    the words the CLI emitted, not their rendering.
    """
    return re.sub(r"\s+", " ", _CSI.sub("", _OSC8.sub("", text))).strip()


@pytest.fixture
def plain() -> Callable[[str], str]:
    """Normalize captured CLI output before a substring assertion."""
    return plain_cli_text
