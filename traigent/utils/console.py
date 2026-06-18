"""Safe console output utilities for cross-platform encoding compatibility.

On Windows with a non-UTF-8 console (e.g. cp1252, cp850), ``print()`` calls
that include characters outside the console's encoding raise
``UnicodeEncodeError``.  This module provides two helpers:

* ``_safe_print`` – a drop-in replacement for ``print()`` that catches
  ``UnicodeEncodeError`` and falls back to ASCII by replacing unencodable
  characters with ``?``.
* ``configure_stdout_encoding`` – wraps ``sys.stdout`` with UTF-8 and
  ``errors='replace'`` at the CLI entry point so that all output — including
  Rich ``Console`` output — is safe.

Usage at the CLI entry point::

    from traigent.utils.console import configure_stdout_encoding
    configure_stdout_encoding()

Usage in progress / diagnostic output::

    from traigent.utils.console import _safe_print as print
"""

# Traceability: CONC-Layer-Core CONC-Quality-Usability FUNC-API-ENTRY

from __future__ import annotations

import io
import sys
from typing import Any


def _safe_print(
    *args: Any, sep: str = " ", end: str = "\n", flush: bool = False
) -> None:
    """Print to stdout, replacing unencodable characters instead of crashing.

    This is a drop-in replacement for the built-in ``print()`` that gracefully
    handles ``UnicodeEncodeError`` on Windows consoles with non-UTF-8 encodings
    (e.g. cp1252, cp850).  Unencodable characters are replaced with ``?``.

    Args:
        *args: Objects to print (converted to str).
        sep: String inserted between objects (default ``" "``).
        end: String appended after the last object (default ``"\\n"``).
        flush: Whether to flush the output buffer.
    """
    text = sep.join(str(a) for a in args) + end
    stream = sys.stdout
    try:
        stream.write(text)
    except UnicodeEncodeError:
        encoding = getattr(stream, "encoding", "ascii") or "ascii"
        safe_text = text.encode(encoding, errors="replace").decode(encoding)
        stream.write(safe_text)
    if flush:
        stream.flush()


def configure_stdout_encoding() -> None:
    """Wrap ``sys.stdout`` with UTF-8 encoding and ``errors='replace'``.

    Call this once at the CLI entry point.  When ``sys.stdout`` already uses
    UTF-8 (most Linux/macOS terminals and Windows 10+ with UTF-8 mode) this
    is a no-op.  On Windows consoles using legacy encodings (cp1252, cp850,
    etc.) it replaces unencodable characters with ``?`` instead of crashing.

    This also covers Rich ``Console`` objects that write to the default
    ``sys.stdout`` because they share the same underlying stream.

    Note: Only applies when ``sys.stdout`` has a ``buffer`` attribute (i.e. it
    is a real text-mode stream, not a StringIO or similar replacement).
    """
    stdout = sys.stdout
    if not hasattr(stdout, "buffer"):
        return  # Already wrapped or replaced (e.g. in tests)

    current_encoding = getattr(stdout, "encoding", None) or ""
    if current_encoding.lower().replace("-", "") in ("utf8", "utf-8"):
        return  # Already UTF-8; nothing to do

    sys.stdout = io.TextIOWrapper(
        stdout.buffer,
        encoding="utf-8",
        errors="replace",
        line_buffering=getattr(stdout, "line_buffering", True),
    )
