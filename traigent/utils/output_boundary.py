"""Contain everything user code and the SDK print while a command runs it.

# Traceability: CONC-Layer-Data FUNC-STORAGE FUNC-ANALYTICS

``capture_output()`` is a context manager that, for the duration of the
``with`` block, collects instead of emitting:

* ``sys.stdout`` and ``sys.stderr`` writes (Python level);
* log records from every logger, including the SDK's (the handlers of every
  existing logger are swapped for one collector and put back afterwards;
  logger levels are never touched);
* warnings raised through the ``warnings`` module (the filters stay as they
  are; only where a shown warning goes changes);
* writes to file descriptors 1 and 2 made below Python (child processes, C
  extensions), by pointing both descriptors at a temporary file with
  ``dup2`` and putting them back in a ``finally``.

Everything is restored on every exit path, including exceptions and
``SystemExit``. The collected text can hold example content (an SDK warning
may quote a metric's exception, a function may print its inputs), so callers
report only counts unless the user opts in to seeing it, and then on stderr.

Limits: a process that ends itself with ``os._exit`` cannot be caught or
contained; output written from another process that outlives the block is not
contained; the descriptor-level capture is skipped (and ``fd_capture`` is
``False``) when descriptors 1 or 2 are not usable, and in that case
descriptor-level writes are not contained.
"""

from __future__ import annotations

import io
import logging
import os
import sys
import tempfile
import warnings
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import IO


class _Collector(logging.Handler):
    def __init__(self) -> None:
        super().__init__(level=logging.NOTSET)
        self.lines: list[str] = []
        self._formatter = logging.Formatter("%(levelname)s %(name)s: %(message)s")

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self.lines.extend(self._formatter.format(record).splitlines() or [""])
        except Exception:  # noqa: BLE001 - a bad record must not escape the boundary
            self.lines.append("(a log record could not be formatted)")


@dataclass
class CapturedOutput:
    """What one ``capture_output()`` window collected."""

    stdout: io.StringIO = field(default_factory=io.StringIO)
    stderr: io.StringIO = field(default_factory=io.StringIO)
    log_lines: list[str] = field(default_factory=list)
    warning_lines: list[str] = field(default_factory=list)
    fd_text: str = ""
    fd_capture: bool = False

    def _blocks(self) -> list[tuple[str, str]]:
        return [
            ("stdout", self.stdout.getvalue()),
            ("stderr", self.stderr.getvalue()),
            ("descriptors 1 and 2", self.fd_text),
            ("log records", "\n".join(self.log_lines)),
            ("warnings", "\n".join(self.warning_lines)),
        ]

    @property
    def line_count(self) -> int:
        return sum(len(text.splitlines()) for _, text in self._blocks())

    def text(self) -> str:
        """The collected text, one labelled block per source (may hold content)."""
        return "\n".join(
            f"--- {label} ---\n{text.rstrip(chr(10))}"
            for label, text in self._blocks()
            if text.strip()
        )


def _flush(stream: object) -> None:
    flush = getattr(stream, "flush", None)
    if callable(flush):
        try:
            flush()
        except Exception:  # noqa: BLE001 - a closed or odd stream is not our problem
            pass


def _flush_c_stdio() -> None:
    """Flush the C library's stdio buffers (``printf`` in an extension)."""
    try:
        import ctypes

        ctypes.CDLL(None).fflush(None)
    except Exception:  # noqa: BLE001 - no libc through ctypes on this platform
        pass


def _flush_everything() -> None:
    """Flush every writer that could still hold captured bytes.

    Called while descriptors 1 and 2 still point at the capture file, so
    buffered text lands in it and not on the real output after the window.
    """
    for stream in (sys.stdout, sys.stderr, sys.__stdout__, sys.__stderr__):
        _flush(stream)
    _flush_c_stdio()


@contextmanager
def capture_output() -> Iterator[CapturedOutput]:
    """Collect all output of the ``with`` block; restore everything on exit."""
    captured = CapturedOutput()
    collector = _Collector()
    loggers: list[logging.Logger] = [logging.getLogger()]
    loggers += [
        logger
        for logger in list(logging.Logger.manager.loggerDict.values())
        if isinstance(logger, logging.Logger)
    ]
    saved_handlers = [(logger, logger.handlers) for logger in loggers]
    saved_streams = (sys.stdout, sys.stderr)
    fd_state: tuple[int, int, IO[bytes]] | None = None

    _flush_everything()  # earlier output must not land in the capture file
    try:
        fd_state = _redirect_descriptors()
    except OSError:
        fd_state = None
    captured.fd_capture = fd_state is not None
    recorded: list[warnings.WarningMessage] = []
    try:
        with warnings.catch_warnings(record=True) as recorded_warnings:
            if recorded_warnings is not None:
                recorded = recorded_warnings
            for logger in loggers:
                logger.handlers = (
                    [collector] if logger is loggers[0] or not logger.propagate else []
                )
            sys.stdout, sys.stderr = captured.stdout, captured.stderr
            try:
                yield captured
            finally:
                sys.stdout, sys.stderr = saved_streams
                for logger, handlers in saved_handlers:
                    logger.handlers = handlers
    finally:
        _finish(captured, collector, recorded, fd_state)


def _finish(
    captured: CapturedOutput,
    collector: _Collector,
    recorded: list[warnings.WarningMessage],
    fd_state: tuple[int, int, IO[bytes]] | None,
) -> None:
    """Close the window. Each step is a ``finally`` of the one before it.

    Descriptors 1 and 2 come first and are put back whatever goes wrong while
    flushing; formatting what was captured happens only after they are back,
    and a failure there becomes a content-free note, never an escape.
    """
    try:
        if fd_state is not None:
            try:
                _flush_everything()
            finally:
                captured.fd_text = _restore_descriptors(fd_state)
    finally:
        captured.log_lines = list(collector.lines)
        captured.warning_lines = _format_warnings(recorded)


def _format_warnings(recorded: list[warnings.WarningMessage]) -> list[str]:
    lines: list[str] = []
    for item in recorded:
        try:
            lines.extend(
                line
                for line in warnings.formatwarning(
                    item.message, item.category, item.filename, item.lineno
                ).splitlines()
                if line.strip()
            )
        except Exception:  # noqa: BLE001 - a user formatter must not escape the boundary
            lines.append("(a warning could not be formatted)")
    return lines


def _redirect_descriptors() -> tuple[int, int, IO[bytes]]:
    """Point descriptors 1 and 2 at a temporary file; return what restores them."""
    saved_out = os.dup(1)
    try:
        saved_err = os.dup(2)
    except OSError:
        os.close(saved_out)
        raise
    sink = tempfile.TemporaryFile()  # noqa: SIM115 - closed in _restore_descriptors
    try:
        os.dup2(sink.fileno(), 1)
        os.dup2(sink.fileno(), 2)
    except OSError:
        os.dup2(saved_out, 1)
        os.dup2(saved_err, 2)
        os.close(saved_out)
        os.close(saved_err)
        sink.close()
        raise
    return saved_out, saved_err, sink


def _restore_descriptors(state: tuple[int, int, IO[bytes]]) -> str:
    """Put descriptors 1 and 2 back, then read the capture file.

    Every step runs whatever happened in the one before; a capture file that
    cannot be read yields an empty text, never an exception.
    """
    saved_out, saved_err, sink = state
    try:
        try:
            os.dup2(saved_out, 1)
        finally:
            os.dup2(saved_err, 2)
    finally:
        try:
            os.close(saved_out)
        finally:
            try:
                os.close(saved_err)
            finally:
                text = _read_sink(sink)
    return text


def _read_sink(sink: IO[bytes]) -> str:
    try:
        try:
            sink.seek(0)
            return sink.read().decode("utf-8", errors="replace")
        except (OSError, ValueError):
            return ""
    finally:
        sink.close()
