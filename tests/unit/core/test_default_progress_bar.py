"""Tests for default progress bar auto-registration."""

from __future__ import annotations

from unittest.mock import patch

from traigent.core.optimized_function import _resolve_callbacks
from traigent.utils.callbacks import ProgressBarCallback


class TestResolveCallbacks:
    def test_auto_enabled_interactive(self):
        """ProgressBarCallback auto-injected when stdin is a TTY."""
        with patch("traigent.core.optimized_function.sys") as mock_sys:
            mock_sys.stdin.isatty.return_value = True
            callbacks = _resolve_callbacks(None, None, progress_bar=None)

        assert len(callbacks) == 1
        assert isinstance(callbacks[0], ProgressBarCallback)

    def test_disabled_explicit(self):
        """progress_bar=False suppresses auto-injection."""
        with patch("traigent.core.optimized_function.sys") as mock_sys:
            mock_sys.stdin.isatty.return_value = True
            callbacks = _resolve_callbacks(None, None, progress_bar=False)

        assert len(callbacks) == 0

    def test_skipped_non_interactive(self):
        """No auto-injection when stdin is not a TTY."""
        with patch("traigent.core.optimized_function.sys") as mock_sys:
            mock_sys.stdin.isatty.return_value = False
            callbacks = _resolve_callbacks(None, None, progress_bar=None)

        assert len(callbacks) == 0

    def test_not_duplicated(self):
        """User-provided ProgressBarCallback is not duplicated."""
        existing = ProgressBarCallback()
        with patch("traigent.core.optimized_function.sys") as mock_sys:
            mock_sys.stdin.isatty.return_value = True
            callbacks = _resolve_callbacks([existing], None, progress_bar=None)

        progress_bars = [cb for cb in callbacks if isinstance(cb, ProgressBarCallback)]
        assert len(progress_bars) == 1
        assert progress_bars[0] is existing

    def test_explicit_callbacks_preserved(self):
        """Explicit callbacks are preserved alongside auto-injected bar."""

        class CustomCallback:
            pass

        custom = CustomCallback()
        with patch("traigent.core.optimized_function.sys") as mock_sys:
            mock_sys.stdin.isatty.return_value = True
            callbacks = _resolve_callbacks([custom], None, progress_bar=None)

        assert len(callbacks) == 2
        assert callbacks[0] is not custom  # ProgressBarCallback inserted first
        assert isinstance(callbacks[0], ProgressBarCallback)
        assert callbacks[1] is custom

    def test_decorator_callbacks_used_when_no_explicit(self):
        """Decorator callbacks used as fallback."""

        class DecoratorCb:
            pass

        dec_cb = DecoratorCb()
        with patch("traigent.core.optimized_function.sys") as mock_sys:
            mock_sys.stdin.isatty.return_value = True
            callbacks = _resolve_callbacks(None, [dec_cb], progress_bar=None)

        assert len(callbacks) == 2
        assert isinstance(callbacks[0], ProgressBarCallback)
        assert callbacks[1] is dec_cb

    def test_force_progress_bar_non_interactive(self):
        """progress_bar=True forces injection even in non-interactive mode."""
        with patch("traigent.core.optimized_function.sys") as mock_sys:
            mock_sys.stdin.isatty.return_value = False
            callbacks = _resolve_callbacks(None, None, progress_bar=True)

        assert len(callbacks) == 1
        assert isinstance(callbacks[0], ProgressBarCallback)

    def test_auto_skips_non_interactive(self):
        """progress_bar=None with isatty=False should not inject."""
        with patch("traigent.core.optimized_function.sys") as mock_sys:
            mock_sys.stdin.isatty.return_value = False
            callbacks = _resolve_callbacks(None, None, progress_bar=None)

        assert len(callbacks) == 0
