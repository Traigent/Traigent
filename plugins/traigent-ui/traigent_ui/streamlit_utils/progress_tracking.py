"""
Progress Tracking for Streamlit Applications.

This module provides utilities for tracking and displaying progress
of long-running operations in Streamlit with real-time updates.
"""

import time
from typing import Any, Callable

import streamlit as st


class ProgressTracker:
    """Manages progress tracking for async operations in Streamlit."""

    def __init__(self, total_steps: int = 100):
        """
        Initialize progress tracker.

        Args:
            total_steps: Total number of steps for progress calculation
        """
        self.total_steps = total_steps
        self.current_step = 0
        self.start_time = time.time()
        self.progress_bar = None
        self.status_text = None
        self.metrics_cols = None

    def setup_ui(self):
        """Set up the progress UI elements."""
        # Progress bar
        self.progress_bar = st.progress(0)

        # Status text
        self.status_text = st.empty()

        # Metrics columns for additional info
        self.metrics_cols = st.columns(3)

        # Initialize metrics
        with self.metrics_cols[0]:
            st.metric("Progress", "0%")
        with self.metrics_cols[1]:
            st.metric("Elapsed", "0s")
        with self.metrics_cols[2]:
            st.metric("Estimated Remaining", "calculating...")

    def update(self, message: str, progress: float):
        """
        Update progress display.

        Args:
            message: Status message to display
            progress: Progress as float between 0.0 and 1.0
        """
        # Ensure progress is within bounds
        progress = max(0.0, min(1.0, progress))

        # Update progress bar
        if self.progress_bar:
            self.progress_bar.progress(progress)

        # Update status text
        if self.status_text:
            self.status_text.text(f"🔄 {message}")

        # Update metrics
        if self.metrics_cols:
            elapsed = time.time() - self.start_time

            with self.metrics_cols[0]:
                st.metric("Progress", f"{progress * 100:.1f}%")

            with self.metrics_cols[1]:
                st.metric("Elapsed", f"{elapsed:.1f}s")

            with self.metrics_cols[2]:
                if progress > 0.1:  # Only estimate after some progress
                    estimated_total = elapsed / progress
                    remaining = estimated_total - elapsed
                    st.metric("Estimated Remaining", f"{remaining:.1f}s")
                else:
                    st.metric("Estimated Remaining", "calculating...")

    def complete(self, message: str = "Complete!"):
        """Mark the operation as complete."""
        if self.progress_bar:
            self.progress_bar.progress(1.0)

        if self.status_text:
            self.status_text.text(f"✅ {message}")

        # Final metrics update
        if self.metrics_cols:
            elapsed = time.time() - self.start_time
            with self.metrics_cols[0]:
                st.metric("Progress", "100%")
            with self.metrics_cols[1]:
                st.metric("Elapsed", f"{elapsed:.1f}s")
            with self.metrics_cols[2]:
                st.metric("Status", "Complete!")

    def error(self, message: str):
        """Mark the operation as failed."""
        if self.progress_bar:
            # Show red progress bar at current state
            self.progress_bar.progress(0.0)

        if self.status_text:
            self.status_text.text(f"❌ {message}")

        # Error metrics
        if self.metrics_cols:
            elapsed = time.time() - self.start_time
            with self.metrics_cols[0]:
                st.metric("Progress", "Failed")
            with self.metrics_cols[1]:
                st.metric("Elapsed", f"{elapsed:.1f}s")
            with self.metrics_cols[2]:
                st.metric("Status", "Error")


async def run_with_progress(
    operation: Callable, progress_tracker: ProgressTracker, *args, **kwargs
) -> Any:
    """
    Run an async operation with progress tracking.

    Args:
        operation: Async function to run
        progress_tracker: ProgressTracker instance
        *args: Arguments for the operation
        **kwargs: Keyword arguments for the operation

    Returns:
        Result of the operation
    """
    try:
        # Create progress callback
        def progress_callback(message: str, progress: float):
            progress_tracker.update(message, progress)
            # Force Streamlit to update
            time.sleep(0.01)  # Small delay to allow UI update

        # Add progress callback to kwargs if the operation expects it
        if "progress_callback" in operation.__code__.co_varnames:
            kwargs["progress_callback"] = progress_callback

        # Run the operation
        result = await operation(*args, **kwargs)

        progress_tracker.complete()
        return result

    except Exception as e:
        progress_tracker.error(str(e))
        raise


class StreamlitProgressCallback:
    """Simple progress callback for Streamlit that updates session state."""

    def __init__(self, session_key: str = "generation_progress"):
        """
        Initialize callback.

        Args:
            session_key: Session state key to store progress info
        """
        self.session_key = session_key

        # Initialize session state
        if self.session_key not in st.session_state:
            st.session_state[self.session_key] = {
                "message": "",
                "progress": 0.0,
                "active": False,
            }

    def __call__(self, message: str, progress: float):
        """Update progress in session state."""
        st.session_state[self.session_key] = {
            "message": message,
            "progress": max(0.0, min(1.0, progress)),
            "active": True,
        }

        # Force rerun to update UI
        st.rerun()

    def reset(self):
        """Reset progress state."""
        st.session_state[self.session_key] = {
            "message": "",
            "progress": 0.0,
            "active": False,
        }

    def get_state(self):
        """Get current progress state."""
        return st.session_state.get(
            self.session_key, {"message": "", "progress": 0.0, "active": False}
        )


def display_progress_from_session(session_key: str = "generation_progress"):
    """
    Display progress from session state.

    Args:
        session_key: Session state key containing progress info
    """
    progress_info = st.session_state.get(session_key, {})

    if progress_info.get("active", False):
        progress = progress_info.get("progress", 0.0)
        message = progress_info.get("message", "Processing...")

        # Show progress bar
        st.progress(progress)

        # Show status message
        if progress < 1.0:
            st.text(f"🔄 {message}")
        else:
            st.text(f"✅ {message}")

        return True

    return False
