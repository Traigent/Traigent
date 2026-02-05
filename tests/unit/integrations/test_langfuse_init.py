"""Tests for traigent.integrations.langfuse package __init__.py exports.

This module tests that all public APIs are properly exported from the
traigent.integrations.langfuse package and are importable by external users.
"""

import pytest

import traigent.integrations.langfuse as langfuse_module


class TestLangfuseModuleImports:
    """Test that traigent.integrations.langfuse module can be imported."""

    def test_langfuse_module_importable(self):
        """Test that traigent.integrations.langfuse can be imported."""
        assert langfuse_module is not None

    def test_langfuse_has_all_attribute(self):
        """Test that langfuse module has __all__ attribute."""
        assert hasattr(langfuse_module, "__all__")


class TestLangfuseExportsAvailable:
    """Test that all exports listed in __all__ are accessible."""

    def test_all_list_populated(self):
        """Test that __all__ list is not empty."""
        assert len(langfuse_module.__all__) > 0

    def test_langfuse_available_flag_accessible(self):
        """Test that LANGFUSE_AVAILABLE flag is accessible."""
        assert hasattr(langfuse_module, "LANGFUSE_AVAILABLE")
        assert "LANGFUSE_AVAILABLE" in langfuse_module.__all__

    def test_langfuse_client_accessible(self):
        """Test that LangfuseClient is accessible."""
        assert hasattr(langfuse_module, "LangfuseClient")
        assert "LangfuseClient" in langfuse_module.__all__

    def test_langfuse_trace_metrics_accessible(self):
        """Test that LangfuseTraceMetrics is accessible."""
        assert hasattr(langfuse_module, "LangfuseTraceMetrics")
        assert "LangfuseTraceMetrics" in langfuse_module.__all__

    def test_langfuse_observation_accessible(self):
        """Test that LangfuseObservation is accessible."""
        assert hasattr(langfuse_module, "LangfuseObservation")
        assert "LangfuseObservation" in langfuse_module.__all__

    def test_langfuse_optimization_callback_accessible(self):
        """Test that LangfuseOptimizationCallback is accessible."""
        assert hasattr(langfuse_module, "LangfuseOptimizationCallback")
        assert "LangfuseOptimizationCallback" in langfuse_module.__all__

    def test_trace_id_resolver_accessible(self):
        """Test that TraceIdResolver is accessible."""
        assert hasattr(langfuse_module, "TraceIdResolver")
        assert "TraceIdResolver" in langfuse_module.__all__

    def test_langfuse_tracker_accessible(self):
        """Test that LangfuseTracker is accessible."""
        assert hasattr(langfuse_module, "LangfuseTracker")
        assert "LangfuseTracker" in langfuse_module.__all__

    def test_create_langfuse_tracker_accessible(self):
        """Test that create_langfuse_tracker is accessible."""
        assert hasattr(langfuse_module, "create_langfuse_tracker")
        assert "create_langfuse_tracker" in langfuse_module.__all__


class TestLangfuseAvailableFlag:
    """Test the LANGFUSE_AVAILABLE flag."""

    def test_langfuse_available_is_boolean(self):
        """Test that LANGFUSE_AVAILABLE is a boolean."""
        assert isinstance(langfuse_module.LANGFUSE_AVAILABLE, bool)

    def test_langfuse_available_defined(self):
        """Test that LANGFUSE_AVAILABLE is defined."""
        assert hasattr(langfuse_module, "LANGFUSE_AVAILABLE")
        # Either True if langfuse is installed, or False if not
        assert langfuse_module.LANGFUSE_AVAILABLE in [True, False]


class TestLangfuseClientClasses:
    """Test that client-related classes are accessible and not None."""

    def test_langfuse_client_not_none(self):
        """Test that LangfuseClient is not None."""
        assert langfuse_module.LangfuseClient is not None

    def test_langfuse_observation_not_none(self):
        """Test that LangfuseObservation is not None."""
        assert langfuse_module.LangfuseObservation is not None

    def test_langfuse_trace_metrics_not_none(self):
        """Test that LangfuseTraceMetrics is not None."""
        assert langfuse_module.LangfuseTraceMetrics is not None


class TestLangfuseCallbackClasses:
    """Test that callback-related classes are accessible."""

    def test_langfuse_optimization_callback_not_none(self):
        """Test that LangfuseOptimizationCallback is not None."""
        assert langfuse_module.LangfuseOptimizationCallback is not None

    def test_trace_id_resolver_not_none(self):
        """Test that TraceIdResolver is not None."""
        assert langfuse_module.TraceIdResolver is not None


class TestLangfuseTrackerClasses:
    """Test that tracker-related classes and functions are accessible."""

    def test_langfuse_tracker_not_none(self):
        """Test that LangfuseTracker is not None."""
        assert langfuse_module.LangfuseTracker is not None

    def test_create_langfuse_tracker_callable(self):
        """Test that create_langfuse_tracker is callable."""
        assert callable(langfuse_module.create_langfuse_tracker)

    def test_create_langfuse_tracker_not_none(self):
        """Test that create_langfuse_tracker is not None."""
        assert langfuse_module.create_langfuse_tracker is not None


class TestLangfuseLogger:
    """Test that logger is accessible."""

    def test_logger_accessible(self):
        """Test that logger is accessible."""
        assert hasattr(langfuse_module, "logger")
        assert langfuse_module.logger is not None


class TestDirectImports:
    """Test that items can be directly imported from traigent.integrations.langfuse."""

    def test_direct_import_langfuse_available(self):
        """Test direct import of LANGFUSE_AVAILABLE."""
        from traigent.integrations.langfuse import LANGFUSE_AVAILABLE  # noqa: F401

    def test_direct_import_langfuse_client(self):
        """Test direct import of LangfuseClient."""
        from traigent.integrations.langfuse import LangfuseClient  # noqa: F401

    def test_direct_import_langfuse_trace_metrics(self):
        """Test direct import of LangfuseTraceMetrics."""
        from traigent.integrations.langfuse import LangfuseTraceMetrics  # noqa: F401

    def test_direct_import_langfuse_observation(self):
        """Test direct import of LangfuseObservation."""
        from traigent.integrations.langfuse import LangfuseObservation  # noqa: F401

    def test_direct_import_langfuse_optimization_callback(self):
        """Test direct import of LangfuseOptimizationCallback."""
        from traigent.integrations.langfuse import (  # noqa: F401
            LangfuseOptimizationCallback,
        )

    def test_direct_import_trace_id_resolver(self):
        """Test direct import of TraceIdResolver."""
        from traigent.integrations.langfuse import TraceIdResolver  # noqa: F401

    def test_direct_import_langfuse_tracker(self):
        """Test direct import of LangfuseTracker."""
        from traigent.integrations.langfuse import LangfuseTracker  # noqa: F401

    def test_direct_import_create_langfuse_tracker(self):
        """Test direct import of create_langfuse_tracker."""
        from traigent.integrations.langfuse import create_langfuse_tracker  # noqa: F401


class TestAllListContents:
    """Test that __all__ list has the expected items."""

    def test_all_list_contains_availability_flag(self):
        """Test that __all__ contains LANGFUSE_AVAILABLE."""
        assert "LANGFUSE_AVAILABLE" in langfuse_module.__all__

    def test_all_list_contains_client_exports(self):
        """Test that __all__ contains client-related exports."""
        client_exports = [
            "LangfuseClient",
            "LangfuseTraceMetrics",
            "LangfuseObservation",
        ]
        for export in client_exports:
            assert export in langfuse_module.__all__, f"{export} not in __all__"

    def test_all_list_contains_callback_exports(self):
        """Test that __all__ contains callback-related exports."""
        callback_exports = ["LangfuseOptimizationCallback", "TraceIdResolver"]
        for export in callback_exports:
            assert export in langfuse_module.__all__, f"{export} not in __all__"

    def test_all_list_contains_tracker_exports(self):
        """Test that __all__ contains tracker-related exports."""
        tracker_exports = ["LangfuseTracker", "create_langfuse_tracker"]
        for export in tracker_exports:
            assert export in langfuse_module.__all__, f"{export} not in __all__"

    def test_all_list_items_are_strings(self):
        """Test that all items in __all__ are strings."""
        for item in langfuse_module.__all__:
            assert isinstance(item, str), f"__all__ contains non-string: {item}"

    def test_all_list_no_duplicates(self):
        """Test that __all__ has no duplicate entries."""
        assert len(langfuse_module.__all__) == len(set(langfuse_module.__all__))


class TestWildcardImport:
    """Test that wildcard import works correctly."""

    def test_wildcard_import_succeeds(self):
        """Test that from traigent.integrations.langfuse import * succeeds."""
        # This should not raise an exception
        namespace = {}
        exec(
            "from traigent.integrations.langfuse import *",  # noqa: S102
            namespace,
        )
        # Verify that expected items are in the namespace
        assert "LangfuseClient" in namespace
        assert "LangfuseTracker" in namespace
        assert "LANGFUSE_AVAILABLE" in namespace
