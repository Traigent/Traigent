"""Tests for traigent.integrations.langfuse package __init__.py exports.

This module tests that all public APIs are properly exported from the
traigent.integrations.langfuse package and are importable by external users.
"""

import dataclasses

import pytest

import traigent.integrations.langfuse as langfuse_module


class TestLangfuseModuleImports:
    """Test that traigent.integrations.langfuse module can be imported."""

    def test_langfuse_module_importable(self):
        """Test that traigent.integrations.langfuse exposes its documented
        public API surface (the quick-start symbols from the module
        docstring), not merely that the module object exists."""
        assert langfuse_module.__name__ == "traigent.integrations.langfuse"
        for name in (
            "LangfuseClient",
            "LangfuseTracker",
            "create_langfuse_tracker",
            "LANGFUSE_AVAILABLE",
        ):
            assert hasattr(langfuse_module, name), f"missing public symbol {name}"
        assert callable(langfuse_module.create_langfuse_tracker)
        assert isinstance(langfuse_module.LangfuseClient, type)

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
        """LangfuseClient must be a class exposing the documented
        get_trace_metrics() API used throughout the module's quick-start."""
        cls = langfuse_module.LangfuseClient
        assert isinstance(cls, type)
        assert hasattr(cls, "get_trace_metrics")
        assert callable(cls.get_trace_metrics)

    def test_langfuse_observation_not_none(self):
        """LangfuseObservation must be a dataclass with the documented
        per-observation fields (id, name, type, cost, latency)."""
        cls = langfuse_module.LangfuseObservation
        assert dataclasses.is_dataclass(cls)
        field_names = {f.name for f in dataclasses.fields(cls)}
        assert {"id", "name", "observation_type", "cost", "latency_ms"} <= field_names

    def test_langfuse_trace_metrics_not_none(self):
        """LangfuseTraceMetrics must be a dataclass exposing the documented
        aggregate fields and the to_measures_dict() conversion method."""
        cls = langfuse_module.LangfuseTraceMetrics
        assert dataclasses.is_dataclass(cls)
        field_names = {f.name for f in dataclasses.fields(cls)}
        assert {"trace_id", "total_cost", "per_agent_costs"} <= field_names
        assert callable(cls.to_measures_dict)


class TestLangfuseCallbackClasses:
    """Test that callback-related classes are accessible."""

    def test_langfuse_optimization_callback_not_none(self):
        """LangfuseOptimizationCallback must be a concrete subclass of the
        shared OptimizationCallback base so it can be registered via
        @optimize(..., callbacks=[...])."""
        from traigent.utils.callbacks import OptimizationCallback

        cls = langfuse_module.LangfuseOptimizationCallback
        assert isinstance(cls, type)
        assert issubclass(cls, OptimizationCallback)

    def test_trace_id_resolver_not_none(self):
        """TraceIdResolver is a runtime-checkable Protocol describing a
        trial -> trace_id callable; a matching callable satisfies it and a
        non-callable does not."""
        resolver_protocol = langfuse_module.TraceIdResolver
        assert isinstance(
            lambda trial: trial.metadata.get("trace_id"), resolver_protocol
        )
        assert not isinstance(object(), resolver_protocol)


class TestLangfuseTrackerClasses:
    """Test that tracker-related classes and functions are accessible."""

    def test_langfuse_tracker_not_none(self):
        """LangfuseTracker must expose the documented client property and
        get_callback() method used by the recommended high-level API."""
        cls = langfuse_module.LangfuseTracker
        assert isinstance(cls, type)
        assert isinstance(cls.client, property)
        assert hasattr(cls, "get_callback")
        assert callable(cls.get_callback)

    def test_create_langfuse_tracker_callable(self):
        """Test that create_langfuse_tracker is callable."""
        assert callable(langfuse_module.create_langfuse_tracker)

    def test_create_langfuse_tracker_not_none(self):
        """create_langfuse_tracker must build a LangfuseTracker whose public
        client is a real LangfuseClient, and whose callback (from
        get_callback()) actually invokes the given resolver and feeds its
        return value into client.get_trace_metrics -- proving the resolver
        passed to the factory is the one wired into the tracker, observed
        purely through the public API."""
        from unittest.mock import MagicMock, patch

        from traigent.core.types import TrialResult
        from traigent.utils.callbacks import ProgressInfo

        resolver = MagicMock(return_value="trace-1")
        tracker = langfuse_module.create_langfuse_tracker(trace_id_resolver=resolver)
        assert isinstance(tracker, langfuse_module.LangfuseTracker)
        assert isinstance(tracker.client, langfuse_module.LangfuseClient)

        trial = MagicMock(spec=TrialResult)
        trial.trial_id = "trial-1"
        trial.metrics = {}
        progress = MagicMock(spec=ProgressInfo)

        callback = tracker.get_callback()
        with patch.object(
            tracker.client, "get_trace_metrics", return_value=None
        ) as mock_get_metrics:
            callback.on_trial_complete(trial, progress)

        resolver.assert_called_once_with(trial)
        mock_get_metrics.assert_called_once_with("trace-1")


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
