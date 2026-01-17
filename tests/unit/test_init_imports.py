"""Tests for __init__.py module imports and availability flags.

These tests ensure that module imports work correctly and that availability
flags accurately reflect installed dependencies.
"""

# Traceability: CONC-Layer-API CONC-Quality-Maintainability
# FUNC-API-ENTRY REQ-API-001

from __future__ import annotations

import pytest


class TestTraigentInit:
    """Tests for traigent/__init__.py imports."""

    def test_main_module_imports(self) -> None:
        """Test that main traigent module imports successfully."""
        import traigent

        # Core exports should be available
        assert hasattr(traigent, "optimize")
        assert hasattr(traigent, "Range")
        assert hasattr(traigent, "IntRange")
        assert hasattr(traigent, "LogRange")
        assert hasattr(traigent, "Choices")
        assert hasattr(traigent, "ParameterRange")

    def test_version_available(self) -> None:
        """Test that version is available."""
        import traigent

        assert hasattr(traigent, "__version__")
        assert isinstance(traigent.__version__, str)
        assert len(traigent.__version__) > 0

    def test_author_and_email(self) -> None:
        """Test that author and email are available."""
        import traigent

        assert hasattr(traigent, "__author__")
        assert hasattr(traigent, "__email__")
        assert traigent.__author__ == "Traigent Team"
        assert "traigent" in traigent.__email__.lower()

    def test_constraint_exports(self) -> None:
        """Test TVL constraint system exports."""
        import traigent

        # TVL constraint exports
        assert hasattr(traigent, "when")
        assert hasattr(traigent, "require")
        assert hasattr(traigent, "implies")
        assert hasattr(traigent, "Constraint")
        assert hasattr(traigent, "Condition")

    def test_exception_exports(self) -> None:
        """Test exception exports."""
        import traigent

        assert hasattr(traigent, "TraigentWarning")
        assert hasattr(traigent, "TraigentDeprecationWarning")
        assert hasattr(traigent, "OptimizationStateError")
        assert hasattr(traigent, "ConfigAccessWarning")
        assert hasattr(traigent, "DataIntegrityError")
        assert hasattr(traigent, "MetricExtractionError")
        assert hasattr(traigent, "DTOSerializationError")

    def test_dto_exports(self) -> None:
        """Test DTO exports for multi-agent workflow tracking."""
        import traigent

        assert hasattr(traigent, "AgentCostBreakdown")
        assert hasattr(traigent, "WorkflowCostSummary")
        assert hasattr(traigent, "MeasuresDict")
        assert hasattr(traigent, "TraigentMetadata")
        assert hasattr(traigent, "is_traigent_metadata")

    def test_utility_exports(self) -> None:
        """Test utility function exports."""
        import traigent

        assert hasattr(traigent, "configure")
        assert hasattr(traigent, "initialize")
        assert hasattr(traigent, "get_config")
        assert hasattr(traigent, "get_trial_config")
        assert hasattr(traigent, "get_trial_context")
        assert hasattr(traigent, "with_usage")

    def test_all_exports_list(self) -> None:
        """Test that __all__ is properly defined."""
        import traigent

        assert hasattr(traigent, "__all__")
        assert isinstance(traigent.__all__, list)
        assert len(traigent.__all__) > 0

        # All items in __all__ should be accessible
        for name in traigent.__all__:
            assert hasattr(traigent, name), f"{name} in __all__ but not accessible"


class TestIntegrationsInit:
    """Tests for traigent/integrations/__init__.py imports."""

    def test_integrations_module_imports(self) -> None:
        """Test that integrations module imports successfully."""
        from traigent import integrations

        # Core framework override exports
        assert hasattr(integrations, "FrameworkOverrideManager")
        assert hasattr(integrations, "enable_framework_overrides")
        assert hasattr(integrations, "disable_framework_overrides")
        assert hasattr(integrations, "override_context")

    def test_activation_state_exports(self) -> None:
        """Test activation state exports."""
        from traigent import integrations

        assert hasattr(integrations, "ActivationState")
        assert hasattr(integrations, "create_activation_state")

    def test_wrapper_exports(self) -> None:
        """Test wrapper utility exports."""
        from traigent import integrations

        assert hasattr(integrations, "OverrideContext")
        assert hasattr(integrations, "apply_parameter_overrides")
        assert hasattr(integrations, "create_wrapper")
        assert hasattr(integrations, "create_method_wrapper")

    def test_workflow_traces_always_available(self) -> None:
        """Test that workflow traces are always available."""
        from traigent import integrations

        assert hasattr(integrations, "WORKFLOW_TRACES_INTEGRATION_AVAILABLE")
        assert integrations.WORKFLOW_TRACES_INTEGRATION_AVAILABLE is True

        # Workflow trace types should be available
        assert hasattr(integrations, "WorkflowTracesTracker")
        assert hasattr(integrations, "WorkflowTracesClient")
        assert hasattr(integrations, "SpanPayload")
        assert hasattr(integrations, "SpanStatus")
        assert hasattr(integrations, "SpanType")

    def test_provider_exports(self) -> None:
        """Test provider exports."""
        from traigent import integrations

        assert hasattr(integrations, "get_models_for_tier")
        assert hasattr(integrations, "list_available_providers")
        assert hasattr(integrations, "get_all_tiers")

    def test_integration_availability_flags(self) -> None:
        """Test that integration availability flags exist."""
        from traigent import integrations

        # These should exist (value depends on installed packages)
        assert hasattr(integrations, "MLFLOW_INTEGRATION_AVAILABLE")
        assert hasattr(integrations, "WANDB_INTEGRATION_AVAILABLE")
        assert isinstance(integrations.MLFLOW_INTEGRATION_AVAILABLE, bool)
        assert isinstance(integrations.WANDB_INTEGRATION_AVAILABLE, bool)


class TestObservabilityInit:
    """Tests for traigent/integrations/observability/__init__.py imports."""

    def test_observability_module_imports(self) -> None:
        """Test that observability module imports successfully."""
        from traigent.integrations import observability

        # Workflow traces should always be available
        assert hasattr(observability, "WorkflowTracesTracker")
        assert hasattr(observability, "WorkflowTracesClient")

    def test_workflow_trace_types(self) -> None:
        """Test workflow trace type exports."""
        from traigent.integrations import observability

        assert hasattr(observability, "SpanPayload")
        assert hasattr(observability, "SpanStatus")
        assert hasattr(observability, "SpanType")
        assert hasattr(observability, "WorkflowNode")
        assert hasattr(observability, "WorkflowEdge")
        assert hasattr(observability, "WorkflowLoop")
        assert hasattr(observability, "WorkflowGraphPayload")

    def test_langgraph_extraction_functions(self) -> None:
        """Test LangGraph extraction function exports."""
        from traigent.integrations import observability

        assert hasattr(observability, "extract_nodes_from_langgraph")
        assert hasattr(observability, "extract_edges_from_langgraph")
        assert hasattr(observability, "detect_loops_in_graph")

    def test_availability_flags(self) -> None:
        """Test availability flags."""
        from traigent.integrations import observability

        assert hasattr(observability, "MLFLOW_AVAILABLE")
        assert hasattr(observability, "WANDB_AVAILABLE")
        assert hasattr(observability, "OTEL_AVAILABLE")
        assert isinstance(observability.MLFLOW_AVAILABLE, bool)
        assert isinstance(observability.WANDB_AVAILABLE, bool)
        assert isinstance(observability.OTEL_AVAILABLE, bool)

    def test_all_exports_list(self) -> None:
        """Test that __all__ is properly defined."""
        from traigent.integrations import observability

        assert hasattr(observability, "__all__")
        assert isinstance(observability.__all__, list)
        assert len(observability.__all__) > 0


class TestAnalyticsInit:
    """Tests for traigent/analytics/__init__.py imports."""

    def test_analytics_module_imports(self) -> None:
        """Test that analytics module imports (may show deprecation warning)."""
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            from traigent import analytics

            # Helper functions should be available
            assert hasattr(analytics, "is_analytics_available")
            assert hasattr(analytics, "is_plugin_installed")

    def test_analytics_availability_check(self) -> None:
        """Test analytics availability check functions."""
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            from traigent.analytics import is_analytics_available, is_plugin_installed

            # is_analytics_available should always return True (embedded fallback)
            assert is_analytics_available() is True
            # is_plugin_installed depends on whether plugin is installed
            assert isinstance(is_plugin_installed(), bool)

    def test_all_exports_defined(self) -> None:
        """Test that __all__ is properly defined."""
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            from traigent import analytics

            assert hasattr(analytics, "__all__")
            assert isinstance(analytics.__all__, list)
            assert "is_analytics_available" in analytics.__all__


class TestOptimizersInit:
    """Tests for traigent/optimizers/__init__.py imports."""

    def test_optimizers_module_imports(self) -> None:
        """Test that optimizers module imports successfully."""
        from traigent import optimizers

        assert optimizers is not None

    def test_pruners_available(self) -> None:
        """Test that pruners are available."""
        from traigent.optimizers import pruners

        assert pruners is not None
