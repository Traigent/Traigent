"""Tests for traigent.integrations package imports and availability flags."""

import pytest


class TestIntegrationsAvailability:
    """Test that integration availability flags are set correctly."""

    def test_import_integrations_module(self):
        """Test that integrations module imports without error."""
        import traigent.integrations as intg

        assert hasattr(intg, "MLFLOW_INTEGRATION_AVAILABLE")
        assert hasattr(intg, "WANDB_INTEGRATION_AVAILABLE")
        assert hasattr(intg, "WORKFLOW_TRACES_INTEGRATION_AVAILABLE")
        assert isinstance(intg.MLFLOW_INTEGRATION_AVAILABLE, bool)
        assert isinstance(intg.WANDB_INTEGRATION_AVAILABLE, bool)
        assert intg.WORKFLOW_TRACES_INTEGRATION_AVAILABLE is True

    def test_core_exports_available(self):
        """Test that core exports are always available."""
        from traigent.integrations import (
            ActivationState,
            BaseOverrideManager,
            FrameworkOverrideManager,
            OverrideContext,
            apply_parameter_overrides,
            create_activation_state,
            create_method_wrapper,
            create_resilient_wrapper,
            create_wrapper,
            disable_framework_overrides,
            enable_framework_overrides,
            override_all_platforms,
            override_anthropic,
            override_cohere,
            override_context,
            override_huggingface,
            override_langchain,
            override_openai_sdk,
            register_framework_mapping,
        )

        assert FrameworkOverrideManager is not None
        assert BaseOverrideManager is not None
        assert ActivationState is not None

    def test_mapping_exports(self):
        """Test static mapping exports."""
        from traigent.integrations import (
            METHOD_MAPPINGS,
            PARAMETER_MAPPINGS,
            get_method_mapping,
            get_parameter_mapping,
            get_supported_frameworks,
        )

        assert isinstance(PARAMETER_MAPPINGS, dict)
        assert isinstance(METHOD_MAPPINGS, dict)
        assert callable(get_parameter_mapping)
        assert callable(get_method_mapping)
        assert callable(get_supported_frameworks)

    def test_provider_exports(self):
        """Test provider exports are always available."""
        from traigent.integrations import (
            get_all_tiers,
            get_models_for_tier,
            list_available_providers,
            register_provider_tiers,
        )

        assert callable(get_all_tiers)
        assert callable(get_models_for_tier)
        assert callable(list_available_providers)
        assert callable(register_provider_tiers)

    def test_workflow_traces_exports(self):
        """Test workflow traces exports are always available."""
        from traigent.integrations import (
            OTEL_AVAILABLE,
            OptiGenSpanExporter,
            SpanPayload,
            SpanStatus,
            SpanType,
            TraceIngestionRequest,
            TraceIngestionResponse,
            WorkflowEdge,
            WorkflowGraphPayload,
            WorkflowLoop,
            WorkflowNode,
            WorkflowTracesClient,
            WorkflowTracesTracker,
            create_workflow_tracker,
            detect_loops_in_graph,
            extract_edges_from_langgraph,
            extract_nodes_from_langgraph,
            setup_workflow_tracing,
        )

        assert isinstance(OTEL_AVAILABLE, bool)
        assert SpanStatus is not None
        assert SpanType is not None

    def test_all_list_is_populated(self):
        """Test __all__ list is populated with exports."""
        import traigent.integrations as intg

        assert isinstance(intg.__all__, list)
        assert len(intg.__all__) > 20
        assert "FrameworkOverrideManager" in intg.__all__

    def test_mlflow_fallback_raises_import_error(self):
        """Test MLflow fallback raises ImportError when unavailable."""
        import traigent.integrations as intg

        if not intg.MLFLOW_INTEGRATION_AVAILABLE:
            with pytest.raises(ImportError, match="MLflow"):
                intg.create_mlflow_tracker()

    def test_wandb_fallback_raises_import_error(self):
        """Test W&B fallback raises ImportError when unavailable."""
        import traigent.integrations as intg

        if not intg.WANDB_INTEGRATION_AVAILABLE:
            with pytest.raises(ImportError, match="W&B"):
                intg.create_wandb_tracker()

    def test_langchain_handler_availability(self):
        """Test LangChain handler availability flag."""
        import traigent.integrations as intg

        assert hasattr(intg, "LANGCHAIN_HANDLER_AVAILABLE")
        assert isinstance(intg.LANGCHAIN_HANDLER_AVAILABLE, bool)

    def test_langfuse_availability(self):
        """Test Langfuse availability flag."""
        import traigent.integrations as intg

        assert hasattr(intg, "LANGFUSE_INTEGRATION_AVAILABLE")
        assert isinstance(intg.LANGFUSE_INTEGRATION_AVAILABLE, bool)

    def test_dspy_availability(self):
        """Test DSPy availability flag."""
        import traigent.integrations as intg

        assert hasattr(intg, "DSPY_INTEGRATION_AVAILABLE")
        assert isinstance(intg.DSPY_INTEGRATION_AVAILABLE, bool)

    def test_langchain_integration_availability(self):
        """Test LangChain integration availability flag."""
        import traigent.integrations as intg

        assert hasattr(intg, "LANGCHAIN_INTEGRATION_AVAILABLE")
        assert isinstance(intg.LANGCHAIN_INTEGRATION_AVAILABLE, bool)

    def test_openai_sdk_availability(self):
        """Test OpenAI SDK integration availability flag."""
        import traigent.integrations as intg

        assert hasattr(intg, "OPENAI_SDK_INTEGRATION_AVAILABLE")
        assert isinstance(intg.OPENAI_SDK_INTEGRATION_AVAILABLE, bool)
