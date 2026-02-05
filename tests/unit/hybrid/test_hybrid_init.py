"""Tests for traigent.hybrid package __init__.py exports.

This module tests that all public APIs are properly exported from the
traigent.hybrid package and are importable by external users.
"""

import pytest

import traigent.hybrid as hybrid_module


class TestHybridExportsImportable:
    """Test that all exports are importable from traigent.hybrid."""

    def test_transport_class_importable(self):
        """Test HybridTransport class is importable."""
        assert hasattr(hybrid_module, "HybridTransport")
        assert hybrid_module.HybridTransport is not None

    def test_create_transport_function_importable(self):
        """Test create_transport function is importable."""
        assert hasattr(hybrid_module, "create_transport")
        assert callable(hybrid_module.create_transport)

    def test_transport_error_exceptions_importable(self):
        """Test all transport error classes are importable."""
        error_classes = [
            "TransportError",
            "TransportConnectionError",
            "TransportTimeoutError",
            "TransportAuthError",
            "TransportRateLimitError",
            "TransportServerError",
        ]
        for error_class in error_classes:
            assert hasattr(hybrid_module, error_class), f"{error_class} not found"
            assert hybrid_module.__dict__[error_class] is not None

    def test_protocol_dto_classes_importable(self):
        """Test all protocol DTO classes are importable."""
        protocol_classes = [
            "BatchOptions",
            "HybridExecuteRequest",
            "HybridExecuteResponse",
            "HybridEvaluateRequest",
            "HybridEvaluateResponse",
            "ServiceCapabilities",
            "TVARDefinition",
            "ConfigSpaceResponse",
            "HealthCheckResponse",
        ]
        for dto_class in protocol_classes:
            assert hasattr(hybrid_module, dto_class), f"{dto_class} not found"
            assert hybrid_module.__dict__[dto_class] is not None

    def test_lifecycle_classes_importable(self):
        """Test lifecycle classes are importable."""
        lifecycle_classes = ["AgentLifecycleManager", "SessionInfo"]
        for cls in lifecycle_classes:
            assert hasattr(hybrid_module, cls)
            assert hybrid_module.__dict__[cls] is not None

    def test_discovery_utilities_importable(self):
        """Test discovery utilities are importable."""
        discovery_items = [
            "ConfigSpaceDiscovery",
            "merge_config_spaces",
            "normalize_tvar_to_config_space",
            "validate_config_against_tvars",
        ]
        for item in discovery_items:
            assert hasattr(hybrid_module, item), f"{item} not found"
            assert hybrid_module.__dict__[item] is not None


class TestHybridAllList:
    """Test that __all__ list is properly defined and complete."""

    def test_all_list_exists(self):
        """Test that __all__ list exists."""
        assert hasattr(hybrid_module, "__all__")
        assert isinstance(hybrid_module.__all__, list)

    def test_all_list_populated(self):
        """Test that __all__ list is not empty."""
        assert len(hybrid_module.__all__) > 0

    def test_all_list_contains_expected_items(self):
        """Test that __all__ contains all expected export names."""
        expected_exports = {
            # Transport
            "HybridTransport",
            "create_transport",
            # Exceptions
            "TransportError",
            "TransportConnectionError",
            "TransportTimeoutError",
            "TransportAuthError",
            "TransportRateLimitError",
            "TransportServerError",
            # Protocol DTOs
            "BatchOptions",
            "HybridExecuteRequest",
            "HybridExecuteResponse",
            "HybridEvaluateRequest",
            "HybridEvaluateResponse",
            "ServiceCapabilities",
            "TVARDefinition",
            "ConfigSpaceResponse",
            "HealthCheckResponse",
            # Lifecycle
            "AgentLifecycleManager",
            "SessionInfo",
            # Discovery
            "ConfigSpaceDiscovery",
            "merge_config_spaces",
            "normalize_tvar_to_config_space",
            "validate_config_against_tvars",
        }
        assert set(hybrid_module.__all__) == expected_exports

    def test_all_list_items_are_strings(self):
        """Test that all items in __all__ are strings."""
        for item in hybrid_module.__all__:
            assert isinstance(item, str), f"__all__ contains non-string: {item}"


class TestTransportErrorClasses:
    """Test that transport error classes exist and are exception subclasses."""

    def test_transport_error_is_exception(self):
        """Test TransportError is an Exception subclass."""
        assert issubclass(hybrid_module.TransportError, Exception)

    def test_transport_connection_error_inherits_from_transport_error(self):
        """Test TransportConnectionError inherits from TransportError."""
        assert issubclass(
            hybrid_module.TransportConnectionError, hybrid_module.TransportError
        )

    def test_transport_timeout_error_inherits_from_transport_error(self):
        """Test TransportTimeoutError inherits from TransportError."""
        assert issubclass(
            hybrid_module.TransportTimeoutError, hybrid_module.TransportError
        )

    def test_transport_auth_error_inherits_from_transport_error(self):
        """Test TransportAuthError inherits from TransportError."""
        assert issubclass(
            hybrid_module.TransportAuthError, hybrid_module.TransportError
        )

    def test_transport_rate_limit_error_inherits_from_transport_error(self):
        """Test TransportRateLimitError inherits from TransportError."""
        assert issubclass(
            hybrid_module.TransportRateLimitError, hybrid_module.TransportError
        )

    def test_transport_server_error_inherits_from_transport_error(self):
        """Test TransportServerError inherits from TransportError."""
        assert issubclass(
            hybrid_module.TransportServerError, hybrid_module.TransportError
        )


class TestProtocolClasses:
    """Test that protocol DTO classes exist and are accessible."""

    def test_batch_options_exists(self):
        """Test BatchOptions DTO exists."""
        assert hasattr(hybrid_module, "BatchOptions")
        assert hybrid_module.BatchOptions is not None

    def test_hybrid_execute_request_exists(self):
        """Test HybridExecuteRequest DTO exists."""
        assert hasattr(hybrid_module, "HybridExecuteRequest")
        assert hybrid_module.HybridExecuteRequest is not None

    def test_hybrid_execute_response_exists(self):
        """Test HybridExecuteResponse DTO exists."""
        assert hasattr(hybrid_module, "HybridExecuteResponse")
        assert hybrid_module.HybridExecuteResponse is not None

    def test_hybrid_evaluate_request_exists(self):
        """Test HybridEvaluateRequest DTO exists."""
        assert hasattr(hybrid_module, "HybridEvaluateRequest")
        assert hybrid_module.HybridEvaluateRequest is not None

    def test_hybrid_evaluate_response_exists(self):
        """Test HybridEvaluateResponse DTO exists."""
        assert hasattr(hybrid_module, "HybridEvaluateResponse")
        assert hybrid_module.HybridEvaluateResponse is not None

    def test_service_capabilities_exists(self):
        """Test ServiceCapabilities DTO exists."""
        assert hasattr(hybrid_module, "ServiceCapabilities")
        assert hybrid_module.ServiceCapabilities is not None

    def test_tvar_definition_exists(self):
        """Test TVARDefinition DTO exists."""
        assert hasattr(hybrid_module, "TVARDefinition")
        assert hybrid_module.TVARDefinition is not None

    def test_config_space_response_exists(self):
        """Test ConfigSpaceResponse DTO exists."""
        assert hasattr(hybrid_module, "ConfigSpaceResponse")
        assert hybrid_module.ConfigSpaceResponse is not None

    def test_health_check_response_exists(self):
        """Test HealthCheckResponse DTO exists."""
        assert hasattr(hybrid_module, "HealthCheckResponse")
        assert hybrid_module.HealthCheckResponse is not None


class TestDirectImports:
    """Test that items can be directly imported from traigent.hybrid."""

    def test_direct_import_hybrid_transport(self):
        """Test direct import of HybridTransport."""
        from traigent.hybrid import HybridTransport  # noqa: F401

    def test_direct_import_create_transport(self):
        """Test direct import of create_transport."""
        from traigent.hybrid import create_transport  # noqa: F401

    def test_direct_import_transport_errors(self):
        """Test direct import of all transport errors."""
        from traigent.hybrid import (  # noqa: F401
            TransportAuthError,
            TransportConnectionError,
            TransportError,
            TransportRateLimitError,
            TransportServerError,
            TransportTimeoutError,
        )

    def test_direct_import_protocol_dtos(self):
        """Test direct import of protocol DTOs."""
        from traigent.hybrid import (  # noqa: F401
            BatchOptions,
            ConfigSpaceResponse,
            HealthCheckResponse,
            HybridEvaluateRequest,
            HybridEvaluateResponse,
            HybridExecuteRequest,
            HybridExecuteResponse,
            ServiceCapabilities,
            TVARDefinition,
        )

    def test_direct_import_lifecycle(self):
        """Test direct import of lifecycle classes."""
        from traigent.hybrid import (  # noqa: F401
            AgentLifecycleManager,
            SessionInfo,
        )

    def test_direct_import_discovery(self):
        """Test direct import of discovery utilities."""
        from traigent.hybrid import (  # noqa: F401
            ConfigSpaceDiscovery,
            merge_config_spaces,
            normalize_tvar_to_config_space,
            validate_config_against_tvars,
        )
