"""Unit tests for traigent.optigen_integration.

Tests for the OptiGenClient which provides high-level optimization
with multiple execution modes (edge analytics, standard/hybrid, cloud/SaaS).
"""

# Traceability: CONC-Layer-Integration CONC-Quality-Reliability CONC-Quality-Performance FUNC-CLOUD-HYBRID FUNC-INVOKERS REQ-CLOUD-009 REQ-INV-006 SYNC-CloudHybrid

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from traigent.config.types import ExecutionMode
from traigent.optigen_integration import OptiGenClient
from traigent.utils.exceptions import OptimizationError


class TestOptiGenClientInitialization:
    """Tests for OptiGenClient initialization."""

    @patch("traigent.optigen_integration.BackendIntegratedClient")
    @patch("traigent.config.backend_config.BackendConfig")
    def test_init_with_explicit_api_key(
        self, mock_backend_config: MagicMock, mock_backend_client: MagicMock
    ) -> None:
        """Test client initialization with explicit API key."""
        mock_backend_config.get_api_key.return_value = "default_key"
        mock_backend_config.get_backend_url.return_value = "https://default.url"

        client = OptiGenClient(
            api_key="explicit_key", backend_url="https://explicit.url"
        )

        assert client.api_key == "explicit_key"
        assert client.backend_url == "https://explicit.url"
        assert client._explicit_api_key is True
        assert client.execution_mode == ExecutionMode.CLOUD

    @patch("traigent.optigen_integration.BackendIntegratedClient")
    @patch("traigent.config.backend_config.BackendConfig")
    def test_init_with_default_config(
        self, mock_backend_config: MagicMock, mock_backend_client: MagicMock
    ) -> None:
        """Test client initialization with default configuration."""
        mock_backend_config.get_api_key.return_value = "default_key"
        mock_backend_config.get_backend_url.return_value = "https://default.url"

        client = OptiGenClient()

        assert client.api_key == "default_key"
        assert client.backend_url == "https://default.url"
        assert client._explicit_api_key is False

    @patch("traigent.optigen_integration.BackendIntegratedClient")
    @patch("traigent.config.backend_config.BackendConfig")
    def test_init_with_edge_analytics_mode(
        self, mock_backend_config: MagicMock, mock_backend_client: MagicMock
    ) -> None:
        """Test client initialization with edge analytics mode."""
        mock_backend_config.get_api_key.return_value = "key"
        mock_backend_config.get_backend_url.return_value = "https://url"

        client = OptiGenClient(execution_mode="edge_analytics")

        assert client.execution_mode == ExecutionMode.EDGE_ANALYTICS

    @patch("traigent.optigen_integration.BackendIntegratedClient")
    @patch("traigent.config.backend_config.BackendConfig")
    def test_init_with_agent_builder(
        self, mock_backend_config: MagicMock, mock_backend_client: MagicMock
    ) -> None:
        """Test client initialization with agent builder."""
        mock_backend_config.get_api_key.return_value = "key"
        mock_backend_config.get_backend_url.return_value = "https://url"

        mock_builder = Mock()
        client = OptiGenClient(agent_builder=mock_builder)

        assert client.agent_builder is mock_builder


class TestDetermineExecutionMode:
    """Tests for execution mode determination logic."""

    @patch("traigent.optigen_integration.BackendIntegratedClient")
    @patch("traigent.config.backend_config.BackendConfig")
    def test_explicit_edge_analytics_mode(
        self, mock_backend_config: MagicMock, mock_backend_client: MagicMock
    ) -> None:
        """Test explicit edge analytics mode selection."""
        mock_backend_config.get_api_key.return_value = "key"
        mock_backend_config.get_backend_url.return_value = "https://url"

        client = OptiGenClient(execution_mode="edge_analytics")
        assert client.execution_mode == ExecutionMode.EDGE_ANALYTICS

    @patch("traigent.optigen_integration.BackendIntegratedClient")
    @patch("traigent.config.backend_config.BackendConfig")
    def test_explicit_standard_mode(
        self, mock_backend_config: MagicMock, mock_backend_client: MagicMock
    ) -> None:
        """Test explicit standard mode selection."""
        mock_backend_config.get_api_key.return_value = "key"
        mock_backend_config.get_backend_url.return_value = "https://url"

        client = OptiGenClient(execution_mode="standard")
        assert client.execution_mode == ExecutionMode.STANDARD

    @patch("traigent.optigen_integration.BackendIntegratedClient")
    @patch("traigent.config.backend_config.BackendConfig")
    def test_explicit_cloud_mode(
        self, mock_backend_config: MagicMock, mock_backend_client: MagicMock
    ) -> None:
        """Test explicit cloud mode selection."""
        mock_backend_config.get_api_key.return_value = "key"
        mock_backend_config.get_backend_url.return_value = "https://url"

        client = OptiGenClient(execution_mode="cloud")
        assert client.execution_mode == ExecutionMode.CLOUD

    @patch("traigent.optigen_integration.BackendIntegratedClient")
    @patch("traigent.config.backend_config.BackendConfig")
    @patch.dict("os.environ", {"TRAIGENT_FORCE_LOCAL": "1"})
    def test_auto_mode_with_force_local(
        self, mock_backend_config: MagicMock, mock_backend_client: MagicMock
    ) -> None:
        """Test auto mode detection with TRAIGENT_FORCE_LOCAL."""
        mock_backend_config.get_api_key.return_value = "key"
        mock_backend_config.get_backend_url.return_value = "https://url"

        client = OptiGenClient(execution_mode="auto")
        assert client.execution_mode == ExecutionMode.EDGE_ANALYTICS

    @patch("traigent.optigen_integration.BackendIntegratedClient")
    @patch("traigent.config.backend_config.BackendConfig")
    @patch.dict("os.environ", {"TRAIGENT_FORCE_HYBRID": "1"})
    def test_auto_mode_with_force_hybrid(
        self, mock_backend_config: MagicMock, mock_backend_client: MagicMock
    ) -> None:
        """Test auto mode detection with TRAIGENT_FORCE_HYBRID."""
        mock_backend_config.get_api_key.return_value = "key"
        mock_backend_config.get_backend_url.return_value = "https://url"

        client = OptiGenClient(execution_mode="auto")
        assert client.execution_mode == ExecutionMode.STANDARD

    @patch("traigent.optigen_integration.BackendIntegratedClient")
    @patch("traigent.config.backend_config.BackendConfig")
    @patch.dict("os.environ", {"TRAIGENT_FORCE_CLOUD": "1"})
    def test_auto_mode_with_force_cloud(
        self, mock_backend_config: MagicMock, mock_backend_client: MagicMock
    ) -> None:
        """Test auto mode detection with TRAIGENT_FORCE_CLOUD."""
        mock_backend_config.get_api_key.return_value = "key"
        mock_backend_config.get_backend_url.return_value = "https://url"

        client = OptiGenClient(execution_mode="auto")
        assert client.execution_mode == ExecutionMode.CLOUD

    @patch("traigent.optigen_integration.BackendIntegratedClient")
    @patch("traigent.config.backend_config.BackendConfig")
    @patch.dict("os.environ", {}, clear=True)
    def test_auto_mode_with_explicit_api_key(
        self, mock_backend_config: MagicMock, mock_backend_client: MagicMock
    ) -> None:
        """Test auto mode defaults to cloud when API key is provided."""
        mock_backend_config.get_api_key.return_value = "key"
        mock_backend_config.get_backend_url.return_value = "https://url"

        client = OptiGenClient(api_key="explicit_key", execution_mode="auto")
        assert client.execution_mode == ExecutionMode.CLOUD

    @patch("traigent.optigen_integration.BackendIntegratedClient")
    @patch("traigent.config.backend_config.BackendConfig")
    @patch.dict("os.environ", {}, clear=True)
    def test_auto_mode_without_api_key(
        self, mock_backend_config: MagicMock, mock_backend_client: MagicMock
    ) -> None:
        """Test auto mode defaults to edge analytics without API key."""
        mock_backend_config.get_api_key.return_value = None
        mock_backend_config.get_backend_url.return_value = "https://url"

        client = OptiGenClient(execution_mode="auto")
        assert client.execution_mode == ExecutionMode.EDGE_ANALYTICS


class TestCheckPrivacyRequirements:
    """Tests for privacy requirements detection."""

    @patch("traigent.optigen_integration.BackendIntegratedClient")
    @patch("traigent.config.backend_config.BackendConfig")
    @patch.dict("os.environ", {"TRAIGENT_PRIVATE_DATA": "1"})
    def test_privacy_with_private_data_env(
        self, mock_backend_config: MagicMock, mock_backend_client: MagicMock
    ) -> None:
        """Test privacy requirement detection with TRAIGENT_PRIVATE_DATA."""
        mock_backend_config.get_api_key.return_value = "key"
        mock_backend_config.get_backend_url.return_value = "https://url"

        client = OptiGenClient(execution_mode="auto")
        assert client._check_privacy_requirements() is True

    @patch("traigent.optigen_integration.BackendIntegratedClient")
    @patch("traigent.config.backend_config.BackendConfig")
    @patch.dict("os.environ", {"TRAIGENT_COMPLIANCE_MODE": "1"})
    def test_privacy_with_compliance_mode(
        self, mock_backend_config: MagicMock, mock_backend_client: MagicMock
    ) -> None:
        """Test privacy requirement detection with TRAIGENT_COMPLIANCE_MODE."""
        mock_backend_config.get_api_key.return_value = "key"
        mock_backend_config.get_backend_url.return_value = "https://url"

        client = OptiGenClient(execution_mode="auto")
        assert client._check_privacy_requirements() is True

    @patch("traigent.optigen_integration.BackendIntegratedClient")
    @patch("traigent.config.backend_config.BackendConfig")
    @patch("os.path.exists", return_value=True)
    @patch.dict("os.environ", {}, clear=True)
    def test_privacy_with_optigen_private_file(
        self,
        mock_exists: MagicMock,
        mock_backend_config: MagicMock,
        mock_backend_client: MagicMock,
    ) -> None:
        """Test privacy requirement detection with .optigen-private file."""
        mock_backend_config.get_api_key.return_value = "key"
        mock_backend_config.get_backend_url.return_value = "https://url"

        client = OptiGenClient(execution_mode="auto")
        assert client._check_privacy_requirements() is True

    @patch("traigent.optigen_integration.BackendIntegratedClient")
    @patch("traigent.config.backend_config.BackendConfig")
    @patch("os.path.exists", return_value=False)
    @patch.dict("os.environ", {}, clear=True)
    def test_no_privacy_requirements(
        self,
        mock_exists: MagicMock,
        mock_backend_config: MagicMock,
        mock_backend_client: MagicMock,
    ) -> None:
        """Test no privacy requirements detected."""
        mock_backend_config.get_api_key.return_value = "key"
        mock_backend_config.get_backend_url.return_value = "https://url"

        client = OptiGenClient(execution_mode="auto")
        assert client._check_privacy_requirements() is False


class TestConfigurationNormalization:
    """Tests for configuration space normalization."""

    @patch("traigent.optigen_integration.BackendIntegratedClient")
    @patch("traigent.config.backend_config.BackendConfig")
    def test_normalize_with_fallback_model(
        self, mock_backend_config: MagicMock, mock_backend_client: MagicMock
    ) -> None:
        """Test configuration normalization with fallback model."""
        mock_backend_config.get_api_key.return_value = "key"
        mock_backend_config.get_backend_url.return_value = "https://url"

        client = OptiGenClient()
        normalized, defaults = client._normalise_configuration_space(
            {}, fallback_model="gpt-3.5-turbo"
        )

        assert normalized["model"] == ["gpt-3.5-turbo"]
        assert normalized["temperature"] == [0.7]
        assert normalized["max_tokens"] == [512]
        assert normalized["top_p"] == [1.0]
        assert defaults["model"] == "gpt-3.5-turbo"

    @patch("traigent.optigen_integration.BackendIntegratedClient")
    @patch("traigent.config.backend_config.BackendConfig")
    def test_normalize_preserves_existing_values(
        self, mock_backend_config: MagicMock, mock_backend_client: MagicMock
    ) -> None:
        """Test normalization preserves existing configuration values."""
        mock_backend_config.get_api_key.return_value = "key"
        mock_backend_config.get_backend_url.return_value = "https://url"

        client = OptiGenClient()
        config_space = {
            "model": ["gpt-4"],
            "temperature": [0.5, 1.0],
            "max_tokens": [256],
        }
        normalized, defaults = client._normalise_configuration_space(config_space)

        assert normalized["model"] == ["gpt-4"]
        assert normalized["temperature"] == [0.5, 1.0]
        assert normalized["max_tokens"] == [256]
        assert defaults["model"] == "gpt-4"
        assert defaults["temperature"] == 0.5

    @patch("traigent.optigen_integration.BackendIntegratedClient")
    @patch("traigent.config.backend_config.BackendConfig")
    def test_normalize_with_agent_platform(
        self, mock_backend_config: MagicMock, mock_backend_client: MagicMock
    ) -> None:
        """Test normalization includes agent_platform in defaults."""
        mock_backend_config.get_api_key.return_value = "key"
        mock_backend_config.get_backend_url.return_value = "https://url"

        client = OptiGenClient()
        config_space = {"agent_platform": "langchain"}
        normalized, defaults = client._normalise_configuration_space(config_space)

        assert defaults["agent_platform"] == "langchain"

    @patch("traigent.optigen_integration.BackendIntegratedClient")
    @patch("traigent.config.backend_config.BackendConfig")
    def test_normalize_default_agent_platform(
        self, mock_backend_config: MagicMock, mock_backend_client: MagicMock
    ) -> None:
        """Test normalization sets default agent_platform."""
        mock_backend_config.get_api_key.return_value = "key"
        mock_backend_config.get_backend_url.return_value = "https://url"

        client = OptiGenClient()
        normalized, defaults = client._normalise_configuration_space({})

        assert defaults["agent_platform"] == "openai"


class TestStaticHelpers:
    """Tests for static helper methods."""

    def test_has_parameter_value_with_none(self) -> None:
        """Test _has_parameter_value returns False for None."""
        assert OptiGenClient._has_parameter_value(None) is False

    def test_has_parameter_value_with_empty_list(self) -> None:
        """Test _has_parameter_value returns False for empty list."""
        assert OptiGenClient._has_parameter_value([]) is False

    def test_has_parameter_value_with_non_empty_list(self) -> None:
        """Test _has_parameter_value returns True for non-empty list."""
        assert OptiGenClient._has_parameter_value([1, 2, 3]) is True

    def test_has_parameter_value_with_empty_dict(self) -> None:
        """Test _has_parameter_value returns False for empty dict."""
        assert OptiGenClient._has_parameter_value({}) is False

    def test_has_parameter_value_with_non_empty_dict(self) -> None:
        """Test _has_parameter_value returns True for non-empty dict."""
        assert OptiGenClient._has_parameter_value({"key": "value"}) is True

    def test_has_parameter_value_with_scalar(self) -> None:
        """Test _has_parameter_value returns True for scalar values."""
        assert OptiGenClient._has_parameter_value(42) is True
        assert OptiGenClient._has_parameter_value("value") is True

    def test_extract_default_value_from_none(self) -> None:
        """Test _extract_default_value returns None for None input."""
        assert OptiGenClient._extract_default_value(None) is None

    def test_extract_default_value_from_list(self) -> None:
        """Test _extract_default_value extracts first element from list."""
        assert OptiGenClient._extract_default_value([1, 2, 3]) == 1
        assert OptiGenClient._extract_default_value([]) is None

    def test_extract_default_value_from_tuple_range(self) -> None:
        """Test _extract_default_value computes midpoint for numeric tuple."""
        assert OptiGenClient._extract_default_value((0.0, 1.0)) == 0.5
        assert OptiGenClient._extract_default_value((10, 20)) == 15

    def test_extract_default_value_from_tuple_first(self) -> None:
        """Test _extract_default_value extracts first element from non-numeric tuple."""
        assert OptiGenClient._extract_default_value(("a", "b", "c")) == "a"
        assert OptiGenClient._extract_default_value(()) is None

    def test_extract_default_value_from_dict(self) -> None:
        """Test _extract_default_value extracts from dict with known keys."""
        assert OptiGenClient._extract_default_value({"value": 42}) == 42
        assert OptiGenClient._extract_default_value({"default": 100}) == 100
        assert OptiGenClient._extract_default_value({"initial": 50}) == 50
        assert OptiGenClient._extract_default_value({"low": 10}) == 10
        assert OptiGenClient._extract_default_value({"min": 5}) == 5

    def test_extract_default_value_from_dict_empty(self) -> None:
        """Test _extract_default_value returns None for empty dict."""
        assert OptiGenClient._extract_default_value({}) is None

    def test_extract_default_value_from_scalar(self) -> None:
        """Test _extract_default_value returns scalar as-is."""
        assert OptiGenClient._extract_default_value(42) == 42
        assert OptiGenClient._extract_default_value("value") == "value"

    def test_apply_config_defaults(self) -> None:
        """Test _apply_config_defaults merges config with defaults."""
        defaults = {"model": "gpt-3.5-turbo", "temperature": 0.7, "extra": None}
        config = {"temperature": 0.9, "max_tokens": 512}

        result = OptiGenClient._apply_config_defaults(config, defaults)

        assert result["model"] == "gpt-3.5-turbo"
        assert result["temperature"] == 0.9
        assert result["max_tokens"] == 512
        assert "extra" not in result

    def test_apply_config_defaults_empty_config(self) -> None:
        """Test _apply_config_defaults with empty config."""
        defaults = {"model": "gpt-4", "temperature": 0.5}
        config = {}

        result = OptiGenClient._apply_config_defaults(config, defaults)

        assert result == defaults


class TestOptimizeValidation:
    """Tests for optimize method validation."""

    @pytest.mark.asyncio
    async def test_optimize_missing_model_raises_error(self) -> None:
        """Test optimize raises error when model is missing in standard mode."""
        with patch("traigent.optigen_integration.BackendIntegratedClient"):
            with patch("traigent.config.backend_config.BackendConfig") as mock_config:
                mock_config.get_api_key.return_value = "key"
                mock_config.get_backend_url.return_value = "https://url"
                # Use standard mode which doesn't auto-fill model
                client = OptiGenClient(execution_mode="standard")

                def test_func() -> str:
                    return "test"

                dataset = {"examples": []}
                config_space = {}  # No model specified
                objectives = ["accuracy"]

                with pytest.raises(ValueError, match="missing required entries: model"):
                    await client.optimize(
                        test_func, dataset, config_space, objectives, max_trials=10
                    )

    @pytest.mark.asyncio
    async def test_optimize_edge_analytics_missing_agent_builder(self) -> None:
        """Test optimize raises error when agent builder is missing in edge analytics mode."""
        with patch("traigent.optigen_integration.BackendIntegratedClient"):
            with patch("traigent.config.backend_config.BackendConfig") as mock_config:
                mock_config.get_api_key.return_value = "key"
                mock_config.get_backend_url.return_value = "https://url"
                # Create client WITHOUT agent_builder
                client = OptiGenClient(execution_mode="edge_analytics")

                def test_func() -> str:
                    return "test"

                dataset = {"examples": []}
                config_space = {"model": ["gpt-3.5-turbo"]}
                objectives = ["accuracy"]

                with pytest.raises(ValueError, match="Agent builder required"):
                    await client.optimize(
                        test_func, dataset, config_space, objectives, max_trials=10
                    )


class TestOptimizeHybrid:
    """Tests for hybrid/standard mode optimization."""

    @pytest.fixture
    def mock_client(self) -> OptiGenClient:
        """Create a mock OptiGenClient in standard mode."""
        with patch("traigent.optigen_integration.BackendIntegratedClient"):
            with patch("traigent.config.backend_config.BackendConfig") as mock_config:
                mock_config.get_api_key.return_value = "key"
                mock_config.get_backend_url.return_value = "https://url"
                mock_builder = Mock()
                return OptiGenClient(
                    execution_mode="standard", agent_builder=mock_builder
                )

    @pytest.mark.asyncio
    async def test_optimize_hybrid_success(self, mock_client: OptiGenClient) -> None:
        """Test successful hybrid optimization."""

        def test_func() -> str:
            return "test"

        dataset = {"examples": [{"input": "test"}]}
        config_space = {"model": ["gpt-3.5-turbo"]}
        objectives = ["accuracy"]

        # Mock backend client
        mock_client.backend_client.__aenter__ = AsyncMock(
            return_value=mock_client.backend_client
        )
        mock_client.backend_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.backend_client.create_hybrid_session = AsyncMock(
            return_value=("session_123", "token_abc", "https://optimizer.url")
        )
        mock_client.backend_client.finalize_hybrid_session = AsyncMock(
            return_value={"best_configuration": None, "status": "completed"}
        )

        # Mock optimizer client
        mock_optimizer = AsyncMock()
        mock_optimizer.__aenter__ = AsyncMock(return_value=mock_optimizer)
        mock_optimizer.__aexit__ = AsyncMock(return_value=None)
        mock_optimizer.get_next_configuration = AsyncMock(
            side_effect=[
                {
                    "has_next": True,
                    "configuration": {"model": "gpt-3.5-turbo"},
                    "trial_id": "trial_1",
                },
                {"has_next": False},
            ]
        )
        mock_optimizer.submit_metrics = AsyncMock()

        with patch(
            "traigent.optigen_integration.OptimizerDirectClient",
            return_value=mock_optimizer,
        ):
            with patch(
                "traigent.optigen_integration.LocalExecutionAdapter"
            ) as mock_adapter_class:
                mock_adapter = AsyncMock()
                mock_adapter.execute_configuration = AsyncMock(
                    return_value={
                        "metrics": {"accuracy": 0.85},
                        "execution_time": 1.5,
                        "metadata": {},
                    }
                )
                mock_adapter_class.return_value = mock_adapter

                result = await mock_client.optimize(
                    test_func, dataset, config_space, objectives, max_trials=5
                )

                assert result["execution_mode"] == "standard"
                assert result["completed_trials"] == 1

    @pytest.mark.asyncio
    async def test_optimize_hybrid_no_agent_builder_raises(
        self, mock_client: OptiGenClient
    ) -> None:
        """Test hybrid optimization raises error without agent builder."""
        mock_client.agent_builder = None

        def test_func() -> str:
            return "test"

        dataset = {"examples": []}
        config_space = {"model": ["gpt-3.5-turbo"]}
        objectives = ["accuracy"]

        mock_client.backend_client.__aenter__ = AsyncMock(
            return_value=mock_client.backend_client
        )
        mock_client.backend_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.backend_client.create_hybrid_session = AsyncMock(
            return_value=("session_123", "token_abc", "https://optimizer.url")
        )

        mock_optimizer = AsyncMock()
        mock_optimizer.__aenter__ = AsyncMock(return_value=mock_optimizer)
        mock_optimizer.__aexit__ = AsyncMock(return_value=None)

        with patch(
            "traigent.optigen_integration.OptimizerDirectClient",
            return_value=mock_optimizer,
        ):
            with pytest.raises(ValueError, match="Agent builder required"):
                await mock_client.optimize(
                    test_func, dataset, config_space, objectives, max_trials=5
                )

    @pytest.mark.asyncio
    async def test_optimize_hybrid_trial_failure_continues(
        self, mock_client: OptiGenClient
    ) -> None:
        """Test hybrid optimization continues after trial failure."""

        def test_func() -> str:
            return "test"

        dataset = {"examples": []}
        config_space = {"model": ["gpt-3.5-turbo"]}
        objectives = ["accuracy"]

        mock_client.backend_client.__aenter__ = AsyncMock(
            return_value=mock_client.backend_client
        )
        mock_client.backend_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.backend_client.create_hybrid_session = AsyncMock(
            return_value=("session_123", "token_abc", "https://optimizer.url")
        )
        mock_client.backend_client.finalize_hybrid_session = AsyncMock(
            return_value={"status": "completed"}
        )

        mock_optimizer = AsyncMock()
        mock_optimizer.__aenter__ = AsyncMock(return_value=mock_optimizer)
        mock_optimizer.__aexit__ = AsyncMock(return_value=None)
        mock_optimizer.get_next_configuration = AsyncMock(
            side_effect=[
                {
                    "has_next": True,
                    "configuration": {"model": "gpt-3.5-turbo"},
                    "trial_id": "trial_1",
                },
                {
                    "has_next": True,
                    "configuration": {"model": "gpt-3.5-turbo"},
                    "trial_id": "trial_2",
                },
                {"has_next": False},
            ]
        )
        mock_optimizer.submit_metrics = AsyncMock()

        with patch(
            "traigent.optigen_integration.OptimizerDirectClient",
            return_value=mock_optimizer,
        ):
            with patch(
                "traigent.optigen_integration.LocalExecutionAdapter"
            ) as mock_adapter_class:
                mock_adapter = AsyncMock()
                mock_adapter.execute_configuration = AsyncMock(
                    side_effect=[
                        Exception("Trial failed"),
                        {
                            "metrics": {"accuracy": 0.9},
                            "execution_time": 1.0,
                            "metadata": {},
                        },
                    ]
                )
                mock_adapter_class.return_value = mock_adapter

                result = await mock_client.optimize(
                    test_func, dataset, config_space, objectives, max_trials=5
                )

                assert result["completed_trials"] == 1


class TestOptimizeSaaS:
    """Tests for SaaS/cloud mode optimization."""

    @pytest.fixture
    def mock_client(self) -> OptiGenClient:
        """Create a mock OptiGenClient in cloud mode."""
        with patch("traigent.optigen_integration.BackendIntegratedClient"):
            with patch("traigent.config.backend_config.BackendConfig") as mock_config:
                mock_config.get_api_key.return_value = "key"
                mock_config.get_backend_url.return_value = "https://url"
                return OptiGenClient(execution_mode="cloud")

    @pytest.mark.asyncio
    async def test_optimize_saas_success(self, mock_client: OptiGenClient) -> None:
        """Test successful SaaS optimization."""

        def test_func() -> str:
            return "test"

        dataset = {"examples": []}
        config_space = {"model": ["gpt-4"]}
        objectives = ["accuracy"]

        mock_client.backend_client.__aenter__ = AsyncMock(
            return_value=mock_client.backend_client
        )
        mock_client.backend_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.backend_client.upload_dataset = AsyncMock(
            return_value={"dataset_id": "dataset_123"}
        )
        mock_client.backend_client.create_optimization_session = AsyncMock(
            return_value={"session_id": "session_456"}
        )
        mock_client.backend_client.get_session_status = AsyncMock(
            side_effect=[
                {"status": "RUNNING", "completed_trials": 1},
                {"status": "COMPLETED", "completed_trials": 5},
            ]
        )
        mock_client.backend_client.get_optimization_results = AsyncMock(
            return_value={
                "best_configuration": {"model": "gpt-4"},
                "metrics": {"accuracy": 0.95},
            }
        )

        result = await mock_client.optimize(
            test_func, dataset, config_space, objectives, max_trials=5
        )

        assert result["execution_mode"] == "cloud"
        assert "best_configuration" in result

    @pytest.mark.asyncio
    async def test_optimize_saas_missing_upload_dataset(
        self, mock_client: OptiGenClient
    ) -> None:
        """Test SaaS optimization raises error when upload_dataset is missing."""

        def test_func() -> str:
            return "test"

        dataset = {"examples": []}
        config_space = {"model": ["gpt-4"]}
        objectives = ["accuracy"]

        mock_client.backend_client.__aenter__ = AsyncMock(
            return_value=mock_client.backend_client
        )
        mock_client.backend_client.__aexit__ = AsyncMock(return_value=None)
        # Remove upload_dataset method
        delattr(mock_client.backend_client, "upload_dataset")

        with pytest.raises(OptimizationError, match="does not support upload_dataset"):
            await mock_client.optimize(
                test_func, dataset, config_space, objectives, max_trials=5
            )

    @pytest.mark.asyncio
    async def test_optimize_saas_custom_poll_interval(
        self, mock_client: OptiGenClient
    ) -> None:
        """Test SaaS optimization with custom poll interval."""

        def test_func() -> str:
            return "test"

        dataset = {"examples": []}
        config_space = {"model": ["gpt-4"]}
        objectives = ["accuracy"]

        mock_client.backend_client.__aenter__ = AsyncMock(
            return_value=mock_client.backend_client
        )
        mock_client.backend_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.backend_client.upload_dataset = AsyncMock(
            return_value={"dataset_id": "dataset_123"}
        )
        mock_client.backend_client.create_optimization_session = AsyncMock(
            return_value={"session_id": "session_456"}
        )
        mock_client.backend_client.get_session_status = AsyncMock(
            return_value={"status": "COMPLETED", "completed_trials": 5}
        )
        mock_client.backend_client.get_optimization_results = AsyncMock(
            return_value={"best_configuration": {"model": "gpt-4"}}
        )

        await mock_client.optimize(
            test_func,
            dataset,
            config_space,
            objectives,
            max_trials=5,
            optimization_config={"poll_interval": 2.0},
        )

        # Verify that get_session_status was called
        mock_client.backend_client.get_session_status.assert_called()

    @pytest.mark.asyncio
    async def test_optimize_saas_invalid_poll_interval(
        self, mock_client: OptiGenClient
    ) -> None:
        """Test SaaS optimization handles invalid poll interval."""

        def test_func() -> str:
            return "test"

        dataset = {"examples": []}
        config_space = {"model": ["gpt-4"]}
        objectives = ["accuracy"]

        mock_client.backend_client.__aenter__ = AsyncMock(
            return_value=mock_client.backend_client
        )
        mock_client.backend_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.backend_client.upload_dataset = AsyncMock(
            return_value={"dataset_id": "dataset_123"}
        )
        mock_client.backend_client.create_optimization_session = AsyncMock(
            return_value={"session_id": "session_456"}
        )
        mock_client.backend_client.get_session_status = AsyncMock(
            return_value={"status": "COMPLETED", "completed_trials": 5}
        )
        mock_client.backend_client.get_optimization_results = AsyncMock(
            return_value={"best_configuration": {"model": "gpt-4"}}
        )

        # Should use 0.1 as minimum poll interval
        result = await mock_client.optimize(
            test_func,
            dataset,
            config_space,
            objectives,
            max_trials=5,
            optimization_config={"poll_interval": -1.0},
        )
        assert result is not None  # Method returns results


class TestOptimizeLocal:
    """Tests for edge analytics/local mode optimization."""

    @pytest.fixture
    def mock_client(self) -> OptiGenClient:
        """Create a mock OptiGenClient in edge analytics mode."""
        with patch("traigent.optigen_integration.BackendIntegratedClient"):
            with patch("traigent.config.backend_config.BackendConfig") as mock_config:
                mock_config.get_api_key.return_value = "key"
                mock_config.get_backend_url.return_value = "https://url"
                mock_builder = Mock()
                return OptiGenClient(
                    execution_mode="edge_analytics", agent_builder=mock_builder
                )

    @pytest.mark.asyncio
    async def test_optimize_local_success(self, mock_client: OptiGenClient) -> None:
        """Test successful local optimization."""

        def test_func() -> str:
            return "test"

        dataset = {"examples": []}
        config_space = {"model": ["gpt-3.5-turbo"]}
        objectives = ["accuracy"]

        mock_optimizer = Mock()
        mock_optimizer.should_stop.return_value = False
        mock_optimizer.suggest_next_trial.side_effect = [
            {"model": "gpt-3.5-turbo", "temperature": 0.7},
            OptimizationError("Search space exhausted"),
        ]
        mock_optimizer.update_best = Mock()

        with patch(
            "traigent.optigen_integration.get_optimizer", return_value=mock_optimizer
        ):
            with patch(
                "traigent.optigen_integration.LocalExecutionAdapter"
            ) as mock_adapter_class:
                mock_adapter = AsyncMock()
                mock_adapter.execute_configuration = AsyncMock(
                    return_value={
                        "metrics": {"accuracy": 0.88},
                        "execution_time": 1.2,
                        "metadata": {},
                    }
                )
                mock_adapter_class.return_value = mock_adapter

                result = await mock_client.optimize(
                    test_func, dataset, config_space, objectives, max_trials=10
                )

                assert result["execution_mode"] == "edge_analytics"
                assert result["completed_trials"] == 1
                assert result["status"] == "completed"
                assert result["best_configuration"] is not None

    @pytest.mark.asyncio
    async def test_optimize_local_no_trials(self, mock_client: OptiGenClient) -> None:
        """Test local optimization with max_trials=0."""

        def test_func() -> str:
            return "test"

        dataset = {"examples": []}
        config_space = {"model": ["gpt-3.5-turbo"]}
        objectives = ["accuracy"]

        result = await mock_client.optimize(
            test_func, dataset, config_space, objectives, max_trials=0
        )

        assert result["execution_mode"] == "edge_analytics"
        assert result["completed_trials"] == 0
        assert result["status"] == "no_trials"
        assert result["best_configuration"] is None

    @pytest.mark.asyncio
    async def test_optimize_local_missing_agent_builder(
        self, mock_client: OptiGenClient
    ) -> None:
        """Test local optimization raises error without agent builder."""
        mock_client.agent_builder = None

        def test_func() -> str:
            return "test"

        dataset = {"examples": []}
        config_space = {"model": ["gpt-3.5-turbo"]}
        objectives = ["accuracy"]

        with pytest.raises(ValueError, match="Agent builder required"):
            await mock_client.optimize(
                test_func, dataset, config_space, objectives, max_trials=10
            )

    @pytest.mark.asyncio
    async def test_optimize_local_trial_failure(
        self, mock_client: OptiGenClient
    ) -> None:
        """Test local optimization handles trial failures."""

        def test_func() -> str:
            return "test"

        dataset = {"examples": []}
        config_space = {"model": ["gpt-3.5-turbo"]}
        objectives = ["accuracy"]

        mock_optimizer = Mock()
        mock_optimizer.should_stop.return_value = False
        mock_optimizer.suggest_next_trial.side_effect = [
            {"model": "gpt-3.5-turbo"},
            OptimizationError("Exhausted"),
        ]
        mock_optimizer.update_best = Mock()

        with patch(
            "traigent.optigen_integration.get_optimizer", return_value=mock_optimizer
        ):
            with patch(
                "traigent.optigen_integration.LocalExecutionAdapter"
            ) as mock_adapter_class:
                mock_adapter = AsyncMock()
                mock_adapter.execute_configuration = AsyncMock(
                    side_effect=Exception("Execution failed")
                )
                mock_adapter_class.return_value = mock_adapter

                result = await mock_client.optimize(
                    test_func, dataset, config_space, objectives, max_trials=10
                )

                assert result["completed_trials"] == 0
                assert len(result["all_results"]) == 0

    @pytest.mark.asyncio
    async def test_optimize_local_optimizer_stop_condition(
        self, mock_client: OptiGenClient
    ) -> None:
        """Test local optimization respects optimizer stop condition."""

        def test_func() -> str:
            return "test"

        dataset = {"examples": []}
        config_space = {"model": ["gpt-3.5-turbo"]}
        objectives = ["accuracy"]

        mock_optimizer = Mock()
        mock_optimizer.should_stop.side_effect = [False, True]
        mock_optimizer.suggest_next_trial.return_value = {"model": "gpt-3.5-turbo"}
        mock_optimizer.update_best = Mock()

        with patch(
            "traigent.optigen_integration.get_optimizer", return_value=mock_optimizer
        ):
            with patch(
                "traigent.optigen_integration.LocalExecutionAdapter"
            ) as mock_adapter_class:
                mock_adapter = AsyncMock()
                mock_adapter.execute_configuration = AsyncMock(
                    return_value={
                        "metrics": {"accuracy": 0.9},
                        "execution_time": 1.0,
                        "metadata": {},
                    }
                )
                mock_adapter_class.return_value = mock_adapter

                result = await mock_client.optimize(
                    test_func, dataset, config_space, objectives, max_trials=10
                )

                assert result["completed_trials"] == 1

    @pytest.mark.asyncio
    async def test_optimize_local_custom_optimizer(
        self, mock_client: OptiGenClient
    ) -> None:
        """Test local optimization with custom optimizer algorithm."""

        def test_func() -> str:
            return "test"

        dataset = {"examples": []}
        config_space = {"model": ["gpt-3.5-turbo"]}
        objectives = ["accuracy"]

        mock_optimizer = Mock()
        mock_optimizer.should_stop.return_value = False
        mock_optimizer.suggest_next_trial.side_effect = [
            {"model": "gpt-3.5-turbo"},
            OptimizationError("Exhausted"),
        ]
        mock_optimizer.update_best = Mock()

        with patch(
            "traigent.optigen_integration.get_optimizer", return_value=mock_optimizer
        ) as mock_get_optimizer:
            with patch(
                "traigent.optigen_integration.LocalExecutionAdapter"
            ) as mock_adapter_class:
                mock_adapter = AsyncMock()
                mock_adapter.execute_configuration = AsyncMock(
                    return_value={
                        "metrics": {"accuracy": 0.85},
                        "execution_time": 1.0,
                        "metadata": {},
                    }
                )
                mock_adapter_class.return_value = mock_adapter

                await mock_client.optimize(
                    test_func,
                    dataset,
                    config_space,
                    objectives,
                    max_trials=10,
                    optimization_config={"algorithm": "bayesian"},
                )

                mock_get_optimizer.assert_called_once()
                call_args = mock_get_optimizer.call_args
                assert call_args[0][0] == "bayesian"

    @pytest.mark.asyncio
    async def test_optimize_local_optimizer_initialization_failure(
        self, mock_client: OptiGenClient
    ) -> None:
        """Test local optimization handles optimizer initialization failure."""

        def test_func() -> str:
            return "test"

        dataset = {"examples": []}
        config_space = {"model": ["gpt-3.5-turbo"]}
        objectives = ["accuracy"]

        with patch(
            "traigent.optigen_integration.get_optimizer",
            side_effect=OptimizationError("Invalid optimizer"),
        ):
            with pytest.raises(ValueError, match="Failed to initialize optimizer"):
                await mock_client.optimize(
                    test_func, dataset, config_space, objectives, max_trials=10
                )

    @pytest.mark.asyncio
    async def test_optimize_local_metric_extraction(
        self, mock_client: OptiGenClient
    ) -> None:
        """Test local optimization metric extraction logic."""

        def test_func() -> str:
            return "test"

        dataset = {"examples": []}
        config_space = {"model": ["gpt-3.5-turbo"]}
        objectives = ["custom_metric"]

        mock_optimizer = Mock()
        mock_optimizer.should_stop.return_value = False
        mock_optimizer.suggest_next_trial.side_effect = [
            {"model": "gpt-3.5-turbo"},
            {"model": "gpt-3.5-turbo"},
            {"model": "gpt-3.5-turbo"},
            OptimizationError("Exhausted"),
        ]
        mock_optimizer.update_best = Mock()

        with patch(
            "traigent.optigen_integration.get_optimizer", return_value=mock_optimizer
        ):
            with patch(
                "traigent.optigen_integration.LocalExecutionAdapter"
            ) as mock_adapter_class:
                mock_adapter = AsyncMock()
                mock_adapter.execute_configuration = AsyncMock(
                    side_effect=[
                        {
                            "metrics": {"custom_metric": 0.8},
                            "execution_time": 1.0,
                            "metadata": {},
                        },
                        {
                            "metrics": {"score": 0.9},
                            "execution_time": 1.1,
                            "metadata": {},
                        },
                        {
                            "metrics": {"other": 0.7},
                            "execution_time": 1.2,
                            "metadata": {},
                        },
                    ]
                )
                mock_adapter_class.return_value = mock_adapter

                result = await mock_client.optimize(
                    test_func, dataset, config_space, objectives, max_trials=10
                )

                assert result["completed_trials"] == 3
                assert result["best_configuration"] is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
