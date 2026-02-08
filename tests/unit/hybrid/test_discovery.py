"""Unit tests for hybrid mode configuration space discovery."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from traigent.hybrid.discovery import (
    ConfigSpaceDiscovery,
    merge_config_spaces,
    normalize_tvar_to_config_space,
    validate_config_against_tvars,
)
from traigent.hybrid.protocol import ConfigSpaceResponse, TVARDefinition


class TestConfigSpaceDiscovery:
    """Tests for ConfigSpaceDiscovery class."""

    @pytest.fixture
    def mock_transport(self) -> MagicMock:
        """Create mock transport for testing."""
        transport = MagicMock()
        transport.discover_config_space = AsyncMock(
            return_value=ConfigSpaceResponse(
                schema_version="0.9",
                capability_id="test_agent",
                tvars=[
                    TVARDefinition(
                        name="model",
                        type="enum",
                        domain={"values": ["gpt-4", "claude-3"]},
                    ),
                    TVARDefinition(
                        name="temperature",
                        type="float",
                        domain={"range": [0.0, 2.0], "resolution": 0.1},
                    ),
                ],
                constraints={
                    "structural": [
                        {
                            "id": "temp_bound",
                            "expr": "params.temperature >= 0.0",
                        }
                    ]
                },
                objectives=[
                    {"name": "accuracy", "direction": "maximize", "weight": 2.0},
                    {"name": "cost", "direction": "minimize", "weight": 1.0},
                ],
                exploration={
                    "strategy": "nsga2",
                    "budgets": {"max_trials": 25, "max_spend_usd": 5.0},
                    "convergence": {
                        "metric": "hypervolume_improvement",
                        "window": 5,
                        "threshold": 0.01,
                    },
                },
                promotion_policy={
                    "dominance": "epsilon_pareto",
                    "alpha": 0.05,
                    "min_effect": {"accuracy": 0.02},
                },
                defaults={"model": "gpt-4", "temperature": 0.4},
                measures=["accuracy", "cost"],
            )
        )
        return transport

    @pytest.fixture
    def discovery(self, mock_transport: MagicMock) -> ConfigSpaceDiscovery:
        """Create discovery instance for testing."""
        return ConfigSpaceDiscovery(mock_transport)

    @pytest.mark.asyncio
    async def test_fetch(
        self, discovery: ConfigSpaceDiscovery, mock_transport: MagicMock
    ) -> None:
        """Test fetching config space."""
        response = await discovery.fetch()

        assert response.schema_version == "0.9"
        assert response.capability_id == "test_agent"
        assert len(response.tvars) == 2
        mock_transport.discover_config_space.assert_called_once()

    @pytest.mark.asyncio
    async def test_fetch_caches_result(
        self, discovery: ConfigSpaceDiscovery, mock_transport: MagicMock
    ) -> None:
        """Test that fetch caches the result."""
        await discovery.fetch()
        await discovery.fetch()

        # Should only call transport once
        mock_transport.discover_config_space.assert_called_once()

    @pytest.mark.asyncio
    async def test_fetch_and_normalize(self, discovery: ConfigSpaceDiscovery) -> None:
        """Test fetching and normalizing to Traigent format."""
        config_space = await discovery.fetch_and_normalize()

        assert config_space["model"] == ["gpt-4", "claude-3"]
        assert config_space["temperature"]["low"] == 0.0
        assert config_space["temperature"]["high"] == 2.0
        assert config_space["temperature"]["step"] == 0.1

    @pytest.mark.asyncio
    async def test_get_capability_id(self, discovery: ConfigSpaceDiscovery) -> None:
        """Test getting capability ID."""
        # Before fetch
        assert discovery.get_capability_id() is None

        await discovery.fetch()
        assert discovery.get_capability_id() == "test_agent"

    @pytest.mark.asyncio
    async def test_get_tvars(self, discovery: ConfigSpaceDiscovery) -> None:
        """Test getting TVARs."""
        assert discovery.get_tvars() == []

        await discovery.fetch()
        tvars = discovery.get_tvars()
        assert len(tvars) == 2

    @pytest.mark.asyncio
    async def test_get_optional_optimization_sections(
        self, discovery: ConfigSpaceDiscovery
    ) -> None:
        """Optional optimization metadata is available after discovery."""
        await discovery.fetch()
        assert discovery.get_objectives() is not None
        assert discovery.get_exploration() is not None
        assert discovery.get_promotion_policy() is not None
        assert discovery.get_defaults() == {"model": "gpt-4", "temperature": 0.4}
        assert discovery.get_measures() == ["accuracy", "cost"]

    @pytest.mark.asyncio
    async def test_build_optimization_spec(
        self, discovery: ConfigSpaceDiscovery
    ) -> None:
        """Build optimizer-compatible spec from config-space metadata."""
        spec = await discovery.build_optimization_spec()
        assert spec["configuration_space"]["model"] == ["gpt-4", "claude-3"]
        assert spec["default_config"] == {"model": "gpt-4", "temperature": 0.4}
        assert spec["objective_schema"] is not None
        assert spec["promotion_policy"] is not None
        assert spec["runtime_overrides"]["algorithm"] == "nsga2"
        assert spec["runtime_overrides"]["max_trials"] == 25
        assert spec["runtime_overrides"]["cost_limit"] == 5.0
        assert (
            spec["runtime_overrides"]["convergence_metric"] == "hypervolume_improvement"
        )
        assert len(spec["constraints"]) >= 1
        assert spec["measures"] == ["accuracy", "cost"]

    @pytest.mark.asyncio
    async def test_build_tvl_artifact_alias(
        self, discovery: ConfigSpaceDiscovery
    ) -> None:
        """Legacy alias delegates to build_optimization_spec()."""
        spec = await discovery.build_tvl_artifact()
        assert "configuration_space" in spec

    @pytest.mark.asyncio
    async def test_clear_cache(
        self, discovery: ConfigSpaceDiscovery, mock_transport: MagicMock
    ) -> None:
        """Test clearing cache."""
        await discovery.fetch()
        discovery.clear_cache()

        # Should fetch again after cache clear
        await discovery.fetch()
        assert mock_transport.discover_config_space.call_count == 2

    @pytest.mark.asyncio
    async def test_build_optimization_spec_with_legacy_text_constraints(self) -> None:
        """Legacy textual constraint maps are normalized for parsing."""
        transport = MagicMock()
        transport.discover_config_space = AsyncMock(
            return_value=ConfigSpaceResponse(
                schema_version="0.9",
                capability_id="legacy_agent",
                tvars=[
                    TVARDefinition(
                        name="temperature",
                        type="float",
                        domain={"range": [0.0, 1.0]},
                    )
                ],
                constraints={"hard": ["params.temperature <= 1.0"]},
            )
        )
        discovery = ConfigSpaceDiscovery(transport)
        spec = await discovery.build_optimization_spec()
        assert len(spec["constraints"]) == 1


class TestConfigSpaceDiscoveryWithTools:
    """Tests for discovery with tool TVARs."""

    @pytest.fixture
    def mock_transport(self) -> MagicMock:
        """Create mock transport with tool TVARs."""
        transport = MagicMock()
        transport.discover_config_space = AsyncMock(
            return_value=ConfigSpaceResponse(
                schema_version="0.9",
                capability_id="multi_agent",
                tvars=[
                    TVARDefinition(
                        name="search_tool",
                        type="enum",
                        domain={"values": ["google", "bing"]},
                        is_tool=True,
                    ),
                    TVARDefinition(
                        name="model",
                        type="enum",
                        domain={"values": ["gpt-4"]},
                        agent="qa_agent",
                    ),
                    TVARDefinition(
                        name="temperature",
                        type="float",
                        domain={"range": [0.0, 1.0]},
                        agent="qa_agent",
                    ),
                    TVARDefinition(
                        name="summarizer_model",
                        type="enum",
                        domain={"values": ["gpt-3.5"]},
                        agent="summarizer",
                    ),
                ],
            )
        )
        return transport

    @pytest.mark.asyncio
    async def test_get_tool_tvars(self, mock_transport: MagicMock) -> None:
        """Test getting tool TVARs."""
        discovery = ConfigSpaceDiscovery(mock_transport)
        await discovery.fetch()

        tool_tvars = discovery.get_tool_tvars()
        assert len(tool_tvars) == 1
        assert tool_tvars[0].name == "search_tool"

    @pytest.mark.asyncio
    async def test_get_agents(self, mock_transport: MagicMock) -> None:
        """Test getting agent names."""
        discovery = ConfigSpaceDiscovery(mock_transport)
        await discovery.fetch()

        agents = discovery.get_agents()
        assert agents == ["qa_agent", "summarizer"]

    @pytest.mark.asyncio
    async def test_get_tvars_for_agent(self, mock_transport: MagicMock) -> None:
        """Test getting TVARs for specific agent."""
        discovery = ConfigSpaceDiscovery(mock_transport)
        await discovery.fetch()

        qa_tvars = discovery.get_tvars_for_agent("qa_agent")
        assert len(qa_tvars) == 2
        assert {t.name for t in qa_tvars} == {"model", "temperature"}


class TestNormalizeTvarToConfigSpace:
    """Tests for normalize_tvar_to_config_space function."""

    def test_enum_tvar(self) -> None:
        """Test normalizing enum TVAR."""
        tvar = TVARDefinition(
            name="model",
            type="enum",
            domain={"values": ["a", "b", "c"]},
        )
        result = normalize_tvar_to_config_space(tvar)
        assert result == ["a", "b", "c"]

    def test_bool_tvar(self) -> None:
        """Test normalizing bool TVAR."""
        tvar = TVARDefinition(name="flag", type="bool", domain={})
        result = normalize_tvar_to_config_space(tvar)
        assert result == [True, False]

    def test_int_tvar(self) -> None:
        """Test normalizing int TVAR."""
        tvar = TVARDefinition(
            name="count",
            type="int",
            domain={"range": [1, 10]},
        )
        result = normalize_tvar_to_config_space(tvar)
        assert result == {"low": 1, "high": 10, "type": "int"}

    def test_float_tvar_with_step(self) -> None:
        """Test normalizing float TVAR with resolution."""
        tvar = TVARDefinition(
            name="temp",
            type="float",
            domain={"range": [0.0, 1.0], "resolution": 0.05},
        )
        result = normalize_tvar_to_config_space(tvar)
        assert result == {"low": 0.0, "high": 1.0, "step": 0.05}

    def test_float_tvar_without_step(self) -> None:
        """Test normalizing float TVAR without resolution."""
        tvar = TVARDefinition(
            name="temp",
            type="float",
            domain={"range": [0.0, 1.0]},
        )
        result = normalize_tvar_to_config_space(tvar)
        assert result == {"low": 0.0, "high": 1.0}

    def test_str_tvar(self) -> None:
        """Test normalizing str TVAR."""
        tvar = TVARDefinition(
            name="format",
            type="str",
            domain={"values": ["json", "text"]},
        )
        result = normalize_tvar_to_config_space(tvar)
        assert result == ["json", "text"]


class TestMergeConfigSpaces:
    """Tests for merge_config_spaces function."""

    def test_merge_with_none(self) -> None:
        """Test merging with None override."""
        base = {"a": [1, 2], "b": [3, 4]}
        result = merge_config_spaces(base, None)
        assert result == base
        # Should be a copy
        assert result is not base

    def test_merge_with_override(self) -> None:
        """Test merging with override."""
        base = {"a": [1, 2], "b": [3, 4]}
        override = {"a": [5, 6], "c": [7, 8]}
        result = merge_config_spaces(base, override)

        assert result["a"] == [5, 6]  # Overridden
        assert result["b"] == [3, 4]  # From base
        assert result["c"] == [7, 8]  # From override

    def test_merge_empty_base(self) -> None:
        """Test merging empty base."""
        result = merge_config_spaces({}, {"a": [1]})
        assert result == {"a": [1]}


class TestValidateConfigAgainstTvars:
    """Tests for validate_config_against_tvars function."""

    @pytest.fixture
    def tvars(self) -> list[TVARDefinition]:
        """Sample TVAR definitions."""
        return [
            TVARDefinition(
                name="model",
                type="enum",
                domain={"values": ["gpt-4", "claude-3"]},
            ),
            TVARDefinition(
                name="temperature",
                type="float",
                domain={"range": [0.0, 2.0]},
            ),
            TVARDefinition(
                name="max_tokens",
                type="int",
                domain={"range": [100, 4096]},
            ),
            TVARDefinition(
                name="use_cache",
                type="bool",
                domain={},
                default=True,
            ),
        ]

    def test_valid_config(self, tvars: list[TVARDefinition]) -> None:
        """Test validating valid configuration."""
        config = {
            "model": "gpt-4",
            "temperature": 0.7,
            "max_tokens": 1000,
            "use_cache": True,
        }
        errors = validate_config_against_tvars(config, tvars)
        assert errors == []

    def test_missing_required_tvar(self, tvars: list[TVARDefinition]) -> None:
        """Test missing required TVAR."""
        config = {"temperature": 0.7}
        errors = validate_config_against_tvars(config, tvars)

        assert any("model" in e for e in errors)

    def test_unknown_config_key(self, tvars: list[TVARDefinition]) -> None:
        """Test unknown configuration key."""
        config = {"model": "gpt-4", "unknown_key": "value"}
        errors = validate_config_against_tvars(config, tvars)

        assert any("unknown_key" in e for e in errors)

    def test_invalid_enum_value(self, tvars: list[TVARDefinition]) -> None:
        """Test invalid enum value."""
        config = {"model": "invalid-model"}
        errors = validate_config_against_tvars(config, tvars)

        assert any("invalid-model" in e and "allowed values" in e for e in errors)

    def test_value_out_of_range(self, tvars: list[TVARDefinition]) -> None:
        """Test value out of range."""
        config = {"model": "gpt-4", "temperature": 3.0}
        errors = validate_config_against_tvars(config, tvars)

        assert any("temperature" in e and "not in range" in e for e in errors)

    def test_wrong_type_bool(self, tvars: list[TVARDefinition]) -> None:
        """Test wrong type for bool."""
        config = {"model": "gpt-4", "use_cache": "yes"}
        errors = validate_config_against_tvars(config, tvars)

        assert any("use_cache" in e and "expected bool" in e for e in errors)

    def test_wrong_type_int(self, tvars: list[TVARDefinition]) -> None:
        """Test wrong type for int."""
        config = {"model": "gpt-4", "max_tokens": "many"}
        errors = validate_config_against_tvars(config, tvars)

        assert any("max_tokens" in e and "expected int" in e for e in errors)

    def test_wrong_type_float(self, tvars: list[TVARDefinition]) -> None:
        """Test wrong type for float."""
        config = {"model": "gpt-4", "temperature": "warm"}
        errors = validate_config_against_tvars(config, tvars)

        assert any("temperature" in e and "expected float" in e for e in errors)


class TestNormalizeConstraintsForParsing:
    """Tests for _normalize_constraints_for_parsing static method."""

    def test_none_returns_none(self) -> None:
        """None constraints pass through."""
        result = ConfigSpaceDiscovery._normalize_constraints_for_parsing(None)
        assert result is None

    def test_list_returns_list(self) -> None:
        """List constraints pass through unchanged."""
        constraints = [{"id": "c1", "type": "expression", "rule": "x > 0"}]
        result = ConfigSpaceDiscovery._normalize_constraints_for_parsing(constraints)
        assert result is constraints

    def test_non_dict_non_list_returns_none(self) -> None:
        """Non-dict, non-list input returns None."""
        result = ConfigSpaceDiscovery._normalize_constraints_for_parsing(42)  # type: ignore[arg-type]
        assert result is None

    def test_typed_constraints_pass_through(self) -> None:
        """TVL 0.9 typed constraints with 'structural'/'derived' keys pass through."""
        constraints = {"structural": [{"expr": "x > 0"}], "derived": []}
        result = ConfigSpaceDiscovery._normalize_constraints_for_parsing(constraints)
        assert result is constraints

    def test_legacy_text_constraints_with_non_list_entries_skipped(self) -> None:
        """Legacy constraints where group value is not a list are skipped."""
        constraints = {"hard": "not_a_list", "soft": ["x > 0"]}
        result = ConfigSpaceDiscovery._normalize_constraints_for_parsing(constraints)
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]["rule"] == "x > 0"

    def test_legacy_dict_entries_appended(self) -> None:
        """Legacy constraint entries that are already dicts are passed through."""
        entry = {"id": "c1", "type": "expression", "rule": "x > 0"}
        constraints = {"hard": [entry]}
        result = ConfigSpaceDiscovery._normalize_constraints_for_parsing(constraints)
        assert isinstance(result, list)
        assert result[0] is entry


class TestDiscoveryGettersBeforeFetch:
    """Test discovery getters return None/empty before fetch."""

    def test_get_promotion_policy_before_fetch(self) -> None:
        """get_promotion_policy returns None when not yet fetched."""
        transport = MagicMock()
        discovery = ConfigSpaceDiscovery(transport)
        assert discovery.get_promotion_policy() is None

    def test_get_objectives_before_fetch(self) -> None:
        """get_objectives returns None before fetch."""
        transport = MagicMock()
        discovery = ConfigSpaceDiscovery(transport)
        assert discovery.get_objectives() is None

    def test_get_exploration_before_fetch(self) -> None:
        """get_exploration returns None before fetch."""
        transport = MagicMock()
        discovery = ConfigSpaceDiscovery(transport)
        assert discovery.get_exploration() is None

    def test_get_defaults_before_fetch(self) -> None:
        """get_defaults returns None before fetch."""
        transport = MagicMock()
        discovery = ConfigSpaceDiscovery(transport)
        assert discovery.get_defaults() is None

    def test_get_measures_before_fetch(self) -> None:
        """get_measures returns None before fetch."""
        transport = MagicMock()
        discovery = ConfigSpaceDiscovery(transport)
        assert discovery.get_measures() is None


class TestBuildOptimizationSpecDerivedConstraints:
    """Test build_optimization_spec with derived constraints."""

    @pytest.mark.asyncio
    async def test_derived_constraints_compiled(self) -> None:
        """Derived constraints from response are compiled into wrappers."""
        transport = MagicMock()
        transport.discover_config_space = AsyncMock(
            return_value=ConfigSpaceResponse(
                schema_version="0.9",
                capability_id="derived_agent",
                tvars=[
                    TVARDefinition(
                        name="temperature",
                        type="float",
                        domain={"range": [0.0, 1.0]},
                    )
                ],
                constraints={
                    "derived": [
                        {"require": "params.temperature <= 0.9", "when": "True"}
                    ]
                },
            )
        )
        discovery = ConfigSpaceDiscovery(transport)
        spec = await discovery.build_optimization_spec()
        assert len(spec["constraints"]) >= 1
        assert spec["derived_constraints"] is not None
