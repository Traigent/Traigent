"""Tests for TraigentConfig type and execution mode helpers."""

import inspect

import pytest

from traigent.config.types import (
    ExecutionMode,
    TraigentConfig,
    _reset_deprecation_warning_state_for_tests,
    accepted_algorithm_values,
    accepted_execution_mode_values,
    resolve_execution_mode,
    validate_algorithm_name,
    validate_execution_mode,
)
from traigent.utils.exceptions import ConfigurationError, ValidationError


@pytest.fixture(autouse=True)
def reset_deprecation_warning_state():
    _reset_deprecation_warning_state_for_tests()
    yield
    _reset_deprecation_warning_state_for_tests()


class TestTraigentConfig:
    """Test suite for TraigentConfig."""

    def test_create_empty_config(self):
        """Test creating empty configuration."""
        config = TraigentConfig()
        assert config.model is None
        assert config.temperature is None
        assert config.custom_params == {}

    def test_create_config_with_values(self):
        """Test creating configuration with values."""
        config = TraigentConfig(model="GPT-4o", temperature=0.7, max_tokens=1000)
        assert config.model == "GPT-4o"
        assert config.temperature == 0.7
        assert config.max_tokens == 1000

    def test_deprecated_edge_analytics_config_warns_once_but_still_works(self):
        """Deprecated edge_analytics config remains behavior-compatible."""
        import warnings

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            first = TraigentConfig(execution_mode="edge_analytics")
            second = TraigentConfig(execution_mode="edge_analytics")

        messages = [
            str(warning.message)
            for warning in caught
            if issubclass(warning.category, DeprecationWarning)
        ]
        assert len(messages) == 1
        assert "execution_mode='edge_analytics' is deprecated" in messages[0]
        assert "algorithm='grid'" in messages[0]
        assert "algorithm='random'" in messages[0]
        assert "prefer local over edge_analytics" in messages[0].lower()
        assert "future major" in messages[0]
        assert first.execution_mode == "local"
        assert second.execution_mode == "local"
        assert first.is_local_mode() is True
        assert second.is_local_mode() is True

    @pytest.mark.parametrize("algorithm", ("grid", "random"))
    def test_preferred_local_config_emits_no_deprecation_warning(self, algorithm):
        import warnings

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            config = TraigentConfig(algorithm=algorithm, offline=True)

        assert config.algorithm == algorithm
        assert config.offline is True
        assert not [
            warning
            for warning in caught
            if issubclass(warning.category, DeprecationWarning)
        ]

    def test_temperature_validation(self):
        """Test temperature validation."""
        # Valid temperatures
        TraigentConfig(temperature=0.0)
        TraigentConfig(temperature=1.0)
        TraigentConfig(temperature=2.0)

        # Invalid temperatures
        with pytest.raises(ValidationError, match="temperature.*below minimum"):
            TraigentConfig(temperature=-0.1)

        with pytest.raises(ValidationError, match="temperature.*exceeds maximum"):
            TraigentConfig(temperature=2.1)

    def test_max_tokens_validation(self):
        """Test max_tokens validation."""
        # Valid max_tokens
        TraigentConfig(max_tokens=1)
        TraigentConfig(max_tokens=1000)

        # Invalid max_tokens
        with pytest.raises(ValidationError, match="max_tokens.*positive"):
            TraigentConfig(max_tokens=0)

        with pytest.raises(ValidationError, match="max_tokens.*positive"):
            TraigentConfig(max_tokens=-1)

    def test_top_p_validation(self):
        """Test top_p validation."""
        # Valid top_p
        TraigentConfig(top_p=0.0)
        TraigentConfig(top_p=0.5)
        TraigentConfig(top_p=1.0)

        # Invalid top_p
        with pytest.raises(ValidationError, match="top_p.*below minimum"):
            TraigentConfig(top_p=-0.1)

        with pytest.raises(ValidationError, match="top_p.*exceeds maximum"):
            TraigentConfig(top_p=1.1)

    def test_penalty_validation(self):
        """Test penalty validation."""
        # Valid penalties
        TraigentConfig(frequency_penalty=-2.0)
        TraigentConfig(frequency_penalty=0.0)
        TraigentConfig(frequency_penalty=2.0)
        TraigentConfig(presence_penalty=-2.0)
        TraigentConfig(presence_penalty=2.0)

        # Invalid penalties
        with pytest.raises(ValidationError, match="frequency_penalty.*below minimum"):
            TraigentConfig(frequency_penalty=-2.1)

        with pytest.raises(ValidationError, match="presence_penalty.*exceeds maximum"):
            TraigentConfig(presence_penalty=2.1)

    def test_to_dict(self):
        """Test converting config to dictionary."""
        config = TraigentConfig(
            model="GPT-4o",
            temperature=0.7,
            custom_params={"custom_key": "custom_value"},
        )

        result = config.to_dict()
        expected = {
            "model": "GPT-4o",
            "temperature": 0.7,
            "custom_key": "custom_value",
        }
        assert result == expected

    def test_to_dict_excludes_none_values(self):
        """Test that to_dict excludes None values."""
        config = TraigentConfig(model="GPT-4o")
        result = config.to_dict()

        assert "model" in result
        assert "temperature" not in result
        assert "max_tokens" not in result

    def test_to_dict_excludes_default_execution_mode(self):
        """Default edge_analytics must not leak into partial config merges."""
        assert TraigentConfig().to_dict() == {}
        assert TraigentConfig(model="GPT-4o").to_dict() == {"model": "GPT-4o"}

    def test_to_dict_omits_legacy_execution_surface(self):
        """Public serialization should expose algorithm/offline, not legacy knobs."""
        with pytest.warns(DeprecationWarning):
            config = TraigentConfig(
                algorithm="grid",
                offline=True,
                execution_mode="hybrid",
                privacy_enabled=True,
                custom_params={
                    "custom_key": "custom_value",
                    "execution_mode": "custom",
                    "privacy_enabled": "custom",
                },
            )

        assert config.to_dict() == {
            "algorithm": "grid",
            "offline": True,
            "custom_key": "custom_value",
        }

    def test_public_signature_uses_algorithm_offline_surface(self):
        """TraigentConfig help/inspect output should not advertise legacy knobs."""
        public_signature = str(inspect.signature(TraigentConfig))

        assert "algorithm" in public_signature
        assert "offline" in public_signature
        for legacy_name in (
            "execution_mode",
            "privacy_enabled",
            "edge_analytics",
            "hybrid_api",
            "cloud_fallback_policy",
        ):
            assert legacy_name not in public_signature

    def test_from_dict(self):
        """Test creating config from dictionary."""
        config_dict = {
            "model": "GPT-4o",
            "temperature": 0.7,
            "custom_key": "custom_value",
        }

        config = TraigentConfig.from_dict(config_dict)

        assert config.model == "GPT-4o"
        assert config.temperature == 0.7
        assert config.custom_params == {"custom_key": "custom_value"}

    def test_merge_with_config(self):
        """Test merging with another config."""
        config1 = TraigentConfig(model="gpt-4o-mini", temperature=0.5)
        config2 = TraigentConfig(temperature=0.8, max_tokens=1000)

        merged = config1.merge(config2)

        assert merged.model == "gpt-4o-mini"  # From config1
        assert merged.temperature == 0.8  # From config2 (overrides)
        assert merged.max_tokens == 1000  # From config2

    def test_merge_with_dict(self):
        """Test merging with dictionary."""
        config = TraigentConfig(model="GPT-4o", temperature=0.5)
        override = {"temperature": 0.8, "max_tokens": 1000}

        merged = config.merge(override)

        assert merged.model == "GPT-4o"
        assert merged.temperature == 0.8
        assert merged.max_tokens == 1000

    def test_merge_partial_config_preserves_execution_mode(self):
        """A default-valued override must not reset a hybrid base config."""
        base = TraigentConfig(execution_mode="hybrid", privacy_enabled=True)
        override = TraigentConfig(model="GPT-4o")

        merged = base.merge(override)

        assert merged.model == "GPT-4o"
        assert merged.execution_mode == "hybrid"
        assert merged.privacy_enabled is True

    def test_merge_dict_can_explicitly_reset_execution_mode_to_default(self):
        """Dict overrides preserve explicitly supplied default values."""
        base = TraigentConfig(execution_mode="hybrid", privacy_enabled=True)

        merged = base.merge({"execution_mode": "local"})

        assert merged.execution_mode == "local"
        assert merged.privacy_enabled is True

    def test_repr(self):
        """Test string representation."""
        config = TraigentConfig(model="GPT-4o", temperature=0.7)
        repr_str = repr(config)

        assert "TraigentConfig" in repr_str
        assert "model='GPT-4o'" in repr_str
        assert "temperature=0.7" in repr_str
        assert "execution_mode" not in repr_str
        assert "privacy_enabled" not in repr_str

    def test_repr_omits_legacy_execution_surface(self):
        """Public repr should expose algorithm/offline, not legacy knobs."""
        with pytest.warns(DeprecationWarning):
            config = TraigentConfig(
                algorithm="grid",
                offline=True,
                execution_mode="hybrid",
                privacy_enabled=True,
            )

        repr_str = repr(config)

        assert "algorithm='grid'" in repr_str
        assert "offline=True" in repr_str
        assert "execution_mode" not in repr_str
        assert "privacy_enabled" not in repr_str

    def test_custom_params(self):
        """Test handling of custom parameters."""
        config = TraigentConfig(
            model="GPT-4o",
            custom_params={"custom_setting": True, "api_version": "2023-07-01"},
        )

        result = config.to_dict()
        assert result["custom_setting"] is True
        assert result["api_version"] == "2023-07-01"
        assert result["model"] == "GPT-4o"


class TestValidateAlgorithmName:
    """Tests for validate_algorithm_name function."""

    def test_unknown_algorithm_message_lists_canonical_names(self) -> None:
        """Unknown algorithm errors should enumerate the accepted public names."""
        with pytest.raises(ValueError) as exc_info:
            validate_algorithm_name("optuna_foo")

        message = str(exc_info.value)
        for algorithm_name in accepted_algorithm_values():
            assert algorithm_name in message
        assert "optuna_foo" in message


class TestValidateExecutionMode:
    """Tests for validate_execution_mode function."""

    def test_resolve_execution_mode_none_defaults_to_local(self) -> None:
        """Omitted execution mode follows the public SDK default."""
        assert resolve_execution_mode(None) is ExecutionMode.LOCAL

    def test_validate_execution_mode_none_defaults_to_local(self) -> None:
        """Validation should accept the omitted-mode public default."""
        assert validate_execution_mode(None) is ExecutionMode.LOCAL

    @pytest.mark.parametrize(
        ("mode", "expected"),
        [
            ("edge_analytics", "local"),
            ("hybrid", "hybrid"),
            ("hybrid_api", "hybrid_api"),
        ],
    )
    def test_config_direct_legacy_execution_mode_warns_but_works(
        self, mode: str, expected: str
    ) -> None:
        with pytest.warns(DeprecationWarning) as caught:
            config = TraigentConfig(execution_mode=mode)

        messages = [str(w.message) for w in caught]
        assert config.execution_mode == expected
        assert any("algorithm" in message for message in messages)
        assert any("offline=True" in message for message in messages)

    def test_config_direct_privacy_enabled_warns_but_works(self) -> None:
        with pytest.warns(DeprecationWarning) as caught:
            config = TraigentConfig(privacy_enabled=True)

        messages = [str(w.message) for w in caught]
        assert config.privacy_enabled is True
        assert any("privacy_enabled is deprecated" in message for message in messages)
        assert any("offline=True" in message for message in messages)

    def test_legacy_config_attribute_assignments_warn_but_work(self) -> None:
        config = TraigentConfig()

        with pytest.warns(DeprecationWarning) as mode_warnings:
            config.execution_mode = "hybrid"
        with pytest.warns(DeprecationWarning) as privacy_warnings:
            config.privacy_enabled = True

        assert config.execution_mode == "hybrid"
        assert config.privacy_enabled is True
        assert any("algorithm" in str(w.message) for w in mode_warnings)
        assert any("offline=True" in str(w.message) for w in privacy_warnings)

    def test_invalid_mode_string_raises_configuration_error(self) -> None:
        """Invalid compatibility selectors should guide to algorithm/offline."""
        with pytest.raises(ConfigurationError) as exc_info:
            validate_execution_mode("nonexistent_mode")

        message = str(exc_info.value)
        assert "algorithm='auto', 'grid', or 'random'" in message
        assert "offline=True" in message
        assert "execution_mode" not in message
        assert "edge_analytics" not in message
        assert "hybrid" not in message

    def test_resolve_execution_mode_invalid_message_uses_public_knobs(self) -> None:
        with pytest.raises(ValueError) as exc_info:
            resolve_execution_mode("nonexistent_mode")

        message = str(exc_info.value)
        assert "algorithm='auto', 'grid', or 'random'" in message
        assert "offline=True" in message
        assert "execution_mode" not in message
        assert "edge_analytics" not in message
        assert "hybrid" not in message

    def test_privacy_alias_fails_closed(self) -> None:
        """The legacy privacy alias no longer normalizes to hybrid."""
        with pytest.raises(ConfigurationError, match="fails closed"):
            validate_execution_mode("privacy")

    def test_local_validates_as_canonical_local(self) -> None:
        """Local is accepted as the preferred compatibility wire value."""
        import warnings

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            assert resolve_execution_mode("local") is ExecutionMode.LOCAL
            assert validate_execution_mode("local") is ExecutionMode.LOCAL
            assert TraigentConfig(execution_mode="local").execution_mode == "local"
        assert not [
            warning
            for warning in caught
            if issubclass(warning.category, DeprecationWarning)
        ]

    def test_deprecated_edge_analytics_public_alias_still_resolves_to_local(
        self,
    ) -> None:
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            assert ExecutionMode.EDGE_ANALYTICS is ExecutionMode.LOCAL
            assert ExecutionMode["EDGE_ANALYTICS"] is ExecutionMode.LOCAL
            assert ExecutionMode("edge_analytics") is ExecutionMode.LOCAL

    def test_deprecated_is_edge_analytics_mode_delegates_to_is_local_mode(
        self,
    ) -> None:
        config = TraigentConfig(execution_mode="local")

        with pytest.warns(DeprecationWarning, match="is_local_mode"):
            assert config.is_edge_analytics_mode() == config.is_local_mode()

    def test_deprecated_standard_mode_warns_and_resolves_to_hybrid(self) -> None:
        """The removed standard mode emits DeprecationWarning and maps to hybrid."""
        import warnings

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = validate_execution_mode("standard")
        assert result is ExecutionMode.HYBRID
        assert any(issubclass(w.category, DeprecationWarning) for w in caught)
        assert list(accepted_execution_mode_values()) == [
            "edge_analytics",
            "hybrid",
            "hybrid_api",
            "local",
        ]
        for mode in accepted_execution_mode_values():
            result = validate_execution_mode(mode)
            assert result in {
                ExecutionMode.LOCAL,
                ExecutionMode.HYBRID,
                ExecutionMode.HYBRID_API,
            }

    def test_deprecated_cloud_mode_fails_closed(self) -> None:
        """The deprecated cloud mode no longer normalizes to hybrid."""
        with pytest.raises(ConfigurationError, match="fails closed"):
            validate_execution_mode("cloud")

    def test_config_privacy_alias_fails_closed(self) -> None:
        """TraigentConfig rejects the privacy alias before normalization."""
        with pytest.raises(ConfigurationError, match="fails closed"):
            TraigentConfig(execution_mode="privacy")

    def test_config_accepts_deprecated_standard_mode_with_warning(self) -> None:
        """TraigentConfig still accepts standard with DeprecationWarning."""
        import warnings

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            config = TraigentConfig(execution_mode="standard")
        assert config.execution_mode == "hybrid"
        assert any(issubclass(w.category, DeprecationWarning) for w in caught)

    def test_config_cloud_mode_fails_closed(self) -> None:
        """TraigentConfig rejects cloud before it can normalize to hybrid."""
        with pytest.raises(ConfigurationError, match="fails closed"):
            TraigentConfig(execution_mode="cloud")
