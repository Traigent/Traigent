"""Comprehensive tests for TRAIGENT_STRICT_METRICS_NULLS feature flag."""

import os
from unittest import mock

import pytest

from traigent.config.types import TraigentConfig
from traigent.core.utils import safe_float_convert
from traigent.evaluators.metrics_tracker import MetricsTracker


class TestStrictMetricsNullsConfig:
    """Test TraigentConfig integration with strict metrics nulls feature."""

    def test_config_default_disabled(self):
        """Test that strict_metrics_nulls is disabled by default."""
        config = TraigentConfig()
        assert config.strict_metrics_nulls is False
        assert not config.is_strict_metrics_nulls_enabled()

    def test_config_from_environment_enabled(self):
        """Test loading strict_metrics_nulls from environment variable."""
        with mock.patch.dict(os.environ, {"TRAIGENT_STRICT_METRICS_NULLS": "true"}):
            config = TraigentConfig.from_environment()
            assert config.strict_metrics_nulls is True
            assert config.is_strict_metrics_nulls_enabled()

    def test_config_from_environment_disabled(self):
        """Test that strict_metrics_nulls stays disabled when env var is not set."""
        with mock.patch.dict(os.environ, {}, clear=True):
            config = TraigentConfig.from_environment()
            assert config.strict_metrics_nulls is False

    def test_config_to_dict_includes_flag(self):
        """Test that to_dict includes strict_metrics_nulls when enabled."""
        config = TraigentConfig(strict_metrics_nulls=True)
        config_dict = config.to_dict()
        assert "strict_metrics_nulls" in config_dict
        assert config_dict["strict_metrics_nulls"] is True

    def test_config_from_dict_preserves_flag(self):
        """Test that from_dict preserves strict_metrics_nulls."""
        config_dict = {"strict_metrics_nulls": True, "model": "gpt-4"}
        config = TraigentConfig.from_dict(config_dict)
        assert config.strict_metrics_nulls is True
        assert config.model == "gpt-4"

    @pytest.mark.parametrize("env_value", ["true", "1", "yes", "True", "TRUE", "YES"])
    def test_environment_parsing_true_values(self, env_value):
        """Test various truthy environment variable values."""
        with mock.patch.dict(os.environ, {"TRAIGENT_STRICT_METRICS_NULLS": env_value}):
            config = TraigentConfig.from_environment()
            assert config.strict_metrics_nulls is True

    @pytest.mark.parametrize("env_value", ["false", "0", "no", "False", "", "random"])
    def test_environment_parsing_false_values(self, env_value):
        """Test various falsy environment variable values."""
        with mock.patch.dict(os.environ, {"TRAIGENT_STRICT_METRICS_NULLS": env_value}):
            config = TraigentConfig.from_environment()
            assert config.strict_metrics_nulls is False


class TestSafeFloatConvert:
    """Test safe_float_convert with strict nulls mode."""

    def test_safe_float_convert_normal_mode(self):
        """Test safe_float_convert returns 0.0 in normal mode."""
        with mock.patch.dict(os.environ, {}, clear=True):
            assert safe_float_convert(None) == 0.0
            assert safe_float_convert("invalid") == 0.0
            assert safe_float_convert(None, default=5.0) == 5.0

    def test_safe_float_convert_strict_mode(self):
        """Test safe_float_convert returns None in strict mode."""
        with mock.patch.dict(os.environ, {"TRAIGENT_STRICT_METRICS_NULLS": "true"}):
            assert safe_float_convert(None) is None
            assert safe_float_convert("invalid") is None
            assert safe_float_convert(None, default=5.0) is None

    def test_safe_float_convert_valid_values(self):
        """Test that valid values are converted correctly in both modes."""
        # Normal mode
        with mock.patch.dict(os.environ, {}, clear=True):
            assert safe_float_convert(3.14) == 3.14
            assert safe_float_convert("2.5") == 2.5
            assert safe_float_convert(10) == 10.0

        # Strict mode
        with mock.patch.dict(os.environ, {"TRAIGENT_STRICT_METRICS_NULLS": "true"}):
            assert safe_float_convert(3.14) == 3.14
            assert safe_float_convert("2.5") == 2.5
            assert safe_float_convert(10) == 10.0


class TestMetricsTracker:
    """Test MetricsTracker with strict nulls mode."""

    def test_empty_backend_format_normal_mode(self):
        """Test _empty_backend_format returns 0.0 in normal mode."""
        with mock.patch.dict(os.environ, {}, clear=True):
            tracker = MetricsTracker()
            empty_format = tracker._empty_backend_format()

            assert empty_format["score"] == 0.0
            assert empty_format["accuracy"] == 0.0
            assert empty_format["response_time_ms"] == 0.0
            assert empty_format["cost"] == 0.0
            assert empty_format["input_tokens"] == 0.0

    def test_empty_backend_format_strict_mode(self):
        """Test _empty_backend_format returns None in strict mode."""
        with mock.patch.dict(os.environ, {"TRAIGENT_STRICT_METRICS_NULLS": "true"}):
            tracker = MetricsTracker()
            empty_format = tracker._empty_backend_format()

            assert empty_format["score"] is None
            assert empty_format["accuracy"] is None
            assert empty_format["response_time_ms"] is None
            assert empty_format["cost"] is None
            assert empty_format["input_tokens"] is None

    def test_empty_aggregated_metrics_normal_mode(self):
        """Test _empty_aggregated_metrics returns 0.0 in normal mode."""
        with mock.patch.dict(os.environ, {}, clear=True):
            tracker = MetricsTracker()
            empty_metrics = tracker._empty_aggregated_metrics()

            assert empty_metrics["success_rate"] == 0.0
            assert empty_metrics["duration"] == 0.0
            assert empty_metrics["input_tokens"]["mean"] == 0.0
            assert empty_metrics["response_time_ms"]["median"] == 0.0

    def test_empty_aggregated_metrics_strict_mode(self):
        """Test _empty_aggregated_metrics returns None in strict mode."""
        with mock.patch.dict(os.environ, {"TRAIGENT_STRICT_METRICS_NULLS": "true"}):
            tracker = MetricsTracker()
            empty_metrics = tracker._empty_aggregated_metrics()

            assert empty_metrics["success_rate"] is None
            assert empty_metrics["duration"] is None
            assert empty_metrics["input_tokens"]["mean"] is None
            assert empty_metrics["response_time_ms"]["median"] is None

    def test_safe_get_function_normal_mode(self):
        """Test safe_get nested function returns 0.0 in normal mode."""
        with mock.patch.dict(os.environ, {}, clear=True):
            tracker = MetricsTracker()

            # Create test data

            # Test in format_for_backend context
            formatted = tracker.format_for_backend()
            # These should use safe_get internally
            assert formatted["score"] == 0.0  # When no data
            assert formatted["response_time_ms"] == 0.0

    def test_safe_get_function_strict_mode(self):
        """Test safe_get nested function returns None in strict mode."""
        with mock.patch.dict(os.environ, {"TRAIGENT_STRICT_METRICS_NULLS": "true"}):
            tracker = MetricsTracker()

            # Test in format_for_backend context
            formatted = tracker.format_for_backend()
            # These should use safe_get internally with None defaults
            assert formatted["score"] is None  # When no data
            assert formatted["response_time_ms"] is None


class TestEndToEndIntegration:
    """End-to-end integration tests."""

    def test_json_output_normal_mode(self):
        """Test that JSON output contains 0.0 in normal mode."""
        import json

        with mock.patch.dict(os.environ, {}, clear=True):
            tracker = MetricsTracker()
            formatted = tracker.format_for_backend()

            # Convert to JSON and back to ensure serialization works
            json_str = json.dumps(formatted)
            parsed = json.loads(json_str)

            assert parsed["response_time_ms"] == 0.0
            assert parsed["cost"] == 0.0

    def test_json_output_strict_mode(self):
        """Test that JSON output contains null in strict mode."""
        import json

        with mock.patch.dict(os.environ, {"TRAIGENT_STRICT_METRICS_NULLS": "true"}):
            tracker = MetricsTracker()
            formatted = tracker.format_for_backend()

            # Convert to JSON and back to ensure serialization works
            json_str = json.dumps(formatted)
            parsed = json.loads(json_str)

            assert parsed["response_time_ms"] is None
            assert parsed["cost"] is None

    def test_backward_compatibility(self):
        """Test that existing code works without changes."""
        # Without the flag, everything should work as before
        with mock.patch.dict(os.environ, {}, clear=True):
            config = TraigentConfig()
            assert config.strict_metrics_nulls is False

            tracker = MetricsTracker()
            empty = tracker._empty_backend_format()
            assert all(
                v == 0.0 or v == 0
                for k, v in empty.items()
                if k not in ["total_examples", "successful_examples"]
            )

    def test_flag_toggle_runtime(self):
        """Test that flag can be toggled at runtime."""
        # Start with flag disabled
        with mock.patch.dict(os.environ, {}, clear=True):
            result1 = safe_float_convert(None)
            assert result1 == 0.0

        # Enable flag
        with mock.patch.dict(os.environ, {"TRAIGENT_STRICT_METRICS_NULLS": "true"}):
            result2 = safe_float_convert(None)
            assert result2 is None

        # Disable again
        with mock.patch.dict(os.environ, {}, clear=True):
            result3 = safe_float_convert(None)
            assert result3 == 0.0
