"""Integration tests for configuration injection with decorator."""

import asyncio
import json
import tempfile

import pytest

from traigent.api.decorators import optimize
from traigent.config import get_config
from traigent.config.types import TraigentConfig
from traigent.utils.exceptions import ConfigurationError


class TestConfigurationIntegration:
    """Integration tests for configuration injection."""

    def setup_method(self):
        """Set up test dataset."""
        # Create a simple test dataset
        self.test_data = [
            {"input": {"text": "hello"}, "output": "HELLO"},
            {"input": {"text": "world"}, "output": "WORLD"},
        ]

        # Create temporary dataset file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            for item in self.test_data:
                json.dump(item, f)
                f.write("\n")
            self.dataset_path = f.name

    def test_context_based_injection(self):
        """Test context-based configuration injection."""

        @optimize(
            eval_dataset=self.dataset_path,
            configuration_space={"multiplier": [1, 2]},
            injection_mode="context",
        )
        def test_function(text: str) -> str:
            config = get_config()
            if isinstance(config, dict):
                multiplier = config.get("multiplier", 1)
            else:
                multiplier = config.custom_params.get("multiplier", 1)
            return text.upper() * multiplier

        # Test that function works with default config
        result = test_function("hello")
        assert result in ["HELLO", "HELLOHELLO"]  # Depends on optimization

        # Check that the wrapped function has the right type
        assert hasattr(test_function, "optimize")
        assert hasattr(test_function, "configuration_space")

    def test_parameter_based_injection(self):
        """Test parameter-based configuration injection."""

        @optimize(
            eval_dataset=self.dataset_path,
            configuration_space={"multiplier": [1, 2]},
            injection_mode="parameter",
            config_param="config",
        )
        def test_function(text: str, config: TraigentConfig) -> str:
            multiplier = config.custom_params.get("multiplier", 1)
            return text.upper() * multiplier

        # Test that function works
        result = test_function("hello")
        assert result in ["HELLO", "HELLOHELLO"]

        # Verify the function signature is preserved
        import inspect

        sig = inspect.signature(test_function.func)  # Original function
        assert "config" in sig.parameters

    def test_context_based_injection(self):
        """Test context-based configuration injection."""
        from traigent import get_config

        @optimize(
            eval_dataset=self.dataset_path,
            configuration_space={"multiplier": [1, 2]},
            injection_mode="context",
        )
        def test_function(text: str) -> str:
            config = get_config()
            multiplier = getattr(config, "multiplier", 1)
            return text.upper() * multiplier

        # Test that function works
        result = test_function("hello")
        assert result in ["HELLO", "HELLOHELLO"]

    def test_invalid_injection_mode(self):
        """Test error for invalid injection mode."""
        with pytest.raises(
            (ValueError, TypeError, ConfigurationError)
        ):  # Should raise ConfigurationError

            @optimize(eval_dataset=self.dataset_path, injection_mode="invalid_mode")
            def test_function(text: str) -> str:
                return text

    def test_parameter_injection_missing_param(self):
        """Test error when parameter injection used but param missing."""
        with pytest.raises((ValueError, TypeError)):  # Should raise ConfigurationError

            @optimize(eval_dataset=self.dataset_path, injection_mode="parameter")
            def test_function(text: str) -> str:  # Missing config parameter
                return text

    @pytest.mark.asyncio
    async def test_async_function_context_injection(self):
        """Test context injection with async function."""

        @optimize(
            eval_dataset=self.dataset_path,
            configuration_space={"multiplier": [1, 2]},
            injection_mode="context",
        )
        async def test_function(text: str) -> str:
            config = get_config()
            multiplier = config.get("multiplier", 1)
            await asyncio.sleep(0.001)  # Simulate async work
            return text.upper() * multiplier

        # Test that async function works
        result = await test_function("hello")
        assert result in ["HELLO", "HELLOHELLO"]

    @pytest.mark.asyncio
    async def test_async_function_parameter_injection(self):
        """Test parameter injection with async function."""

        @optimize(
            eval_dataset=self.dataset_path,
            configuration_space={"multiplier": [1, 2]},
            injection_mode="parameter",
        )
        async def test_function(text: str, config: TraigentConfig) -> str:
            multiplier = config.custom_params.get("multiplier", 1)
            await asyncio.sleep(0.001)  # Simulate async work
            return text.upper() * multiplier

        # Test that async function works
        result = await test_function("hello")
        assert result in ["HELLO", "HELLOHELLO"]

    def test_configuration_space_integration(self):
        """Test that configuration space is properly integrated."""

        @optimize(
            eval_dataset=self.dataset_path,
            configuration_space={
                "model": ["o4-mini", "GPT-4o"],
                "temperature": [0.5, 0.7, 0.9],
            },
            injection_mode="context",
        )
        def test_function(text: str) -> str:
            config = get_config()
            if isinstance(config, dict):
                model = config.get("model", "unknown")
                temp = config.get("temperature", 0.0)
            else:
                model = config.custom_params.get("model", "unknown")
                temp = config.custom_params.get("temperature", 0.0)
            return f"{text}_{model}_{temp}"

        # Verify configuration space is accessible
        assert "model" in test_function.configuration_space
        assert "temperature" in test_function.configuration_space

        # Test function execution
        result = test_function("test")
        # Result should contain model and temperature from config space
        assert "_gpt-" in result or "_unknown_" in result
