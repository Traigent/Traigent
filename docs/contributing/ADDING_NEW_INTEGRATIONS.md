# Adding New LLM Integrations to Traigent

This guide explains how to contribute new LLM provider integrations to Traigent. Whether you're adding support for a new AI model provider or enhancing an existing integration, this document covers the standards and requirements for high-quality contributions.

## Overview

Traigent uses a plugin-based architecture for LLM integrations. Each provider (OpenAI, Anthropic, Mistral, etc.) has its own plugin that handles:

- **Parameter Mapping**: Converting Traigent's canonical parameter names to provider-specific names
- **Validation**: Ensuring parameters are within valid ranges for the provider
- **Model Discovery**: Dynamic discovery of available models from the provider's API
- **Framework Override**: Intercepting and modifying SDK calls at runtime

## Prerequisites

Before starting, ensure you have:

1. **Development Environment Setup**:
   ```bash
   git clone https://github.com/YOUR_USERNAME/Traigent.git
   cd Traigent
   make install-dev
   ```

2. **Provider SDK Documentation**: Familiarize yourself with the provider's Python SDK

3. **API Access**: Have test API credentials for the provider (or use mock mode)

## Step-by-Step Guide

### Step 1: Add Framework to Parameter Normalizer

First, register your framework in `traigent/integrations/utils/parameter_normalizer.py`:

```python
class Framework(Enum):
    """Supported LLM frameworks."""
    TRAIGENT = "traigent"
    OPENAI = "openai"
    # ... existing frameworks ...
    YOUR_PROVIDER = "your_provider"  # Add your framework
```

### Step 2: Create the Plugin File

Create a new plugin file in `traigent/integrations/llms/`:

```bash
touch traigent/integrations/llms/your_provider_plugin.py
```

### Step 3: Implement the Plugin Class

Follow this template structure:

```python
"""Your Provider integration plugin for Traigent.

This module provides the Your Provider-specific plugin implementation for
parameter mappings and framework overrides.
"""

# Traceability tag (required for all new files):
# Traceability: CONC-Layer-Integration CONC-Quality-Compatibility FUNC-INTEGRATIONS REQ-INT-008 SYNC-IntegrationHook

import logging
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from traigent.integrations.base_plugin import (
    IntegrationPriority,
    PluginMetadata,
    ValidationRule,
)
from traigent.integrations.llms.base_llm_plugin import LLMPlugin
from traigent.integrations.utils import Framework

if TYPE_CHECKING:
    from traigent.config.types import TraigentConfig

logger = logging.getLogger(__name__)


class YourProviderPlugin(LLMPlugin):
    """Plugin for Your Provider SDK integration.

    Supports the official Your Provider Python SDK.
    Handles parameter mapping for chat completions including streaming,
    tool use, and provider-specific parameters.
    """

    FRAMEWORK = Framework.YOUR_PROVIDER

    def _get_metadata(self) -> PluginMetadata:
        """Return plugin metadata."""
        return PluginMetadata(
            name="your_provider",
            version="1.0.0",
            supported_packages=["your_provider_sdk"],
            priority=IntegrationPriority.HIGH,
            description="Your Provider SDK integration",
            author="Your Name",
            requires_packages=["your_provider_sdk>=1.0.0"],
            supports_versions={"your_provider_sdk": "1."},
        )

    def _get_extra_mappings(self) -> dict[str, str]:
        """Return provider-specific parameter mappings.

        These mappings handle parameters unique to your provider
        that aren't in the standard ParameterNormalizer.
        """
        return {
            # Provider-specific parameters
            "provider_specific_param": "sdk_param_name",
            # API configuration
            "your_api_key": "api_key",
            # Common aliases
            "seed": "random_seed",  # Example: map common 'seed' to provider name
        }

    def _get_provider_specific_rules(self) -> dict[str, ValidationRule]:
        """Return provider-specific validation rules.

        Define valid ranges, allowed values, and required parameters.
        """
        return {
            "model": ValidationRule(
                required=True,
                custom_validator="_validate_model",
            ),
            "temperature": ValidationRule(min_value=0.0, max_value=2.0),
            "top_p": ValidationRule(min_value=0.0, max_value=1.0),
            "max_tokens": ValidationRule(min_value=1, max_value=100000),
            "stream": ValidationRule(allowed_values=[True, False]),
        }

    def _validate_model(self, param_name: str, value: Any) -> list[str]:
        """Validate model ID using dynamic discovery."""
        errors = []
        if not isinstance(value, str):
            errors.append(f"Parameter '{param_name}' must be a string")
            return errors

        if not value:
            errors.append(f"Parameter '{param_name}' cannot be empty")
            return errors

        try:
            from traigent.integrations.model_discovery import get_model_discovery

            discovery = get_model_discovery(self.FRAMEWORK)
            if discovery and not discovery.is_valid_model(value):
                # Warn but don't block - model might be new
                logger.warning(
                    f"Unrecognized model: {value}. Proceeding anyway."
                )
        except ImportError:
            logger.debug("Model discovery not available")

        return errors

    def get_target_classes(self) -> list[str]:
        """Return list of SDK classes to override."""
        return [
            "your_provider_sdk.Client",
            "your_provider_sdk.AsyncClient",
        ]

    def get_target_methods(self) -> dict[str, list[str]]:
        """Return mapping of classes to methods to override."""
        return {
            "your_provider_sdk.Client": [
                "chat.completions.create",
                "completions.create",
            ],
            "your_provider_sdk.AsyncClient": [
                "chat.completions.create",
                "completions.create",
            ],
        }

    def apply_overrides(
        self, kwargs: dict[str, Any], config: "TraigentConfig | dict[str, Any]"
    ) -> dict[str, Any]:
        """Apply provider-specific overrides.

        Extend base implementation for provider-specific logic.
        """
        config_obj = self._normalize_config(config)
        overridden = super().apply_overrides(kwargs, config_obj)

        # Add provider-specific override logic here
        # Example: Message formatting, special parameter handling

        return overridden
```

### Step 4: Add Model Discovery (Optional but Recommended)

Create a model discovery module in `traigent/integrations/model_discovery/`:

```python
"""Model discovery for Your Provider."""

import logging
import os
import re
from typing import Any

from traigent.integrations.model_discovery.base import ModelDiscovery
from traigent.integrations.utils import Framework

logger = logging.getLogger(__name__)

# Pattern for validating model names
YOUR_PROVIDER_MODEL_PATTERN = r"^(your-model-prefix-|another-prefix-)"


class YourProviderDiscovery(ModelDiscovery):
    """Model discovery service for Your Provider."""

    PROVIDER = "your_provider"
    FRAMEWORK = Framework.YOUR_PROVIDER

    def _fetch_models_from_sdk(self) -> list[str]:
        """Fetch models directly from SDK if API key available."""
        api_key = os.getenv("YOUR_PROVIDER_API_KEY")
        if not api_key:
            return []

        try:
            from your_provider_sdk import Client
            client = Client(api_key=api_key)
            models_response = client.models.list()
            return sorted([m.id for m in models_response.data])
        except Exception as e:
            logger.debug(f"Failed to fetch models: {e}")
            return []

    def _get_config_key(self) -> str:
        """Return config key for loading known models."""
        return "your_provider"

    def _get_model_pattern(self) -> re.Pattern[str]:
        """Return regex pattern for model validation."""
        return re.compile(YOUR_PROVIDER_MODEL_PATTERN)
```

Register the discovery in `traigent/integrations/model_discovery/registry.py`:

```python
from traigent.integrations.model_discovery.your_provider_discovery import (
    YourProviderDiscovery,
)
# ...
register_discovery("your_provider", YourProviderDiscovery)
```

Add known models to `traigent/config/models.yaml`:

```yaml
your_provider:
  known_models:
    - your-model-small
    - your-model-large
    - your-model-latest
  pattern: "^(your-model-prefix-|another-prefix-)"
```

### Step 5: Export the Plugin

Add your plugin to `traigent/integrations/llms/__init__.py`:

```python
from traigent.integrations.llms.your_provider_plugin import YourProviderPlugin

__all__ = [
    # ... existing exports ...
    "YourProviderPlugin",
]
```

### Step 6: Write Comprehensive Tests

Create test file at `tests/unit/integrations/test_your_provider_plugin.py`:

```python
"""Tests for the Your Provider integration plugin."""

import pytest

from traigent.integrations.llms.your_provider_plugin import YourProviderPlugin
from traigent.integrations.utils import Framework


class TestYourProviderPlugin:
    """Basic plugin behavior tests."""

    def setup_method(self) -> None:
        self.plugin = YourProviderPlugin()

    def test_apply_overrides_with_dict_config(self) -> None:
        """Plugin should handle raw dict payloads."""
        config_payload = {
            "model": "your-model-latest",
            "stream": True,
            "temperature": 0.7,
            "max_tokens": 1000,
        }

        overridden = self.plugin.apply_overrides({}, config_payload)

        assert overridden["model"] == "your-model-latest"
        assert overridden["stream"] is True
        assert overridden["temperature"] == pytest.approx(0.7)
        assert overridden["max_tokens"] == 1000


class TestYourProviderParameterMappings:
    """Test parameter mapping via ParameterNormalizer."""

    def setup_method(self) -> None:
        self.plugin = YourProviderPlugin()

    def test_model_preserved(self) -> None:
        """Test that model parameter is preserved."""
        config = {"model": "your-model-small"}
        overridden = self.plugin.apply_overrides({}, config)
        assert overridden["model"] == "your-model-small"

    def test_temperature_preserved(self) -> None:
        """Test that temperature is preserved."""
        config = {"model": "your-model-small", "temperature": 0.5}
        overridden = self.plugin.apply_overrides({}, config)
        assert overridden["temperature"] == pytest.approx(0.5)

    def test_user_kwarg_not_overwritten(self) -> None:
        """Test that user-provided kwargs take precedence."""
        kwargs = {"model": "user-specified-model"}
        config = {"model": "config-model"}
        overridden = self.plugin.apply_overrides(kwargs, config)
        assert overridden["model"] == "user-specified-model"


class TestYourProviderValidationRules:
    """Test validation rules."""

    def setup_method(self) -> None:
        self.plugin = YourProviderPlugin()

    def test_temperature_validation_range(self) -> None:
        """Test temperature has valid range."""
        rules = self.plugin._get_provider_specific_rules()
        assert "temperature" in rules
        assert rules["temperature"].min_value == 0.0
        assert rules["temperature"].max_value == 2.0


class TestYourProviderPluginMetadata:
    """Test plugin metadata."""

    def setup_method(self) -> None:
        self.plugin = YourProviderPlugin()

    def test_framework_is_correct(self) -> None:
        """Test plugin identifies correct framework."""
        assert self.plugin.FRAMEWORK == Framework.YOUR_PROVIDER

    def test_metadata_name(self) -> None:
        """Test plugin metadata has correct name."""
        assert self.plugin.metadata.name == "your_provider"

    def test_supported_packages(self) -> None:
        """Test plugin lists supported packages."""
        packages = self.plugin.metadata.supported_packages
        assert "your_provider_sdk" in packages
```

## Code Quality Standards

### Required Elements

Every plugin must include:

1. **Traceability Comment**: First comment after docstring:
   ```python
   # Traceability: CONC-Layer-Integration CONC-Quality-Compatibility FUNC-INTEGRATIONS REQ-INT-008 SYNC-IntegrationHook
   ```

2. **Type Hints**: All public methods must have complete type annotations

3. **Docstrings**: Google-style docstrings for all classes and public methods

4. **Logging**: Use `logging.getLogger(__name__)` for debug and warning messages

### Validation Rules Guidelines

- **Required Parameters**: Mark essential parameters like `model` as `required=True`
- **Range Validation**: Use `min_value` and `max_value` for numeric parameters
- **Allowed Values**: Use `allowed_values` for enum-like parameters
- **Custom Validators**: Use `custom_validator` for complex validation logic

### Parameter Mapping Best Practices

1. **Canonical Names**: Map to Traigent's canonical names when possible
2. **Aliases**: Support common aliases (e.g., `seed` → `random_seed`)
3. **Pass-through**: Provider-specific params should pass through unchanged
4. **Documentation**: Comment each mapping explaining the purpose

## Testing Requirements

### Minimum Test Coverage

Your contribution must include tests for:

1. **Basic Override Tests**: Verify parameters are correctly mapped
2. **Parameter Preservation Tests**: Each standard parameter should have a test
3. **User Precedence Tests**: User kwargs should override config values
4. **Validation Rule Tests**: Verify min/max ranges and allowed values
5. **Metadata Tests**: Framework, name, supported packages

### Running Tests

```bash
# Run your plugin tests
TRAIGENT_MOCK_MODE=true pytest tests/unit/integrations/test_your_provider_plugin.py -v

# Run all integration tests
TRAIGENT_MOCK_MODE=true pytest tests/unit/integrations/ -v

# Run with coverage
TRAIGENT_MOCK_MODE=true pytest tests/ --cov=traigent.integrations.llms.your_provider_plugin
```

### Mock Mode

Always use `TRAIGENT_MOCK_MODE=true` for tests to avoid real API calls:

```bash
export TRAIGENT_MOCK_MODE=true
pytest tests/unit/integrations/test_your_provider_plugin.py
```

## Formatting and Linting

Before submitting, ensure your code passes all checks:

```bash
# Format code
make format
# Or manually:
isort traigent/ tests/
black traigent/ tests/

# Lint code
make lint
# Or manually:
ruff check traigent/ --fix
black --check traigent/
isort --check-only traigent/
```

## Submitting Your Contribution

### Pull Request Checklist

- [ ] Plugin implements all required abstract methods
- [ ] Traceability comment included
- [ ] Type hints for all public methods
- [ ] Docstrings for all classes and public methods
- [ ] Model discovery implemented (if provider has API)
- [ ] Known models added to `models.yaml`
- [ ] Plugin exported in `__init__.py`
- [ ] Comprehensive test suite (minimum 10 tests)
- [ ] All tests pass with `TRAIGENT_MOCK_MODE=true`
- [ ] Code formatted with `make format`
- [ ] Code passes `make lint`

### PR Description Template

```markdown
## New Integration: [Provider Name]

### Summary
Adds support for [Provider Name] LLM SDK.

### Features
- Parameter mapping for [list key parameters]
- Validation rules for [list validated parameters]
- Model discovery via SDK API
- Support for [streaming/tool use/etc.]

### Testing
- X unit tests covering plugin functionality
- Tested with mock mode
- [Optional: Tested with real API]

### Documentation
- Added to models.yaml
- Exported in __init__.py

### Related Issues
Closes #XXX
```

## Examples of Well-Implemented Plugins

For reference, study these existing implementations:

- **OpenAI Plugin**: `traigent/integrations/llms/openai_plugin.py`
- **Anthropic Plugin**: `traigent/integrations/llms/anthropic_plugin.py`
- **Mistral Plugin**: `traigent/integrations/llms/mistral_plugin.py`
- **Gemini Plugin**: `traigent/integrations/llms/gemini_plugin.py`

## Getting Help

- **GitHub Issues**: For questions about implementation
- **GitHub Discussions**: For design discussions
- **Existing Plugins**: Study implementations for patterns

## License

By contributing, you agree that your contributions will be licensed under the same license as the Traigent project.
