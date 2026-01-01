# LLM Plugin Migration Guide

This guide explains how to migrate existing LLM plugins to use the `LLMPlugin` base class. Use it when you maintain a custom LLM integration that still subclasses `IntegrationPlugin`.

## Why Migrate?

The new `LLMPlugin` base class provides:

1. **Automatic parameter mapping** via `ParameterNormalizer`
2. **Reduced boilerplate** - no need to define common mappings manually
3. **Consistent validation rules** across all LLM providers
4. **Extension hooks** for provider-specific customization

## Migration Steps

### Step 1: Change Base Class

```python
# Before
from traigent.integrations.base_plugin import IntegrationPlugin

class OpenAIPlugin(IntegrationPlugin):
    ...

# After
from traigent.integrations.llms import LLMPlugin
from traigent.integrations.utils import Framework

class OpenAIPlugin(LLMPlugin):
    FRAMEWORK = Framework.OPENAI
    ...
```

### Step 2: Remove Redundant Mappings

Delete mappings that are now provided by `ParameterNormalizer`:

```python
# Before - DELETE THIS:
def _get_default_mappings(self) -> dict[str, str]:
    return {
        "model": "model",
        "temperature": "temperature",
        "max_tokens": "max_tokens",
        "top_p": "top_p",
        "stream": "stream",
        "stop": "stop",
        # ... many more lines
    }

# After - Only add EXTRA mappings not in normalizer:
def _get_extra_mappings(self) -> dict[str, str]:
    return {
        "logit_bias": "logit_bias",  # OpenAI-specific
        "logprobs": "logprobs",       # OpenAI-specific
        "n": "n",                      # OpenAI-specific
    }
```

### Step 3: Convert Validation Rules

Use `_get_provider_specific_rules()` instead of overriding `_get_validation_rules()`:

```python
# Before:
def _get_validation_rules(self) -> dict[str, ValidationRule]:
    return {
        "model": ValidationRule(required=True),
        "temperature": ValidationRule(min_value=0.0, max_value=2.0),  # Common - now auto
        "top_p": ValidationRule(min_value=0.0, max_value=1.0),        # Common - now auto
        "frequency_penalty": ValidationRule(min_value=-2.0, max_value=2.0),  # Provider-specific
    }

# After:
def _get_provider_specific_rules(self) -> dict[str, ValidationRule]:
    return {
        "model": ValidationRule(required=True),  # Override to make required
        "frequency_penalty": ValidationRule(min_value=-2.0, max_value=2.0),
        "presence_penalty": ValidationRule(min_value=-2.0, max_value=2.0),
    }
```

### Step 4: Keep Provider-Specific Logic

The `apply_overrides()` method still works - keep any provider-specific logic:

```python
def apply_overrides(
    self, kwargs: dict[str, Any], config: "TraigentConfig | dict[str, Any]"
) -> dict[str, Any]:
    # Apply base overrides (handles standard mapping)
    overridden = super().apply_overrides(kwargs, config)

    # Add provider-specific transformations
    if "functions" in overridden and "tools" not in overridden:
        # Convert functions to tools format
        ...

    return overridden
```

## Extension Hooks Reference

| Hook | Purpose | Default |
|------|---------|---------|
| `FRAMEWORK` | Set to `Framework.XXX` enum | `None` |
| `_should_use_normalizer()` | Return `False` to bypass normalizer | `True` |
| `_get_supported_canonical_params()` | Limit which params are auto-mapped | `None` (all) |
| `_get_extra_mappings()` | Add provider-specific params | `{}` |
| `_get_provider_specific_rules()` | Add custom validation | `{}` |

### Critical: `_get_supported_canonical_params()`

**IMPORTANT**: If your provider doesn't support all canonical LLM params, you MUST override `_get_supported_canonical_params()` to prevent runtime errors.

```python
def _get_supported_canonical_params(self) -> set[str]:
    """Return only the params this provider accepts.

    Without this, the normalizer maps ALL canonical params, which causes:
    - Bedrock: ValidationException for frequency_penalty, presence_penalty
    - Gemini: TypeError for unsupported params in generate_content()
    - Cohere: Unexpected keyword argument errors
    """
    return {"model", "max_tokens", "temperature", "top_p", "stop"}
```

**Return values:**

- `None` (default): Support ALL canonical params from normalizer
- `set[str]`: Support ONLY the listed canonical params

Extra mappings from `_get_extra_mappings()` are NOT filtered - they are always added.

## Example: Full Migration

### Before (OpenAI Plugin)

```python
from traigent.integrations.base_plugin import IntegrationPlugin, ValidationRule

class OpenAIPlugin(IntegrationPlugin):

    def _get_default_mappings(self) -> dict[str, str]:
        return {
            "model": "model",
            "temperature": "temperature",
            "max_tokens": "max_tokens",
            "top_p": "top_p",
            "frequency_penalty": "frequency_penalty",
            "presence_penalty": "presence_penalty",
            "stop": "stop",
            "stream": "stream",
            "functions": "functions",
            "function_call": "function_call",
            "tools": "tools",
            "tool_choice": "tool_choice",
            "response_format": "response_format",
            "seed": "seed",
            "logit_bias": "logit_bias",
            "logprobs": "logprobs",
            "top_logprobs": "top_logprobs",
            "n": "n",
        }

    def _get_validation_rules(self) -> dict[str, ValidationRule]:
        return {
            "model": ValidationRule(required=True),
            "temperature": ValidationRule(min_value=0.0, max_value=2.0),
            "max_tokens": ValidationRule(min_value=1, max_value=128000),
            "top_p": ValidationRule(min_value=0.0, max_value=1.0),
            "frequency_penalty": ValidationRule(min_value=-2.0, max_value=2.0),
            "presence_penalty": ValidationRule(min_value=-2.0, max_value=2.0),
            "n": ValidationRule(min_value=1, max_value=10),
        }
```

### After (OpenAI Plugin)

```python
from traigent.integrations.llms import LLMPlugin
from traigent.integrations.utils import Framework
from traigent.integrations.base_plugin import ValidationRule

class OpenAIPlugin(LLMPlugin):
    FRAMEWORK = Framework.OPENAI

    def _get_extra_mappings(self) -> dict[str, str]:
        # Only OpenAI-specific params not in ParameterNormalizer
        return {
            "logit_bias": "logit_bias",
            "logprobs": "logprobs",
            "top_logprobs": "top_logprobs",
            "n": "n",
            "functions": "functions",
            "function_call": "function_call",
        }

    def _get_provider_specific_rules(self) -> dict[str, ValidationRule]:
        return {
            "model": ValidationRule(required=True),
            "max_tokens": ValidationRule(min_value=1, max_value=128000),
            "frequency_penalty": ValidationRule(min_value=-2.0, max_value=2.0),
            "presence_penalty": ValidationRule(min_value=-2.0, max_value=2.0),
            "n": ValidationRule(min_value=1, max_value=10),
            "top_logprobs": ValidationRule(min_value=0, max_value=5),
        }
```

## Built-In LLM Plugins

Traigent ships LLM plugins for:

- OpenAI (`traigent/integrations/llms/openai_plugin.py`)
- Anthropic (`traigent/integrations/llms/anthropic_plugin.py`)
- Azure OpenAI (`traigent/integrations/llms/azure_openai_plugin.py`)
- Bedrock (`traigent/integrations/llms/bedrock_plugin.py`)
- Gemini (`traigent/integrations/llms/gemini_plugin.py`)
- Mistral (`traigent/integrations/llms/mistral_plugin.py`)
- Cohere (`traigent/integrations/llms/cohere_plugin.py`)
- HuggingFace (`traigent/integrations/llms/huggingface_plugin.py`)
- LlamaIndex (`traigent/integrations/llms/llamaindex_plugin.py`)
- LangChain (`traigent/integrations/llms/langchain_plugin.py`)

## Testing Your Migration

Run the plugin tests after migration:

```bash
TRAIGENT_MOCK_MODE=true pytest tests/unit/integrations/test_<plugin>_plugin.py -v
```

Run the full integration test suite:

```bash
TRAIGENT_MOCK_MODE=true pytest tests/unit/integrations/ -v
```
