# Quick Reference: Adding a New LLM Provider

This is a condensed, actionable guide for adding new LLM provider support to Traigent.

## 🚀 5-Minute Quick Start

```bash
# 1. Generate boilerplate (replaces hours of manual setup)
python scripts/scaffold_llm_plugin.py <provider_name>

# 2. Customize these files (the scaffold adds TODO comments):
#    - traigent/integrations/llms/<provider>_plugin.py
#    - tests/unit/integrations/test_<provider>_plugin.py

# 3. Test in mock mode (no API costs!)
TRAIGENT_MOCK_LLM=true pytest tests/unit/integrations/test_<provider>_plugin.py -v

# 4. Format and lint
make format && make lint

# 5. Submit PR using the LLM Integration template
```

## 📝 What the Scaffold Generates

The `scaffold_llm_plugin.py` script creates:

1. **Plugin Implementation** (`traigent/integrations/llms/<provider>_plugin.py`)
   - Base class setup
   - Parameter mappings
   - Validation rules
   - Metadata
   - All required methods with TODO comments

2. **Comprehensive Tests** (`tests/unit/integrations/test_<provider>_plugin.py`)
   - Basic behavior tests
   - Parameter mapping tests
   - Validation tests
   - Metadata tests

3. **Framework Registration**
   - Adds Framework enum value
   - Exports plugin in `__init__.py`

## 🎯 Customization Checklist

After scaffolding, customize these sections (marked with TODO):

### In Plugin File (`<provider>_plugin.py`)

- [ ] **Update `_get_extra_mappings()`**: Add provider-specific parameters
  ```python
  def _get_extra_mappings(self) -> dict[str, str]:
      return {
          "seed": "random_seed",  # Example: map 'seed' to provider's name
          "custom_param": "provider_param_name",
      }
  ```

- [ ] **Update `_get_provider_specific_rules()`**: Adjust validation ranges
  ```python
  def _get_provider_specific_rules(self) -> dict[str, ValidationRule]:
      return {
          "temperature": ValidationRule(min_value=0.0, max_value=1.0),  # Adjust range
          "max_tokens": ValidationRule(min_value=1, max_value=50000),   # Provider limit
      }
  ```

- [ ] **Update `get_target_classes()`**: List SDK classes to instrument
  ```python
  def get_target_classes(self) -> list[str]:
      return [
          "provider_sdk.Client",
          "provider_sdk.AsyncClient",
      ]
  ```

- [ ] **Update `get_target_methods()`**: List methods to override
  ```python
  def get_target_methods(self) -> dict[str, list[str]]:
      return {
          "provider_sdk.Client": [
              "chat.completions.create",
              "completions.create",  # Add if supported
          ],
          "provider_sdk.AsyncClient": [
              "chat.completions.create",
          ],
      }
  ```

- [ ] **Optional: Override `apply_overrides()`**: For message format transformations
  ```python
  def apply_overrides(self, kwargs, config):
      overridden = super().apply_overrides(kwargs, config)
      
      # Example: Transform messages format
      if "messages" in overridden:
          overridden["messages"] = self._transform_messages(overridden["messages"])
      
      return overridden
  ```

### In Test File (`test_<provider>_plugin.py`)

- [ ] **Add provider-specific parameter tests**: Test unique parameters
- [ ] **Test edge cases**: Invalid models, parameter ranges
- [ ] **Optional: Add integration tests**: With real SDK (mark with `@pytest.mark.integration`)

## 🧪 Testing Strategy

### Unit Tests (Always Required)
```bash
# Run tests in mock mode (no API calls, no costs)
TRAIGENT_MOCK_LLM=true pytest tests/unit/integrations/test_<provider>_plugin.py -v

# Run with coverage
TRAIGENT_MOCK_LLM=true pytest tests/unit/integrations/test_<provider>_plugin.py \
  --cov=traigent.integrations.llms.<provider>_plugin --cov-report=term-missing
```

### Integration Tests (Optional)
```bash
# Test with real SDK (requires API key)
export <PROVIDER>_API_KEY="your-key-here"
pytest tests/integration/test_<provider>_plugin.py -v
```

## 🔍 Common Patterns

### Provider with Limited Parameter Support

If provider doesn't support all standard parameters (e.g., no `frequency_penalty`):

```python
def _get_supported_canonical_params(self) -> set[str]:
    """Only map params this provider supports."""
    return {"model", "max_tokens", "temperature", "top_p", "stop"}
```

### Provider with Special Message Format

If provider needs custom message transformation:

```python
def apply_overrides(self, kwargs, config):
    overridden = super().apply_overrides(kwargs, config)
    
    # Example: Convert OpenAI-style messages to provider format
    if "messages" in overridden:
        overridden["prompt"] = self._messages_to_prompt(overridden.pop("messages"))
    
    return overridden
```

### Provider with Unique Authentication

Add custom authentication parameters:

```python
def _get_extra_mappings(self) -> dict[str, str]:
    return {
        "api_key": "api_key",
        "base_url": "base_url",
        "organization_id": "org_id",  # Provider-specific
    }
```

## 📋 Pre-PR Checklist

Before submitting your PR:

- [ ] All TODOs in generated code are resolved
- [ ] Tests pass: `TRAIGENT_MOCK_LLM=true pytest tests/unit/integrations/test_<provider>_plugin.py -v`
- [ ] Code formatted: `make format` (or `isort` + `black`)
- [ ] Code linted: `make lint` (or `ruff check --fix`)
- [ ] Traceability comments present in all new files
- [ ] Type hints added to all public methods
- [ ] Docstrings added (Google style)
- [ ] Updated examples (if needed)
- [ ] Tested with real SDK (optional, but recommended)

## 🆘 Troubleshooting

### Issue: `Framework.<PROVIDER> not found`
**Solution**: Run the scaffold script again, or manually add to `parameter_normalizer.py`:
```python
class Framework(Enum):
    # ...
    YOUR_PROVIDER = "your_provider"
```

### Issue: Import errors in tests
**Solution**: Ensure plugin is exported in `traigent/integrations/llms/__init__.py`:
```python
from traigent.integrations.llms.<provider>_plugin import <Provider>Plugin

__all__ = [
    # ...
    "<Provider>Plugin",
]
```

### Issue: Tests fail with "unexpected keyword argument"
**Solution**: Provider might not support that parameter. Override `_get_supported_canonical_params()`:
```python
def _get_supported_canonical_params(self) -> set[str]:
    return {"model", "temperature", "max_tokens"}  # Only supported params
```

## 📚 Additional Resources

- **Full Guide**: [docs/contributing/ADDING_NEW_INTEGRATIONS.md](../ADDING_NEW_INTEGRATIONS.md)
- **Migration Guide**: [docs/guides/llm_plugin_migration_guide.md](../../guides/llm_plugin_migration_guide.md)
- **Examples**: See `traigent/integrations/llms/openai_plugin.py`, `anthropic_plugin.py`
- **Scaffold Tool Help**: `python scripts/scaffold_llm_plugin.py --help`

## 🎬 Real Example: Adding Groq

```bash
# 1. Scaffold
python scripts/scaffold_llm_plugin.py groq --sdk groq

# 2. Customize groq_plugin.py:
# - Update SDK class names: "groq.Groq", "groq.AsyncGroq"
# - Update method names: "chat.completions.create"
# - Add Groq-specific params (if any)
# - Update validation ranges based on Groq docs

# 3. Test
TRAIGENT_MOCK_LLM=true pytest tests/unit/integrations/test_groq_plugin.py -v

# 4. Format and lint
make format && make lint

# 5. Test with real API (optional)
export GROQ_API_KEY="your-key"
pytest tests/integration/test_groq_plugin.py -v

# 6. Submit PR using LLM Integration template
```

## 💡 Pro Tips

1. **Start with OpenAI plugin** as a reference - it's well-documented
2. **Use mock mode** during development to avoid API costs
3. **Test parameter precedence** - user kwargs should override config
4. **Add model discovery** (optional) for better validation
5. **Document unique features** in plugin docstring (streaming, tools, vision)

---

**Need help?** Open an issue with the `question` label or ask in GitHub Discussions!
