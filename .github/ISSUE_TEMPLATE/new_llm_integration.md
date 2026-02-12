---
name: New LLM Integration
about: Request support for a new LLM provider
title: '[LLM Integration] Add support for [Provider Name]'
labels: ['enhancement', 'integration', 'llm']
assignees: ''
---

## Provider Information

**Provider Name:** 
<!-- e.g., Groq, Together AI, Perplexity, etc. -->

**Provider SDK:**
<!-- Name of the Python SDK package (e.g., `groq`, `together`, `perplexity-python`) -->

**SDK Documentation:**
<!-- Link to the provider's Python SDK documentation -->

**Provider API Documentation:**
<!-- Link to the provider's REST API documentation -->

## Integration Details

### Available Models
<!-- List some of the key models available from this provider -->

- 
- 
- 

### Unique Features
<!-- Does this provider have any unique features or parameters that need special handling? -->

- [ ] Streaming support
- [ ] Tool/function calling
- [ ] Vision/multimodal
- [ ] JSON mode
- [ ] Custom parameters (describe below)

**Custom Parameters:**
<!-- List any provider-specific parameters not common to other LLMs -->


### API Authentication
**Environment Variable(s):**
<!-- e.g., `GROQ_API_KEY`, `TOGETHER_API_KEY` -->


## Implementation Checklist

### Getting Started
- [ ] Read [docs/contributing/ADDING_NEW_INTEGRATIONS.md](../../docs/contributing/ADDING_NEW_INTEGRATIONS.md)
- [ ] Install the provider's SDK: `pip install [provider-sdk]`
- [ ] Get API credentials for testing (optional with mock mode)

### Scaffolding
- [ ] Run scaffold script: `python scripts/scaffold_llm_plugin.py [provider_name]`
- [ ] Review generated files and customize TODO sections

### Implementation
- [ ] Add Framework enum value to `traigent/integrations/utils/parameter_normalizer.py`
- [ ] Implement plugin in `traigent/integrations/llms/[provider]_plugin.py`
  - [ ] Set `FRAMEWORK` attribute
  - [ ] Implement `_get_metadata()`
  - [ ] Implement `_get_extra_mappings()` for provider-specific params
  - [ ] Implement `_get_provider_specific_rules()` for validation
  - [ ] Implement `get_target_classes()` with SDK classes
  - [ ] Implement `get_target_methods()` with SDK methods
  - [ ] Override `apply_overrides()` if needed for special logic
- [ ] Export plugin in `traigent/integrations/llms/__init__.py`

### Model Discovery (Optional)
- [ ] Create `traigent/integrations/model_discovery/[provider]_discovery.py`
- [ ] Implement `_fetch_models_from_sdk()`
- [ ] Implement `_get_model_pattern()`
- [ ] Register in model discovery registry
- [ ] Add known models to `traigent/config/models.yaml`

### Testing
- [ ] Create test file: `tests/unit/integrations/test_[provider]_plugin.py`
- [ ] Test basic plugin behavior (framework, metadata)
- [ ] Test parameter mappings (model, temperature, max_tokens, etc.)
- [ ] Test user kwargs precedence
- [ ] Test validation rules
- [ ] Run tests: `TRAIGENT_MOCK_LLM=true pytest tests/unit/integrations/test_[provider]_plugin.py -v`

### Code Quality
- [ ] Add traceability comment to all new files
- [ ] Add type hints to all public methods
- [ ] Add Google-style docstrings to all classes and public methods
- [ ] Format code: `make format`
- [ ] Lint code: `make lint`
- [ ] All tests pass

### Documentation
- [ ] Update examples if needed
- [ ] Add provider to list in README (if major provider)

## Additional Context
<!-- Add any other context, screenshots, or examples about the integration request -->


## Quick Start Command

Once the integration is implemented, users should be able to use it like this:

```python
import os
import traigent

# Set API key
os.environ["[PROVIDER]_API_KEY"] = "your-api-key-here"

@traigent.optimize(
    objectives=["accuracy"],
    config_space={
        "model": ["[provider-model-1]", "[provider-model-2]"],
        "temperature": (0.0, 1.0),
    }
)
def my_llm_task(messages):
    from [provider_sdk] import Client
    client = Client()
    response = client.chat.completions.create(
        messages=messages
    )
    return response.choices[0].message.content
```

---

## Resources

- **Documentation**: [Adding New Integrations Guide](../../docs/contributing/ADDING_NEW_INTEGRATIONS.md)
- **Examples**: See existing plugins in `traigent/integrations/llms/`
- **Scaffold Tool**: `python scripts/scaffold_llm_plugin.py [provider_name] --help`
- **Testing Guide**: Use `TRAIGENT_MOCK_LLM=true` for development without API costs
