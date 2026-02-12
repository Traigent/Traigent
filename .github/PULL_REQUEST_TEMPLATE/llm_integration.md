## New LLM Integration: [Provider Name]

<!--
Thank you for contributing a new LLM integration! This template helps ensure
your PR includes all necessary components for a high-quality integration.

Please fill out the sections below and check off items as you complete them.
-->

### Summary
Adds support for [Provider Name] LLM SDK integration.

**Provider**: [Provider Name]  
**SDK**: `[provider-sdk-package-name]`  
**Documentation**: [Link to provider docs]

### Features
<!-- Check all that apply -->

- [ ] Parameter mapping for standard LLM parameters (model, temperature, max_tokens, etc.)
- [ ] Provider-specific parameter mappings
- [ ] Validation rules for all parameters
- [ ] Support for streaming
- [ ] Support for tool/function calling
- [ ] Support for vision/multimodal (if applicable)
- [ ] Model discovery via SDK API
- [ ] Comprehensive test suite

### Implementation Checklist

#### Plugin Implementation
- [ ] Created plugin file: `traigent/integrations/llms/[provider]_plugin.py`
- [ ] Added Framework enum value to `parameter_normalizer.py`
- [ ] Implemented all required methods:
  - [ ] `_get_metadata()`
  - [ ] `_get_extra_mappings()`
  - [ ] `_get_provider_specific_rules()`
  - [ ] `get_target_classes()`
  - [ ] `get_target_methods()`
  - [ ] `apply_overrides()` (if needed)
- [ ] Exported plugin in `__init__.py`

#### Model Discovery (Optional)
- [ ] Created model discovery: `traigent/integrations/model_discovery/[provider]_discovery.py`
- [ ] Implemented `_fetch_models_from_sdk()`
- [ ] Implemented `_get_model_pattern()`
- [ ] Registered in model discovery registry
- [ ] Added known models to `traigent/config/models.yaml`

#### Testing
- [ ] Created test file: `tests/unit/integrations/test_[provider]_plugin.py`
- [ ] Tests for basic plugin behavior (≥3 tests)
- [ ] Tests for parameter mappings (≥4 tests)
- [ ] Tests for validation rules (≥3 tests)
- [ ] Tests for plugin metadata (≥3 tests)
- [ ] All tests pass with `TRAIGENT_MOCK_LLM=true`
- [ ] Test coverage ≥ 80%

#### Code Quality
- [ ] Traceability comments added to all new files
- [ ] Type hints for all public methods
- [ ] Google-style docstrings for all classes and public methods
- [ ] Formatted with `make format`
- [ ] Passes `make lint` (Ruff, Black, isort, MyPy, Bandit)
- [ ] No security issues flagged

### Testing Evidence

**Test Results:**
```bash
# Paste output from: TRAIGENT_MOCK_LLM=true pytest tests/unit/integrations/test_[provider]_plugin.py -v
```

**Coverage Report:**
```bash
# Paste output from: TRAIGENT_MOCK_LLM=true pytest tests/unit/integrations/test_[provider]_plugin.py --cov=traigent.integrations.llms.[provider]_plugin
```

### Validation

**Parameter Mappings Tested:**
<!-- List all parameters you've tested -->

- [ ] model
- [ ] temperature
- [ ] max_tokens
- [ ] top_p
- [ ] stream
- [ ] stop
- [ ] [Add provider-specific parameters]

**SDK Methods Instrumented:**
<!-- List all SDK methods your plugin overrides -->

- [ ] `[SDK.Class.method]` (e.g., `openai.OpenAI.chat.completions.create`)
- [ ] `[SDK.AsyncClass.method]` (e.g., `openai.AsyncOpenAI.chat.completions.create`)

### Example Usage

```python
import os
import traigent

# Set API key
os.environ["[PROVIDER]_API_KEY"] = "your-api-key-here"

@traigent.optimize(
    objectives=["accuracy"],
    config_space={
        "model": ["[model-1]", "[model-2]"],
        "temperature": (0.0, 1.0),
    }
)
def my_task(input_text):
    from [provider_sdk] import Client
    client = Client()
    response = client.chat.completions.create(
        messages=[{"role": "user", "content": input_text}]
    )
    return response.choices[0].message.content

# Run optimization
result = my_task.optimize(
    dataset=[
        {"input_text": "What is AI?"},
        {"input_text": "Explain machine learning"},
    ]
)
print(f"Best config: {result.best_config}")
```

### Breaking Changes
<!-- Does this PR introduce any breaking changes? -->

- [ ] No breaking changes
- [ ] Breaking changes (describe below)

**Breaking Changes:**
<!-- If yes, describe what breaks and how users should migrate -->


### Documentation
<!-- Check all that apply -->

- [ ] Updated examples (if applicable)
- [ ] Updated README with provider (if major provider)
- [ ] Followed [ADDING_NEW_INTEGRATIONS.md](../../docs/contributing/ADDING_NEW_INTEGRATIONS.md) guide

### Related Issues
<!-- Link to related issues or discussions -->

Closes #[issue_number]

### Additional Context
<!-- Add any other context, screenshots, or examples -->


---

### Reviewer Checklist
<!-- For maintainers reviewing the PR -->

- [ ] Plugin follows established patterns from existing integrations
- [ ] All required files present and correctly structured
- [ ] Tests are comprehensive and pass
- [ ] Code quality standards met (formatting, linting, type hints, docstrings)
- [ ] No security issues
- [ ] Documentation is clear and complete
- [ ] Example usage works as expected
