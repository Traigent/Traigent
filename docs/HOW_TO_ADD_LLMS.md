# How to Add More LLMs to Traigent (Like GitHub Copilot)

This document explains how the new LLM integration workflow makes it easy to add new LLM providers to Traigent.

## 🎯 Goal

Make it as easy as possible for contributors to add support for new LLM providers (Groq, Together AI, Perplexity, etc.) to Traigent, similar to how GitHub Copilot provides an intuitive workflow.

## 🚀 Quick Start (5 Minutes)

Adding a new LLM provider now takes just 5 minutes:

```bash
# 1. Generate boilerplate
python scripts/scaffold_llm_plugin.py <provider_name>

# 2. Customize the generated files (follow TODO comments)

# 3. Test in mock mode (no API costs!)
TRAIGENT_MOCK_LLM=true pytest tests/unit/integrations/test_<provider>_plugin.py -v

# 4. Format and lint
make format && make lint

# 5. Submit PR using the template
```

## 📦 What's New

### 1. **Scaffold Script** (`scripts/scaffold_llm_plugin.py`)

Automatically generates:
- ✅ Plugin implementation with all required methods
- ✅ Comprehensive test suite (10+ tests)
- ✅ Framework registration and exports
- ✅ Type hints, docstrings, and traceability comments
- ✅ Helpful TODO comments for customization

**Example:**
```bash
$ python scripts/scaffold_llm_plugin.py groq

INFO: SUCCESS! Plugin scaffolding complete.
Created:
  - traigent/integrations/llms/groq_plugin.py
  - tests/unit/integrations/test_groq_plugin.py
```

### 2. **GitHub Templates**

#### Issue Template
Use `.github/ISSUE_TEMPLATE/new_llm_integration.md` when requesting a new LLM integration:
- Provider information checklist
- Integration details
- Implementation roadmap
- Example usage

#### PR Template
Use `.github/PULL_REQUEST_TEMPLATE/llm_integration.md` when submitting:
- Implementation checklist
- Testing evidence
- Code quality verification
- Reviewer guidelines

### 3. **Documentation**

#### Quick Reference (`docs/contributing/QUICK_REFERENCE_LLM_INTEGRATION.md`)
- 5-minute quick start
- Customization checklist with examples
- Common patterns and solutions
- Troubleshooting guide

#### Comprehensive Guide (`docs/contributing/ADDING_NEW_INTEGRATIONS.md`)
- Step-by-step instructions
- Detailed examples
- Best practices
- Testing requirements

### 4. **Enhanced Discoverability**

#### README.md
Now includes prominent "Adding New LLM Integrations" section with:
- Quick start commands
- Resource links
- Current supported LLMs list

#### CONTRIBUTING.md
Added dedicated section with quick-start guide and templates

## 📋 Complete Workflow Example

### Step 1: Create Issue
1. Go to GitHub Issues
2. Click "New Issue"
3. Select "New LLM Integration" template
4. Fill in provider details

### Step 2: Scaffold Plugin
```bash
python scripts/scaffold_llm_plugin.py groq --sdk groq
```

### Step 3: Customize Generated Code
Edit `traigent/integrations/llms/groq_plugin.py`:

```python
# Update SDK classes
def get_target_classes(self) -> list[str]:
    return [
        "groq.Groq",        # ← Update with real SDK class
        "groq.AsyncGroq",   # ← Update with real SDK class
    ]

# Update SDK methods
def get_target_methods(self) -> dict[str, list[str]]:
    return {
        "groq.Groq": [
            "chat.completions.create",  # ← Update with real methods
        ],
        "groq.AsyncGroq": [
            "chat.completions.create",
        ],
    }

# Add provider-specific parameters
def _get_extra_mappings(self) -> dict[str, str]:
    return {
        "seed": "seed",  # ← Add Groq-specific params
    }
```

### Step 4: Test
```bash
TRAIGENT_MOCK_LLM=true pytest tests/unit/integrations/test_groq_plugin.py -v
```

Output:
```
tests/unit/integrations/test_groq_plugin.py::TestGroqPlugin::test_framework_is_correct PASSED
tests/unit/integrations/test_groq_plugin.py::TestGroqPlugin::test_metadata_name PASSED
tests/unit/integrations/test_groq_plugin.py::TestGroqParameterMappings::test_model_preserved PASSED
... (10+ tests)

=============== 13 passed in 0.5s ===============
```

### Step 5: Format and Lint
```bash
make format && make lint
```

### Step 6: Submit PR
1. Create PR using "LLM Integration" template
2. Fill in checklist
3. Add test results
4. Submit for review

## 🎓 Learning Path

1. **Start here**: [Quick Reference](docs/contributing/QUICK_REFERENCE_LLM_INTEGRATION.md)
2. **Deep dive**: [Adding New Integrations Guide](docs/contributing/ADDING_NEW_INTEGRATIONS.md)
3. **Examples**: Look at existing plugins in `traigent/integrations/llms/`
4. **Migration**: [LLM Plugin Migration Guide](docs/guides/llm_plugin_migration_guide.md)

## 🔧 Tools & Resources

| Resource | Description | Link |
|----------|-------------|------|
| **Scaffold Script** | Generate boilerplate | `scripts/scaffold_llm_plugin.py` |
| **Issue Template** | Request integration | `.github/ISSUE_TEMPLATE/new_llm_integration.md` |
| **PR Template** | Submit integration | `.github/PULL_REQUEST_TEMPLATE/llm_integration.md` |
| **Quick Reference** | 5-min guide | `docs/contributing/QUICK_REFERENCE_LLM_INTEGRATION.md` |
| **Full Guide** | Comprehensive docs | `docs/contributing/ADDING_NEW_INTEGRATIONS.md` |

## 📊 Current Supported LLMs

| Provider | SDK | Status |
|----------|-----|--------|
| OpenAI | `openai` | ✅ Supported |
| Anthropic | `anthropic` | ✅ Supported |
| Google Gemini | `google-generativeai` | ✅ Supported |
| Mistral AI | `mistralai` | ✅ Supported |
| Cohere | `cohere` | ✅ Supported |
| AWS Bedrock | `boto3` | ✅ Supported |
| Azure OpenAI | `openai` | ✅ Supported |
| Hugging Face | `huggingface_hub` | ✅ Supported |
| LangChain | `langchain` | ✅ Supported |
| LlamaIndex | `llama-index` | ✅ Supported |

## 💡 Pro Tips

1. **Use mock mode** during development to avoid API costs
2. **Start with OpenAI plugin** as a reference
3. **Test parameter precedence** - user kwargs should override config
4. **Add model discovery** for better validation (optional)
5. **Document unique features** in plugin docstring

## ❓ Common Questions

### Q: How long does it take to add a new LLM?
**A:** ~5-10 minutes with the scaffold script, plus customization time based on provider complexity.

### Q: Do I need API credentials?
**A:** No! Use `TRAIGENT_MOCK_LLM=true` for development without API calls.

### Q: What if the provider has unique features?
**A:** Override `apply_overrides()` method for custom transformations. See examples in existing plugins.

### Q: How do I handle providers with limited parameters?
**A:** Override `_get_supported_canonical_params()` to list only supported parameters.

## 🐛 Troubleshooting

### Issue: Scaffold script fails
```bash
# Check Python version
python --version  # Should be 3.11+

# Run with verbose output
python scripts/scaffold_llm_plugin.py <provider> --help
```

### Issue: Tests fail with import errors
```bash
# Ensure plugin is exported
# Check traigent/integrations/llms/__init__.py

# Format and lint
make format && make lint
```

### Issue: Provider doesn't support parameter
```python
# Override supported params
def _get_supported_canonical_params(self) -> set[str]:
    return {"model", "temperature", "max_tokens"}
```

## 🎉 Success Stories

Contributors have successfully added:
- OpenAI (GPT models)
- Anthropic (Claude)
- Google Gemini
- Mistral AI
- And more!

Your integration could be next!

## 📞 Get Help

- **Questions**: Open GitHub issue with `question` label
- **Bugs**: Open GitHub issue with `bug` label
- **Discussions**: GitHub Discussions
- **Examples**: Browse `traigent/integrations/llms/`

---

**Ready to contribute?** Run the scaffold script and start building! 🚀
