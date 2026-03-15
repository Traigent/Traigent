# Example Template for Traigent

## 📋 Template Structure

```python
"""
[ONE-LINE DESCRIPTION]

This example demonstrates [FEATURE/CONCEPT] in Traigent.
Shows how to [PRIMARY USE CASE].
"""

# [REQUIRED] Enable mock mode for safe testing
os.environ["TRAIGENT_MOCK_LLM"] = "true"

import traigent

# [OPTIONAL] Configuration
# [REQUIRED] Core example code (minimal, focused)
# [OPTIONAL] Results display (minimal output)
# [REQUIRED] Key takeaways/comments
```

## 🎯 Organization Principles

### 1. **File Naming Convention**
```
[category]_[feature]_[aspect].py
# Examples:
# basic_optimization_single.py
# advanced_caching_multi.py
# integration_openai_simple.py
```

### 2. **Content Structure**
```python
"""
ONE-LINE: What this example does
MULTI-LINE: Why it's useful, what it teaches
"""

# 1. Setup (2-3 lines)
# 2. Core functionality (5-15 lines)
# 3. Results (2-3 lines)
# 4. Key insights (comments)
```

### 3. **Consistency Rules**
- ✅ Always enable mock mode first
- ✅ Minimal imports (only what's needed)
- ✅ Clear, descriptive variable names
- ✅ One feature per example
- ✅ Comments explain "why", not "what"
- ✅ Results show actual value/outcome

### 4. **Quality Checklist**
- [ ] Runs without external dependencies
- [ ] Clear, focused purpose
- [ ] Minimal but complete code
- [ ] Educational value
- [ ] Easy to modify/experiment
- [ ] Good error handling
- [ ] Performance considerations shown

---

## 📁 Recommended Directory Structure

```
examples/
├── core/                 # Core feature examples (start here)
├── core-concepts/        # Fundamental features
├── advanced-patterns/    # Complex use cases
├── integrations/         # External service examples
├── use-cases/           # Real-world applications
├── shared_utils/        # Common helper code
└── docs/               # Documentation
```
