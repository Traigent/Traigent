# Release Readiness Testing Prompt (Extended v2)

You are a senior AI engineer who just discovered Traigent and is excited to optimize your LLM agents. You have **NO prior knowledge** of Traigent's internals—you only know what's in the README.

**Your mission**: Validate that Traigent is release-ready by:
1. Following the README exactly as a first-time user would
2. Creating your own experiments to stress-test the SDK
3. Documenting every point of friction, confusion, or failure

---

## PART 0: Environment Setup (Critical)

Before starting, ensure a completely clean environment:

```bash
# Create fresh directory OUTSIDE any existing projects
cd /tmp  # or another clean location
mkdir traigent-test-$(date +%s)
cd traigent-test-*

# Verify no prior Traigent state
pip list | grep -i traigent  # Should show nothing
ls ~/.traigent 2>/dev/null   # Note if this exists (prior state)
```

**Environment variables to set for ALL testing:**
```bash
export TRAIGENT_MOCK_MODE=true
export TRAIGENT_LOG_LEVEL=INFO  # See what's happening
```

---

## PART 1: Installation & Quickstart Validation

### 1.1 Installation (Time yourself - should be <5 min)

```bash
# Clone and install - follow README EXACTLY
git clone https://github.com/traigent/traigent-sdk.git
cd traigent-sdk

# Follow README instructions for your preferred method (pip or uv)
# Document any deviations or errors
```

**Checklist:**
- [ ] Clone succeeds without authentication issues
- [ ] Virtual environment creation works
- [ ] `pip install -e ".[dev,integrations,analytics,security]"` completes without errors
- [ ] No deprecation warnings during install (note any that appear)
- [ ] Total install time: _____ minutes

### 1.2 Version Verification

```bash
traigent --version
traigent info
python -c "import traigent; print(traigent.__version__)"
```

**Checklist:**
- [ ] All three show version `0.9.0`
- [ ] `traigent info` shows expected features and integrations
- [ ] No import warnings when loading traigent module

### 1.3 Quickstart Examples (Time each one)

Run each in order, noting any issues:

```bash
export TRAIGENT_MOCK_MODE=true

# Example 1
python examples/quickstart/01_simple_qa.py
# Expected: Runs optimization, shows trials, prints best config
# Time: _____ seconds

# Example 2
python examples/quickstart/02_customer_support_rag.py
# Expected: Runs RAG optimization
# Time: _____ seconds

# Example 3
python examples/quickstart/03_custom_objectives.py
# Expected: Custom objective optimization
# Time: _____ seconds
```

**For each example, note:**
- [ ] Completes without errors
- [ ] Output is understandable (can you tell what happened?)
- [ ] No scary warnings (distinguish INFO from WARNING/ERROR)
- [ ] Results make sense (best_score, trial count, etc.)

---

## PART 2: Documentation Validation

Read ONLY the README and try to answer these questions. Document where answers are found or if they're missing:

### 2.1 Core Concepts
- [ ] What is a "Tuned Variable"? (Section: _____)
- [ ] What objectives are supported? (Section: _____)
- [ ] What's the difference between `edge_analytics` and `cloud` mode? (Section: _____)
- [ ] How do I create a dataset? (Section: _____)

### 2.2 API Questions
- [ ] How do I access the current trial's config inside my function?
- [ ] How do I get the best config after optimization?
- [ ] What parameters does `@traigent.optimize` accept?
- [ ] How do I use a custom evaluator?

### 2.3 Error Recovery
- [ ] If I see 0% accuracy, what should I check? (Section: _____)
- [ ] If I get import errors, what should I do? (Section: _____)
- [ ] How do I debug optimization issues?

---

## PART 3: Creative Experiments (Mock Mode)

### 3.1 Create Your Test Dataset

First, create a test dataset. The format should match what README documents:

```bash
cat > my_test_data.jsonl << 'EOF'
{"input": {"question": "What is Python?"}, "output": "Python is a programming language"}
{"input": {"question": "What is 2+2?"}, "output": "4"}
{"input": {"question": "Who wrote Hamlet?"}, "output": "William Shakespeare"}
{"input": {"question": "What color is the sky?"}, "output": "Blue"}
{"input": {"question": "What is the capital of France?"}, "output": "Paris"}
EOF
```

### 3.2 Parameter-Sensitive Mock Function

Create a function where Traigent can actually discover meaningful differences:

```python
# my_experiment.py
import asyncio
import traigent

@traigent.optimize(
    configuration_space={
        "temperature": [0.1, 0.5, 0.9],
        "model": ["gpt-3.5-turbo", "gpt-4o-mini", "gpt-4o"],
        "max_tokens": [50, 150, 300]
    },
    objectives=["accuracy", "cost"],
    eval_dataset="my_test_data.jsonl",
)
def smart_mock_agent(question: str) -> str:
    """
    Mock agent that simulates quality differences based on config.
    Lower temperature + better model = better factual accuracy.
    """
    config = traigent.get_config()
    temp = config.get("temperature", 0.7)
    model = config.get("model", "gpt-3.5-turbo")
    max_tokens = config.get("max_tokens", 150)

    # Simulate quality based on config
    quality_score = (1 - temp) * 0.4 + (max_tokens / 300) * 0.3
    if "gpt-4" in model:
        quality_score += 0.3

    # Return answers that reflect quality
    if "Python" in question:
        if quality_score > 0.7:
            return "Python is a programming language"
        else:
            return "Python is a language"
    elif "2+2" in question:
        return "4" if quality_score > 0.3 else "four"
    elif "Hamlet" in question:
        return "William Shakespeare" if quality_score > 0.5 else "Shakespeare"
    elif "sky" in question:
        return "Blue" if quality_score > 0.4 else "blue"
    elif "France" in question:
        return "Paris" if quality_score > 0.6 else "paris"
    else:
        return f"Answer about {question}"

async def main():
    result = await smart_mock_agent.optimize(
        algorithm="random",
        max_trials=10
    )
    print(f"\nBest score: {result.best_score:.3f}")
    print(f"Best config: {result.best_config}")
    print(f"Trials completed: {len(result.trials)}")

if __name__ == "__main__":
    asyncio.run(main())
```

**Run and verify:**
```bash
TRAIGENT_MOCK_MODE=true python my_experiment.py
```

- [ ] Optimization completes
- [ ] Best config favors low temperature / better model (as expected from logic)
- [ ] Can access `result.best_config` and `result.best_score`

### 3.3 Edge Case Matrix

Test each scenario and document results:

#### Configuration Space Variations

| Test | Code | Expected | Actual | Severity |
|------|------|----------|--------|----------|
| Empty space | `configuration_space={}` | Error or single trial | | |
| Single param | `configuration_space={"temp": [0.5]}` | 1 config tested | | |
| Large space | 4+ params, 5+ values each | Many trials or sampling | | |
| Continuous range | `{"temp": (0.0, 1.0)}` | Works with random/bayesian | | |
| Mixed types | categorical + continuous | Works | | |
| Invalid values | `{"temp": "invalid"}` | Clear error message | | |

```python
# Test: Empty configuration space
@traigent.optimize(
    configuration_space={},
    objectives=["accuracy"],
    eval_dataset="my_test_data.jsonl",
)
def empty_config_test(question: str) -> str:
    return "answer"
```

#### Dataset Variations

| Test | Dataset Content | Expected | Actual | Severity |
|------|-----------------|----------|--------|----------|
| Minimal (1 example) | Single line | Works | | |
| Missing output field | `{"input": {...}}` only | Accuracy=0 or warning | | |
| Wrong field names | `{"query": ...}` instead of `{"input": ...}` | Clear error | | |
| Empty file | 0 bytes | Clear error | | |
| Invalid JSON | Malformed line | Clear error with line number | | |
| Very long outputs | 10K+ chars | Works | | |
| Unicode/emoji | Special characters | Works | | |

```bash
# Test: Wrong field names
echo '{"query": "test", "answer": "response"}' > bad_dataset.jsonl
# Then try to use it and note the error message
```

#### Objective Variations

| Test | Objectives | Expected | Actual | Severity |
|------|------------|----------|--------|----------|
| Single | `["accuracy"]` | Works | | |
| Dual | `["accuracy", "cost"]` | Works | | |
| Triple | `["accuracy", "cost", "latency"]` | Works or clear warning | | |
| Invalid | `["invalid_metric"]` | Clear error | | |
| Empty | `[]` | Error or default | | |

#### Algorithm Variations

| Test | Algorithm | Expected | Actual | Severity |
|------|-----------|----------|--------|----------|
| Grid | `algorithm="grid"` | Exhaustive search | | |
| Random | `algorithm="random"` | Random sampling | | |
| Bayesian | `algorithm="bayesian"` | Works if optuna installed | | |
| Invalid | `algorithm="invalid"` | Clear error | | |
| Grid + continuous | Grid with `(0.0, 1.0)` range | Error or auto-discretize | | |

### 3.4 Lifecycle Tests

Test the full optimization lifecycle:

```python
import asyncio
import traigent

@traigent.optimize(
    configuration_space={"temp": [0.1, 0.5, 0.9]},
    objectives=["accuracy"],
    eval_dataset="my_test_data.jsonl",
)
def lifecycle_test(question: str) -> str:
    config = traigent.get_config()
    return f"Answer with temp={config.get('temp')}"

async def main():
    # 1. Run optimization
    result = await lifecycle_test.optimize(max_trials=5)

    # 2. Check result attributes
    print(f"best_score: {result.best_score}")
    print(f"best_config: {result.best_config}")
    print(f"trials: {len(result.trials)}")
    print(f"duration: {result.duration}")

    # 3. Apply best config
    lifecycle_test.apply_best_config()

    # 4. Call function normally (should use best config)
    normal_result = lifecycle_test("test question")
    print(f"Normal call result: {normal_result}")

    # 5. Check current_config
    print(f"current_config: {lifecycle_test.current_config}")

if __name__ == "__main__":
    asyncio.run(main())
```

**Checklist:**
- [ ] `result.best_score` is accessible and numeric
- [ ] `result.best_config` is a dict with expected keys
- [ ] `result.trials` is a list with trial objects
- [ ] `apply_best_config()` works without error
- [ ] Normal function call after optimization uses best config
- [ ] `func.current_config` shows the applied config

### 3.5 Error Handling Tests

Intentionally trigger errors and evaluate the error messages:

```python
# Test 1: get_trial_config outside optimization
import traigent
try:
    config = traigent.get_trial_config()  # Should error
except Exception as e:
    print(f"Error type: {type(e).__name__}")
    print(f"Error message: {e}")
    # Is this message helpful? Does it suggest what to do?
```

```python
# Test 2: Invalid dataset path
@traigent.optimize(
    configuration_space={"temp": [0.5]},
    objectives=["accuracy"],
    eval_dataset="nonexistent_file.jsonl",  # Does not exist
)
def bad_dataset_test(q: str) -> str:
    return "answer"

# Try to optimize - what error do you get?
```

```python
# Test 3: Exception inside decorated function
@traigent.optimize(
    configuration_space={"temp": [0.5]},
    objectives=["accuracy"],
    eval_dataset="my_test_data.jsonl",
)
def error_prone_func(question: str) -> str:
    if "Python" in question:
        raise ValueError("Intentional error!")
    return "answer"

# Does the trial fail gracefully? Is the error reported?
```

```python
# Test 4: Non-string return value
@traigent.optimize(
    configuration_space={"temp": [0.5]},
    objectives=["accuracy"],
    eval_dataset="my_test_data.jsonl",
)
def returns_dict(question: str) -> dict:  # Note: returns dict, not str
    return {"answer": "test"}

# What happens? Error? Works? Weird accuracy?
```

---

## PART 4: CLI Exploration

### 4.1 Help & Info Commands

```bash
traigent --help
traigent info
traigent algorithms
traigent examples
```

**Checklist:**
- [ ] All commands show helpful output
- [ ] No errors or tracebacks
- [ ] Version info is consistent

### 4.2 Template Generation

```bash
# Generate each template type
traigent generate -t basic -o test_basic.py
traigent generate -t multi-objective -o test_multi.py
traigent generate -t langchain -o test_langchain.py
traigent generate -t openai -o test_openai.py

# Try to run the generated templates
TRAIGENT_MOCK_MODE=true python test_basic.py
```

**For each template:**
- [ ] Generation succeeds
- [ ] Generated file has correct syntax
- [ ] Dataset path in template exists or is clearly documented
- [ ] Can run the template in mock mode
- [ ] Template demonstrates the advertised feature

### 4.3 CLI Optimize Command

```bash
# Create a simple file to optimize
cat > cli_test.py << 'EOF'
import traigent

@traigent.optimize(
    configuration_space={"temp": [0.1, 0.5, 0.9]},
    objectives=["accuracy"],
    eval_dataset="my_test_data.jsonl",
)
def my_func(question: str) -> str:
    return "answer"
EOF

# Run via CLI
TRAIGENT_MOCK_MODE=true traigent optimize cli_test.py --algorithm random --max-trials 5
```

**Checklist:**
- [ ] Command runs without error
- [ ] Progress is shown during optimization
- [ ] Results are displayed at the end
- [ ] Results are saved (check with `traigent results`)

### 4.4 Validation Command

```bash
# Validate dataset
traigent validate my_test_data.jsonl -o accuracy -o cost

# Try validating bad dataset
echo "invalid json" > bad.jsonl
traigent validate bad.jsonl -o accuracy
```

**Checklist:**
- [ ] Valid dataset passes validation
- [ ] Invalid dataset shows clear error
- [ ] Error message indicates what's wrong

### 4.5 Results Management

```bash
traigent results              # List results
traigent results --last 5     # Recent results
traigent plot <result_name>   # If results exist
```

**Checklist:**
- [ ] Results from previous runs are listed
- [ ] Can view details of specific result
- [ ] Plot command works (or gives clear error if no results)

---

## PART 5: Integration Scenarios

### 5.1 LangChain Integration (if installed)

```python
import traigent
try:
    from langchain_openai import ChatOpenAI
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    print("LangChain not installed - skip this test")

if LANGCHAIN_AVAILABLE:
    @traigent.optimize(
        configuration_space={
            "model": ["gpt-3.5-turbo", "gpt-4o-mini"],
            "temperature": [0.1, 0.5]
        },
        objectives=["accuracy"],
        eval_dataset="my_test_data.jsonl",
    )
    def langchain_test(question: str) -> str:
        llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.5)
        # In mock mode, this should work without API key
        response = llm.invoke(question)
        return response.content
```

### 5.2 Persistence Test

```python
from traigent.utils.persistence import PersistenceManager
import asyncio

# First, run an optimization
@traigent.optimize(
    configuration_space={"temp": [0.1, 0.5]},
    objectives=["accuracy"],
    eval_dataset="my_test_data.jsonl",
)
def persist_test(q: str) -> str:
    return "answer"

async def main():
    result = await persist_test.optimize(max_trials=3)

    # Save result
    pm = PersistenceManager(".traigent_test")
    path = pm.save_result(result, "my_test_result")
    print(f"Saved to: {path}")

    # List results
    results = pm.list_results()
    print(f"Found {len(results)} results")

    # Load result
    loaded = pm.load_result("my_test_result")
    print(f"Loaded result with {len(loaded.trials)} trials")

asyncio.run(main())
```

**Checklist:**
- [ ] Save completes without error
- [ ] Load returns equivalent result
- [ ] list_results shows saved result

---

## Reporting Format

For each gap found, create an entry like this:

```markdown
### Gap: [Short Title]

**Category**: [Installation | Documentation | API | CLI | Error Handling | Performance]

**Severity**:
- 🔴 Blocker: User cannot proceed, would abandon Traigent
- 🟡 Friction: Confusing but workaround exists, poor error message
- 🟢 Polish: Works but could be improved

**What I tried**:
```
[exact code or command]
```

**What I expected**:
[based on README or intuition]

**What actually happened**:
```
[exact output, error message, or behavior]
```

**Suggested fix**:
[optional - your recommendation]
```

---

## Summary Checklist

After completing all tests, fill out this summary:

### Installation & Setup
- [ ] Install time < 5 minutes
- [ ] Version numbers consistent everywhere
- [ ] No scary warnings during install

### Quickstarts
- [ ] All 3 quickstart examples run
- [ ] Output is understandable
- [ ] Total quickstart time < 10 minutes

### Documentation
- [ ] Can create dataset from README alone
- [ ] API parameters are documented
- [ ] Error recovery steps are clear

### Core Functionality
- [ ] Optimization completes for custom experiment
- [ ] Best config is accessible and makes sense
- [ ] apply_best_config() works
- [ ] Multiple algorithms work (grid, random)

### CLI
- [ ] All CLI commands work
- [ ] Generated templates are runnable
- [ ] Validation command catches errors

### Error Handling
- [ ] Invalid inputs give clear errors
- [ ] Errors include actionable suggestions
- [ ] No silent failures

### Edge Cases
- [ ] Empty/minimal datasets handled
- [ ] Invalid objectives caught
- [ ] Mixed config types work

---

## Success Criteria

Traigent is **release-ready** when:

1. ✅ A new user can install and run quickstarts in **under 10 minutes**
2. ✅ Creating a custom optimization experiment is **intuitive from README alone**
3. ✅ Error messages **guide users toward solutions** (not just "error occurred")
4. ✅ No **silent failures** or confusing behaviors in common use cases
5. ✅ Version numbers are **consistent** in CLI, Python import, and README
6. ✅ Mock mode works **completely offline** with no warnings about missing backends
7. ✅ Generated templates are **immediately runnable** without modification
8. ✅ At least **80% of edge cases** pass or fail gracefully with clear errors

---

## Notes for Testers

1. **README is your only guide** — Do NOT look at source code to figure things out
2. **Document confusion** — If something isn't clear from docs, that's a gap
3. **Try to break it** — Edge cases reveal real issues
4. **Fresh environment** — No cached packages or prior Traigent state
5. **Mock mode only** — Use `TRAIGENT_MOCK_MODE=true` for all experiments
6. **Be creative** — Try combinations the developers might not have tested
7. **Report everything** — Even minor friction points matter for release readiness
8. **Time yourself** — Note how long each section takes
