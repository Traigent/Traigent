# Creative Stress Testing Prompt

You are a senior AI engineer exploring Traigent's capabilities. The README quickstarts have been validated - your job is to **push the boundaries** by creating your own experiments.

**Your mission**: Design creative experiments that stress-test Traigent's flexibility, error handling, and edge cases. Everything runs in mock mode.

---

## Setup

```bash
# Ensure you're in the Traigent checkout with the SDK installed
cd /path/to/Traigent
export TRAIGENT_MOCK_LLM=true
export TRAIGENT_OFFLINE_MODE=true
export TRAIGENT_LOG_LEVEL=INFO
```

---

## SCENARIO 1: Domain-Specific Agents

Create agents for specific domains and observe how Traigent handles them.

### 1.1 Code Review Agent

```python
# code_review_experiment.py
import asyncio
import traigent

# Create a code review dataset
DATASET_CONTENT = '''{"input": {"code": "def add(a, b): return a + b"}, "output": "Clean, simple function. Consider adding type hints."}
{"input": {"code": "x=1;y=2;z=x+y;print(z)"}, "output": "Use proper spacing and line breaks for readability."}
{"input": {"code": "def factorial(n):\\n    if n <= 1: return 1\\n    return n * factorial(n-1)"}, "output": "Good recursion. Consider adding base case validation."}
{"input": {"code": "import os; os.system('rm -rf /')"}, "output": "CRITICAL: Dangerous system command. Never execute user input directly."}
{"input": {"code": "async def fetch(url): return await aiohttp.get(url)"}, "output": "Missing error handling for network failures."}'''

# Write dataset
with open("code_review_data.jsonl", "w") as f:
    f.write(DATASET_CONTENT)

@traigent.optimize(
    configuration_space={
        "review_depth": ["quick", "standard", "thorough"],
        "focus_area": ["security", "performance", "style", "all"],
        "strictness": [1, 3, 5, 7, 10],
    },
    objectives=["accuracy", "cost"],
    eval_dataset="code_review_data.jsonl",
)
def code_review_agent(code: str) -> str:
    """Review code based on configuration."""
    config = traigent.get_config()
    depth = config.get("review_depth", "standard")
    focus = config.get("focus_area", "all")
    strictness = config.get("strictness", 5)

    # Simulate different quality based on config
    if "rm -rf" in code or "system(" in code:
        if focus in ["security", "all"] and strictness >= 5:
            return "CRITICAL: Dangerous system command. Never execute user input directly."
        else:
            return "Code contains system calls."

    if depth == "thorough" and strictness >= 7:
        return "Clean, simple function. Consider adding type hints."
    elif depth in ["standard", "thorough"]:
        return "Good code structure."
    else:
        return "Looks fine."

async def main():
    print("🔍 Code Review Agent Optimization")
    print("=" * 50)

    result = await code_review_agent.optimize(
        algorithm="random",
        max_trials=15
    )

    print(f"\n📊 Results:")
    print(f"   Best score: {result.best_score:.3f}")
    print(f"   Best config: {result.best_config}")
    print(f"   Trials: {len(result.trials)}")

    # Test: Does deeper review + security focus + high strictness win?
    best = result.best_config
    if best.get("focus_area") in ["security", "all"] and best.get("strictness", 0) >= 5:
        print("   ✅ Optimization found security-focused config (expected)")
    else:
        print("   ⚠️ Unexpected best config - review mock logic")

if __name__ == "__main__":
    asyncio.run(main())
```

**Questions to answer:**
- [ ] Does Traigent handle string-based categorical values correctly?
- [ ] Can it optimize for security-sensitive outputs?
- [ ] What happens with integer parameters like `strictness`?

### 1.2 Multi-Language Translation Agent

```python
# translation_experiment.py
import asyncio
import traigent

DATASET = '''{"input": {"text": "Hello, how are you?", "target_lang": "es"}, "output": "Hola, ¿cómo estás?"}
{"input": {"text": "Good morning", "target_lang": "fr"}, "output": "Bonjour"}
{"input": {"text": "Thank you very much", "target_lang": "de"}, "output": "Vielen Dank"}
{"input": {"text": "I love programming", "target_lang": "ja"}, "output": "プログラミングが大好きです"}
{"input": {"text": "The weather is nice", "target_lang": "it"}, "output": "Il tempo è bello"}'''

with open("translation_data.jsonl", "w") as f:
    f.write(DATASET)

@traigent.optimize(
    configuration_space={
        "model": ["gpt-3.5-turbo", "gpt-4o-mini", "gpt-4o"],
        "temperature": [0.0, 0.3, 0.7],
        "formality": ["casual", "formal", "auto"],
    },
    objectives=["accuracy"],
    eval_dataset="translation_data.jsonl",
)
def translate(text: str, target_lang: str) -> str:
    """Translate text to target language."""
    config = traigent.get_config()
    temp = config.get("temperature", 0.3)
    model = config.get("model", "gpt-3.5-turbo")

    # Simulate: lower temp + better model = more accurate
    translations = {
        "es": {"Hello, how are you?": "Hola, ¿cómo estás?"},
        "fr": {"Good morning": "Bonjour"},
        "de": {"Thank you very much": "Vielen Dank"},
        "ja": {"I love programming": "プログラミングが大好きです"},
        "it": {"The weather is nice": "Il tempo è bello"},
    }

    quality = (1 - temp) * 0.5 + (0.5 if "gpt-4" in model else 0.2)

    if quality > 0.6 and target_lang in translations:
        return translations.get(target_lang, {}).get(text, f"[{target_lang}] {text}")
    else:
        return f"Translation of '{text}' to {target_lang}"

async def main():
    print("🌍 Translation Agent Optimization")
    result = await translate.optimize(algorithm="grid", max_trials=27)
    print(f"Best: {result.best_config} (score: {result.best_score:.3f})")

if __name__ == "__main__":
    asyncio.run(main())
```

**Questions to answer:**
- [ ] Does Traigent handle functions with MULTIPLE input parameters?
- [ ] Grid search with 3x3x3 = 27 configurations - does it run all?
- [ ] Unicode in expected outputs - handled correctly?

---

## SCENARIO 2: Edge Case Configurations

### 2.1 Extreme Configuration Spaces

```python
# extreme_configs.py
import asyncio
import traigent

# Minimal dataset
with open("minimal.jsonl", "w") as f:
    f.write('{"input": {"q": "test"}, "output": "answer"}\n')

# Test 1: Single value in config space
@traigent.optimize(
    configuration_space={"only_option": ["single"]},
    objectives=["accuracy"],
    eval_dataset="minimal.jsonl",
)
def single_config(q: str) -> str:
    return "answer"

# Test 2: Many parameters
@traigent.optimize(
    configuration_space={
        "p1": ["a", "b"],
        "p2": ["x", "y"],
        "p3": [1, 2],
        "p4": [0.1, 0.9],
        "p5": ["low", "high"],
        "p6": [True, False],
    },
    objectives=["accuracy"],
    eval_dataset="minimal.jsonl",
)
def many_params(q: str) -> str:
    config = traigent.get_config()
    # 2^6 = 64 possible configs
    return "answer"

# Test 3: Boolean parameters
@traigent.optimize(
    configuration_space={
        "use_caching": [True, False],
        "verbose": [True, False],
        "strict_mode": [True, False],
    },
    objectives=["accuracy"],
    eval_dataset="minimal.jsonl",
)
def boolean_params(q: str) -> str:
    config = traigent.get_config()
    return "answer"

# Test 4: Numeric-only space
@traigent.optimize(
    configuration_space={
        "threshold": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        "count": [1, 2, 3, 4, 5],
    },
    objectives=["accuracy"],
    eval_dataset="minimal.jsonl",
)
def numeric_only(q: str) -> str:
    config = traigent.get_config()
    threshold = config.get("threshold", 0.5)
    return "answer" if threshold > 0.4 else "wrong"

async def main():
    print("🧪 Extreme Configuration Tests")
    print("=" * 50)

    tests = [
        ("Single config", single_config, 1),
        ("Many params (64 combos)", many_params, 10),
        ("Boolean params", boolean_params, 8),
        ("Numeric only", numeric_only, 15),
    ]

    for name, func, trials in tests:
        print(f"\n▶ Testing: {name}")
        try:
            result = await func.optimize(algorithm="random", max_trials=trials)
            print(f"  ✅ Success: {len(result.trials)} trials, best={result.best_score:.3f}")
            print(f"  Config: {result.best_config}")
        except Exception as e:
            print(f"  ❌ Failed: {type(e).__name__}: {e}")

if __name__ == "__main__":
    asyncio.run(main())
```

**Questions to answer:**
- [ ] Single-value config space - does it still run?
- [ ] 64 possible configs with max_trials=10 - does random sampling work?
- [ ] Boolean values in config - handled correctly?
- [ ] Does best_config reflect the actual best configuration?

### 2.2 Dataset Edge Cases

```python
# dataset_edges.py
import asyncio
import traigent
import json

# Test various dataset formats
datasets = {
    "single_example": [
        {"input": {"q": "only one"}, "output": "single answer"}
    ],
    "no_output_field": [
        {"input": {"q": "what happens?"}},
        {"input": {"q": "no expected output"}},
    ],
    "long_content": [
        {"input": {"q": "x" * 1000}, "output": "y" * 1000},
        {"input": {"q": "z" * 500}, "output": "w" * 500},
    ],
    "special_chars": [
        {"input": {"q": "Test with 'quotes' and \"double quotes\""}, "output": "Handled"},
        {"input": {"q": "Newlines:\nand\ttabs"}, "output": "Also handled"},
        {"input": {"q": "Unicode: 你好 🎉 émoji"}, "output": "Supported: ✓"},
    ],
    "nested_input": [
        {"input": {"context": {"user": "Alice", "role": "admin"}, "query": "test"}, "output": "response"},
    ],
}

for name, data in datasets.items():
    with open(f"test_{name}.jsonl", "w") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

@traigent.optimize(
    configuration_space={"mode": ["a", "b"]},
    objectives=["accuracy"],
    eval_dataset="test_single_example.jsonl",
)
def test_single(q: str) -> str:
    return "single answer"

@traigent.optimize(
    configuration_space={"mode": ["a", "b"]},
    objectives=["accuracy"],
    eval_dataset="test_no_output_field.jsonl",
)
def test_no_output(q: str) -> str:
    return "something"

@traigent.optimize(
    configuration_space={"mode": ["a", "b"]},
    objectives=["accuracy"],
    eval_dataset="test_special_chars.jsonl",
)
def test_special(q: str) -> str:
    if "quotes" in q:
        return "Handled"
    elif "Unicode" in q:
        return "Supported: ✓"
    return "Also handled"

async def main():
    print("📁 Dataset Edge Case Tests")
    print("=" * 50)

    tests = [
        ("Single example dataset", test_single),
        ("No output field (accuracy=?)", test_no_output),
        ("Special characters", test_special),
    ]

    for name, func in tests:
        print(f"\n▶ {name}")
        try:
            result = await func.optimize(algorithm="random", max_trials=3)
            print(f"  ✅ Completed: {len(result.trials)} trials")
            print(f"  Best score: {result.best_score:.3f}")

            # Check for any failed trials
            failed = [t for t in result.trials if t.status != "completed"]
            if failed:
                print(f"  ⚠️ {len(failed)} non-completed trials")
        except Exception as e:
            print(f"  ❌ Error: {type(e).__name__}: {e}")

if __name__ == "__main__":
    asyncio.run(main())
```

---

## SCENARIO 3: Algorithm Comparison

### 3.1 Same Problem, Different Algorithms

```python
# algorithm_comparison.py
import asyncio
import traigent
import time

# Dataset where one config is clearly better
DATASET = '''{"input": {"x": "1"}, "output": "optimal"}
{"input": {"x": "2"}, "output": "optimal"}
{"input": {"x": "3"}, "output": "optimal"}
{"input": {"x": "4"}, "output": "optimal"}
{"input": {"x": "5"}, "output": "optimal"}'''

with open("algo_test.jsonl", "w") as f:
    f.write(DATASET)

def create_optimizable_func(name: str):
    """Factory to create fresh decorated functions."""

    @traigent.optimize(
        configuration_space={
            "quality": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "mode": ["fast", "balanced", "accurate"],
        },
        objectives=["accuracy"],
        eval_dataset="algo_test.jsonl",
    )
    def agent(x: str) -> str:
        config = traigent.get_config()
        quality = config.get("quality", 5)
        mode = config.get("mode", "balanced")

        # Only quality=10 + mode=accurate gives optimal
        if quality == 10 and mode == "accurate":
            return "optimal"
        elif quality >= 7 and mode in ["balanced", "accurate"]:
            return "good"
        else:
            return "suboptimal"

    agent.__name__ = name
    return agent

async def main():
    print("🔬 Algorithm Comparison")
    print("=" * 50)
    print("Target: quality=10, mode=accurate")
    print()

    algorithms = ["random", "grid"]
    # Note: Add "bayesian" if optuna is installed

    for algo in algorithms:
        func = create_optimizable_func(f"agent_{algo}")

        print(f"▶ Algorithm: {algo}")
        start = time.time()

        try:
            result = await func.optimize(
                algorithm=algo,
                max_trials=15
            )
            elapsed = time.time() - start

            print(f"  Time: {elapsed:.2f}s")
            print(f"  Trials: {len(result.trials)}")
            print(f"  Best score: {result.best_score:.3f}")
            print(f"  Best config: {result.best_config}")

            # Did it find the optimal?
            best = result.best_config
            if best.get("quality") == 10 and best.get("mode") == "accurate":
                print("  ✅ Found optimal configuration!")
            else:
                print("  ⚠️ Did not find optimal (may need more trials)")

        except Exception as e:
            print(f"  ❌ Error: {type(e).__name__}: {e}")

        print()

if __name__ == "__main__":
    asyncio.run(main())
```

---

## SCENARIO 4: Lifecycle & State Management

### 4.1 Full Lifecycle Test

```python
# lifecycle_test.py
import asyncio
import traigent

with open("lifecycle.jsonl", "w") as f:
    f.write('{"input": {"q": "test1"}, "output": "answer1"}\n')
    f.write('{"input": {"q": "test2"}, "output": "answer2"}\n')
    f.write('{"input": {"q": "test3"}, "output": "answer3"}\n')

@traigent.optimize(
    configuration_space={
        "version": ["v1", "v2", "v3"],
        "speed": [1, 2, 3],
    },
    objectives=["accuracy"],
    eval_dataset="lifecycle.jsonl",
)
def lifecycle_agent(q: str) -> str:
    config = traigent.get_config()
    version = config.get("version", "v1")
    speed = config.get("speed", 1)

    # v3 + speed=3 is best
    if version == "v3" and speed == 3:
        return f"answer{q[-1]}"  # Matches expected
    return "generic answer"

async def main():
    print("🔄 Lifecycle Test")
    print("=" * 50)

    # Phase 1: Optimize
    print("\n1️⃣ Running optimization...")
    result = await lifecycle_agent.optimize(
        algorithm="random",
        max_trials=10
    )

    print(f"   Trials completed: {len(result.trials)}")
    print(f"   Best score: {result.best_score:.3f}")
    print(f"   Best config: {result.best_config}")

    # Phase 2: Examine result object
    print("\n2️⃣ Examining result object...")
    print(f"   result.optimization_id: {result.optimization_id}")
    print(f"   result.duration: {result.duration:.2f}s")
    print(f"   result.algorithm: {result.algorithm}")
    print(f"   result.objectives: {result.objectives}")

    # Check trial details
    if result.trials:
        trial = result.trials[0]
        print(f"\n   First trial details:")
        print(f"     trial_id: {trial.trial_id}")
        print(f"     status: {trial.status}")
        print(f"     config: {trial.config}")
        print(f"     metrics: {trial.metrics}")
        print(f"     duration: {trial.duration:.3f}s")

    # Phase 3: Apply best config
    print("\n3️⃣ Applying best config...")
    try:
        lifecycle_agent.apply_best_config()
        print("   ✅ apply_best_config() succeeded")
    except Exception as e:
        print(f"   ❌ apply_best_config() failed: {e}")

    # Phase 4: Check current_config
    print("\n4️⃣ Checking current_config...")
    try:
        current = lifecycle_agent.current_config
        print(f"   current_config: {current}")
    except Exception as e:
        print(f"   ❌ current_config access failed: {e}")

    # Phase 5: Call function normally (post-optimization)
    print("\n5️⃣ Calling function after optimization...")
    try:
        # This should use the best config automatically
        output = lifecycle_agent("test_query")
        print(f"   Output: {output}")
        print("   ✅ Normal call succeeded")
    except Exception as e:
        print(f"   ❌ Normal call failed: {e}")

    # Phase 6: Reset
    print("\n6️⃣ Testing reset...")
    try:
        lifecycle_agent.reset()
        print("   ✅ reset() succeeded")
        print(f"   Config after reset: {lifecycle_agent.current_config}")
    except Exception as e:
        print(f"   ⚠️ reset() error: {e}")

if __name__ == "__main__":
    asyncio.run(main())
```

### 4.2 State Isolation Test

```python
# state_isolation.py
import asyncio
import traigent

with open("iso.jsonl", "w") as f:
    f.write('{"input": {"q": "test"}, "output": "answer"}\n')

@traigent.optimize(
    configuration_space={"opt": ["A", "B", "C"]},
    objectives=["accuracy"],
    eval_dataset="iso.jsonl",
)
def agent_one(q: str) -> str:
    return "answer"

@traigent.optimize(
    configuration_space={"opt": ["X", "Y", "Z"]},
    objectives=["accuracy"],
    eval_dataset="iso.jsonl",
)
def agent_two(q: str) -> str:
    return "answer"

async def main():
    print("🔒 State Isolation Test")
    print("=" * 50)

    # Run both optimizations
    print("\nOptimizing agent_one...")
    r1 = await agent_one.optimize(algorithm="random", max_trials=3)
    print(f"  Best: {r1.best_config}")

    print("\nOptimizing agent_two...")
    r2 = await agent_two.optimize(algorithm="random", max_trials=3)
    print(f"  Best: {r2.best_config}")

    # Apply best configs
    agent_one.apply_best_config()
    agent_two.apply_best_config()

    # Check isolation
    print("\nChecking state isolation...")
    c1 = agent_one.current_config
    c2 = agent_two.current_config

    print(f"  agent_one.current_config: {c1}")
    print(f"  agent_two.current_config: {c2}")

    # Verify they have different config keys
    if "opt" in c1 and "opt" in c2:
        if c1["opt"] in ["A", "B", "C"] and c2["opt"] in ["X", "Y", "Z"]:
            print("  ✅ States are properly isolated")
        else:
            print("  ❌ State leakage detected!")
    else:
        print("  ⚠️ Config structure unexpected")

if __name__ == "__main__":
    asyncio.run(main())
```

---

## SCENARIO 5: Error Recovery & Robustness

### 5.1 Intentional Failures

```python
# error_handling.py
import asyncio
import traigent

with open("errors.jsonl", "w") as f:
    f.write('{"input": {"q": "normal"}, "output": "ok"}\n')
    f.write('{"input": {"q": "trigger_error"}, "output": "ok"}\n')
    f.write('{"input": {"q": "another"}, "output": "ok"}\n')

@traigent.optimize(
    configuration_space={"mode": ["safe", "risky"]},
    objectives=["accuracy"],
    eval_dataset="errors.jsonl",
)
def sometimes_fails(q: str) -> str:
    config = traigent.get_config()
    mode = config.get("mode", "safe")

    if mode == "risky" and "trigger_error" in q:
        raise ValueError("Simulated failure in risky mode!")

    return "ok"

@traigent.optimize(
    configuration_space={"delay": [0.1, 0.5]},
    objectives=["accuracy"],
    eval_dataset="errors.jsonl",
)
def slow_agent(q: str) -> str:
    import time
    config = traigent.get_config()
    delay = config.get("delay", 0.1)
    time.sleep(delay)
    return "ok"

async def main():
    print("💥 Error Handling Tests")
    print("=" * 50)

    # Test 1: Function that sometimes raises exceptions
    print("\n▶ Test: Function with intermittent errors")
    try:
        result = await sometimes_fails.optimize(
            algorithm="grid",
            max_trials=4  # Will try both safe and risky
        )
        print(f"  Completed with {len(result.trials)} trials")

        # Check trial statuses
        for t in result.trials:
            status_emoji = "✅" if t.status == "completed" else "❌"
            print(f"    {status_emoji} {t.config} -> {t.status}")
            if hasattr(t, 'error_message') and t.error_message:
                print(f"       Error: {t.error_message}")

    except Exception as e:
        print(f"  ❌ Optimization failed entirely: {e}")

    # Test 2: get_config outside optimization
    print("\n▶ Test: get_config() outside optimization")
    try:
        config = traigent.get_config()
        print(f"  ⚠️ No error raised, got: {config}")
    except Exception as e:
        print(f"  ✅ Correctly raised {type(e).__name__}: {e}")

if __name__ == "__main__":
    asyncio.run(main())
```

### 5.2 Invalid Configurations

```python
# invalid_configs.py
import asyncio
import traigent

with open("valid.jsonl", "w") as f:
    f.write('{"input": {"q": "test"}, "output": "answer"}\n')

async def test_invalid_objective():
    """Test with invalid objective name."""
    print("▶ Invalid objective name")
    try:
        @traigent.optimize(
            configuration_space={"x": [1, 2]},
            objectives=["nonexistent_metric"],
            eval_dataset="valid.jsonl",
        )
        def bad_objective(q: str) -> str:
            return "answer"

        result = await bad_objective.optimize(max_trials=2)
        print(f"  ⚠️ No error, completed with score: {result.best_score}")
    except Exception as e:
        print(f"  Result: {type(e).__name__}: {e}")

async def test_missing_dataset():
    """Test with non-existent dataset."""
    print("\n▶ Non-existent dataset")
    try:
        @traigent.optimize(
            configuration_space={"x": [1, 2]},
            objectives=["accuracy"],
            eval_dataset="this_file_does_not_exist.jsonl",
        )
        def bad_dataset(q: str) -> str:
            return "answer"

        result = await bad_dataset.optimize(max_trials=2)
        print(f"  ⚠️ No error raised!")
    except Exception as e:
        print(f"  ✅ Error: {type(e).__name__}: {e}")

async def test_empty_objectives():
    """Test with empty objectives list."""
    print("\n▶ Empty objectives list")
    try:
        @traigent.optimize(
            configuration_space={"x": [1, 2]},
            objectives=[],
            eval_dataset="valid.jsonl",
        )
        def empty_obj(q: str) -> str:
            return "answer"

        result = await empty_obj.optimize(max_trials=2)
        print(f"  Result: {len(result.trials)} trials, score={result.best_score}")
    except Exception as e:
        print(f"  Error: {type(e).__name__}: {e}")

async def main():
    print("🚫 Invalid Configuration Tests")
    print("=" * 50)

    await test_invalid_objective()
    await test_missing_dataset()
    await test_empty_objectives()

if __name__ == "__main__":
    asyncio.run(main())
```

---

## SCENARIO 6: Persistence & Results

### 6.1 Save and Load Results

```python
# persistence_test.py
import asyncio
import traigent
from traigent.utils.persistence import PersistenceManager
import shutil
import os

with open("persist.jsonl", "w") as f:
    f.write('{"input": {"q": "test"}, "output": "answer"}\n')

@traigent.optimize(
    configuration_space={"tier": ["bronze", "silver", "gold"]},
    objectives=["accuracy"],
    eval_dataset="persist.jsonl",
)
def persist_agent(q: str) -> str:
    config = traigent.get_config()
    return "answer" if config.get("tier") == "gold" else "wrong"

async def main():
    print("💾 Persistence Test")
    print("=" * 50)

    # Clean up any previous test data
    test_dir = ".traigent_test_persist"
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)

    pm = PersistenceManager(test_dir)

    # Run optimization
    print("\n1️⃣ Running optimization...")
    result = await persist_agent.optimize(algorithm="grid", max_trials=3)
    print(f"   Best: {result.best_config}, score={result.best_score:.3f}")

    # Save result
    print("\n2️⃣ Saving result...")
    try:
        path = pm.save_result(result, "my_test_run")
        print(f"   ✅ Saved to: {path}")
    except Exception as e:
        print(f"   ❌ Save failed: {e}")
        return

    # List results
    print("\n3️⃣ Listing results...")
    results_list = pm.list_results()
    print(f"   Found {len(results_list)} result(s)")
    for r in results_list:
        print(f"     - {r.get('name')}: {r.get('best_score', 'N/A')} ({r.get('algorithm')})")

    # Load result
    print("\n4️⃣ Loading result...")
    try:
        loaded = pm.load_result("my_test_run")
        print(f"   ✅ Loaded successfully")
        print(f"   Trials: {len(loaded.trials)}")
        print(f"   Best config: {loaded.best_config}")
        print(f"   Best score: {loaded.best_score}")

        # Verify data integrity
        if loaded.best_config == result.best_config:
            print("   ✅ Config matches original")
        else:
            print("   ❌ Config mismatch!")

    except Exception as e:
        print(f"   ❌ Load failed: {e}")

    # Delete result
    print("\n5️⃣ Deleting result...")
    try:
        deleted = pm.delete_result("my_test_run")
        print(f"   Deleted: {deleted}")

        # Verify deletion
        remaining = pm.list_results()
        if len(remaining) == 0:
            print("   ✅ Result removed successfully")
        else:
            print(f"   ⚠️ Still have {len(remaining)} results")
    except Exception as e:
        print(f"   ❌ Delete failed: {e}")

    # Cleanup
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)

if __name__ == "__main__":
    asyncio.run(main())
```

---

## SCENARIO 7: CLI Stress Tests

Run these commands and document results:

```bash
# Generate and run each template
echo "=== Template Tests ==="

for template in basic multi-objective langchain openai; do
    echo "▶ Testing template: $template"
    traigent generate -t $template -o "test_${template}.py"

    if [ -f "test_${template}.py" ]; then
        echo "  Generated successfully"
        # Try to run it
        TRAIGENT_MOCK_LLM=true timeout 60 python "test_${template}.py" 2>&1 | head -20
        echo ""
    else
        echo "  ❌ Generation failed"
    fi
done

# Results commands
echo "=== Results Management ==="
traigent results
traigent results --last 3

# Validation
echo "=== Validation ==="
echo '{"input": {"q": "test"}, "output": "answer"}' > cli_test.jsonl
traigent validate cli_test.jsonl -o accuracy
traigent validate cli_test.jsonl -o accuracy -o cost

# Invalid dataset
echo "not valid json" > bad.jsonl
traigent validate bad.jsonl -o accuracy
```

---

## Reporting Template

For each scenario, report:

```markdown
## Scenario X: [Name]

### Test: [Specific test name]

**Status**: ✅ Pass | ⚠️ Warning | ❌ Fail

**What happened**:
[Describe actual behavior]

**Expected**:
[What should have happened]

**Output/Error**:
```
[Paste relevant output]
```

**Severity**: 🔴 Blocker | 🟡 Friction | 🟢 Polish

**Recommendation**:
[Suggested fix or documentation improvement]
```

---

## Summary Questions

After running all scenarios, answer:

1. **Configuration flexibility**: Can Traigent handle diverse config types (strings, ints, floats, booleans)?
2. **Multi-parameter functions**: Do functions with multiple inputs work correctly?
3. **Algorithm comparison**: Do different algorithms produce reasonable results?
4. **Error handling**: Are errors reported clearly? Does optimization continue after partial failures?
5. **Lifecycle management**: Do apply_best_config(), reset(), and current_config work as expected?
6. **Persistence**: Can results be saved, loaded, and deleted reliably?
7. **State isolation**: Do multiple decorated functions maintain separate state?
8. **Mock mode completeness**: Does everything work offline without API warnings?

---

## Success Criteria

All scenarios should:
- [ ] Complete without crashes
- [ ] Produce meaningful results (not all zeros or nulls)
- [ ] Report errors clearly when intentionally triggered
- [ ] Maintain state isolation between different agents
- [ ] Work entirely offline in mock mode
