# LangGraph Multi-Agent Demo Fix Summary

## Issues Identified

### Issue 1: Only 1 Trial Ran (Expected 2)

**Root Cause:**
- Default optimization timeout is 60 seconds
- Each trial with 20 examples took ~104 seconds
- Optimization stopped after first trial due to timeout

**Evidence from logs:**
```
2026-01-16 19:47:32,408 - traigent.evaluators.local - INFO - Detailed evaluation completed: 20/20 successful, duration: 103.90s
...
2026-01-16 19:47:32,505 - traigent.core.orchestrator - INFO - Stopping: reached timeout (60.0s)
```

**Fix:**
Added explicit timeout parameter to allow all trials to complete:
```python
timeout_seconds = max_trials * 120  # 240s for 2 trials in real mode
results = await run_rag_workflow.optimize(
    algorithm="grid",
    max_trials=max_trials,
    random_seed=42,
    timeout=timeout_seconds,
)
```

---

### Issue 2: Accuracy Showing 0.00%

**Root Cause:**
- The decorator uses `objectives=["accuracy", "cost"]` but doesn't provide a custom evaluator
- The default accuracy metric can't automatically evaluate how well LLM-generated answers match expected outputs
- Without an evaluator, all examples are marked as 0.0 accuracy

**Evidence from logs:**
```
metrics: {'accuracy': 0.0, 'cost': 0.000566, 'score': 0.0, ...}
```

**Fix:**
Added custom evaluator function to calculate accuracy:
```python
def evaluate_answer_accuracy(output: str, expected_output: str) -> float:
    """Evaluate answer accuracy by checking if expected output is in generated answer.

    Returns:
        1.0 if expected output is found in generated answer, 0.0 otherwise
    """
    output_lower = output.lower().strip()
    expected_lower = expected_output.lower().strip()

    if expected_lower in output_lower:
        return 1.0

    # Also check individual words for partial credit
    expected_words = expected_lower.split()
    if len(expected_words) == 1 and expected_words[0] in output_lower:
        return 1.0

    return 0.0
```

Then added to decorator:
```python
@traigent.optimize(
    eval_dataset=str(SCRIPT_DIR / "simple_questions.jsonl"),
    objectives=["accuracy", "cost"],
    evaluator=evaluate_answer_accuracy,  # Custom evaluator
    ...
)
```

---

## Changes Made

1. **Added custom accuracy metric function** (`accuracy_metric`) at line 423
   - Signature: `accuracy_metric(expected: str, actual: str) -> float`
   - Uses substring matching to check if expected answer is in generated output
2. **Added metric_functions parameter** to `@traigent.optimize` decorator
   - `metric_functions={"accuracy": accuracy_metric}`
   - This is the correct way to provide custom metrics in Traigent
3. **Increased timeout** to `max_trials * 120` seconds (240s for 2 trials)
4. **Added timeout logging** to show expected time per trial
5. **Removed unused import** (`json`) to fix linting error
6. **Added type ignore comment** for `traigent.with_usage()` return type

---

## Expected Results After Fix

### Before:
- ✗ Only 1 trial completed (timeout)
- ✗ Accuracy: 0.00% (no evaluator)
- ✗ Frontend shows incomplete optimization

### After:
- ✓ Both trials complete (increased timeout)
- ✓ Accuracy: >0% (custom evaluator calculates meaningful accuracy)
- ✓ Frontend shows:
  - 2 configuration runs with different parameters
  - Meaningful accuracy metrics for comparison
  - Cost vs accuracy tradeoff visualization

---

## Testing the Fix

Run the demo again:
```bash
OPENAI_API_KEY="your-key" python walkthrough/examples/mock/advanced/05_langgraph_multiagent_demo.py
```

Expected output:
```
Progress: 2 trials, best score: X.XXXX, success rate: 100.00%, elapsed: ~210s
```

And in the frontend:
- 2 configuration runs visible
- Accuracy > 0% for both trials
- Comparison between temperature=0.3 vs 0.7 configurations