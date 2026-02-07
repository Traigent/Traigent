---
name: example-review
description: Methodologically review Traigent SDK examples for code quality, documentation accuracy, and production readiness. Use when reviewing examples in examples/, walkthrough/, or any SDK demonstration code.
argument-hint: [example-path]
allowed-tools: Read, Grep, Glob, Bash
---

# Traigent Example Review

Perform a methodological review of the example at: **$ARGUMENTS**

## Review Process

1. **Read the example code** (all .py files in the directory)
2. **Read the README** (if exists, or check parent README)
3. **Run the example in mock mode** to verify it works
4. **Evaluate against all criteria below**
5. **Produce a structured report**

---

## Verification Checklist

### Code Quality & Readability

- [ ] Flow understandable in <2 minutes
- [ ] Names are descriptive (avoid `cfg`, `f`, `x` unless scoped and obvious)
- [ ] Top-level docstring states what the example demonstrates
- [ ] Magic numbers/letters (e.g., `k` in RAG) are named or explained
- [ ] Setup/boilerplate is grouped and not distracting
- [ ] Follows SOLID, DRY, KISS, SRP, YAGNI where applicable
- [ ] User can understand how to adapt this example for production use
- [ ] Follows repo patterns (imports, logging, style)
- [ ] No silent exception swallowing (no bare `except: pass`)
- [ ] All functions used in the example are implemented correctly
- [ ] Inline comments explain non-obvious logic
- [ ] Helper functions are clean and understandable

### Traigent SDK Usage

- [ ] Uses `@traigent.optimize()` (no custom wrapper substitutes)
- [ ] Uses official SDK arguments only (no invented helper APIs)
- [ ] `traigent.get_config()` usage matches the injection mode
- [ ] Injection mode is consistent throughout and explained
- [ ] Mock mode uses Traigent's built-in mock support (not manual fakes)
- [ ] Default arguments are sensible (e.g., log verbosity not too noisy)
- [ ] `max_trials` is reasonable for a demo (not too high)
- [ ] Aligned to Traigent capabilities (e.g., use built-in latency check, not custom time wrappers)

### SDK Initialization & Environment

- [ ] Calls `traigent.initialize(execution_mode="...")` before using `@traigent.optimize`
- [ ] Uses `os.environ.setdefault()` for mock/offline flags (preserves user's existing env)
- [ ] Does NOT hardcode env vars with `os.environ["KEY"] = "value"` (would override user settings)
- [ ] Sets `TRAIGENT_OFFLINE_MODE=true` alongside `TRAIGENT_MOCK_LLM=true` for offline examples

### Dataset Quality

- [ ] Dataset path exists and is reachable
- [ ] Dataset has at least 5 examples
- [ ] Dataset examples are realistic (not "test1/test2")
- [ ] Dataset has varied difficulty (not all trivial or all hard)
- [ ] Dataset schema matches what the evaluator expects
- [ ] Dataset format consistent across walkthrough examples (code_fixes branch)

### Output & User Experience

- [ ] Prints clear start message before optimization
- [ ] Prints clear finish message after optimization
- [ ] Shows progress/trials during run (or explains why hidden)
- [ ] Errors show user-friendly messages (no raw stack traces)
- [ ] Success is explicitly indicated
- [ ] Prints `configuration_space` before running
- [ ] Prints `objectives` before running
- [ ] Prints total number of configuration combinations before running
- [ ] Pre-run info is formatted neatly
- [ ] Prints summary table at the end with results
- [ ] Output is concise and readable
- [ ] Formatting matches walkthrough examples on code_fixes branch
- [ ] Shows optimization results in a table
- [ ] Highlights best config in terms of accuracy, latency, cost, and overall

### README Documentation

- [ ] README exists (or parent README clearly covers this example)
- [ ] Every feature claimed in README is actually present in code
- [ ] No placeholder text or "coming soon" claims
- [ ] Code snippets in README match actual code
- [ ] Output examples in README match real output (or labeled "sample")
- [ ] Any claim about behavior (e.g., "random output in mock") is accurate
- [ ] README states purpose of the example
- [ ] README lists prerequisites
- [ ] README provides run steps
- [ ] README describes expected output
- [ ] README documents required env vars (API keys, mock mode)
- [ ] README links to SDK docs where relevant
- [ ] README explains why results may vary between runs (if applicable)
- [ ] README uses user-facing language (no internal jargon)
- [ ] Helper functions/files are documented in README where they already exist
- [ ] No redundancy in README (no duplicate explanations or unnecessary content)

### Mock Mode Behavior

- [ ] Works with `TRAIGENT_MOCK_LLM=true`
- [ ] Mock runs produce deterministic results (same inputs → same outputs)
- [ ] Accuracy is above 75% in mock mode (unless explained in README)
- [ ] Uses `random_seed` for reproducibility where applicable

### Mock Realism (Varied Results Across Configurations)

- [ ] Different model configs produce different accuracy scores (e.g., GPT-4 > GPT-3.5)
- [ ] Mock accuracy reflects realistic model quality ordering (expensive models score higher)
- [ ] If temperature is a config parameter, mock scoring accounts for it (lower temp → slightly higher accuracy for factual tasks)
- [ ] Mock results are not constant across all trials (optimization has meaningful differences to compare)

### Real Mode Behavior

- [ ] Works with real API keys (if example supports real mode)
- [ ] Shows estimated runtime before start for real-mode examples >10s
- [ ] Logs actual runtime after completion (real mode)
- [ ] Cost estimate or warning shown for real API runs
- [ ] Warnings displayed for potentially expensive runs

### Error Handling & Safety

- [ ] No hard-coded absolute paths
- [ ] Dependencies are documented or included in extras
- [ ] Exit code is 0 on success
- [ ] Missing API keys produce a clear error message
- [ ] Partial failures are handled gracefully
- [ ] No sensitive data leaked in logs (CRITICAL)
- [ ] Defaults are safe for beginners (won't burn API credits)

### Performance & Concurrency

- [ ] No unnecessary code that would hurt runtime (extra loops, excessive trials, heavy logging, extra provider calls)
- [ ] No infinite loops or unbounded retries
- [ ] Blocking calls have timeouts or clear exit conditions
- [ ] Concurrency safety: no shared mutable global state across trials unless isolated
- [ ] Parallel/mock runs are deterministic and thread/async-safe (e.g., contextvars for per-trial state)

### Parallel Configuration (If Example Uses Parallelism)

- [ ] Uses SDK's `ParallelConfig` class (not ad-hoc dict or manual concurrency logic)
- [ ] Prints resolved concurrency settings before optimization (transparency for debugging)
- [ ] Supports environment overrides (e.g., `TRAIGENT_PARALLEL_WORKERS`) for user flexibility
- [ ] High concurrency warning handled by SDK or explicitly printed (trial × example > 8 may cause throttling)

### Terminology & Clarity

- [ ] Uses "LLM API calls" explicitly when referring to provider calls
- [ ] Clear distinction between Traigent SDK calls vs LLM provider calls
- [ ] Mock vs real mode behavior is clearly explained
- [ ] Switching between mock and real mode is straightforward
- [ ] Results storage paths are explained only if results are saved
- [ ] Privacy mode behaves as documented (only if example is about privacy)

### Multi-Provider Examples (If Testing Multiple LLM Providers)

Purpose: When an example tests models from different providers (OpenAI, Anthropic, Google), ensure clear communication about which providers are used and what credentials are needed.

- [ ] Models are grouped by provider (e.g., "OpenAI: gpt-4o, gpt-4o-mini / Anthropic: claude-sonnet-4")
- [ ] Output shows provider breakdown before optimization (user knows what will be called)
- [ ] Each provider's required API key is documented (OPENAI_API_KEY, ANTHROPIC_API_KEY, etc.)
- [ ] Missing API keys produce provider-specific error messages (not generic failures)

### CI / Release Readiness

- [ ] `make format && make lint` passes
- [ ] Running time not obviously regressed versus prior example baseline

---

## Review Output Format

### Summary

Brief 2-3 sentence overview of the example and its review status.

### Status: PASS | FAIL | NEEDS WORK

### Checklist Results

Group results by section. For each section:

- Total items checked
- Items passing
- Items failing (with specific details)

### Critical Issues

List any blockers that must be fixed before the example is ready.

### Improvements

List non-blocking suggestions for better quality.

### Verification Run

```bash
Command: TRAIGENT_MOCK_LLM=true TRAIGENT_OFFLINE_MODE=true python <example.py>
Exit code:
Output summary:
```

### Questions for Maintainer

List anything ambiguous that needs clarification before completing the review.

---

## Notes

- Run the example in mock mode before completing the review
- Check both the code AND the README for consistency
- If a criterion doesn't apply to this example, mark it N/A with explanation
- For README redundancy issues, ask before suggesting deletions
