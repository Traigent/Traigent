Got it—you want a **comprehensive, per-module code-review playbook** (not diff-based) for a Python LLM-agent project. Here’s a clean, repeatable checklist + a fill-in template you can use module by module.

---

# Per-Module Code Review — LLM Agent (Python)

## 0) What to load for each module

* The module file(s) + sibling tests
* `pyproject.toml` / `requirements*.txt`
* Any schemas/prompts the module references
* CI/lint/type configs (ruff, mypy), and logging/tracing setup

---

## 1) Purpose & Boundaries

**Goal:** Verify the module does one cohesive thing and has a clear API.

* [ ] One sentence: what does this module do?
* [ ] Single responsibility (no “god” module; split if doing orchestration + I/O + parsing).
* [ ] Clear boundaries: what it **owns** vs what it **uses** (providers/tools/stores).
* [ ] Public API is small and named intentionally; private helpers prefixed `_`.

**Deliverable (2–4 lines):** Summary + suspected boundary leaks.

---

## 2) Public API & Contracts

**Goal:** The module is predictable, typed, and documented.

* [ ] All public callables have **type hints** (inputs/outputs precise).
* [ ] Docstrings (Google/NumPy) describe purpose, params, return, errors, side effects.
* [ ] Pre/postconditions or invariants stated (even briefly) where non-obvious.
* [ ] Stable exceptions (typed, module-scoped); no `except Exception:` without re-raise.

**Deliverable:** List exported names + contract notes.

---

## 3) Dependencies & Architecture

**Goal:** Good layering; adapters at the edges; no cycles.

* [ ] Imports are acyclic; no upward dependency from core to adapters.
* [ ] Uses Strategy/Adapter for providers and tools (swappable; no vendor lock in code paths).
* [ ] No unexpected global state or side effects on import.
* [ ] Config and I/O abstracted behind interfaces; easy to mock.

**Deliverable:** Dependency concerns (1–3 bullets).

---

## 4) LLM-Specific Hygiene

**Applies to modules that prompt/parse/call tools.**

* [ ] **Prompts** externalized (YAML/Markdown/Jinja) with version tags and placeholders validated.
* [ ] **Schema-bounded outputs** (Pydantic/JSON Schema) with strict parsing and **constrained re-prompt** on parse errors.
* [ ] **Tool use** goes through a registry with input validation, least privilege, and idempotent side effects.
* [ ] Injection safety: no raw concatenation with untrusted input; delimiters/role separation enforced.
* [ ] Reproducibility: record model, params, seeds, tool list; avoid implicit changes.

**Deliverable:** Risks (parse fragility, tool safety, prompt source) + fixes.

---

## 5) Error Handling & Logging

**Goal:** Fail safely with actionable context; never leak secrets/PII.

* [ ] Exceptions are specific; error paths tested; no silent fallbacks that mask failure.
* [ ] **Structured logging** (JSON) with request_id/trace_id; PII/secrets redaction.
* [ ] Log levels appropriate; no hot-loop spam; sampling if necessary.

**Deliverable:** Top 1–3 failure modes + proposed exception classes/messages.

---

## 6) Concurrency & Resilience

**Goal:** Async done right; calls bounded and retriable.

* [ ] Async boundaries correct (`async`/`await` consistent; no mixed blocking I/O).
* [ ] Timeouts, exponential backoff with jitter, and (if needed) circuit-breaker.
* [ ] Bounded concurrency (semaphore/pool); cancellation propagates; cleanup on shutdown.
* [ ] Streaming handled safely (backpressure, partials).

**Deliverable:** Concurrency limits + retry/timeout settings to standardize.

---

## 7) Performance & Cost

**Goal:** Reasonable latency and token/$ budgets.

* [ ] Batching where applicable; connection reuse; N+1 provider calls avoided.
* [ ] Caching keys include model + params; TTLs and invalidation defined.
* [ ] Budget guards (tokens/$/latency) and sensible fail-fast behavior.

**Deliverable:** One measurable improvement (cache/batch/guard to add).

---

## 8) Configuration & Secrets

**Goal:** Typed, validated settings; no secrets in code.

* [ ] Pydantic Settings (or equivalent) with explicit defaults + validation.
* [ ] `.env.sample` lists required vars; secrets never logged; redaction filters exist.
* [ ] Safe defaults for model/provider/tool allow-lists.

**Deliverable:** Missing config validations + default policy changes.

---

## 9) Data, Storage, and SQLAlchemy (if relevant)

**Goal:** Correct sessions and efficient queries.

* [ ] Session scope explicit (`sessionmaker`/`async_sessionmaker`) with context managers.
* [ ] No reflection on hot paths; metadata cached; parameterized queries only.
* [ ] Transactions minimal; lazy loads intentional; indexes for frequent filters.
* [ ] Idempotent writes where retried; migrations documented.

**Deliverable:** Session pattern confirmation + any slow query/index TODOs.

---

## 10) Tests & Evaluation

**Goal:** Behavior locked by tests; prompts have goldens.

* [ ] Unit tests for public API and edge cases; e2e smoke for happy path.
* [ ] **Golden tests** for prompts/schemas; update protocol documented.
* [ ] Offline stubs/record-replay for LLM calls; use `TRAIGENT_MOCK_LLM=true` in CI and local tests.
* [ ] Coverage trend non-decreasing for this module.

**Deliverable:** Missing test list (by function/branch) + fixtures to add.

---

## 11) Maintainability & Style

**Goal:** Readable, idiomatic, and consistent.

* [ ] `ruff`/`black`/`isort`/`mypy` clean or justified ignores.
* [ ] Names are meaningful; modules small; cyclomatic complexity acceptable.
* [ ] Comments explain **why**, not **what**; TODOs are ticket-linked with scope.

**Deliverable:** 2–3 refactors (rename/extract/flatten) that raise clarity.

---

## 12) Observability

**Goal:** Trace what matters.

* [ ] OpenTelemetry spans around provider calls (attrs: model, tokens, latency, cache_hit).
* [ ] Metrics for parse-error rate, retry counts, rate-limit hits, tool failures.

**Deliverable:** Metrics to emit + span attributes to standardize.

---

# Reviewer Output Template (per module)

````md
## Module: <path.to.module>

**Purpose & Boundaries**
- Summary:
- Boundary leaks:

**API & Contracts**
- Public API:
- Contract gaps:

**LLM Hygiene**
- Prompt source/version:
- Schema parsing robustness:
- Tool safety:

**Errors & Logging**
- Top failure modes:
- Redaction/logging notes:

**Concurrency & Resilience**
- Timeouts/retries:
- Concurrency bounds:

**Perf & Cost**
- Cache/batch opportunities:
- Budget guards:

**Config & Secrets**
- Settings validation gaps:

**Data/SQLAlchemy (if any)**
- Session pattern:
- Query/index notes:

**Tests & Eval**
- Missing tests:
- Golden tests status:

**Maintainability & Style**
- Refactors:
- Lint/type status:

**Observability**
- Spans/metrics to add:

### Severity Summary
- Blockers: <n>
- Majors: <n>
- Minors: <n>
- Nits: <n>

### Must-Fix (before merge/release)
1.
2.
3.

### Suggested Patches (optional)
```diff
# example unified diff here
````

```

---

## Quick Scoring Rubric (0–3 each; 3 = excellent)
- Cohesion & SRP
- Contracts & Types
- LLM Hygiene (prompts/schemas/tools)
- Safety (secrets/PII/injection)
- Resilience (timeouts/retries)
- Performance/Cost
- Tests/Eval
- Maintainability
- Observability

> **Gate suggestion:** require average ≥2.5 and no **Blockers**.

---

## Fast Auto-Checks (run before deep review)
- `ruff check && ruff format --check`
- `mypy --strict <module_path>`
- `pytest -q tests/<module_name>` (offline mode)
- Record-replay tests for LLM calls pass
- Search: `grep -R "except Exception" <module>` (ensure no swallow)
- Search: `grep -R "temperature=" <module>` (ensure config-driven, not hard-coded)

---

If you want, tell me a specific module path (e.g., `agents/planner.py` or `adapters/sql_store.py`) and I’ll produce a filled example using this template.
```
