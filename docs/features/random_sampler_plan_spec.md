# RandomSampler Deterministic Plan Specification

## Module Scope

The plan-aware sampling primitives live under `traigent/core/samplers/`:

- `base.py` – shared sampler interface.
- `random_sampler.py` – plan-capable `RandomSampler` implementation.
- `__init__.py` – re-exports `RandomSampler`, `RandomSamplerPlan`, and `SamplerFactory`.

This document describes the externally visible behaviours of those modules so reviewers can confirm the implementation matches the contract.

---

## `traigent.core.samplers.base`

### `class BaseSampler`

Abstract base for samplers that yield individual samples until exhaustion. Implementations are responsible for thread-safety and respecting the contract below.

- **`supports_plans` (property)**  
  - **Purpose:** Indicates whether the concrete sampler can expose deterministic sampling plans.  
  - **Return value:** `bool`. Default implementation returns `False`.  
  - **Post-condition:** No internal state changes.

- **`create_plan(**kwargs) -> Any`**  
  - **Purpose:** Materialise a deterministic plan describing future samples.  
  - **Default behaviour:** Raises `NotImplementedError` for samplers that do not override it.  
  - **Pre-condition:** Caller must only invoke when `supports_plans` is `True`.  
  - **Post-condition:** Sampler state may be inspected but must not change.

- **`apply_plan(plan: Any, **kwargs) -> None`**  
  - **Purpose:** Configure the sampler to replay a precomputed plan.  
  - **Default behaviour:** Raises `NotImplementedError`.  
  - **Pre-condition:** Caller must only invoke when `supports_plans` is `True`.  
  - **Post-condition:** Concrete implementation decides how internal state evolves.

- **`sample(**kwargs) -> Any | None`** *(abstract)*  
  - **Purpose:** Produce the next sample or `None` when exhausted.  
  - **Pre-condition:** None.  
  - **Post-condition:** If `None` is returned the sampler must report `exhausted == True`.

- **`clone() -> BaseSampler`** *(abstract)*  
  - **Purpose:** Create an independent copy with the same configuration but fresh state.  
  - **Post-condition:** Returned sampler must be safe to use in parallel with the original.

---

## `traigent.core.samplers.random_sampler`

### `@dataclass(frozen=True) RandomSamplerPlan`

Serializable description of a deterministic sampling schedule for `RandomSampler`.

- **Fields**
  - `indices: tuple[int, ...]` – positions in the population pulled in order.
  - `replace: bool` – whether sampling assumes replacement.
  - `population_size: int` – expected population length used for validation.
  - `sample_limit: int | None` – maximum number of draws encoded in the plan.
  - `fingerprint: str | None` – SHA-256 digest of the population to detect drift.
- **`__len__`** – returns number of indices.
- **`to_dict() -> dict[str, Any]`**  
  - **Purpose:** Produce a JSON-safe mapping for persistence.  
  - **Post-condition:** New dictionary contains only built-in types.
- **`from_dict(data: Mapping[str, Any]) -> RandomSamplerPlan`** *(class method)*  
  - **Purpose:** Rehydrate a plan from the serialized form.  
  - **Pre-conditions:**  
    - `data["indices"]` must be iterable of ints.  
    - Optional keys (`replace`, `population_size`, `sample_limit`, `fingerprint`) must be type-compatible.  
  - **Failure:** `TypeError` for malformed data; `ValueError` for invalid ints.  
  - **Post-condition:** Returns the reconstructed plan without modifying `data`.

### `class RandomSampler(BaseSampler)`

Thread-safe random sampler supporting optional deterministic plans.

- **Constructor**
  ```python
  RandomSampler(
      population: Sequence[T],
      *,
      sample_limit: int | None = None,
      replace: bool = False,
      seed: int | None = None,
      plan: RandomSamplerPlan | Mapping[str, Any] | None = None,
      resume_random_after_plan: bool = False,
  )
  ```
  - **Pre-conditions:**  
    - `population` is non-empty.  
    - `sample_limit` is `None` or positive.  
    - `plan`, when provided, must be compatible (see `apply_plan`).  
  - **Post-conditions:**  
    - Internal RNG initialised with `seed`.  
    - If `plan` is supplied, it is applied with `strict=False` and without resetting.
  - **Failure:** `ValueError` for invalid population/limit; see `apply_plan` for plan validation errors.

- **`supports_plans` (property)**  
  - **Purpose:** Advertise plan capability.  
  - **Return value:** Always `True`.  
  - **Post-condition:** No state change.

- **`create_plan(*, draws: int | None = None, from_start: bool = True, include_fingerprint: bool = True) -> RandomSamplerPlan`**  
  - **Purpose:** Materialise a deterministic draw order.  
  - **Pre-conditions:**  
    - `draws` is `None` or a non-negative integer.  
    - If the sampler has no remaining-capacity bound (e.g., infinite `sample_limit` with replacement), `draws` must be supplied.  
  - **Post-conditions:**  
    - Returns a `RandomSamplerPlan`.  
    - Sampler state is untouched (read-only operation).  
    - When `include_fingerprint` is `True`, `.fingerprint` contains the SHA-256 population digest.
  - **Failure:** `ValueError` if requested plan exceeds available capacity or inputs are invalid.

- **`apply_plan(plan, *, strict: bool = True, resume_random_after_plan: bool | None = None, reset_state: bool = True) -> None`**  
  - **Purpose:** Configure the sampler to replay the supplied plan.  
  - **Accepts:** `RandomSamplerPlan` or mapping convertible via `RandomSamplerPlan.from_dict`.  
  - **Validation (`strict=True`):**  
    - `plan.replace` matches sampler `replace`.  
    - `plan.population_size` matches current population length.  
    - `plan.sample_limit` (if set) does not exceed sampler `sample_limit`.  
    - `plan.fingerprint` (if set) matches population digest.  
    - For `replace=False`, indices contain no duplicates.  
  - **Pre-conditions:**  
    - All plan indices are within `[0, population_size)`.  
  - **Post-conditions:**  
    - Sampler exhaustion flag cleared.  
    - Future `sample()` calls emit plan entries before falling back to RNG (when `resume_random_after_plan=True`).  
    - `reset_state` restores counters and pools before binding the plan; when `False` the current progress is preserved.
  - **Failure:** `ValueError` for validation or range issues.

- **`sample(**kwargs) -> T | None`**  
  - **Purpose:** Produce the next element.  
  - **Behaviour:**  
    - If a plan is active, consumes the next plan entry.  
    - Otherwise draws randomly with or without replacement using the internal RNG/pool.  
  - **Post-conditions:**  
    - `_samples_drawn` increments on successful draw.  
    - `exhausted` becomes `True` when plan and population/limit are depleted.  
    - When `resume_random_after_plan=True`, sampler seamlessly transitions to RNG after the plan ends.

- **`clone() -> RandomSampler`**  
  - **Purpose:** Create an independent sampler with identical configuration (sans plan).  
  - **Post-condition:** Returned sampler shares the population but has fresh RNG state with `seed=None`.  
  - **Note:** Callers needing the same fixed plan should explicitly reapply the plan to the clone.

---

## Behavioural Guarantees

1. **Thread Safety:** All mutating operations (`sample`, `reset`, `apply_plan`) run under the sampler’s `RLock`; concurrent consumers see a consistent plan order without duplication.
2. **Deterministic Plans:** `RandomSamplerPlan` captures the exact indices returned by the sampler. Applying the same plan to multiple samplers yields identical output sequences until the plan is exhausted.
3. **Validation:** When `strict=True`, plans guard against population drift and configuration mismatches. Consumers may pass `strict=False` to bypass fingerprint/limit checks when reusing plans across compatible populations.
4. **Reset Semantics:** `reset()` clears exhaustion and rewinds progress; any bound plan replays from the beginning on the next `sample()` call.
5. **Resume Mode:** With `resume_random_after_plan=True`, the sampler honours the plan then continues random draws until exhausted (limit reached or population drained).

---

## Related Tests

All behaviours above are exercised in `tests/unit/core/test_samplers.py`, including:

- Sequential plan replay and reset (`test_random_sampler_plan_roundtrip`).
- Plan-driven sampling followed by random continuation (`test_random_sampler_plan_resume_random`).
- Serialization round-trip and fingerprint validation (`test_random_sampler_plan_serialization_and_validation`).
- Threaded consumption of a shared plan (`test_random_sampler_plan_thread_safety`).
- Async `asyncio.to_thread` consumption (`test_random_sampler_plan_async_to_thread`).
- Cross-instance plan reuse for fair trials (`test_random_sampler_plan_shared_across_instances`).

