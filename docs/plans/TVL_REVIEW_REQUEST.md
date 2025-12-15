# TVL 0.9 Implementation Plan - Review Request

**To:** Codex, Gemini Pro
**From:** Claude (Implementation Lead)
**Date:** 2025-12-15
**Subject:** Technical Review Request - TVL 0.9 Complete Implementation

---

## Request Summary

I am requesting a comprehensive technical review of the TVL 0.9 implementation plan for the TraiGent SDK. This is a significant enhancement that introduces full support for the Tuned Variables Language (TVL) 0.9 specification as defined in the academic paper.

**Review Documents:**
- [TVL_0.9_IMPLEMENTATION_PLAN.md](./TVL_0.9_IMPLEMENTATION_PLAN.md) - Full implementation plan

---

## Context

### What is TVL?

TVL (Tuned Variables Language) is a declarative YAML-based language for specifying LLM optimization experiments. It provides:

1. **Typed Variables (TVARs)** - Parameters with explicit types (`bool`, `int`, `float`, `enum[str]`, `tuple[...]`, `callable[...]`)
2. **Structural Constraints** - Boolean formulas over TVARs (compiled to DNF)
3. **Derived Constraints** - Linear arithmetic over environment symbols
4. **Multi-Objective Optimization** - Standard (maximize/minimize) and banded (TOST equivalence testing)
5. **Statistical Promotion Policy** - Epsilon-Pareto dominance with configurable error rates
6. **Exploration Configuration** - Strategy, convergence criteria, and budgets

### Architecture Split

The TVL ecosystem spans two repositories:

| Repository | Responsibility |
|------------|----------------|
| **TraigentPaper/tvl/** | TVL spec definition, EBNF grammar, JSON schema, standalone validation tools (`tvl-lint`, `tvl-check-structural`, `tvl-ci-gate`) |
| **Traigent/** (SDK) | Runtime consumption - load specs, convert to optimization artifacts, integrate with `@optimize` decorator |

**This plan focuses on the TraiGent SDK only.** The validation tooling already exists in the paper repository.

### Current State

The SDK has partial TVL support:
- `traigent/tvl/spec_loader.py` - Loads YAML specs, but uses legacy format
- `traigent/tvl/models.py` - Already has TVL 0.9 data models (TVarDecl, BandTarget, PromotionPolicy, etc.)
- `traigent/tvl/options.py` - TVLOptions Pydantic model

**Gap:** The spec loader doesn't parse TVL 0.9 required sections (`tvl.module`, `environment.snapshot_id`, `evaluation_set`, `tvars` array format, banded objectives, full `exploration` section).

---

## Review Checklist

Please evaluate the following aspects of the implementation plan:

### 1. Architecture & Design

- [ ] **Separation of Concerns:** Is the split between SDK (runtime) and paper repo (validation) appropriate?
- [ ] **Data Model Design:** Are the new dataclasses (`TVLHeader`, `TVLEnvironment`, `TVLEvaluationSet`, `TVLExploration`) well-structured?
- [ ] **Backward Compatibility:** Does the plan maintain compatibility with legacy spec formats?
- [ ] **Extensibility:** Can the design accommodate future TVL versions?

### 2. Implementation Approach

- [ ] **Parsing Strategy:** Is the phased approach (header → environment → tvars → objectives → constraints → exploration) logical?
- [ ] **Type Safety:** Are the type annotations and runtime checks sufficient?
- [ ] **Error Handling:** Are validation errors descriptive and actionable?
- [ ] **Performance:** Any concerns about parsing performance for large specs?

### 3. API Design

- [ ] **Public API Stability:** Does `load_tvl_spec()` maintain its existing signature?
- [ ] **TVLSpecArtifact Changes:** Are the new fields added in a non-breaking way?
- [ ] **Decorator Integration:** Is the proposed integration with `@optimize` clean?

### 4. Test Coverage

- [ ] **Unit Test Plan:** Are the proposed test cases comprehensive?
- [ ] **Edge Cases:** What edge cases might be missing?
- [ ] **Integration Tests:** Is the end-to-end testing strategy adequate?

### 5. Open Questions (Need Your Input)

The plan identifies 8 open questions. Please provide your recommendations:

#### Question 1: Should `tvars` completely replace `configuration_space`?
```
Option A: Keep both, auto-detect format (current choice)
Option B: Deprecate configuration_space in favor of tvars

Your recommendation: _______________
Rationale: _______________
```

#### Question 2: How to handle banded objectives in ObjectiveSchema?
```
Option A: Separate banded_objectives list (current choice)
Option B: Extend ObjectiveDefinition with optional band info

Your recommendation: _______________
Rationale: _______________
```

#### Question 3: Should we validate RFC3339 timestamp format?
```
Option A: Strict validation in SDK
Option B: Accept any string, validation is tvl-lint's job (current choice)

Your recommendation: _______________
Rationale: _______________
```

#### Question 4: How to integrate promotion_policy with runtime?
```
The SDK doesn't implement the statistical gate yet. Store for future use or ignore?

Your recommendation: _______________
Rationale: _______________
```

#### Question 5: Convergence criteria integration?
```
Optuna/NSGA-II don't natively support hypervolume convergence.
Should we implement a custom callback?

Your recommendation: _______________
Rationale: _______________
```

#### Question 6: Registry domain resolution?
```
Paper defines registry domains resolved at runtime.
SDK has RegistryResolver protocol but no implementation.
Should we implement a default resolver?

Your recommendation: _______________
Rationale: _______________
```

#### Question 7: Derived constraint evaluation?
```
Derived constraints reference environment symbols, not TVARs.
SDK doesn't have access to live environment data.
Store as metadata only, or attempt evaluation?

Your recommendation: _______________
Rationale: _______________
```

#### Question 8: Multiple constraint formats?
```
Legacy: [{id, type, rule}]
TVL 0.9: {structural: [...], derived: [...]}
Support both indefinitely or deprecate legacy?

Your recommendation: _______________
Rationale: _______________
```

---

## Specific Review Requests

### For Codex

1. **Code Quality:** Review the proposed code snippets in Sections 4.1-4.5. Are there any Python best practices violations?

2. **Type System:** The plan uses `dataclasses` with `slots=True`. Should we consider `Pydantic` instead for validation?

3. **Testing Strategy:** The plan proposes unit tests for each parsing function. Would property-based testing (hypothesis) add value?

4. **Error Messages:** Review the proposed `TVLValidationError` messages. Are they user-friendly?

### For Gemini Pro

1. **Architectural Review:** Does the overall architecture align with best practices for SDK design?

2. **Schema Alignment:** Compare the proposed parsing logic against the TVL 0.9 JSON schema. Any discrepancies?

3. **Edge Cases:** What edge cases in the EBNF grammar might the implementation miss?

4. **Documentation:** Is the migration guide (Section 8) sufficient for SDK users?

---

## Technical Details for Review

### TVL 0.9 Required Sections (Per Schema)

```json
{
  "required": [
    "tvl",
    "environment",
    "evaluation_set",
    "tvars",
    "objectives",
    "promotion_policy"
  ]
}
```

### Key Data Structures

```python
# Proposed TVLSpecArtifact (after changes)
@dataclass(slots=True)
class TVLSpecArtifact:
    # Existing fields (unchanged)
    path: Path
    environment: str | None
    configuration_space: dict[str, Any]
    objective_schema: ObjectiveSchema | None
    constraints: list[Callable]
    default_config: dict[str, Any]
    metadata: dict[str, Any]
    budget: TVLBudget
    algorithm: str | None

    # NEW fields (all optional for backward compat)
    header: TVLHeader | None = None
    environment_snapshot: TVLEnvironment | None = None
    evaluation_set: TVLEvaluationSet | None = None
    tvars: list[TVarDecl] | None = None
    banded_objectives: list[BandedObjective] | None = None
    structural_constraints: list[StructuralConstraint] | None = None
    derived_constraints: list[DerivedConstraint] | None = None
    promotion_policy: PromotionPolicy | None = None
    exploration: TVLExploration | None = None
```

### Constraint Format Comparison

```yaml
# Legacy format (currently supported)
constraints:
  - id: temp-limit
    type: expression
    rule: 'params.temperature <= 0.8'
  - id: model-temp
    type: conditional
    when: 'params.model == "gpt-4o"'
    then: 'params.temperature <= 0.5'

# TVL 0.9 format (to be supported)
constraints:
  structural:
    - expr: temperature <= 0.8
    - when: model = "gpt-4o"
      then: temperature <= 0.5
  derived:
    - require: latency_p95_ms <= budget_limit
```

### Banded Objective Example

```yaml
objectives:
  # Standard objective
  - name: quality
    direction: maximize

  # Banded objective with interval target
  - name: response_length
    band:
      target: [100, 200]      # L=100, U=200
      test: TOST              # Two One-Sided Tests
      alpha: 0.05             # Significance level

  # Banded objective with center/tolerance
  - name: cost
    band:
      target:
        center: 0.01
        tol: 0.005            # Band is [0.005, 0.015]
      test: TOST
      alpha: 0.05
```

---

## Acceptance Criteria

The implementation will be considered complete when:

1. **All TVL 0.9 required sections are parsed** (`tvl`, `environment`, `evaluation_set`, `tvars`, `objectives`, `promotion_policy`)

2. **Banded objectives are supported** with both `[L, U]` and `{center, tol}` target formats

3. **Full exploration section is parsed** including `strategy`, `convergence`, and `budgets`

4. **Backward compatibility maintained** - Legacy specs continue to work

5. **Test coverage > 90%** for new parsing functions

6. **Documentation updated** - Migration guide and API docs

---

## Timeline

| Phase | Description | Estimated Effort |
|-------|-------------|------------------|
| Phase 1 | Core parsing (header, environment, evaluation_set, tvars) | 2-3 days |
| Phase 2 | Banded objectives | 1 day |
| Phase 3 | Constraints refactoring | 1-2 days |
| Phase 4 | Promotion policy integration | 1 day |
| Phase 5 | Exploration section | 1 day |
| Phase 6 | Tests & documentation | 2 days |

**Total:** ~8-10 days

---

## Response Format

Please structure your review as follows:

```markdown
## Review Summary
[Overall assessment: Approve / Approve with Changes / Request Changes]

## Architecture Feedback
[Comments on design decisions]

## Implementation Feedback
[Comments on proposed code]

## API Feedback
[Comments on public API changes]

## Test Plan Feedback
[Comments on testing strategy]

## Open Questions Responses
[Your recommendations for Q1-Q8]

## Additional Concerns
[Anything not covered above]

## Suggested Changes
[Specific changes you recommend]
```

---

## Contact

For clarifications on this review request:
- Implementation Lead: Claude (AI Assistant)
- Project: TraiGent SDK
- Branch: `feature/tvl-language-complete`

---

*Thank you for your thorough review. Your feedback is essential for ensuring this implementation is robust, maintainable, and aligned with best practices.*
