# Sonar Triage

This document tracks the current Sonar findings so the repo can add
static-analysis enforcement without treating every existing smell as an
immediate release blocker.

## Current Policy

- `fix-now`: real bugs, unsafe behavior, compatibility defects, and low-risk
  correctness fixes
- `schedule-later`: maintainability issues that should be reduced over time but
  do not change runtime behavior today
- `accept/document`: intentional patterns or bounded tradeoffs that are known
  and documented

## Current Findings

### Fix Now

- `src/integrations/registry.ts`
  - `Array.sort()` without an explicit comparator
  - Resolved by using `localeCompare(...)`

### Schedule Later

- Cognitive complexity hotspots:
  - `src/optimization/tvl.ts`
  - `src/optimization/tvl-expression.ts`
  - `src/optimization/native.ts`
  - `src/optimization/native-promotion.ts`
  - `src/optimization/native-bayesian.ts`
  - `src/optimization/native-scoring.ts`
  - `src/cli/config-validation.ts`
- Nested ternaries in optimization / TVL helpers where behavior is correct but
  readability should improve
- Optional-chaining / alias cleanup issues that do not affect behavior

### Accept / Document

- Bounded native-vs-hybrid parity decisions where the implementation is
  intentionally explicit rather than abstracted further
- Experimental runtime seamless path, which remains opt-in, documented as
  trusted-local only, and is not a default execution path

## Exit Criteria

- `fix-now` findings should be eliminated as part of normal hardening work
- `schedule-later` findings should only be closed when they do not create
  product or release churn
- newly introduced Sonar issues should be treated as `fix-now` unless explicitly
  justified and documented
