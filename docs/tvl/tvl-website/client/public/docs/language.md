# Language Reference (Non-normative)

This reference mirrors the structure of the normative spec (see `spec/archive/TVL-0.1.md` for archived version; current spec in `spec/grammar/tvl.schema.json` and examples) and explains the language with examples.

- Core objects and notation
- Syntax: AST and concrete YAML
- Static semantics: typing and constraints
- Denotational semantics: objectives, bands, chance constraints
- Exploration sub-language
- CI gate semantics
- Configuration objects and measurement bundles
- Validation semantics (structural, operational, chance)

Refer to the spec for MUST/SHOULD/MAY requirements.

## Validation Tooling (Phase 1A)

The `tvl-lint` CLI now enforces the Phase 1A typed-surface checks described in the validation roadmap:

- TVAR declarations are type-checked and specialised; empty domains (including unresolved registry-backed enums) raise `empty_domain`.
- Structural constraints are parsed into DNF literals (no parentheses or negation yet) and typed against Γ. Mis-typed atoms emit `constraint_type_mismatch`, `constraint_value_out_of_domain`, or `constraint_operator_type_mismatch`. Floating-point equality produces a `float_equality` warning.
- Derived linear constraints are guarded against TVAR references (`derived_references_tvar`) and non-linear tokens (`*`, `/`, `^`).
- Legacy per-objective `epsilon` keys emit a warning directing authors to `promotion_policy.min_effect`.

Run the lints with:

```bash
tvl-lint path/to/module.yml
```

Tests covering these behaviours live in `tvl/tests/test_lints.py` and can be executed via:

```bash
python -m unittest tvl.tests.test_lints
```

## Validation Tooling (Phase 1B)

Structural constraints now accept the full surface grammar (parentheses, `not`, and `=>` implications). The linter canonicalises every expression to typed DNF, which enables:

- Stable literal ordering for clause hashing and diagnostics.
- Support for implication sugar (`when`/`then` is compiled to `¬when ∨ then`).
- Accurate literal paths even inside nested parentheses (`constraints.structural[i].expr.literal[j]`).
- Every literal-level issue now carries a deterministic `clause_id` (`{index}#${hash}`) derived from the canonical clause string, making unsat-core reporting and IDE decorations stable across formatting changes.

Example: `not (agent = 'pro') or (max_calls >= 1 and max_calls <= 3)` parses successfully, with negated literals type-checked against the same domains as their positive counterparts.

## Validation Tooling (Phase 2)

`tvl-check-structural` validates that structural constraints admit at least one assignment. The checker reports either a satisfying configuration or an UNSAT core (list of structural entries) when constraints conflict. The implementation falls back to a bounded search when OR-Tools is unavailable.

```bash
tvl-check-structural path/to/module.yml --json
```

## Validation Tooling (Phase 3)

`tvl-check-operational` inspects derived constraints and budgets (`max_trials`, `max_spend_usd`, `max_wallclock_s`). The initial release keeps the checks lightweight and honours `tvl.validation.skip_budget_checks` / `skip_cost_estimation` options so teams can suppress budget enforcement when costs cannot be estimated yet.

- `tvl.validation.skip_budget_checks`: bypass hard budget validation (useful during early prototyping).
- `tvl.validation.skip_cost_estimation`: flag future integrations that require cost estimates; currently informational.
