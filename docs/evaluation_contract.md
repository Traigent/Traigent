# Evaluation Compatibility Contract (no-execution)

`traigent.validate_evaluation_contract` answers one question **without running
anything**:

> Would this `(function, dataset, evaluator, injection)` combination actually
> work under a real optimization run?

It reuses the *exact* production decision points — dataset normalization,
injection-mode resolution, per-example call shaping, and metric-callback binding
— but stops precisely at each execution boundary. No target function, metric,
or custom evaluator is ever called; no configuration context is entered; no
backend/network client is imported; no cost is ever incurred.

```python
import traigent

report = traigent.validate_evaluation_contract(
    func=my_agent,                       # raw callable OR @traigent.optimize'd function
    dataset=[{"input": {"q": "hi"}, "expected": "hello"}],
    scoring_function=my_scorer,
)
if not report.ok:
    for finding in report.findings:
        if finding.severity == "error":
            print(finding.code, finding.message)
```

`report` is an `EvaluationContractReport` (a frozen dataclass). Call
`report.to_dict()` for a JSON-serializable view.

## What is guaranteed vs. merely reported

The contract distinguishes **stable guarantees** (safe to build tooling on)
from facts it merely **reports** (may evolve as the SDK evolves).

**Guaranteed stable** (changing these is a *major* version bump):

- The **report schema** — the fields on `EvaluationContractReport` and its
  nested dataclasses (`ContractFinding`, `NormalizedExampleSummary`,
  `InjectionSummary`, `ExampleCallShape`, `EvaluatorBinding`) and their
  `to_dict()` shape.
- The **`ContractCode` vocabulary** — codes are additive-only; an existing code
  never silently changes meaning.
- The **no-execution guarantee** — the API never calls your function, a metric,
  or a custom evaluator, and never enters a configuration context or opens a
  network connection.
- **Keys-only reporting** — configuration is reported as *keys only*
  (`InjectionSummary.config_keys`); values never appear anywhere in the report,
  so a serialized report cannot leak a secret.
- **`report.ok` semantics** — `ok` is `True` iff no finding has
  `severity == "error"`.

**Reported, not individually guaranteed** (may change in a *minor* release):

- The evaluator alias sets (which parameter names count as "output",
  "expected", "llm_metrics", …).
- The dict-expansion rule for mapping inputs.
- The metric-callback candidate order.

These are surfaced through the report so you can *observe* current behavior, but
the SDK may refine them without a major bump. Assert on the report fields, not
on these internal heuristics.

## Contract version

`traigent.EVALUATION_CONTRACT_VERSION` (currently `"1.0.0"`) versions the report
schema and the `ContractCode` vocabulary. Adding a code or field is a **minor**
bump; removing/renaming one or changing an existing code's meaning is a
**major** bump.

## Injection modes

- **context** (default): config is read via `get_config()`; the call shape is
  computed from the raw function signature.
- **parameter**: the config object is injected as a named parameter. The report
  models the *effective* call shape (`ExampleCallShape.effective_*`) as the
  runtime shape plus the injected config parameter, and flags
  `INJECTION_CONFIG_PARAM_MISSING` when the function lacks that parameter.
- **seamless**: AST transformation. The contract runs a *structural probe*
  (`SeamlessParameterProvider._transform_function`) that compiles but never runs
  the body, and reports the variables it would rewrite in
  `InjectionSummary.seamless_injected_names`. Seamless coverage is **best
  effort** — a function without importable source yields a
  `SEAMLESS_INJECTION_UNAVAILABLE` *warning* (never a hard error).

For a raw callable, injection is resolved from the flat `injection_mode` /
`config_param` kwargs or a grouped `injection_options` bundle (conflicts between
the two raise `INJECTION_OPTIONS_CONFLICT`). For an `@traigent.optimize`'d
function, injection settings come from the decorator.

## Custom evaluators

A custom evaluator callback cannot be inspected without executing it, so it is
**never run**. Its presence is reported as `unsupported == ("custom_evaluator",)`
plus a `CUSTOM_EVALUATOR_UNSUPPORTED` **warning** (it does not fail
`report.ok`). Validate custom-evaluator behavior with a dedicated runtime test.

## Diagnostic codes (`ContractCode`)

| Code | Severity (typical) | Meaning |
| --- | --- | --- |
| `DATASET_NORMALIZATION_FAILED` | error | Dataset could not be normalized (original message preserved). |
| `DATASET_EMPTY` | error | Dataset normalized to zero examples. |
| `DATASET_MISSING_INPUT` | warning | An example has an empty input payload. |
| `INJECTION_OPTIONS_CONFLICT` | error | Same option via both flat kwargs and a bundle. |
| `INJECTION_MODE_UNSUPPORTED` | error | Removed (`attribute`/`decorator`) or unknown mode. |
| `INJECTION_CONFIG_PARAM_MISSING` | error | Parameter mode needs a config param the function lacks. |
| `INJECTION_PROVIDER_SETUP_FAILED` | error | Provider could not construct its wrapper. |
| `SEAMLESS_INJECTION_UNAVAILABLE` | warning | Seamless AST injection cannot transform this function. |
| `AGENT_SIGNATURE_UNAVAILABLE` | warning | Target function signature could not be introspected. |
| `AGENT_BIND_FAILED` | error | Resolved call shape does not bind to the function. |
| `EVALUATOR_SIGNATURE_UNAVAILABLE` | warning | Metric signature could not be introspected. |
| `EVALUATOR_BIND_FAILED` | error | No candidate binds to the metric's signature. |
| `EVALUATOR_NO_RECOGNIZED_PARAMS` | warning | Metric binds only by bare positional fallback (silent mis-scoring hazard). |
| `CUSTOM_EVALUATOR_UNSUPPORTED` | warning | A custom evaluator is present and cannot be inspected. |
| `CONTRACT_INTERNAL_ERROR` | error | Unexpected internal error while building the report. |

## Open questions (pending owner confirmation)

The following design points are implemented with the defaults below but are
**one-way public-surface commitments awaiting owner sign-off**:

1. **Public naming.** `validate_evaluation_contract` /
   `EvaluationContractReport` / `ContractFinding` / `ContractCode` /
   `EVALUATION_CONTRACT_VERSION` are the chosen names (vs. alternatives like
   `EvaluationContractDiagnostic` / `EvaluationContractCode` / a module-scoped
   `CONTRACT_VERSION`).
2. **Stability granularity.** Promised stable: the report schema, the
   `ContractCode` vocabulary, and the no-execution guarantee. Explicitly *not*
   individually guaranteed: the alias sets, the expand rule, and the metric
   candidate order (reported, not promised).
3. **Custom-evaluator policy.** Chosen: detect-and-report as `unsupported`
   (warning, does not fail `report.ok`) rather than a hard error.
4. **Seamless in the contract.** Chosen: report seamless as best-effort
   (warnings only, never a hard `error`), since seamless is currently bundled
   but flagged for possible extraction to a plugin.
5. **`contract_version` starting value & bump policy.** Chosen: `"1.0.0"`, with
   minor = additive (new code/field), major = semantic change to an existing
   code.
