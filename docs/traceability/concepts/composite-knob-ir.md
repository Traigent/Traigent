# Composite-Knob IR

Traigent 0.12.0 includes the RFC 0002 P5 composite-knob intermediate
representation in `traigent.knobs.composites` and catalog factories in
`traigent.knobs.patterns`.

This is internal declaration-only IR — not wired to optimizer/cloud effectuation.
It lets the SDK declare grouped control-flow shapes without changing the RFC
0001 one-knob binding model.

## What It Represents

`traigent.knobs.composites` defines a sealed algebra for composite declarations:

- `cascade`
- `ensemble`
- `loop`

The module models declarations such as `CompositeNode`, `CompositeProgram`,
`StageArm`, `CompositeArm`, gates, stop rules, aggregate declarations, and signal
uses. Validation checks structural well-formedness, including missing composite
references, cycles, ambiguous arms, and parent coverage.

## Pattern Catalog

`traigent.knobs.patterns` provides named factories over the composite algebra,
including:

- `binary_cascade`
- `n_cascade`
- `best_of_n`
- `self_consistency`
- `self_refine`
- `self_debug`

Each factory returns a `CompositeKnob` declaration bundle with a structure,
member knob declarations, provenance, and standard telemetry names.

## Boundary

Composite knobs do not bind values. Value binding remains on regular `Tuned`,
`Fixed`, and `Calibrated` knobs. The pattern factories return declarations only;
they do not change orchestrator behavior and do not trigger cloud effectuation.
