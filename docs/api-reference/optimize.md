# `optimize()`

`optimize(spec)(fn)` wraps a trial function and attaches optimization metadata.

Key spec fields:

- `configurationSpace`
- `objectives`
- `budget`
- `constraints`
- `defaultConfig`
- `execution`
- `evaluation`
- `promotionPolicy`
- `autoLoadBest`
- `loadFrom`

Execution modes:

- default hybrid execution
- explicit native execution with `mode: "native"`
