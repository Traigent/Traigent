# Configuration Spaces

JS configuration spaces are authored in code with `param.*` helpers.

Supported parameter types:

- `param.enum(...)`
- `param.int(...)`
- `param.float(...)`
- `param.bool()`

Hybrid mode additionally supports:

- conditional parameters with equality conditions
- required default fallbacks for inactive parameters
- structural constraints
- derived constraints

Legacy `toHybridConfigSpace()` is still available for the older `/config-space`
wire format, but it is narrower than the typed session contract.
