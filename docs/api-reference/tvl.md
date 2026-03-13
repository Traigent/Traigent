# TVL Support

The JS SDK supports a focused TVL subset through `parseTvlSpec()` and
`loadTvlSpec()`:

- `tvars`
- objectives, including banded objectives
- budgets
- structural and derived constraints
- default config
- promotion-policy metadata

Current caveat:

- promotion policy is parsed and transported, but not yet behaviorally enforced
