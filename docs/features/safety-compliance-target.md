# Safety constraints: the required compliance rate

`SafetyValidator.validate()` asks whether trials comply with a constraint often
enough, with statistical confidence. Each trial is first checked against its own
per-trial bar (`.above(x)`, `.below(x)`, ...). The validator then needs a second
number: the **required compliance rate**, the share of trials that must comply.

That rate is derived from the threshold's operator and value; there is no
separate parameter for it.

| Constraint | Required compliance rate |
|---|---|
| `.above(x)` (`>=`, `>`) | `x` |
| `.below(x)` (`<=`, `<`) | `1 - x` |
| `==` | `x` |
| `.below(x)` with `x` outside [0, 1] | `1.0` (fails closed) |

So `.above(0.9)` and `.below(0.1)` both require 90% of trials to comply: a
stricter per-trial bar means a higher required rate. The result reports the
derived value as `required_compliance_rate`. A constraint passes only when the
lower confidence bound of the observed compliance rate reaches it.
