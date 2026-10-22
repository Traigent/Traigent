# Local-guard parity vectors

`local_guard_vectors.json` is SDK-owned. It is consumed by the Python tests
(`tests/unit/runtime/test_local_guards.py`) and is vendored byte-for-byte by the
JS SDK (`traigent-js`), which must reproduce every decision.

local_guard_vectors.json sha256: 4a0278a264c4b1ac5e1e4a4986d9ac2687d3d3dbd6e24a489e37aaa82f4c4ce1

## Derived from

TVL `origin/main` 098bcdfba0b7347eda9cdbad2e351aad39d25f95 (`Traigent/tvl`),
`formalization/`:

- `rfc-0003-runtime-requirements-monitors-responses.md`: section 7.5 (local guards
  and `on_check_error`), 6.3 (events, `ref`, `correlation_id`, revocation), 6.4
  (classifications, prefix-fixed violations), 6.5 (unit predicates), 6.6 (resource
  limits), 5.8 (`response_guard_not_enforceable`), 11.4.4 (`tvl-runtime-local`),
  15.3 (the payment-authorization example).
- `tvl-formal-semantics.md`: Definition 12.4 (events), 12.6 (violation predicates
  and prefix-fixed violations), 12.7 (an evaluator profile is outside TVL), 12.8
  (classification contract and the local-guard paragraph).

Each case names the sections it rests on in `cites`.

## What the vectors pin that the RFC leaves to the evaluator profile (Def 12.7)

- An event that cannot be a violating step for a rule (it matches none of the
  predicate's relevant selectors) is allowed without consulting `on_check_error`.
- The violating step is the offered event itself: the matching `a` (absence), the
  (N+1)-th matching `a` (bounded_existence), a matching `b` without a legal
  authorization (precedence), an in-alphabet event ending in a rejecting state
  (protocol; rejecting is absorbing, so later in-alphabet events stay violating).
- History is trusted only while each producer numbers its events 0, 1, 2, ... with
  no gap or repeat; several producers need an attested order; a unit contradicting
  `single_producer` is never decided.
- `absence` and the `bounded_existence` count are monotone, so they are decided from
  the events seen even when the history has a gap.
- Resource limits: `max_tracked_correlations` bounds the distinct keys among the
  events a precedence predicate looks at; `max_active_obligations` bounds grants still
  pending (neither consumed nor revoked).
- Precedence assignment: each `b` spends the eligible grant that is revoked soonest.
