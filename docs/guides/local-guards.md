# Block destructive tool calls with local guards

A local guard stops a tool call before it runs. You describe the rule in a TVL
module ("a volume delete needs a recorded approval, one approval per delete"),
compile it, and wrap the tool. When the rule is violated the wrapped function
raises `LocalGuardError("local_guard_blocked")` and its body never executes.

The runnable version of this guide is in `examples/advanced/local-guards/`.

## What you need

    pip install "traigent[runtime-tvl]"

This installs `tvl-spec`, which provides the `tvl-compile-runtime` command. The
guard classes live in `traigent.runtime`: `LocalGuardSet`, `guarded` and
`LocalGuardError`.

## 1. Write the rule in a TVL module

The rule is a `precedence` requirement: an `authorization_granted` event for
`destructive_change` must come before each `infra.delete_volume` tool call
(`mode: single_use`, matched by `correlation_id`). The response uses
`block_action`, and `on_check_error: block` makes the guard fail closed when it
cannot decide.

```yaml
requirements:
  - name: delete_needs_approval
    kind: zero_tolerance
    unit: invocation
    violation:
      pattern: precedence
      a: { event: authorization_granted, ref: destructive_change }
      b: { event: tool_call, tool: infra.delete_volume }
      mode: single_use
      correlate_by: correlation_id

responses:
  - name: delete_guard
    rules:
      - { id: block_delete, trigger: { monitor: delete_needs_approval_prod, event: unit_violation }, do: block_action, on_check_error: block }
```

The complete module, with the monitors it needs, is
`examples/advanced/local-guards/delete_needs_approval.runtime.yml`.

## 2. Compile it

    tvl-compile-runtime --token-map token-map.json --out-dir out delete_needs_approval.runtime.yml

The token map pairs each name in the module with an opaque token. The compiler
writes `out/runtime-local.json`, which the SDK loads.

## 3. Load the guards and wrap the tool

```python
import json
from traigent.runtime import LocalGuardError, LocalGuardSet, guarded

guards = LocalGuardSet.from_local(json.load(open("out/runtime-local.json")))

@guarded("infra.delete_volume")
def delete_volume(volume: str) -> str:
    ...  # the destructive call
```

## 4. Run work inside a unit and record approvals

```python
cid = "change-42"
with guards.unit(correlation_id=cid, ordered_attested=True) as unit:
    unit.observe({"type": "authorization_granted", "ref": "destructive_change",
                  "correlation_id": cid, "producer": "approver", "sequence": 0})
    delete_volume("vol-1")      # allowed: one approval, one delete
    try:
        delete_volume("vol-2")  # blocked: the approval was used up
    except LocalGuardError:
        ...
```

Expected result of the example: with no approval the call is refused; with one
approval the first delete runs; a second delete under the same approval is
refused and its side effect does not happen.

## Limits

- **In process only.** Guards run inside your Python process. They are not a
  sandbox.
- **Only wrapped tools are covered.** A tool you did not wrap with `@guarded` is
  not checked. An agent that also has a shell, or a broad credential, can reach
  the same resource another way.
- **Manual wiring.** `@traigent.optimize` does not apply guards for you. You load
  the guard set and decorate each tool yourself.
- **Approvals must be recorded correctly.** Each producer's `sequence` numbers must
  run 0, 1, 2 with no gap or repeat, and with more than one producer you attest the
  order with `ordered_attested=True`. If the guard cannot establish the history,
  the check is undecidable and the rule's `on_check_error` decides: `block` refuses,
  `allow` lets the call through and marks the unit unresolved.
- **Nothing is sent anywhere.** Guards make no network calls and send nothing to
  Traigent.
- **Deterministic.** Decisions follow the compiled rule. No model is involved.
