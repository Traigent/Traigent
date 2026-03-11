# Minimal Adoption Examples

These examples show the closest JS equivalent to the Python SDK pattern where a
client already has an agent/runnable function and wants to make the smallest
possible Traigent change.

Because this branch does not support JS decorator syntax, the equivalent of the
Python decorator is:

```js
export const optimizedThing = optimize(spec)(existingAgentFunction);
```

## Start Here

- plain client code: [minimal-change/original-agent.mjs](./minimal-change/original-agent.mjs)
- minimally Traigent-enabled version: [minimal-change/optimized-agent.mjs](./minimal-change/optimized-agent.mjs)
- runner: [minimal-change/runner.mjs](./minimal-change/runner.mjs)
- seamless original: [seamless/original-agent.mjs](./seamless/original-agent.mjs)
- seamless codemod target: [seamless/codemod-target.mjs](./seamless/codemod-target.mjs)
- seamless transformed: [seamless/transformed-agent.mjs](./seamless/transformed-agent.mjs)
- seamless runner: [seamless/runner.mjs](./seamless/runner.mjs)

## What This Demonstrates

1. Existing client function stays mostly unchanged
2. The original agent function is wrapped directly
3. The SDK owns evaluation through `evaluation.scoringFunction` and `metricFunctions`
4. `applyBestConfig()` stores the best config on the wrapped function
5. Future normal calls can reuse that applied config automatically

## Seamless Adoption in This Checkout

There are now three minimal-adoption stories:

1. `context` or `parameter` injection for durable explicit tuned variables
2. framework seamless for wrapped OpenAI, LangChain, and Vercel AI clients
3. code rewrite seamless for hardcoded locals via:
   - `traigent migrate seamless`
   - `@traigent/sdk/babel-plugin-seamless`

The seamless example folder shows the post-transform result users end up with:
explicit `getTrialParam()` reads, but without hand-writing them one by one.
