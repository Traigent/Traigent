# Quickstart: Where Injection Happens

The quickstart examples are intentionally small, but that also makes the flow
easy to miss. The default path optimizes a plain agent function, not a manual
trial loop.

## Mental Model

You write three things:

1. The optimization spec
2. The agent function
3. The evaluator settings

The SDK does the search loop, calls your agent on each evaluation row, and
aggregates metrics from `scoringFunction`, `metricFunctions`, or
`customEvaluator`.

`customEvaluator` supports two shapes in this checkout:

- context style: `customEvaluator({ output, expectedOutput, runtimeMetrics, row, config })`
- Python-like style: `customEvaluator(agentFn, config, row)`

## Where `tvars` Are Declared

In this JS SDK branch, the tunables are declared in:

```js
optimize({
  configurationSpace: {
    model: param.enum(['cheap', 'balanced', 'accurate']),
    temperature: param.float({ min: 0, max: 0.4, step: 0.2 }),
  },
  objectives: ['accuracy', 'cost'],
  evaluation: { data: rows },
});
```

That `configurationSpace` block is the JS equivalent of defining tunable
variables.

## Where the Agent Code Is

In the small quickstarts, the "agent" is the wrapped function itself. The SDK
owns the dataset iteration and evaluation.

In a more realistic demo, the app logic is separated:

- agent/business logic: [agent.ts](../../demos/arkia-sales-agent/src/agent.ts)
- optimization wrapper: [trial.ts](../../demos/arkia-sales-agent/src/trial.ts)

## Where Injection Happens

The injection point is inside the SDK runtime:

1. the optimizer chooses a config
2. the wrapper resolves the active config in [spec.ts](../../src/optimization/spec.ts)
3. the agent evaluator binds trial context in [agent.ts](../../src/optimization/agent.ts)
4. your agent reads it via:
   - `trialConfig.config`
   - [getTrialConfig()](../../src/core/context.ts#L162)
   - [getTrialParam()](../../src/core/context.ts#L173)
5. the SDK evaluates outputs using your `evaluation` block

For `seamless` mode in this checkout, config can reach your code in three ways:

1. framework interception through the wrappers below
2. codemod / build-time transformed `getTrialParam()` calls
3. experimental runtime rewriting for self-contained plain Node functions

Framework interception flows through:

- [openai/index.ts](../../src/integrations/openai/index.ts)
- [langchain/model.ts](../../src/integrations/langchain/model.ts)
- [vercel-ai/index.ts](../../src/integrations/vercel-ai/index.ts)
- [auto-wrap.ts](../../src/integrations/auto-wrap.ts)

If you already have a few framework objects in scope, the least-friction path is:

```js
import { autoWrapFrameworkTargets } from '@traigent/sdk';

const wrapped = autoWrapFrameworkTargets({
  openaiClient,
  chatModel,
});
```

## Best File To Read First

Start with:

- [00_agent_injection_map.mjs](./00_agent_injection_map.mjs)

That file is purpose-built to show:

- where tunables are declared
- where the agent code lives
- where the chosen config is read
- how `.optimize(...)` owns the search loop

If you want minimal-change seamless adoption for hardcoded locals, read:

- [examples/adoption/README.md](../adoption/README.md)
