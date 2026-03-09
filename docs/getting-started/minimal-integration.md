# Minimal Integration

Use `optimize(spec)(fn)` to attach optimization metadata to a trial function.

```ts
import { optimize, param } from "@traigent/sdk";

const runTrial = optimize({
  configurationSpace: {
    model: param.enum(["cheap", "accurate"]),
  },
  objectives: ["accuracy"],
  evaluation: { data: [{ id: 1 }] },
})(async (trialConfig) => ({
  metrics: {
    accuracy: trialConfig.config.model === "accurate" ? 1 : 0.5,
  },
}));
```

Then choose either:

- backend-guided hybrid: `runTrial.optimize({ algorithm: "optuna", ... })`
- local native: `runTrial.optimize({ mode: "native", algorithm: "grid", ... })`
