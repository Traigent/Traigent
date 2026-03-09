# Interactive Optimization

`.optimize(...)` defaults to backend-guided hybrid execution.

```ts
await wrapped.optimize({
  algorithm: "optuna",
  maxTrials: 12,
  backendUrl: process.env.TRAIGENT_BACKEND_URL,
  apiKey: process.env.TRAIGENT_API_KEY,
});
```

Use `mode: "native"` when you want the optimizer to stay in-process:

```ts
await wrapped.optimize({
  mode: "native",
  algorithm: "grid",
  maxTrials: 12,
});
```
