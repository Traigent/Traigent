#!/usr/bin/env node

import { optimize, param } from "../../../dist/index.js";

const runTrial = optimize({
  configurationSpace: {
    model: param.enum(["cheap", "accurate"]),
    maxTokens: param.int({
      min: 64,
      max: 256,
      step: 64,
      conditions: { model: "accurate" },
      default: 64,
    }),
  },
  objectives: [
    { metric: "accuracy", direction: "maximize", weight: 2 },
    { metric: "cost", direction: "minimize", weight: 1 },
  ],
  constraints: {
    structural: [
      {
        when: 'params.model == "accurate"',
        then: "params.maxTokens >= 64",
      },
    ],
  },
  evaluation: {
    data: [{ id: 1 }, { id: 2 }],
  },
})(async (trialConfig) => ({
  metrics: {
    accuracy: trialConfig.config.model === "accurate" ? 0.95 : 0.75,
    cost: trialConfig.config.model === "accurate" ? 0.2 : 0.05,
  },
}));

if (
  !(process.env.TRAIGENT_BACKEND_URL ?? process.env.TRAIGENT_API_URL) ||
  !process.env.TRAIGENT_API_KEY
) {
  console.log(
    "[skip] examples/core/conditional-constraints/run.mjs requires the typed Traigent backend session API plus TRAIGENT_API_KEY.",
  );
  process.exit(0);
}

const result = await runTrial.optimize({
  mode: "hybrid",
  algorithm: "optuna",
  maxTrials: 4,
  backendUrl: process.env.TRAIGENT_BACKEND_URL ?? process.env.TRAIGENT_API_URL,
  apiKey: process.env.TRAIGENT_API_KEY,
});

console.log(JSON.stringify(result, null, 2));
