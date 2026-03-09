#!/usr/bin/env node

import { optimize, param } from "../../dist/index.js";
import { average, loadDataset, printSummary } from "../utils/base-example.mjs";

const rows = await loadDataset("classification.jsonl");

const runTrial = optimize({
  configurationSpace: {
    model: param.enum(["cheap", "accurate"]),
    temperature: param.float({ min: 0, max: 0.8, step: 0.4 }),
  },
  objectives: [
    { metric: "quality_score", direction: "maximize" },
    { metric: "latency", direction: "minimize" },
    { metric: "response_length", band: { low: 1, high: 2 } },
  ],
  execution: { mode: "native" },
  evaluation: { data: rows },
})(async (trialConfig) => {
  const selectedRows = trialConfig.dataset_subset.indices.map((index) => rows[index]);
  const responseLength = trialConfig.config.model === "accurate" ? 1 : 3;
  return {
    metrics: {
      quality_score: average(
        selectedRows.map((row) =>
          row.output === "positive" || trialConfig.config.model === "accurate"
            ? 1
            : 0.6,
        ),
      ),
      latency: trialConfig.config.model === "accurate" ? 0.9 : 0.4,
      response_length: responseLength,
    },
  };
});

const result = await runTrial.optimize({
  algorithm: "grid",
  maxTrials: 4,
});

printSummary("quickstart/03_custom_objectives", result);
