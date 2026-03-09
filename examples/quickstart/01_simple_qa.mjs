#!/usr/bin/env node

import { optimize, param } from "../../dist/index.js";
import {
  average,
  exactMatchScore,
  loadDataset,
  printSummary,
} from "../utils/base-example.mjs";

const rows = await loadDataset("simple_questions.jsonl");

const runTrial = optimize({
  configurationSpace: {
    model: param.enum(["cheap", "balanced", "accurate"]),
    temperature: param.float({ min: 0, max: 0.4, step: 0.2 }),
  },
  objectives: ["accuracy", "cost"],
  execution: { mode: "native" },
  evaluation: { data: rows },
})(async (trialConfig) => {
  const selectedRows = trialConfig.dataset_subset.indices.map((index) => rows[index]);
  const quality =
    trialConfig.config.model === "accurate"
      ? 0.95
      : trialConfig.config.model === "balanced"
        ? 0.82
        : 0.68;

  const accuracies = selectedRows.map((row) =>
    exactMatchScore(
      quality >= 0.8 ? row.output : "unknown",
      row.output,
    ),
  );

  return {
    metrics: {
      accuracy: average(accuracies),
      cost:
        trialConfig.config.model === "accurate"
          ? 0.22
          : trialConfig.config.model === "balanced"
            ? 0.12
            : 0.05,
      latency:
        trialConfig.config.model === "accurate"
          ? 1.2
          : trialConfig.config.model === "balanced"
            ? 0.8
            : 0.4,
    },
  };
});

const result = await runTrial.optimize({
  algorithm: "grid",
  maxTrials: 9,
});

printSummary("quickstart/01_simple_qa", result);
