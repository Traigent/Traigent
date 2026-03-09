#!/usr/bin/env node

import { optimize, param } from "../../dist/index.js";
import {
  average,
  exactMatchScore,
  loadDataset,
  printSummary,
} from "../utils/base-example.mjs";

const rows = await loadDataset("rag_questions.jsonl");

const runTrial = optimize({
  configurationSpace: {
    retrievalK: param.int({ min: 1, max: 4, step: 1 }),
    retrievalMethod: param.enum(["keyword", "dense"]),
  },
  objectives: ["accuracy", "latency", "cost"],
  execution: { mode: "native" },
  evaluation: { data: rows },
})(async (trialConfig) => {
  const selectedRows = trialConfig.dataset_subset.indices.map((index) => rows[index]);
  const isStrong =
    trialConfig.config.retrievalMethod === "dense" &&
    Number(trialConfig.config.retrievalK) >= 2;

  const accuracies = selectedRows.map((row) =>
    exactMatchScore(isStrong ? row.output : "partial answer", row.output),
  );

  return {
    metrics: {
      accuracy: average(accuracies),
      latency: Number(trialConfig.config.retrievalK) * 0.2,
      cost:
        trialConfig.config.retrievalMethod === "dense"
          ? 0.08 + Number(trialConfig.config.retrievalK) * 0.01
          : 0.04 + Number(trialConfig.config.retrievalK) * 0.01,
    },
  };
});

const result = await runTrial.optimize({
  algorithm: "grid",
  maxTrials: 8,
});

printSummary("quickstart/02_customer_support_rag", result);
