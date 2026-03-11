import OpenAI from "openai";

import { optimize, param } from "../../dist/index.js";
import {
  average,
  exactMatchScore,
  loadWalkthroughDataset,
  printOptimizationSummary,
  resolveRealProviderConfig,
  subsetRows,
} from "./helpers.mjs";

const DATASET_BY_SCENARIO = {
  tuning_qa: "simple_questions.jsonl",
  zero_code_change: "simple_questions.jsonl",
  parameter_mode: "simple_questions.jsonl",
  multi_objective: "classification.jsonl",
  rag_parallel: "rag_questions.jsonl",
};

function scenarioConfigSpace(id) {
  switch (id) {
    case "parameter_mode":
      return {
        model: param.enum(["cheap", "balanced", "accurate"]),
        temperature: param.float({ min: 0, max: 0.4, step: 0.2 }),
        max_tokens: param.int({ min: 64, max: 192, step: 64 }),
      };
    case "rag_parallel":
      return {
        model: param.enum(["balanced", "accurate"]),
        retrieval_k: param.int({ min: 1, max: 4, step: 1 }),
        retrieval_method: param.enum(["keyword", "dense"]),
      };
    default:
      return {
        model: param.enum(["cheap", "balanced", "accurate"]),
        temperature: param.float({ min: 0, max: 0.4, step: 0.2 }),
      };
  }
}

function scenarioObjectives(id) {
  switch (id) {
    case "multi_objective":
      return [
        { metric: "accuracy", direction: "maximize", weight: 2 },
        { metric: "cost", direction: "minimize", weight: 1 },
        { metric: "latency", direction: "minimize", weight: 1 },
      ];
    default:
      return ["accuracy", "cost"];
  }
}

function mockText(row, config, id) {
  if (id === "rag_parallel") {
    return Number(config.retrieval_k ?? 1) >= 2 ? row.output : "partial answer";
  }
  if (id === "multi_objective") {
    return config.model === "accurate" ? row.output : "neutral";
  }
  return config.model === "cheap" ? "unknown" : row.output;
}

function mockCost(config, id) {
  if (id === "rag_parallel") {
    return 0.06 + Number(config.retrieval_k ?? 1) * 0.02;
  }
  return config.model === "accurate" ? 0.18 : config.model === "balanced" ? 0.1 : 0.05;
}

function mockLatency(config, id) {
  if (id === "rag_parallel") {
    return 0.4 + Number(config.retrieval_k ?? 1) * 0.1;
  }
  return config.model === "accurate" ? 1.1 : config.model === "balanced" ? 0.7 : 0.4;
}

async function callRealModel(prompt, config) {
  const provider = resolveRealProviderConfig();
  if (!provider) {
    throw new Error(
      "Real walkthrough examples require OPENAI_API_KEY or OPENROUTER_API_KEY.",
    );
  }

  const client = new OpenAI({
    apiKey: provider.apiKey,
    ...(provider.baseURL ? { baseURL: provider.baseURL } : {}),
  });

  const model =
    config.model === "accurate"
      ? provider.defaultModel.replace("mini", "")
      : provider.defaultModel;

  const started = Date.now();
  const response = await client.chat.completions.create({
    model,
    temperature: Number(config.temperature ?? 0.2),
    max_tokens: Number(config.max_tokens ?? 128),
    messages: [{ role: "user", content: prompt }],
  });

  return {
    text: response.choices[0]?.message?.content ?? "",
    latency: (Date.now() - started) / 1000,
    cost:
      typeof response.usage?.total_tokens === "number"
        ? response.usage.total_tokens * 0.000002
        : 0,
  };
}

function scoreRow(id, output, row) {
  if (id === "multi_objective") {
    return row.output === output ? 1 : 0.6;
  }
  if (id === "rag_parallel") {
    if (output === row.output) return 1;
    if (output === "partial answer") return 0.72;
    return 0.4;
  }
  return exactMatchScore(output, row.output);
}

export async function runScenario(id, mode) {
  const rows = await loadWalkthroughDataset(DATASET_BY_SCENARIO[id]);

  const wrapped = optimize({
    configurationSpace: scenarioConfigSpace(id),
    objectives: scenarioObjectives(id),
    evaluation: { data: rows },
    execution: { contract: "trial" },
  })(async (trialConfig) => {
    const selectedRows = subsetRows(rows, trialConfig);
    const outputs = [];
    const costs = [];
    const latencies = [];

    for (const row of selectedRows) {
      if (mode === "real") {
        const prompt =
          row.input.question ??
          row.input.text ??
          row.input.task ??
          JSON.stringify(row.input);
        const real = await callRealModel(prompt, trialConfig.config);
        outputs.push(real.text);
        costs.push(real.cost);
        latencies.push(real.latency);
      } else {
        outputs.push(mockText(row, trialConfig.config, id));
        costs.push(mockCost(trialConfig.config, id));
        latencies.push(mockLatency(trialConfig.config, id));
      }
    }

    const scores = selectedRows.map((row, index) =>
      scoreRow(id, outputs[index], row),
    );

    return {
      metrics: {
        accuracy: average(scores),
        cost: average(costs),
        latency: average(latencies),
      },
      metadata: {
        sampleOutput: outputs[0],
      },
    };
  });

  const result = await wrapped.optimize({
    algorithm: mode === "real" ? "random" : "grid",
    maxTrials: mode === "real" ? 4 : 6,
    randomSeed: 7,
  });

  printOptimizationSummary(`${mode}/${id}`, result);
  return result;
}
