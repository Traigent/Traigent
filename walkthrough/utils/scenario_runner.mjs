import OpenAI from "openai";

import { optimize, param } from "../../dist/index.js";
import {
  average,
  loadWalkthroughDataset,
  printOptimizationSummary,
  resolveRealProviderConfig,
  subsetRows,
} from "./helpers.mjs";
import {
  classificationScore,
  codeGenerationScore,
  exactMatchScore,
  semanticSimilarityScore,
} from "./scoring.mjs";
import { mockCost, mockLatency, mockText } from "./mock_answers.mjs";

const DATASET_BY_SCENARIO = {
  tuning_qa: "simple_questions.jsonl",
  zero_code_change: "simple_questions.jsonl",
  parameter_mode: "simple_questions.jsonl",
  multi_objective: "classification.jsonl",
  rag_parallel: "rag_questions.jsonl",
  custom_evaluator: "code_gen.jsonl",
  multi_provider: "simple_questions_10.jsonl",
  privacy_modes: "simple_questions.jsonl",
};

function scenarioConfigSpace(id) {
  switch (id) {
    case "rag_parallel":
      return {
        model: param.enum(["balanced", "accurate"]),
        retrievalK: param.int({ min: 1, max: 4, step: 1 }),
        retrievalMethod: param.enum(["keyword", "dense"]),
      };
    case "multi_provider":
      return {
        providerModel: param.enum([
          "cheap",
          "balanced",
          "accurate",
        ]),
        temperature: param.float({ min: 0, max: 0.4, step: 0.2 }),
      };
    case "privacy_modes":
      return {
        model: param.enum(["cheap", "accurate"]),
        temperature: param.float({ min: 0, max: 0.4, step: 0.2 }),
      };
    default:
      return {
        model: param.enum(["cheap", "balanced", "accurate"]),
        temperature: param.float({ min: 0, max: 0.4, step: 0.2 }),
      };
  }
}

function scenarioObjectives(id, useHybrid) {
  switch (id) {
    case "multi_objective":
      return ["accuracy", "cost", "latency"];
    case "custom_evaluator":
      return useHybrid
        ? [{ metric: "quality", direction: "maximize", weight: 2 }, "cost"]
        : [{ metric: "quality", direction: "maximize" }, "cost"];
    case "privacy_modes":
      return ["accuracy", "cost"];
    default:
      return ["accuracy", "cost"];
  }
}

function scoreRow(id, output, row) {
  switch (id) {
    case "multi_objective":
      return classificationScore(output, row.output);
    case "rag_parallel":
      return semanticSimilarityScore(output, row.output);
    case "custom_evaluator":
      return codeGenerationScore(output, row.input.task);
    default:
      return exactMatchScore(output, row.output);
  }
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
    typeof config.providerModel === "string" &&
    ["cheap", "balanced", "accurate"].includes(config.providerModel)
      ? config.providerModel === "accurate"
        ? provider.defaultModel.replace("mini", "")
        : provider.defaultModel
      : provider.defaultModel;

  const started = Date.now();
  const response = await client.chat.completions.create({
    model,
    temperature: Number(config.temperature ?? 0.2),
    max_tokens: Number(config.maxTokens ?? 128),
    messages: [{ role: "user", content: prompt }],
  });
  const text = response.choices[0]?.message?.content ?? "";
  const usage = response.usage;
  return {
    text,
    latency: (Date.now() - started) / 1000,
    cost:
      typeof usage?.total_tokens === "number"
        ? usage.total_tokens * 0.000002
        : 0,
  };
}

export async function runScenario(id, mode) {
  const rows = await loadWalkthroughDataset(DATASET_BY_SCENARIO[id]);
  const useHybrid =
    mode === "real" &&
    Boolean(
      (process.env.TRAIGENT_BACKEND_URL ?? process.env.TRAIGENT_API_URL) &&
        process.env.TRAIGENT_API_KEY,
    );
  const executionMode =
    useHybrid ? undefined : "native";

  const wrapped = optimize({
    configurationSpace: scenarioConfigSpace(id),
    objectives: scenarioObjectives(id, useHybrid),
    ...(id === "privacy_modes"
      ? {
          defaultConfig: { temperature: 0.2 },
          autoLoadBest: true,
          loadFrom: "./walkthrough/.best-config.json",
        }
      : {}),
    ...(executionMode ? { execution: { mode: executionMode } } : {}),
    evaluation: { data: rows },
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
        costs.push(mockCost(trialConfig.config));
        latencies.push(mockLatency(trialConfig.config));
      }
    }

    const scores = selectedRows.map((row, index) =>
      scoreRow(id, outputs[index], row),
    );

    return {
      metrics: {
        ...(id === "custom_evaluator"
          ? { quality: average(scores) }
          : { accuracy: average(scores) }),
        cost: average(costs),
        latency: average(latencies),
      },
      metadata: {
        sampleOutput: outputs[0],
      },
    };
  });

  const result = await wrapped.optimize({
    ...(executionMode ? { mode: "native", algorithm: "grid" } : { algorithm: "optuna" }),
    maxTrials: mode === "real" ? 4 : 6,
    ...(executionMode
      ? {}
      : {
          backendUrl: process.env.TRAIGENT_BACKEND_URL ?? process.env.TRAIGENT_API_URL,
          apiKey: process.env.TRAIGENT_API_KEY,
        }),
  });

  printOptimizationSummary(`${mode}/${id}`, result);
  return result;
}
