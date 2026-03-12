import OpenAI from "openai";
import { ChatOpenAI } from "@langchain/openai";

import {
  autoWrapFrameworkTarget,
  deleteOptimizationSession,
  finalizeOptimizationSession,
  listOptimizationSessions,
  getOptimizationSessionStatus,
  getTrialParam,
  optimize,
  param,
} from "../../../dist/index.js";
import { createTraigentOpenAI } from "../../../dist/integrations/openai/index.js";

export const SECTION_MAX_TRIALS = Number.parseInt(
  process.env.TRAIGENT_SHOWCASE_MAX_TRIALS ?? "5",
  10,
);

export const SECTION_DATASET_SIZE = Number.parseInt(
  process.env.TRAIGENT_SHOWCASE_DATASET_SIZE ?? "10",
  10,
);

export const FIVE_TEMPERATURES = [0, 0.2, 0.4, 0.6, 0.8];
export const FIVE_MAX_TOKENS = [8, 12, 16, 24, 32];
export const FIVE_STYLES = [
  "token-only",
  "strict",
  "minimal",
  "concise",
  "literal",
];

const TOKEN_DATASET = [
  "ALPHA",
  "ALPHA",
  "ALPHA",
  "ALPHA",
  "ALPHA",
  "ALPHA",
  "ALPHA",
  "ALPHA",
  "ALPHA",
  "ALPHA",
].map((token) => ({
  input: `Reply with exactly this uppercase token and nothing else: ${token}`,
  output: token,
}));

export function getDataset() {
  return TOKEN_DATASET.slice(0, Math.max(1, SECTION_DATASET_SIZE));
}

export function resolveConnection() {
  const backendUrl =
    process.env.TRAIGENT_BACKEND_URL ?? process.env.TRAIGENT_API_URL;
  const apiKey = process.env.TRAIGENT_API_KEY;

  if (!backendUrl || !apiKey) {
    throw new Error(
      "Set TRAIGENT_BACKEND_URL (or TRAIGENT_API_URL) and TRAIGENT_API_KEY before running the online showcase.",
    );
  }

  return { backendUrl, apiKey };
}

export function resolveProvider() {
  if (process.env.OPENROUTER_API_KEY) {
    return {
      provider: "openrouter",
      mode: "free",
      apiKey: process.env.OPENROUTER_API_KEY,
      baseURL: "https://openrouter.ai/api/v1",
      model:
        process.env.OPENROUTER_FREE_MODEL ??
        "nvidia/nemotron-3-super-120b-a12b:free",
      headers: {
        "HTTP-Referer": "https://traigent.ai",
        "X-Title": "Traigent JS Interactive Showcase",
      },
    };
  }

  if (process.env.OPENAI_API_KEY) {
    return {
      provider: "openai",
      mode: "fallback",
      apiKey: process.env.OPENAI_API_KEY,
      baseURL: undefined,
      model: process.env.OPENAI_MODEL ?? "gpt-4o-mini",
      headers: undefined,
    };
  }

  throw new Error(
    "Set OPENROUTER_API_KEY (preferred) or OPENAI_API_KEY before running the online showcase.",
  );
}

export function summarizeProvider(provider) {
  return {
    provider: provider.provider,
    mode: provider.mode,
    model: provider.model,
    baseURL: provider.baseURL ?? null,
  };
}

export function createWrappedOpenAIClient(options = {}) {
  const provider = resolveProvider();
  const rawClient = new OpenAI({
    apiKey: provider.apiKey,
    baseURL: provider.baseURL,
    defaultHeaders: provider.headers,
  });

  const client =
    options.wrapper === "manual"
      ? createTraigentOpenAI(rawClient)
      : autoWrapFrameworkTarget(rawClient);

  return { client, provider };
}

export function createWrappedLangChainModel() {
  const provider = resolveProvider();
  const rawModel = new ChatOpenAI({
    apiKey: provider.apiKey,
    model: provider.model,
    temperature: 0,
    configuration: provider.baseURL
      ? {
          baseURL: provider.baseURL,
          defaultHeaders: provider.headers,
        }
      : undefined,
  });

  return {
    model: autoWrapFrameworkTarget(rawModel),
    provider,
  };
}

export function extractToken(value) {
  const tokens = String(value ?? "")
    .toUpperCase()
    .match(/[A-Z]+/g);
  return tokens?.at(-1) ?? "";
}

export function scoreTokenAccuracy(output, row) {
  return extractToken(output) === String(row.output ?? "") ? 1 : 0;
}

export function buildMetricFunctions() {
  return {
    cost: (_output, _expectedOutput, runtimeMetrics) =>
      runtimeMetrics.total_cost ?? runtimeMetrics.cost ?? 0,
    total_cost: (_output, _expectedOutput, runtimeMetrics) =>
      runtimeMetrics.total_cost ?? runtimeMetrics.cost ?? 0,
    latency: (_output, _expectedOutput, runtimeMetrics) =>
      runtimeMetrics.latency ?? 0,
  };
}

export function buildBaseEvaluation() {
  return {
    data: getDataset(),
    scoringFunction: (output, _expectedOutput, _runtimeMetrics, row) =>
      scoreTokenAccuracy(output, row),
    metricFunctions: buildMetricFunctions(),
  };
}

export function createHybridOptions(connection, overrides = {}) {
  return {
    mode: "hybrid",
    algorithm: "optuna",
    maxTrials: SECTION_MAX_TRIALS,
    requestTimeoutMs: 20_000,
    timeoutMs: 60_000,
    includeFullHistory: true,
    ...connection,
    ...overrides,
  };
}

export async function maybeDeleteSession(sessionId, connection) {
  if (!sessionId) {
    return { attempted: false, deleted: false };
  }

  if (process.env.TRAIGENT_SHOWCASE_DELETE_AFTER_SECTION !== "1") {
    return { attempted: false, deleted: false };
  }

  const deleted = await deleteOptimizationSession(sessionId, {
    ...connection,
    cascade: false,
  });
  return { attempted: true, deleted: deleted.success };
}

export async function collectSessionHelpers(sessionId, connection) {
  if (!sessionId) {
    return {
      listed: null,
      status: null,
      finalized: null,
      deleted: { attempted: false, deleted: false },
      helperErrors: [],
    };
  }

  const settled = await Promise.allSettled([
    listOptimizationSessions({
      ...connection,
      pattern: sessionId,
    }),
    getOptimizationSessionStatus(sessionId, connection),
    finalizeOptimizationSession(sessionId, {
      ...connection,
      includeFullHistory: true,
    }),
  ]);

  const helperNames = ["listed", "status", "finalized"];
  const helperValues = Object.fromEntries(
    settled.map((result, index) => [
      helperNames[index],
      result.status === "fulfilled" ? result.value : null,
    ]),
  );
  const helperErrors = settled.flatMap((result, index) => {
    if (result.status === "fulfilled") {
      return [];
    }

    return [
      {
        helper: helperNames[index],
        message:
          result.reason instanceof Error
            ? result.reason.message
            : String(result.reason),
      },
    ];
  });
  const deleted = await maybeDeleteSession(sessionId, connection);

  return { ...helperValues, deleted, helperErrors };
}

export function summarizeResult(name, result, extras = {}) {
  return {
    name,
    sessionId: result.sessionId,
    stopReason: result.stopReason,
    errorMessage: result.errorMessage ?? null,
    trialCount: result.trials.length,
    bestConfig: result.bestConfig,
    bestMetrics: result.bestMetrics,
    reporting: result.reporting
      ? {
          totalTrials: result.reporting.totalTrials,
          successfulTrials: result.reporting.successfulTrials,
          totalDuration: result.reporting.totalDuration,
          fullHistoryCount: result.reporting.fullHistory?.length ?? 0,
          convergenceCount: result.reporting.convergenceHistory?.length ?? 0,
        }
      : null,
    ...extras,
  };
}

export function summarizeSessionEvidence(helpers) {
  const status = helpers.status;
  const listed = helpers.listed;
  const metadata =
    status && typeof status.metadata === "object" && status.metadata
      ? status.metadata
      : null;
  const experimentId =
    metadata && typeof metadata.experiment_id === "string"
      ? metadata.experiment_id
      : null;
  const experimentRunId =
    metadata && typeof metadata.experiment_run_id === "string"
      ? metadata.experiment_run_id
      : null;
  const frontendBase =
    process.env.TRAIGENT_SHOWCASE_FE_URL ?? "http://localhost:3000";

  return {
    // NOTE: the current backend session listing may exclude completed sessions,
    // so `listedSessionCount` can be 0 even when `status` shows a completed run.
    listedSessionCount:
      listed && Array.isArray(listed.sessions) ? listed.sessions.length : 0,
    listedTotal: listed?.total ?? null,
    status: status?.status ?? null,
    progress:
      status && typeof status.progress === "object" && status.progress
        ? {
            completed: status.progress.completed ?? null,
            total: status.progress.total ?? null,
            failed: status.progress.failed ?? null,
          }
        : null,
    experimentId,
    experimentRunId,
    experimentUrl:
      experimentId && experimentId.length > 0
        ? `${frontendBase.replace(/\/$/, "")}/experiments/view/${experimentId}`
        : null,
    helperErrors:
      Array.isArray(helpers.helperErrors) && helpers.helperErrors.length > 0
        ? helpers.helperErrors
        : [],
  };
}

export function getCachedCompletion(cache, cacheKey, createRequest) {
  const existing = cache.get(cacheKey);
  if (existing) {
    return existing;
  }

  const created = Promise.resolve().then(createRequest);
  cache.set(cacheKey, created);
  return created;
}

export function createTokenOnlyPrompt(input) {
  return [
    {
      role: "system",
      content:
        "You are a deterministic evaluator. Reply with the exact uppercase token only. No punctuation, no explanation.",
    },
    {
      role: "user",
      content: String(input),
    },
  ];
}

export function createContextStylePrompt(input) {
  const style = getTrialParam("style", "token-only");

  const guidanceByStyle = {
    "token-only":
      "Return only the uppercase token. No punctuation. No other words.",
    strict:
      "Return the token only. Do not add punctuation, markdown, or extra text.",
    minimal:
      "Return exactly the token.",
    concise:
      "Reply with the requested token only.",
    literal:
      "Echo the requested uppercase token literally and nothing else.",
  };

  return [
    {
      role: "system",
      content:
        guidanceByStyle[style] ?? guidanceByStyle["token-only"],
    },
    {
      role: "user",
      content: String(input),
    },
  ];
}

export function createParameterizedPrompt(input, config) {
  const prefix = String(config.prefix ?? "token-only");
  const instruction =
    prefix === "literal"
      ? "Return the exact token literally."
      : prefix === "strict"
        ? "Return only the requested uppercase token."
        : prefix === "minimal"
          ? "Reply with the token."
          : prefix === "concise"
            ? "Return only the token and no explanation."
            : "Return the exact uppercase token only.";

  return [
    {
      role: "system",
      content: instruction,
    },
    {
      role: "user",
      content: String(input),
    },
  ];
}

export function createBaseSpec(overrides = {}) {
  return {
    objectives: ["accuracy", "cost"],
    evaluation: buildBaseEvaluation(),
    ...overrides,
  };
}

export { getTrialParam, optimize, param };
