import { readFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const WALKTHROUGH_ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "..");

export async function loadWalkthroughDataset(name) {
  const raw = await readFile(resolve(WALKTHROUGH_ROOT, "datasets", name), "utf8");
  return raw
    .trim()
    .split("\n")
    .filter(Boolean)
    .map((line) => JSON.parse(line));
}

export function subsetRows(rows, trialConfig) {
  const indices = trialConfig.dataset_subset?.indices ?? rows.map((_, index) => index);
  return indices.map((index) => rows[index]);
}

export function average(values) {
  return values.length === 0
    ? 0
    : values.reduce((total, value) => total + value, 0) / values.length;
}

export function printOptimizationSummary(name, result) {
  console.log(`\n=== ${name} ===`);
  console.log(
    JSON.stringify(
      {
        mode: result.mode,
        bestConfig: result.bestConfig,
        bestMetrics: result.bestMetrics,
        stopReason: result.stopReason,
        trialCount: result.trials.length,
      },
      null,
      2,
    ),
  );
}

export function resolveRealProviderConfig() {
  if (process.env.OPENROUTER_API_KEY) {
    return {
      apiKey: process.env.OPENROUTER_API_KEY,
      baseURL: "https://openrouter.ai/api/v1",
      defaultModel: "openai/gpt-4o-mini",
    };
  }

  if (process.env.OPENAI_API_KEY) {
    return {
      apiKey: process.env.OPENAI_API_KEY,
      baseURL: undefined,
      defaultModel: "gpt-4o-mini",
    };
  }

  return null;
}
