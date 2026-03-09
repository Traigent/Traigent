import { readFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const EXAMPLES_ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "..");

export async function loadDataset(name) {
  const raw = await readFile(resolve(EXAMPLES_ROOT, "datasets", name), "utf8");
  return raw
    .trim()
    .split("\n")
    .filter(Boolean)
    .map((line) => JSON.parse(line));
}

export function normalizeText(value) {
  return String(value)
    .trim()
    .toLowerCase()
    .replace(/[^\p{L}\p{N}\s]/gu, " ")
    .replace(/\s+/g, " ");
}

export function exactMatchScore(actual, expected) {
  return normalizeText(actual).includes(normalizeText(expected)) ? 1 : 0;
}

export function average(values) {
  return values.length === 0
    ? 0
    : values.reduce((total, value) => total + value, 0) / values.length;
}

export function printSummary(name, result) {
  console.log(
    JSON.stringify(
      {
        example: name,
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

