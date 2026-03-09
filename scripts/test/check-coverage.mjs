#!/usr/bin/env node

import { readFile } from "node:fs/promises";
import { resolve } from "node:path";

const COVERAGE_PATH = resolve(process.cwd(), "coverage", "coverage-final.json");
const MIN_LINES = 85;
const MIN_STATEMENTS = 85;
const MIN_FUNCTIONS = 85;

function percentage(hit, total) {
  return total === 0 ? 100 : (hit / total) * 100;
}

function summarize(entry) {
  const statementHits = Object.values(entry.s).filter((value) => value > 0).length;
  const functionHits = Object.values(entry.f).filter((value) => value > 0).length;
  const statementTotal = Object.keys(entry.s).length;
  const functionTotal = Object.keys(entry.f).length;

  const lineHits = new Set();
  const lineTotal = new Set();
  for (const [statementId, count] of Object.entries(entry.s)) {
    const location = entry.statementMap[statementId];
    for (let line = location.start.line; line <= location.end.line; line += 1) {
      lineTotal.add(line);
      if (count > 0) {
        lineHits.add(line);
      }
    }
  }

  return {
    lines: percentage(lineHits.size, lineTotal.size),
    statements: percentage(statementHits, statementTotal),
    functions: percentage(functionHits, functionTotal),
  };
}

const raw = await readFile(COVERAGE_PATH, "utf8");
const coverage = JSON.parse(raw);
const failures = [];

for (const [file, entry] of Object.entries(coverage)) {
  if (
    !file.includes("/src/") ||
    file.endsWith(".d.ts") ||
    file.endsWith("/index.ts") ||
    file.includes("/integrations/openai/") ||
    file.includes("/integrations/vercel-ai/") ||
    file.endsWith("/cli/runner.ts")
  ) {
    continue;
  }

  const summary = summarize(entry);
  const reasons = [];
  if (summary.lines < MIN_LINES) {
    reasons.push(`lines ${summary.lines.toFixed(2)} < ${MIN_LINES}`);
  }
  if (summary.statements < MIN_STATEMENTS) {
    reasons.push(`statements ${summary.statements.toFixed(2)} < ${MIN_STATEMENTS}`);
  }
  if (summary.functions < MIN_FUNCTIONS) {
    reasons.push(`functions ${summary.functions.toFixed(2)} < ${MIN_FUNCTIONS}`);
  }

  if (reasons.length > 0) {
    failures.push(
      `${file.replace(`${process.cwd()}/`, "")}: ${reasons.join(", ")}`,
    );
  }
}

if (failures.length > 0) {
  console.error("Per-file coverage check failed:");
  for (const failure of failures) {
    console.error(`- ${failure}`);
  }
  process.exit(1);
}

console.log(
  `Per-file coverage thresholds satisfied: lines/statements/functions >= ${MIN_LINES}%.`,
);
