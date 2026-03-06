import { execFileSync } from 'node:child_process';
import { existsSync } from 'node:fs';
import { performance } from 'node:perf_hooks';
import { resolve } from 'node:path';

import {
  getTrialConfig,
  getTrialParam,
  optimize,
  param,
} from '../dist/index.js';

const pythonRepoRoot = resolve(process.cwd(), '../Traigent');
const oracleScript = resolve(
  pythonRepoRoot,
  'tests/cross_sdk_oracles/generate_native_js_oracles.py',
);
const pythonBenchmarkScript = resolve(
  pythonRepoRoot,
  'tests/cross_sdk_oracles/run_async_scheduler_benchmark.py',
);

function delay(ms) {
  return new Promise((resolveDelay) => {
    setTimeout(resolveDelay, ms);
  });
}

function percentile(values, pct) {
  if (values.length === 0) return 0;
  const ordered = [...values].sort((left, right) => left - right);
  const index = Math.max(
    0,
    Math.min(
      ordered.length - 1,
      Math.round((pct / 100) * (ordered.length - 1)),
    ),
  );
  return ordered[index];
}

function mean(values) {
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function loadOraclePayload() {
  if (!existsSync(oracleScript)) {
    throw new Error(`Missing oracle script: ${oracleScript}`);
  }
  const stdout = execFileSync('python3', [oracleScript], {
    cwd: pythonRepoRoot,
    env: {
      ...process.env,
      PYTHONPATH: pythonRepoRoot,
    },
    encoding: 'utf8',
    stdio: ['ignore', 'pipe', 'pipe'],
  });
  return JSON.parse(stdout);
}

function toParamDefinition(definition) {
  switch (definition.type) {
    case 'enum':
      return param.enum(definition.values);
    case 'int':
      return param.int({
        min: definition.min,
        max: definition.max,
        step: definition.step,
        scale: definition.scale,
      });
    default:
      throw new Error(`Unsupported benchmark parameter type: ${definition.type}`);
  }
}

function summarizeRuns(runs) {
  const wallClockMs = runs.map((run) => run.wallClockMs);
  const throughput = runs.map((run) => run.throughput);
  const overheadRatio = runs.map((run) => run.overheadRatio);
  const rssDeltaMb = runs.map((run) => run.rssDeltaMb);
  return {
    wallClockMs: {
      mean: mean(wallClockMs),
      p50: percentile(wallClockMs, 50),
      p95: percentile(wallClockMs, 95),
    },
    throughput: {
      mean: mean(throughput),
      p50: percentile(throughput, 50),
      p95: percentile(throughput, 95),
    },
    overheadRatio: {
      mean: mean(overheadRatio),
      p50: percentile(overheadRatio, 50),
      p95: percentile(overheadRatio, 95),
    },
    duplicateConfigRate: mean(runs.map((run) => run.duplicateConfigRate)),
    contextLeakCount: runs.reduce((sum, run) => sum + run.contextLeakCount, 0),
    rssDeltaMb: {
      mean: mean(rssDeltaMb),
      p50: percentile(rssDeltaMb, 50),
      p95: percentile(rssDeltaMb, 95),
    },
    bestConfig: runs.at(-1)?.bestConfig ?? {},
    bestLatency: runs.at(-1)?.bestLatency ?? 0,
    stopReason: runs.at(-1)?.stopReason ?? 'maxTrials',
    trialCount: runs.at(-1)?.trialCount ?? 0,
  };
}

async function runSingleJsBenchmark(spec, concurrency) {
  const wrapped = optimize({
    configurationSpace: Object.fromEntries(
      Object.entries(spec.configurationSpace).map(([name, definition]) => [
        name,
        toParamDefinition(definition),
      ]),
    ),
    objectives: ['latency'],
    evaluation: {
      data: [{ id: 1 }],
    },
  })(async () => {
    const lane = String(getTrialParam('lane'));
    const slot = Number(getTrialParam('slot'));
    const currentConfig = getTrialConfig();
    const before = lane;
    const sleepMs = spec.sleepScheduleMs[lane] + slot * 2;
    await delay(sleepMs);
    const after = String(getTrialParam('lane'));
    const contextLeak = before !== after || currentConfig.lane !== lane;
    return {
      metrics: {
        latency: sleepMs,
        cost: sleepMs / 1000,
      },
      metadata: {
        contextLeak,
        sleepMs,
      },
    };
  });

  const rssBeforeMb = process.memoryUsage().rss / (1024 * 1024);
  const start = performance.now();
  const result = await wrapped.optimize({
    algorithm: spec.algorithm,
    maxTrials: spec.maxTrials,
    randomSeed: spec.randomSeed,
    trialConcurrency: concurrency,
  });
  const wallClockMs = performance.now() - start;
  const rssAfterMb = process.memoryUsage().rss / (1024 * 1024);

  const theoreticalMinMs =
    result.trials.reduce(
      (sum, trial) =>
        sum +
        spec.sleepScheduleMs[String(trial.config.lane)] +
        Number(trial.config.slot) * 2,
      0,
    ) / concurrency;
  const uniqueConfigs = new Set(
    result.trials.map((trial) => JSON.stringify(trial.config)),
  );

  return {
    wallClockMs,
    throughput: result.trials.length / (wallClockMs / 1000),
    theoreticalMinMs,
    overheadRatio: wallClockMs / theoreticalMinMs,
    duplicateConfigRate:
      1 - uniqueConfigs.size / Math.max(result.trials.length, 1),
    contextLeakCount: result.trials.filter(
      (trial) => trial.metadata?.contextLeak === true,
    ).length,
    rssDeltaMb: Math.max(rssAfterMb - rssBeforeMb, 0),
    bestConfig: result.bestConfig ?? {},
    bestLatency: Number(result.bestMetrics?.latency ?? 0),
    stopReason: result.stopReason,
    trialCount: result.trials.length,
  };
}

async function runJsBenchmark(benchmarkSpec) {
  const results = {
    generatedAt: new Date().toISOString(),
    runtime: 'js',
    nodeVersion: process.version,
    concurrencyLevels: {},
  };

  for (const concurrency of benchmarkSpec.concurrencyLevels) {
    for (let index = 0; index < benchmarkSpec.warmupRuns; index += 1) {
      await runSingleJsBenchmark(benchmarkSpec, concurrency);
    }

    const measuredRuns = [];
    for (let index = 0; index < benchmarkSpec.measuredRuns; index += 1) {
      measuredRuns.push(await runSingleJsBenchmark(benchmarkSpec, concurrency));
    }
    results.concurrencyLevels[String(concurrency)] = summarizeRuns(measuredRuns);
  }

  return results;
}

function runPythonBenchmark() {
  if (!existsSync(pythonBenchmarkScript)) {
    throw new Error(`Missing Python benchmark script: ${pythonBenchmarkScript}`);
  }
  const stdout = execFileSync('python3', [pythonBenchmarkScript], {
    cwd: pythonRepoRoot,
    env: {
      ...process.env,
      PYTHONPATH: pythonRepoRoot,
    },
    encoding: 'utf8',
    stdio: ['ignore', 'pipe', 'pipe'],
  });
  return JSON.parse(stdout);
}

function compareBenchmarks(jsResults, pythonResults) {
  const comparison = {};
  for (const level of Object.keys(jsResults.concurrencyLevels)) {
    const js = jsResults.concurrencyLevels[level];
    const python = pythonResults.concurrencyLevels[level];
    comparison[level] = {
      jsOverheadMean: js.overheadRatio.mean,
      pythonOverheadMean: python.overheadRatio.mean,
      overheadGap: js.overheadRatio.mean - python.overheadRatio.mean,
      jsThroughputMean: js.throughput.mean,
      pythonThroughputMean: python.throughput.mean,
      jsContextLeakCount: js.contextLeakCount,
      pythonContextLeakCount: python.contextLeakCount,
    };
  }
  return comparison;
}

const oraclePayload = loadOraclePayload();
const benchmarkSpec = oraclePayload.benchmark_spec;
const jsResults = await runJsBenchmark(benchmarkSpec);
const pythonResults = runPythonBenchmark();

console.log(
  JSON.stringify(
    {
      benchmarkSpec,
      js: jsResults,
      python: pythonResults,
      comparison: compareBenchmarks(jsResults, pythonResults),
    },
    null,
    2,
  ),
);
