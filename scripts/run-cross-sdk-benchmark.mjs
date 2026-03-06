import { execFileSync } from 'node:child_process';
import { existsSync } from 'node:fs';
import { performance } from 'node:perf_hooks';
import { resolve } from 'node:path';

import {
  TrialContext,
  getTrialConfig,
  getTrialParam,
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
    stopReason: runs.at(-1)?.stopReason ?? 'completed',
    trialCount: runs.at(-1)?.trialCount ?? 0,
  };
}

function createTrialConfig(config, trialNumber) {
  return {
    trial_id: `bench_${trialNumber}`,
    trial_number: trialNumber,
    experiment_run_id: 'cross_sdk_benchmark',
    config,
    dataset_subset: {
      indices: [0],
      total: 1,
    },
  };
}

async function runSingleJsBenchmark(spec, concurrency) {
  const results = new Array(spec.configs.length);
  let nextIndex = 0;
  const rssBeforeMb = process.memoryUsage().rss / (1024 * 1024);
  const start = performance.now();

  async function worker() {
    while (true) {
      const index = nextIndex;
      nextIndex += 1;

      if (index >= spec.configs.length) {
        return;
      }

      const config = spec.configs[index];
      const trialConfig = createTrialConfig(config, index + 1);
      const lane = String(config.lane);
      const sleepMs = spec.sleepScheduleMs[lane] + Number(config.slot) * 2;

      results[index] = await TrialContext.run(trialConfig, async () => {
        const before = String(getTrialParam('lane'));
        const currentConfig = getTrialConfig();
        await delay(sleepMs);
        const after = String(getTrialParam('lane'));
        return {
          config,
          sleepMs,
          contextLeak: before !== after || currentConfig.lane !== lane,
        };
      });
    }
  }

  await Promise.all(
    Array.from({ length: concurrency }, () => worker()),
  );

  const wallClockMs = performance.now() - start;
  const rssAfterMb = process.memoryUsage().rss / (1024 * 1024);
  const uniqueConfigs = new Set(results.map((result) => JSON.stringify(result.config)));
  const best = results.reduce((currentBest, result) => {
    if (!currentBest || result.sleepMs < currentBest.sleepMs) {
      return result;
    }
    return currentBest;
  }, null);
  const totalSleepMs = results.reduce((sum, result) => sum + result.sleepMs, 0);

  return {
    wallClockMs,
    throughput: results.length / (wallClockMs / 1000),
    theoreticalMinMs: totalSleepMs / concurrency,
    overheadRatio: wallClockMs / (totalSleepMs / concurrency),
    duplicateConfigRate: 1 - uniqueConfigs.size / Math.max(results.length, 1),
    contextLeakCount: results.filter((result) => result.contextLeak).length,
    rssDeltaMb: Math.max(rssAfterMb - rssBeforeMb, 0),
    bestConfig: best?.config ?? {},
    bestLatency: best?.sleepMs ?? 0,
    stopReason: 'completed',
    trialCount: results.length,
  };
}

async function runJsBenchmark(benchmarkSpec) {
  const results = {
    generatedAt: new Date().toISOString(),
    runtime: 'js',
    nodeVersion: process.version,
    mode: 'standalone_async_scheduler',
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
if (!Array.isArray(benchmarkSpec?.configs) || benchmarkSpec.configs.length === 0) {
  throw new Error('benchmark_spec.configs is required for the JS benchmark harness.');
}

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
