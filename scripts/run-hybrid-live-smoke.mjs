import assert from 'node:assert/strict';
import process from 'node:process';

import {
  getTrialConfig,
  getTrialParam,
  optimize,
  param,
} from '../dist/index.js';

const backendUrl =
  process.env.TRAIGENT_BACKEND_URL ?? process.env.TRAIGENT_API_URL;
const apiKey = process.env.TRAIGENT_API_KEY;
const maxTrials = Number.parseInt(
  process.env.TRAIGENT_HYBRID_SMOKE_MAX_TRIALS ?? '4',
  10,
);

if (!backendUrl || !apiKey) {
  throw new Error(
    'Set TRAIGENT_BACKEND_URL (or TRAIGENT_API_URL) and TRAIGENT_API_KEY before running the hybrid live smoke test.',
  );
}

if (!Number.isInteger(maxTrials) || maxTrials <= 0) {
  throw new Error('TRAIGENT_HYBRID_SMOKE_MAX_TRIALS must be a positive integer.');
}

const dataset = [{ id: 1 }, { id: 2 }, { id: 3 }, { id: 4 }];

const runTrial = optimize({
  configurationSpace: {
    model: param.enum(['gpt-4o-mini', 'gpt-4o']),
    temperature: param.float({ min: 0, max: 1, step: 0.2 }),
  },
  objectives: ['accuracy', 'cost'],
  evaluation: {
    data: dataset,
  },
})(async (trialConfig) => {
  const currentConfig = getTrialConfig();
  const model = String(getTrialParam('model'));
  const temperature = Number(getTrialParam('temperature'));

  assert.deepEqual(currentConfig, trialConfig.config);
  assert.equal(trialConfig.dataset_subset.total, dataset.length);
  assert.ok(trialConfig.dataset_subset.indices.length > 0);

  for (const index of trialConfig.dataset_subset.indices) {
    assert.ok(Number.isInteger(index));
    assert.ok(index >= 0 && index < dataset.length);
  }

  const accuracy =
    model === 'gpt-4o'
      ? 0.9 - Math.abs(temperature - 0.4) * 0.05
      : 0.82 - Math.abs(temperature - 0.2) * 0.04;
  const cost = model === 'gpt-4o' ? 0.35 : 0.08;

  return {
    metrics: {
      accuracy,
      cost,
    },
    metadata: {
      model,
      temperature,
      observedSubsetSize: trialConfig.dataset_subset.indices.length,
    },
  };
});

const startedAt = Date.now();
const result = await runTrial.optimize({
  mode: 'hybrid',
  algorithm: 'optuna',
  maxTrials,
  backendUrl,
  apiKey,
  timeoutMs: 10_000,
  requestTimeoutMs: 30_000,
  datasetMetadata: {
    suite: 'js-hybrid-live-smoke',
  },
});

if (['error', 'timeout', 'cancelled'].includes(result.stopReason)) {
  throw new Error(
    `Hybrid live smoke failed with stopReason=${result.stopReason}: ${result.errorMessage ?? 'no backend error message provided'}`,
  );
}

assert.equal(result.mode, 'hybrid');
assert.ok(
  result.sessionId,
  `Expected a sessionId from the backend. stopReason=${result.stopReason}`,
);
assert.ok(result.trials.length > 0);
assert.ok(result.bestConfig && typeof result.bestConfig === 'object');
assert.ok(result.bestMetrics && typeof result.bestMetrics === 'object');

console.log(
  JSON.stringify(
    {
      sessionId: result.sessionId,
      stopReason: result.stopReason,
      trialCount: result.trials.length,
      bestConfig: result.bestConfig,
      bestMetrics: result.bestMetrics,
      durationMs: Date.now() - startedAt,
      metadata: result.metadata,
    },
    null,
    2,
  ),
);
