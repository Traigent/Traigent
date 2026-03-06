#!/usr/bin/env node

import { fileURLToPath } from 'node:url';

import { optimize, param } from '../dist/index.js';

export async function runExample() {
  const evaluatePrompt = optimize({
    configurationSpace: {
      model: param.enum(['cheap', 'accurate']),
      temperature: param.float({
        min: 0,
        max: 0.5,
        step: 0.5,
        scale: 'linear',
      }),
    },
    objectives: ['accuracy', 'cost'],
    budget: {
      maxCostUsd: 2,
    },
    evaluation: {
      data: [{ id: 'row-1' }, { id: 'row-2' }],
    },
  })(async (trialConfig) => {
    const model = String(trialConfig.config.model);
    const temperature = Number(trialConfig.config.temperature ?? 0);

    return {
      metrics: {
        accuracy: model === 'accurate' && temperature === 0 ? 0.96 : 0.72,
        cost: model === 'accurate' ? 0.4 : 0.1,
        latency: model === 'accurate' ? 1.2 : 0.6,
      },
      metadata: {
        evaluatedRows: trialConfig.dataset_subset.total,
      },
    };
  });

  return evaluatePrompt.optimize({
    algorithm: 'grid',
    maxTrials: 10,
  });
}

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  const result = await runExample();
  console.log(
    JSON.stringify(
      {
        bestConfig: result.bestConfig,
        bestMetrics: result.bestMetrics,
        stopReason: result.stopReason,
        totalCostUsd: result.totalCostUsd,
        trialCount: result.trials.length,
      },
      null,
      2,
    ),
  );
}
