#!/usr/bin/env node

import { fileURLToPath } from 'node:url';

import { getTrialParam, optimize, param } from '../dist/index.js';

const rows = [
  { input: 'What is 2+2?', output: '4' },
  { input: 'What is the capital of France?', output: 'Paris' },
];

const answerQuestion = optimize({
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
    data: rows,
    scoringFunction: (output, expectedOutput) =>
      output === expectedOutput ? 1 : 0,
    metricFunctions: {
      cost: (_output, _expectedOutput, _runtimeMetrics, row) =>
        row.input.includes('capital') ? 0.2 : 0.1,
    },
  },
})(async (question) => {
  const model = String(getTrialParam('model', 'cheap'));
  const temperature = Number(getTrialParam('temperature', 0));

  if (model === 'accurate' && temperature === 0) {
    return question.includes('capital') ? 'Paris' : '4';
  }

  return 'unknown';
});

export async function runExample() {
  return answerQuestion.optimize({
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
