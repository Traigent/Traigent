#!/usr/bin/env node

import { getTrialParam, optimize, param } from '../../dist/index.js';
import { loadDataset, printSummary } from '../utils/base-example.mjs';

const rows = await loadDataset('classification.jsonl');

const classifySentiment = optimize({
  configurationSpace: {
    model: param.enum(['cheap', 'accurate']),
    temperature: param.float({ min: 0, max: 0.8, step: 0.4 }),
  },
  objectives: [
    { metric: 'quality_score', direction: 'maximize', weight: 2 },
    { metric: 'cost', direction: 'minimize', weight: 1 },
    { metric: 'latency', direction: 'minimize', weight: 1 },
  ],
  evaluation: {
    data: rows,
    metricFunctions: {
      quality_score: (output, expectedOutput) =>
        output.label === expectedOutput ? output.quality : 0.55,
      cost: (output) => output.cost,
      latency: (output) => output.latency,
    },
  },
})(async (rowInput) => {
  const model = String(getTrialParam('model', 'cheap'));
  const temperature = Number(getTrialParam('temperature', 0));
  const qualityBase = model === 'accurate' ? 0.94 : 0.7;
  const quality = Math.max(0.5, qualityBase - temperature * 0.1);

  return {
    label: model === 'accurate' || String(rowInput.text).includes('love') ? 'positive' : 'neutral',
    quality,
    cost: model === 'accurate' ? 0.18 : 0.06,
    latency: model === 'accurate' ? 0.9 : 0.35,
  };
});

const result = await classifySentiment.optimize({
  algorithm: 'grid',
  maxTrials: 6,
});

printSummary('quickstart/03_custom_objectives', result);
