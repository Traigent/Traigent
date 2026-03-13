#!/usr/bin/env node

import { getTrialParam, optimize, param } from '../../dist/index.js';
import { exactMatchScore, loadDataset, printSummary } from '../utils/base-example.mjs';

const rows = await loadDataset('simple_questions.jsonl');

const answerQuestion = optimize({
  configurationSpace: {
    model: param.enum(['cheap', 'balanced', 'accurate']),
    temperature: param.float({ min: 0, max: 0.4, step: 0.2 }),
  },
  objectives: ['accuracy', 'cost'],
  evaluation: {
    data: rows,
    scoringFunction: (output, expectedOutput) => exactMatchScore(output, expectedOutput),
    metricFunctions: {
      cost: (_output, _expectedOutput, _runtimeMetrics, row) =>
        row.input.question.includes('capital') ? 0.12 : 0.05,
    },
  },
})(async (rowInput) => {
  const model = String(getTrialParam('model', 'cheap'));
  const question = String(rowInput.question);

  if (model === 'accurate') {
    return question.includes('capital') ? 'Paris' : '4';
  }

  if (model === 'balanced' && question.includes('2+2')) {
    return '4';
  }

  return 'unknown';
});

const result = await answerQuestion.optimize({
  algorithm: 'grid',
  maxTrials: 9,
});

printSummary('quickstart/01_simple_qa', result);
