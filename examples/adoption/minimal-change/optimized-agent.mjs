#!/usr/bin/env node

import { optimize, param } from '../../../dist/index.js';
import { answerCustomer } from './original-agent.mjs';
import { exactMatchScore, loadDataset } from '../../utils/base-example.mjs';

const evaluationRows = (await loadDataset('simple_questions_10.jsonl')).map((row) => ({
  input: row.input.question,
  output: row.output,
}));

function inferAnswerQuality(question, config) {
  if (config.model === 'accurate') {
    if (question.includes('2+2')) return '4';
    if (question.toLowerCase().includes('capital of france')) return 'Paris';
    return 'Helpful answer';
  }

  if (config.model === 'balanced') {
    if (question.includes('2+2')) return '4';
    return 'Maybe';
  }

  return 'unknown';
}

// Minimal JS equivalent of the Python decorator:
// wrap the existing agent function directly.
export const optimizedSupportAgent = optimize({
  configurationSpace: {
    model: param.enum(['cheap', 'balanced', 'accurate']),
    tone: param.enum(['concise', 'helpful']),
    maxTokens: param.int({ min: 64, max: 192, step: 64 }),
  },
  objectives: ['accuracy', 'cost'],
  injection: {
    mode: 'parameter',
  },
  evaluation: {
    data: evaluationRows,
    scoringFunction: (output, expectedOutput) =>
      exactMatchScore(output.text, expectedOutput),
    metricFunctions: {
      cost: (output) =>
        output.configUsed.model === 'accurate'
          ? 0.18
          : output.configUsed.model === 'balanced'
            ? 0.09
            : 0.03,
      latency: (output) =>
        output.configUsed.model === 'accurate'
          ? 1.0
          : output.configUsed.model === 'balanced'
            ? 0.6
            : 0.25,
    },
  },
})(async (question, runtimeConfig = {}) => {
  const reply = await answerCustomer(question, runtimeConfig);
  return {
    ...reply,
    text: inferAnswerQuality(question, reply.configUsed),
  };
});

if (import.meta.url === `file://${process.argv[1]}`) {
  const result = await optimizedSupportAgent.optimize({
    algorithm: 'grid',
    maxTrials: 12,
  });
  console.log(JSON.stringify(result, null, 2));
}
