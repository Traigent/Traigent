import { getTrialParam, optimize, param } from '../../../dist/index.js';

import { answerQuestion as originalAnswerQuestion } from './original-agent.mjs';

export async function answerQuestion(question) {
  const model = getTrialParam('model', 'cheap');
  const temperature = getTrialParam('temperature', 0.2);

  if (model === 'accurate' && temperature === 0) {
    if (question.includes('capital of France')) {
      return 'Paris';
    }

    return 'Helpful answer';
  }

  return originalAnswerQuestion(question);
}

export const optimizedAnswerQuestion = optimize({
  configurationSpace: {
    model: param.enum(['cheap', 'accurate']),
    temperature: param.float({ min: 0, max: 0.2, step: 0.2 }),
  },
  objectives: ['accuracy', 'cost'],
  injection: {
    mode: 'seamless',
  },
  evaluation: {
    data: [
      { input: 'What is the capital of France?', output: 'Paris' },
      { input: 'What is 2+2?', output: 'Helpful answer' },
    ],
    scoringFunction: (output, expectedOutput) => (output === expectedOutput ? 1 : 0),
    metricFunctions: {
      cost: (output) => (output === 'Paris' ? 0.2 : 0.05),
    },
  },
})(answerQuestion);
