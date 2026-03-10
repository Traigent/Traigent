#!/usr/bin/env node

import { getTrialParam, optimize, param } from '../../dist/index.js';

// This is the application / agent code.
async function answerQuestion(question) {
  // This is where config injection becomes visible inside app code.
  const model = getTrialParam('model', 'cheap');
  const tone = getTrialParam('tone', 'concise');
  return `${tone}:${model}:${question}`;
}

// This is where the tunable variables are declared.
const answerQuestionOptimized = optimize({
  configurationSpace: {
    model: param.enum(['cheap', 'accurate']),
    tone: param.enum(['concise', 'helpful']),
  },
  objectives: ['accuracy', 'cost'],
  evaluation: {
    data: [{ input: 'What is 2+2?', output: 'helpful:accurate:What is 2+2?' }],
    scoringFunction: (output, expectedOutput) =>
      output === expectedOutput ? 1 : 0,
    metricFunctions: {
      cost: () => (getTrialParam('model') === 'accurate' ? 0.2 : 0.05),
    },
  },
})(answerQuestion);

// The SDK owns the search loop from here.
const result = await answerQuestionOptimized.optimize({
  algorithm: 'grid',
  maxTrials: 4,
});

console.log(
  JSON.stringify(
    {
      explanation: {
        tvarsDeclaredIn: 'optimize({ configurationSpace: ... })',
        agentCodeIn: 'answerQuestion()',
        injectionReadVia: 'getTrialParam() inside the original agent function',
        evaluationOwnedBy: 'the SDK via evaluation.scoringFunction + metricFunctions',
      },
      bestConfig: result.bestConfig,
      bestMetrics: result.bestMetrics,
      trials: result.trials.map((trial) => ({
        trialNumber: trial.trialNumber,
        config: trial.config,
        metrics: trial.metrics,
      })),
    },
    null,
    2,
  ),
);
