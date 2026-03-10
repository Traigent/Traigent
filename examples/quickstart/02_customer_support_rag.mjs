#!/usr/bin/env node

import { getTrialParam, optimize, param } from '../../dist/index.js';
import { loadDataset, printSummary } from '../utils/base-example.mjs';

const rows = await loadDataset('rag_questions.jsonl');

const answerWithRag = optimize({
  configurationSpace: {
    model: param.enum(['cheap', 'balanced', 'accurate']),
    retrieval_k: param.int({ min: 1, max: 3, step: 1 }),
    prompt_style: param.enum(['concise', 'grounded']),
  },
  objectives: ['accuracy', 'cost', 'latency'],
  evaluation: {
    data: rows,
    scoringFunction: (output, expectedOutput) =>
      output.text.includes(String(expectedOutput)) ? 1 : 0.48,
    metricFunctions: {
      cost: (output) => output.cost,
      latency: (output) => output.latency,
    },
  },
})(async (rowInput) => {
  const retrievalK = Number(getTrialParam('retrieval_k', 1));
  const grounded = getTrialParam('prompt_style', 'concise') === 'grounded';
  const model = String(getTrialParam('model', 'cheap'));
  const question = String(rowInput.question);

  const responseQuality =
    (model === 'accurate' ? 1 : model === 'balanced' ? 0.8 : 0.65) +
    (retrievalK >= 2 ? 0.12 : 0) +
    (grounded ? 0.08 : 0);

  return {
    text:
      responseQuality >= 0.95
        ? question.includes('seamless')
          ? 'Seamless mode intercepts and overrides hardcoded LLM parameters'
          : question.includes('LangChain')
            ? 'Via adapters that intercept LangChain LLM calls and inject optimized configurations'
            : 'Traigent optimizes AI applications without code changes'
        : responseQuality >= 0.85
          ? 'Traigent can improve your prompts'
          : 'unknown',
    cost: 0.06 + retrievalK * 0.03 + (model === 'accurate' ? 0.14 : 0.05),
    latency: 0.3 + retrievalK * 0.15 + (grounded ? 0.1 : 0),
  };
});

const result = await answerWithRag.optimize({
  algorithm: 'random',
  maxTrials: 10,
  randomSeed: 7,
});

printSummary('quickstart/02_customer_support_rag', result);
