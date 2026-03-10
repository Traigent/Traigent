#!/usr/bin/env node

import { optimizedAnswerQuestion } from './transformed-agent.mjs';

const result = await optimizedAnswerQuestion.optimize({
  algorithm: 'grid',
  maxTrials: 4,
});

console.log('seamlessResolution', optimizedAnswerQuestion.seamlessResolution());
console.log('bestConfig', result.bestConfig);
console.log('bestMetrics', result.bestMetrics);

optimizedAnswerQuestion.applyBestConfig(result);
console.log(
  'postApply',
  await optimizedAnswerQuestion('What is the capital of France?'),
);
