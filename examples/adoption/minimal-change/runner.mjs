#!/usr/bin/env node

import { answerCustomer } from './original-agent.mjs';
import { optimizedSupportAgent } from './optimized-agent.mjs';

console.log('\n=== Before optimization ===');
const before = await answerCustomer('Can I change my flight date?');
console.log(JSON.stringify(before, null, 2));

console.log('\n=== Running optimization ===');
const result = await optimizedSupportAgent.optimize({
  algorithm: 'grid',
  maxTrials: 12,
});
console.log(
  JSON.stringify(
    {
      bestConfig: result.bestConfig,
      bestMetrics: result.bestMetrics,
      stopReason: result.stopReason,
      trialCount: result.trials.length,
    },
    null,
    2
  )
);

console.log('\n=== Applying best config for future normal calls ===');
optimizedSupportAgent.applyBestConfig(result);

const after = await optimizedSupportAgent('Can I change my flight date?');
console.log(JSON.stringify(after, null, 2));
