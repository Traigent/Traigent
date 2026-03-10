#!/usr/bin/env node

import {
  autoWrapFrameworkTargets,
  describeFrameworkAutoOverride,
  optimize,
  param,
} from '../../../dist/index.js';

const wrapped = autoWrapFrameworkTargets({
  chatModel: {
    modelName: 'gpt-3.5-turbo',
    bind(config) {
      return {
        async invoke(input) {
          return `${config.model}:${config.temperature}:${input}`;
        },
      };
    },
    async invoke(input) {
      return `fallback:${input}`;
    },
  },
});

const askQuestion = optimize({
  configurationSpace: {
    model: param.enum(['gpt-4o-mini', 'gpt-4o']),
    temperature: param.float({ min: 0, max: 0.4, step: 0.2, scale: 'linear' }),
  },
  objectives: ['accuracy'],
  injection: {
    mode: 'seamless',
  },
  evaluation: {
    data: [
      {
        input: 'hello',
        output: 'gpt-4o:0.2:hello',
      },
    ],
    scoringFunction: (output, expectedOutput) =>
      output === expectedOutput ? 1 : 0,
  },
})(async (input) => wrapped.chatModel.invoke(input));

const result = await askQuestion.optimize({
  algorithm: 'grid',
  maxTrials: 6,
});

askQuestion.applyBestConfig(result);

console.log(
  JSON.stringify(
    {
      frameworkAutoOverride: describeFrameworkAutoOverride(undefined, true),
      seamlessResolution: askQuestion.seamlessResolution(),
      bestConfig: result.bestConfig,
      bestMetrics: result.bestMetrics,
      appliedOutput: await askQuestion('hello'),
    },
    null,
    2,
  ),
);
