#!/usr/bin/env node

// This is plain client code with no Traigent dependency.

const DEFAULT_RUNTIME_CONFIG = {
  model: 'balanced',
  tone: 'helpful',
  maxTokens: 120,
};

function buildAnswer(question, config) {
  const prefix = config.tone === 'helpful' ? 'Happy to help.' : 'Short answer.';
  const quality =
    config.model === 'accurate'
      ? 'Detailed and correct'
      : config.model === 'balanced'
        ? 'Mostly correct'
        : 'Cheap guess';

  return `${prefix} ${quality}: ${question}`;
}

export async function answerCustomer(question, runtimeConfig = {}) {
  const config = {
    ...DEFAULT_RUNTIME_CONFIG,
    ...runtimeConfig,
  };

  return {
    text: buildAnswer(question, config),
    configUsed: config,
  };
}

if (import.meta.url === `file://${process.argv[1]}`) {
  const result = await answerCustomer('Can I change my flight date?');
  console.log(JSON.stringify(result, null, 2));
}
