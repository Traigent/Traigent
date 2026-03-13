import { optimize, param } from '../dist/index.js';

const backendUrl = process.env.TRAIGENT_BACKEND_URL ?? process.env.TRAIGENT_API_URL;
const apiKey = process.env.TRAIGENT_API_KEY;

if (!backendUrl || !apiKey) {
  throw new Error(
    'Set TRAIGENT_BACKEND_URL (or TRAIGENT_API_URL) and TRAIGENT_API_KEY before running the hybrid example.'
  );
}

const runTrial = optimize({
  configurationSpace: {
    model: param.enum(['gpt-4o-mini', 'gpt-4o']),
    temperature: param.float({ min: 0, max: 1, step: 0.2 }),
  },
  objectives: ['accuracy', 'cost'],
  evaluation: {
    data: [{ id: 1 }, { id: 2 }, { id: 3 }],
  },
})(async (trialConfig) => {
  const model = String(trialConfig.config.model);
  const temperature = Number(trialConfig.config.temperature);

  return {
    metrics: {
      accuracy:
        model === 'gpt-4o'
          ? 0.88 - Math.abs(temperature - 0.4) * 0.05
          : 0.78 - Math.abs(temperature - 0.2) * 0.04,
      cost: model === 'gpt-4o' ? 0.35 : 0.08,
    },
    metadata: {
      model,
      temperature,
    },
  };
});

const result = await runTrial.optimize({
  mode: 'hybrid',
  algorithm: 'optuna',
  maxTrials: 8,
  backendUrl,
  apiKey,
  timeoutMs: 5_000,
});

console.log(JSON.stringify(result, null, 2));
