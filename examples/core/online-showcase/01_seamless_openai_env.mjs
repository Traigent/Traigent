import { fileURLToPath } from 'node:url';

import {
  collectSessionHelpers,
  createBaseSpec,
  getCachedCompletion,
  createHybridOptions,
  createTokenOnlyPrompt,
  createWrappedOpenAIClient,
  FIVE_TEMPERATURES,
  getTrialParam,
  optimize,
  param,
  resolveConnection,
  summarizeSessionEvidence,
  summarizeProvider,
  summarizeResult,
} from './shared.mjs';

export const metadata = {
  id: '1',
  title: 'Seamless OpenAI + env auth',
  description:
    'Backend-guided Optuna with a wrapped OpenAI-compatible client, seamless injection, env-based Traigent auth, and session helper evidence.',
  codePath: fileURLToPath(import.meta.url),
};

export async function runSection() {
  resolveConnection();
  const { client, provider } = createWrappedOpenAIClient({ wrapper: 'auto' });
  const completionCache = new Map();

  const answerToken = optimize(
    createBaseSpec({
      configurationSpace: {
        temperature: param.enum(FIVE_TEMPERATURES),
      },
      injection: {
        mode: 'seamless',
      },
    })
  )(async (input) => {
    const temperature = getTrialParam('temperature', 0.9);
    const response = await getCachedCompletion(
      completionCache,
      JSON.stringify({ temperature, input }),
      () =>
        client.chat.completions.create({
          model: provider.model,
          // Intentionally hard-coded: the seamless wrapper should override this
          // from the active trial config at runtime.
          temperature: 0.9,
          max_tokens: 24,
          messages: createTokenOnlyPrompt(input),
        })
    );

    return response.choices[0]?.message?.content ?? '';
  });

  const result = await answerToken.optimize(createHybridOptions(undefined));
  const helpers = await collectSessionHelpers(result.sessionId, undefined);

  return summarizeResult(metadata.title, result, {
    provider: summarizeProvider(provider),
    frameworkAutoOverride: answerToken.frameworkAutoOverrideStatus(),
    seamlessResolution: answerToken.seamlessResolution(),
    deleteAttempted: helpers.deleted.attempted,
    ...summarizeSessionEvidence(helpers),
  });
}
