import { fileURLToPath } from "node:url";

import {
  collectSessionHelpers,
  createBaseSpec,
  createHybridOptions,
  createTokenOnlyPrompt,
  createWrappedOpenAIClient,
  FIVE_TEMPERATURES,
  optimize,
  param,
  resolveConnection,
  summarizeProvider,
  summarizeResult,
} from "./shared.mjs";

export const metadata = {
  id: "1",
  title: "Seamless OpenAI + env auth",
  description:
    "Backend-guided Optuna with a wrapped OpenAI-compatible client, seamless injection, env-based Traigent auth, and session helper evidence.",
  codePath: fileURLToPath(import.meta.url),
};

export async function runSection() {
  resolveConnection();
  const { client, provider } = createWrappedOpenAIClient({ wrapper: "auto" });

  const answerToken = optimize(
    createBaseSpec({
      configurationSpace: {
        temperature: param.enum(FIVE_TEMPERATURES),
      },
      injection: {
        mode: "seamless",
      },
    }),
  )(async (input) => {
    const response = await client.chat.completions.create({
      model: provider.model,
      temperature: 0.9,
      max_tokens: 24,
      messages: createTokenOnlyPrompt(input),
    });

    return response.choices[0]?.message?.content ?? "";
  });

  const result = await answerToken.optimize(createHybridOptions(undefined));
  const helpers = await collectSessionHelpers(result.sessionId, undefined);

  return summarizeResult(metadata.title, result, {
    provider: summarizeProvider(provider),
    frameworkAutoOverride: answerToken.frameworkAutoOverrideStatus(),
    seamlessResolution: answerToken.seamlessResolution(),
    status: helpers.status?.status ?? null,
    finalizedStatus: helpers.finalized?.status ?? null,
    deleteAttempted: helpers.deleted.attempted,
  });
}
