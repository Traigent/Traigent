import { fileURLToPath } from "node:url";

import {
  collectSessionHelpers,
  createBaseSpec,
  createContextStylePrompt,
  createHybridOptions,
  createWrappedOpenAIClient,
  FIVE_STYLES,
  optimize,
  param,
  resolveConnection,
  summarizeProvider,
  summarizeResult,
} from "./shared.mjs";

export const metadata = {
  id: "2",
  title: "Context injection + explicit options",
  description:
    "Uses getTrialParam() inside the agent prompt, explicit backend options, and the manual OpenAI wrapper path.",
  codePath: fileURLToPath(import.meta.url),
};

export async function runSection() {
  const connection = resolveConnection();
  const { client, provider } = createWrappedOpenAIClient({ wrapper: "manual" });

  const answerToken = optimize(
    createBaseSpec({
      configurationSpace: {
        style: param.enum(FIVE_STYLES),
      },
      injection: {
        mode: "context",
      },
    }),
  )(async (input) => {
    const response = await client.chat.completions.create({
      model: provider.model,
      temperature: 0.1,
      max_tokens: 20,
      messages: createContextStylePrompt(input),
    });

    return response.choices[0]?.message?.content ?? "";
  });

  const result = await answerToken.optimize(createHybridOptions(connection));
  const helpers = await collectSessionHelpers(result.sessionId, connection);

  return summarizeResult(metadata.title, result, {
    provider: summarizeProvider(provider),
    status: helpers.status?.status ?? null,
    finalizedStatus: helpers.finalized?.status ?? null,
    explicitOptions: true,
  });
}
