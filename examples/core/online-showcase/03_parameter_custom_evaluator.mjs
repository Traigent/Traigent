import { fileURLToPath } from "node:url";

import {
  collectSessionHelpers,
  getCachedCompletion,
  createHybridOptions,
  createParameterizedPrompt,
  createWrappedOpenAIClient,
  FIVE_STYLES,
  getDataset,
  optimize,
  param,
  resolveConnection,
  scoreTokenAccuracy,
  summarizeSessionEvidence,
  summarizeProvider,
  summarizeResult,
} from "./shared.mjs";

export const metadata = {
  id: "3",
  title: "Parameter injection + custom evaluator",
  description:
    "Passes config as the second argument, uses weighted objective objects, and evaluates with a customEvaluator context.",
  codePath: fileURLToPath(import.meta.url),
};

export async function runSection() {
  const connection = resolveConnection();
  const { client, provider } = createWrappedOpenAIClient({ wrapper: "auto" });
  const completionCache = new Map();

  const answerToken = optimize({
    configurationSpace: {
      prefix: param.enum(FIVE_STYLES),
    },
    objectives: [
      { metric: "accuracy", direction: "maximize", weight: 0.8 },
      { metric: "cost", direction: "minimize", weight: 0.2 },
    ],
    evaluation: {
      data: getDataset(),
      customEvaluator: ({ output, runtimeMetrics, row }) => ({
        accuracy: scoreTokenAccuracy(output, row),
        cost: runtimeMetrics.total_cost ?? runtimeMetrics.cost ?? 0,
        brevity: String(output ?? "").trim().split(/\s+/).filter(Boolean).length,
      }),
    },
    injection: {
      mode: "parameter",
    },
    defaultConfig: {
      prefix: "token-only",
    },
  })(async (input, config) => {
    const normalizedConfig = config ?? {};
    const prompt = createParameterizedPrompt(input, normalizedConfig);
    const response = await getCachedCompletion(
      completionCache,
      JSON.stringify({ input, config: normalizedConfig }),
      () =>
        client.chat.completions.create({
          model: provider.model,
          temperature: 0.2,
          max_tokens: 18,
          messages: prompt,
        }),
    );

    return response.choices[0]?.message?.content ?? "";
  });

  const result = await answerToken.optimize(createHybridOptions(connection));
  const helpers = await collectSessionHelpers(result.sessionId, connection);

  return summarizeResult(metadata.title, result, {
    provider: summarizeProvider(provider),
    ...summarizeSessionEvidence(helpers),
  });
}
