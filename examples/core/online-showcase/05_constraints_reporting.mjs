import { fileURLToPath } from "node:url";

import {
  collectSessionHelpers,
  getCachedCompletion,
  createHybridOptions,
  createWrappedOpenAIClient,
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
  id: "5",
  title: "Conditional params + constraints + reporting",
  description:
    "Shows hybrid-only conditional parameters, structural constraints, full-history reporting, and bounded control-plane cleanup.",
  codePath: fileURLToPath(import.meta.url),
};

export async function runSection() {
  const connection = resolveConnection();
  const { client, provider } = createWrappedOpenAIClient({ wrapper: "auto" });
  const completionCache = new Map();

  const answerToken = optimize({
    configurationSpace: {
      style: param.enum(["strict", "careful", "minimal", "concise", "literal"]),
      maxTokens: param.int({
        min: 8,
        max: 24,
        step: 4,
        conditions: { style: "careful" },
        default: 8,
      }),
    },
    objectives: [
      { metric: "accuracy", direction: "maximize", weight: 0.8 },
      { metric: "cost", direction: "minimize", weight: 0.2 },
    ],
    constraints: {
      structural: [
        {
          when: 'params.style == "careful"',
          then: "params.maxTokens >= 8",
        },
      ],
    },
    evaluation: {
      data: getDataset(),
      scoringFunction: (output, _expected, _runtime, row) =>
        scoreTokenAccuracy(output, row),
      metricFunctions: {
        cost: (_output, _expectedOutput, runtimeMetrics) =>
          runtimeMetrics.total_cost ?? runtimeMetrics.cost ?? 0,
        total_cost: (_output, _expectedOutput, runtimeMetrics) =>
          runtimeMetrics.total_cost ?? runtimeMetrics.cost ?? 0,
      },
    },
    injection: {
      mode: "context",
    },
  })(async (input) => {
    const prompt = [
      {
        role: "system",
        content:
          'If the request says "Reply with exactly this uppercase token", return only that token.',
      },
      {
        role: "user",
        content: String(input),
      },
    ];
    const response = await getCachedCompletion(
      completionCache,
      JSON.stringify({ input, prompt }),
      () =>
        client.chat.completions.create({
          model: provider.model,
          temperature: 0.2,
          max_tokens: 20,
          messages: prompt,
        }),
    );

    return response.choices[0]?.message?.content ?? "";
  });

  const result = await answerToken.optimize(
    createHybridOptions(connection, {
      includeFullHistory: true,
    }),
  );
  const helpers = await collectSessionHelpers(result.sessionId, connection);

  return summarizeResult(metadata.title, result, {
    provider: summarizeProvider(provider),
    finalizeFullHistoryCount:
      helpers.finalized?.reporting?.fullHistory?.length ?? 0,
    ...summarizeSessionEvidence(helpers),
  });
}
