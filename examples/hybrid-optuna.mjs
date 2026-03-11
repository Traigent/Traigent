import { fileURLToPath } from "node:url";

import OpenAI from "openai";

import { autoWrapFrameworkTarget, optimize, param } from "../dist/index.js";

export async function runExample() {
  const backendUrl =
    process.env.TRAIGENT_BACKEND_URL ?? process.env.TRAIGENT_API_URL;
  const apiKey = process.env.TRAIGENT_API_KEY;

  if (!backendUrl || !apiKey) {
    throw new Error(
      "Set TRAIGENT_BACKEND_URL (or TRAIGENT_API_URL) and TRAIGENT_API_KEY before running the hybrid example.",
    );
  }

  const client = autoWrapFrameworkTarget(
    new OpenAI({ apiKey: process.env.OPENAI_API_KEY ?? "demo-key" }),
  );

  const answerQuestion = optimize({
    configurationSpace: {
      model: param.enum(["gpt-4o-mini", "gpt-4o"]),
      temperature: param.float({ min: 0, max: 1, step: 0.2 }),
    },
    objectives: ["accuracy", "cost"],
    evaluation: {
      data: [
        { input: "hello", output: "HELLO!" },
        { input: "world", output: "WORLD!" },
      ],
      scoringFunction: (output, expectedOutput) =>
        output === expectedOutput ? 1 : 0,
      metricFunctions: {
        cost: (_output, _expectedOutput, _runtimeMetrics, row) =>
          String((row.input ?? "")).length > 4 ? 0.2 : 0.05,
      },
    },
    injection: {
      mode: "seamless",
    },
  })(async (input) => {
    const response = await client.chat.completions.create({
      model: "gpt-3.5-turbo",
      temperature: 0.9,
      messages: [{ role: "user", content: String(input) }],
    });

    return response.choices[0]?.message?.content ?? "";
  });

  return answerQuestion.optimize({
    mode: "hybrid",
    algorithm: "optuna",
    maxTrials: 8,
    backendUrl,
    apiKey,
    timeoutMs: 5_000,
    includeFullHistory: true,
  });
}

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  const result = await runExample();
  console.log(JSON.stringify(result, null, 2));
}
