import { fileURLToPath } from "node:url";

import {
  autoWrapFrameworkTarget,
  deleteOptimizationSession,
  finalizeOptimizationSession,
  getOptimizationSessionStatus,
  optimize,
  param,
} from "../../../dist/index.js";

function readConnectionFromEnv() {
  const backendUrl =
    process.env.TRAIGENT_BACKEND_URL ?? process.env.TRAIGENT_API_URL;
  const apiKey = process.env.TRAIGENT_API_KEY;
  return backendUrl && apiKey ? { backendUrl, apiKey } : null;
}

function createAgent() {
  const client = autoWrapFrameworkTarget({
    chat: {
      completions: {
        async create(params) {
          const content = String(params.messages?.[0]?.content ?? "");
          return {
            choices: [
              {
                message: {
                  content:
                    Number(params.temperature ?? 0) >= 0.5
                      ? `${content.toUpperCase()}!`
                      : content.toUpperCase(),
                },
              },
            ],
            usage: {
              prompt_tokens: 2,
              completion_tokens: 1,
            },
            model: params.model,
          };
        },
      },
    },
  });

  return optimize({
    configurationSpace: {
      model: param.enum(["gpt-4o-mini", "gpt-4o"]),
      temperature: param.float({ min: 0, max: 1, step: 0.5 }),
    },
    objectives: ["accuracy", "cost"],
    evaluation: {
      data: [{ input: "hello", output: "HELLO!" }],
      scoringFunction: (output, expectedOutput) =>
        output === expectedOutput ? 1 : 0,
    },
    injection: {
      mode: "seamless",
    },
  })(async (input) => {
    const response = await client.chat.completions.create({
      model: "gpt-3.5-turbo",
      temperature: 0,
      messages: [{ role: "user", content: String(input) }],
    });

    return response.choices[0]?.message?.content ?? "";
  });
}

async function runEnvBasedFlow() {
  const agent = createAgent();
  const frameworkAutoOverride = agent.frameworkAutoOverrideStatus();
  const seamlessResolution = agent.seamlessResolution();
  const result = await agent.optimize({
    mode: "hybrid",
    algorithm: "optuna",
    maxTrials: 1,
  });

  if (!result.sessionId) {
    throw new Error("Hybrid optimize result did not include a sessionId.");
  }

  const status = await getOptimizationSessionStatus(result.sessionId);
  const finalized = await finalizeOptimizationSession(result.sessionId);
  const deleted = await deleteOptimizationSession(result.sessionId);

  return { result, status, finalized, deleted, frameworkAutoOverride, seamlessResolution };
}

async function runExplicitFlow(connection) {
  const agent = createAgent();
  const frameworkAutoOverride = agent.frameworkAutoOverrideStatus();
  const seamlessResolution = agent.seamlessResolution();
  const result = await agent.optimize({
    mode: "hybrid",
    algorithm: "optuna",
    maxTrials: 1,
    ...connection,
  });

  if (!result.sessionId) {
    throw new Error("Hybrid optimize result did not include a sessionId.");
  }

  const status = await getOptimizationSessionStatus(result.sessionId, connection);
  const finalized = await finalizeOptimizationSession(result.sessionId, {
    ...connection,
    includeFullHistory: true,
  });
  const deleted = await deleteOptimizationSession(result.sessionId, {
    ...connection,
    cascade: false,
  });

  return { result, status, finalized, deleted, frameworkAutoOverride, seamlessResolution };
}

export async function runExample() {
  const connection = readConnectionFromEnv();
  if (!connection) {
    console.log(
      [
        "Skipping hybrid-session-control example.",
        "Set TRAIGENT_BACKEND_URL (or TRAIGENT_API_URL) and TRAIGENT_API_KEY to run it.",
      ].join(" "),
    );
    return null;
  }

  const envFlow = await runEnvBasedFlow();
  const explicitFlow = await runExplicitFlow(connection);

  const summary = {
    env: {
      sessionId: envFlow.result.sessionId,
      stopReason: envFlow.result.stopReason,
      reportingKeys: Object.keys(envFlow.result.reporting ?? {}),
      frameworkAutoOverride: envFlow.frameworkAutoOverride,
      seamlessResolution: envFlow.seamlessResolution,
      status: envFlow.status.status,
      finalizeStatus: envFlow.finalized.status,
      deleteSuccess: envFlow.deleted.success,
    },
    explicit: {
      sessionId: explicitFlow.result.sessionId,
      stopReason: explicitFlow.result.stopReason,
      reportingKeys: Object.keys(explicitFlow.finalized.reporting ?? {}),
      frameworkAutoOverride: explicitFlow.frameworkAutoOverride,
      seamlessResolution: explicitFlow.seamlessResolution,
      status: explicitFlow.status.status,
      finalizeStatus: explicitFlow.finalized.status,
      deleteSuccess: explicitFlow.deleted.success,
    },
  };

  console.log(JSON.stringify(summary, null, 2));
  return summary;
}

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  await runExample();
}
