import { fileURLToPath } from "node:url";

import {
  checkOptimizationServiceStatus,
  deleteOptimizationSession,
  finalizeOptimizationSession,
  getOptimizationSessionStatus,
  listOptimizationSessions,
  optimize,
  param,
  prepareFrameworkTargets,
} from "../../../dist/index.js";

function readConnectionFromEnv() {
  const backendUrl =
    process.env.TRAIGENT_BACKEND_URL ?? process.env.TRAIGENT_API_URL;
  const apiKey = process.env.TRAIGENT_API_KEY;
  return backendUrl && apiKey ? { backendUrl, apiKey } : null;
}

function createAgent() {
  const runtime = {
    providers: {
      primary: {
        client: {
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
        },
      },
    },
  };
  const preparedTargets = prepareFrameworkTargets(runtime);
  const wrappedRuntime = preparedTargets.wrapped;
  const client = wrappedRuntime.providers.primary.client;

  const agent = optimize({
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

  return {
    agent,
    discoveredTargets: preparedTargets.discovered,
    preparedAutoOverride: preparedTargets.autoOverrideStatus,
  };
}

async function runEnvBasedFlow() {
  const { agent, discoveredTargets, preparedAutoOverride } = createAgent();
  const frameworkAutoOverride = agent.frameworkAutoOverrideStatus();
  const seamlessResolution = agent.seamlessResolution();
  const service = await checkOptimizationServiceStatus();
  const result = await agent.optimize({
    mode: "hybrid",
    algorithm: "optuna",
    maxTrials: 1,
  });

  if (!result.sessionId) {
    throw new Error("Hybrid optimize result did not include a sessionId.");
  }

  const status = await getOptimizationSessionStatus(result.sessionId);
  const listed = await listOptimizationSessions({
    pattern: result.sessionId,
  });
  const finalized = await finalizeOptimizationSession(result.sessionId);
  const deleted = await deleteOptimizationSession(result.sessionId);

  return {
    result,
    service,
    status,
    listed,
    finalized,
    deleted,
    frameworkAutoOverride,
    seamlessResolution,
    discoveredTargets,
    preparedAutoOverride,
  };
}

async function runExplicitFlow(connection) {
  const { agent, discoveredTargets, preparedAutoOverride } = createAgent();
  const frameworkAutoOverride = agent.frameworkAutoOverrideStatus();
  const seamlessResolution = agent.seamlessResolution();
  const service = await checkOptimizationServiceStatus(connection);
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
  const listed = await listOptimizationSessions({
    ...connection,
    pattern: result.sessionId,
  });
  const finalized = await finalizeOptimizationSession(result.sessionId, {
    ...connection,
    includeFullHistory: true,
  });
  const deleted = await deleteOptimizationSession(result.sessionId, {
    ...connection,
    cascade: false,
  });

  return {
    result,
    service,
    status,
    listed,
    finalized,
    deleted,
    frameworkAutoOverride,
    seamlessResolution,
    discoveredTargets,
    preparedAutoOverride,
  };
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
      preparedAutoOverride: envFlow.preparedAutoOverride,
      seamlessResolution: envFlow.seamlessResolution,
      discoveredTargets: envFlow.discoveredTargets,
      serviceStatus: envFlow.service.status,
      status: envFlow.status.status,
      functionName: envFlow.status.functionName,
      datasetSize: envFlow.status.datasetSize,
      experimentId: envFlow.status.experimentId,
      experimentRunId: envFlow.status.experimentRunId,
      listedCount: envFlow.listed.sessions.length,
      finalizeStatus: envFlow.finalized.status,
      deleteSuccess: envFlow.deleted.success,
    },
    explicit: {
      sessionId: explicitFlow.result.sessionId,
      stopReason: explicitFlow.result.stopReason,
      reportingKeys: Object.keys(explicitFlow.finalized.reporting ?? {}),
      frameworkAutoOverride: explicitFlow.frameworkAutoOverride,
      preparedAutoOverride: explicitFlow.preparedAutoOverride,
      seamlessResolution: explicitFlow.seamlessResolution,
      discoveredTargets: explicitFlow.discoveredTargets,
      serviceStatus: explicitFlow.service.status,
      status: explicitFlow.status.status,
      functionName: explicitFlow.status.functionName,
      datasetSize: explicitFlow.status.datasetSize,
      experimentId: explicitFlow.status.experimentId,
      experimentRunId: explicitFlow.status.experimentRunId,
      listedCount: explicitFlow.listed.sessions.length,
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
