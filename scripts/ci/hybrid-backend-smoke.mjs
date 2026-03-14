#!/usr/bin/env node

import {
  createOptimizationSession,
  deleteOptimizationSession,
  finalizeOptimizationSession,
  getNextOptimizationTrial,
  getOptimizationSessionStatus,
  listOptimizationSessions,
  param,
  submitOptimizationTrialResult,
} from '../../dist/index.js';

function resolveConnection() {
  const backendUrl = process.env['TRAIGENT_API_URL'];
  const apiKey = process.env['TRAIGENT_API_KEY'];

  if (!backendUrl) {
    throw new Error('TRAIGENT_API_URL is required for hybrid smoke.');
  }
  if (!apiKey) {
    throw new Error('TRAIGENT_API_KEY is required for hybrid smoke.');
  }

  return { backendUrl, apiKey };
}

async function cleanupStaleSessions(connection) {
  const listed = await listOptimizationSessions(connection);
  const staleSessions = (listed.sessions ?? []).filter(
    (session) => session.functionName === 'hybrid_smoke_agent'
  );

  for (const session of staleSessions) {
    await deleteOptimizationSession(session.sessionId, { ...connection, cascade: true }).catch(
      () => {}
    );
  }
}

async function main() {
  const connection = resolveConnection();
  await cleanupStaleSessions(connection);

  const created = await createOptimizationSession(
    {
      functionName: 'hybrid_smoke_agent',
      configurationSpace: {
        model: param.enum(['smoke-a', 'smoke-b']),
        temperature: param.float({ min: 0, max: 1, step: 0.5 }),
      },
      objectives: ['accuracy'],
      datasetMetadata: { size: 1 },
      maxTrials: 1,
    },
    connection
  );

  const sessionId = created.sessionId;
  if (!sessionId) {
    throw new Error('Hybrid smoke failed: sessionId missing from create response.');
  }

  try {
    const listed = await listOptimizationSessions({ ...connection, pattern: sessionId });
    const next = await getNextOptimizationTrial(sessionId, connection);

    if (!next.suggestion) {
      throw new Error('Hybrid smoke failed: expected a suggestion from next-trial.');
    }

    await submitOptimizationTrialResult(
      sessionId,
      {
        trialId: next.suggestion.trialId,
        metrics: {
          accuracy: 0.75,
          cost: 0.01,
        },
        duration: 0.01,
      },
      connection
    );

    const finalized = await finalizeOptimizationSession(sessionId, {
      ...connection,
      includeFullHistory: false,
    });
    const status = await getOptimizationSessionStatus(sessionId, connection);

    console.log(
      JSON.stringify(
        {
          sessionId,
          listedTotal: listed.total,
          nextShouldContinue: next.shouldContinue,
          finalizedStatus: finalized.stopReason ?? finalized.status,
          status: status.status,
          progress: status.progress,
        },
        null,
        2
      )
    );
  } finally {
    await deleteOptimizationSession(sessionId, { ...connection, cascade: true });
  }
}

main().catch((error) => {
  console.error(error instanceof Error ? (error.stack ?? error.message) : String(error));
  process.exit(1);
});
