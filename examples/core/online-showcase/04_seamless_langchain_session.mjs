import { fileURLToPath } from "node:url";

import {
  collectSessionHelpers,
  createBaseSpec,
  createHybridOptions,
  createTokenOnlyPrompt,
  createWrappedLangChainModel,
  FIVE_MAX_TOKENS,
  optimize,
  param,
  resolveConnection,
  summarizeProvider,
  summarizeResult,
} from "./shared.mjs";

export const metadata = {
  id: "4",
  title: "Seamless LangChain + session helpers",
  description:
    "Uses a wrapped LangChain ChatOpenAI model, seamless override injection, and explicit session status/finalize helper calls.",
  codePath: fileURLToPath(import.meta.url),
};

export async function runSection() {
  const connection = resolveConnection();
  const { model, provider } = createWrappedLangChainModel();

  const answerToken = optimize(
    createBaseSpec({
      configurationSpace: {
        maxTokens: param.enum(FIVE_MAX_TOKENS),
      },
      injection: {
        mode: "seamless",
      },
    }),
  )(async (input) => {
    const response = await model.invoke(createTokenOnlyPrompt(input));
    return typeof response?.content === "string"
      ? response.content
      : Array.isArray(response?.content)
        ? response.content
            .map((part) => (typeof part?.text === "string" ? part.text : ""))
            .join(" ")
        : "";
  });

  const result = await answerToken.optimize(createHybridOptions(connection));
  const helpers = await collectSessionHelpers(result.sessionId, connection);

  return summarizeResult(metadata.title, result, {
    provider: summarizeProvider(provider),
    frameworkAutoOverride: answerToken.frameworkAutoOverrideStatus(),
    seamlessResolution: answerToken.seamlessResolution(),
    status: helpers.status?.status ?? null,
    finalizedStatus: helpers.finalized?.status ?? null,
  });
}
