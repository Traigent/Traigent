/**
 * LangChain.js integration for Traigent SDK.
 *
 * @example
 * ```typescript
 * import { TraigentHandler } from '@traigent/sdk/langchain';
 * import { ChatOpenAI } from '@langchain/openai';
 *
 * const handler = new TraigentHandler();
 * const llm = new ChatOpenAI({ callbacks: [handler] });
 *
 * await llm.invoke("Hello!");
 * console.log(handler.toMeasuresDict());
 * ```
 */
export {
  TraigentHandler,
  type LLMCallMetrics,
  type TraigentHandlerMetrics,
} from './handler.js';
export { withTraigentModel } from './model.js';
