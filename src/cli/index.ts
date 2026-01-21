/**
 * CLI module exports.
 */
export {
  PROTOCOL_VERSION,
  parseRequest,
  createSuccessResponse,
  createErrorResponse,
  serializeResponse,
  type CLIRequest,
  type CLIResponse,
  type Action,
  type ResponseStatus,
  type ErrorPayload,
} from './protocol.js';
