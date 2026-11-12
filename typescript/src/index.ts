import { ensureOpenAIAgentsPatchInstalled } from "./patch.js";

/**
 * Record OpenAI Agents function tools inside Sovara traces; idempotent.
 *
 * Await it once at startup, before creating agents. Model calls are recorded by
 * `autoInstrument()` from `@sovara/sdk` through the HTTP client.
 */
export function instrumentOpenAIAgents(): Promise<void> {
  return ensureOpenAIAgentsPatchInstalled();
}
