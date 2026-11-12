import type { Agent, FunctionTool, RunContext } from "@openai/agents";
import { begin, isCaptureActive, jsonValue, report } from "@sovara/sdk/integration";
import type { JsonValue, ToolCall } from "@sovara/sdk";
import { warnOnce } from "./log.js";

const AGENT_PATCHED_SYMBOL = Symbol.for("sovara.openAIAgentsAgentPatched");
const TOOL_PATCHED_SYMBOL = Symbol.for("sovara.openAIAgentsFunctionToolPatched");

type AnyFunctionTool = FunctionTool<any, any, any> & {
  [TOOL_PATCHED_SYMBOL]?: true;
};

type AgentWithTools = Agent<any, any> & {
  tools: unknown[];
};

type AgentPrototype = {
  getAllTools: (this: AgentWithTools, runContext: RunContext<any>) => Promise<unknown[]>;
  [AGENT_PATCHED_SYMBOL]?: true;
};

type AgentConstructor = {
  prototype: AgentPrototype;
};

let installPromise: Promise<void> | undefined;

/** The tool call for a raw JSON argument string; non-JSON input is kept in metadata. */
function toolCall(name: string, input: string, callId: string | undefined): ToolCall {
  const call: ToolCall = { kind: "tool", name, call_id: callId?.trim() || null };
  if (!input) {
    call.arguments = {};
    return call;
  }
  let parsed: JsonValue;
  try {
    parsed = jsonValue(JSON.parse(input));
  } catch {
    call.metadata = { tool_arguments_raw: input };
    return call;
  }
  call.arguments = parsed;
  return call;
}

function patchFunctionTool(tool: unknown): void {
  if (
    !tool
    || typeof tool !== "object"
    || (tool as { type?: unknown }).type !== "function"
    || typeof (tool as { invoke?: unknown }).invoke !== "function"
  ) {
    return;
  }
  const functionTool = tool as AnyFunctionTool;
  if (functionTool[TOOL_PATCHED_SYMBOL]) return;

  const originalInvoke = functionTool.invoke;
  functionTool.invoke = async function sovaraPatchedFunctionTool(runContext, input, details) {
    if (!isCaptureActive()) return originalInvoke.call(this, runContext, input, details);
    const step = await begin(toolCall(functionTool.name, input, details?.toolCall?.callId));
    let result: unknown;
    try {
      result = await originalInvoke.call(this, runContext, input, details);
    } catch (error) {
      await report(step, "fail", [error], { propagating: true });
      throw error;
    }
    await report(step, "finish", [jsonValue(result)]);
    return result;
  };
  Object.defineProperty(functionTool, TOOL_PATCHED_SYMBOL, { value: true });
}

function patchAgentConstructor(AgentClass: AgentConstructor): void {
  const prototype = AgentClass.prototype;
  if (prototype[AGENT_PATCHED_SYMBOL]) return;
  const originalGetAllTools = prototype.getAllTools;
  if (typeof originalGetAllTools !== "function") {
    throw new TypeError("OpenAI Agents Agent.getAllTools() is unavailable.");
  }
  prototype.getAllTools = async function sovaraPatchedGetAllTools(runContext) {
    for (const tool of this.tools) patchFunctionTool(tool);
    return originalGetAllTools.call(this, runContext);
  };
  Object.defineProperty(prototype, AGENT_PATCHED_SYMBOL, { value: true });
}

export function ensureOpenAIAgentsPatchInstalled(): Promise<void> {
  installPromise ??= import("@openai/agents")
    .then((module) => {
      patchAgentConstructor(module.Agent as unknown as AgentConstructor);
    })
    .catch((error: unknown) => {
      const code = error && typeof error === "object" ? (error as { code?: unknown }).code : undefined;
      if (code === "ERR_MODULE_NOT_FOUND" || code === "MODULE_NOT_FOUND") return;
      warnOnce("openai-agents-patch", "[sovara] Failed to install OpenAI Agents tool patch.");
    });
  return installPromise;
}
