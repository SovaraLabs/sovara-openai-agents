import { Agent, RunContext, tool } from "@openai/agents";
import { beforeAll, describe, expect, it, vi } from "vitest";
import { z } from "zod";
import { ensureOpenAIAgentsPatchInstalled } from "../src/patch.js";
import { inline, traced } from "./support/traced.js";

function toolCallDetails(name: string, input: Record<string, unknown>, callId = "call-1") {
  return {
    toolCall: {
      type: "function_call",
      name,
      callId,
      arguments: JSON.stringify(input),
    },
  } as const;
}

function weatherTool(execute: (input: { city: string }) => unknown) {
  return tool({
    name: "get_weather",
    description: "Get the weather for a city.",
    parameters: z.object({ city: z.string() }),
    execute,
  });
}

function weatherAgent(...tools: ReturnType<typeof weatherTool>[]) {
  return new Agent({ name: "Weather", instructions: "Use the tool.", tools });
}

beforeAll(() => ensureOpenAIAgentsPatchInstalled());

describe("OpenAI Agents function-tool patch", () => {
  it("records configured function tools as tool steps", async () => {
    const execute = vi.fn(({ city }: { city: string }) => `${city}: 22 C`);
    const configuredTool = weatherTool(execute);
    const agent = weatherAgent(configuredTool);
    const details = toolCallDetails("get_weather", { city: "Zurich" });

    const { result, calls, finalizes } = await traced(async () => {
      await agent.getAllTools(new RunContext());
      return configuredTool.invoke(new RunContext(), details.toolCall.arguments, details);
    });

    expect(result).toBe("Zurich: 22 C");
    expect(execute).toHaveBeenCalledWith({ city: "Zurich" }, expect.any(RunContext), expect.anything());
    expect(calls).toEqual([{ kind: "tool", name: "get_weather", call_id: "call-1", arguments: { city: "Zurich" } }]);
    expect(inline(finalizes[0]!, "outcome")).toEqual({ status: "success", output: "Zurich: 22 C" });
  });

  it("keeps arguments that are not JSON as raw metadata", async () => {
    const configuredTool = weatherTool(({ city }) => city);
    const agent = weatherAgent(configuredTool);

    const { calls } = await traced(async () => {
      await agent.getAllTools(new RunContext());
      await configuredTool.invoke(new RunContext(), "{not json", toolCallDetails("get_weather", {})).catch(() => null);
    });

    expect(calls).toEqual([{
      kind: "tool",
      name: "get_weather",
      call_id: "call-1",
      metadata: { tool_arguments_raw: "{not json" },
    }]);
  });

  it("records and rethrows function-tool errors", async () => {
    const error = new Error("weather service unavailable");
    const configuredTool = tool({
      name: "get_weather",
      description: "Get the weather for a city.",
      parameters: z.object({ city: z.string() }),
      execute: () => {
        throw error;
      },
      errorFunction: null,
    });
    const agent = new Agent({ name: "Weather", instructions: "Use the tool.", tools: [configuredTool] });
    const details = toolCallDetails("get_weather", { city: "Zurich" });

    let thrown: unknown;
    const { finalizes } = await traced(async () => {
      await agent.getAllTools(new RunContext());
      thrown = await configuredTool.invoke(new RunContext(), details.toolCall.arguments, details)
        .catch((caught: unknown) => caught);
    });

    expect(thrown).toBe(error);
    expect(inline(finalizes[0]!, "outcome")).toEqual({
      status: "error",
      error: { type: "Error", message: "weather service unavailable" },
    });
  });

  it("patches each configured tool only once", async () => {
    const configuredTool = weatherTool(({ city }) => city);
    const agent = weatherAgent(configuredTool);
    const details = toolCallDetails("get_weather", { city: "Zurich" });

    const { calls } = await traced(async () => {
      await agent.getAllTools(new RunContext());
      await agent.getAllTools(new RunContext());
      await configuredTool.invoke(new RunContext(), details.toolCall.arguments, details);
    });

    expect(calls).toHaveLength(1);
  });

  it("leaves MCP-derived function tools to the MCP instrumentation", async () => {
    const configuredTool = weatherTool(({ city }) => `configured:${city}`);
    const mcpTool = weatherTool(({ city }) => `mcp:${city}`);
    const agent = weatherAgent(configuredTool);
    vi.spyOn(agent, "getMcpTools").mockResolvedValue([mcpTool]);

    const { calls } = await traced(async () => {
      await agent.getAllTools(new RunContext());
      const configuredDetails = toolCallDetails("get_weather", { city: "Zurich" }, "call-1");
      const mcpDetails = toolCallDetails("get_weather", { city: "Boston" }, "call-2");
      await configuredTool.invoke(new RunContext(), configuredDetails.toolCall.arguments, configuredDetails);
      await mcpTool.invoke(new RunContext(), mcpDetails.toolCall.arguments, mcpDetails);
    });

    expect(calls).toEqual([expect.objectContaining({ call_id: "call-1" })]);
  });

  it("is a passthrough outside a Sovara trace", async () => {
    const execute = vi.fn(({ city }: { city: string }) => city);
    const configuredTool = weatherTool(execute);
    const details = toolCallDetails("get_weather", { city: "Zurich" });
    await weatherAgent(configuredTool).getAllTools(new RunContext());

    await expect(
      configuredTool.invoke(new RunContext(), details.toolCall.arguments, details),
    ).resolves.toBe("Zurich");
    expect(execute).toHaveBeenCalledTimes(1);
  });
});
