/// <reference types="node" />
import { SovaraClient, autoInstrument } from "@sovara/sdk";
import { instrumentOpenAIAgents } from "@sovara/openai-agents";
import { Agent, MCPServerStdio, run } from "@openai/agents";
import path from "node:path";
import { fileURLToPath } from "node:url";

await autoInstrument();
await instrumentOpenAIAgents();

function buildMcpServer(): MCPServerStdio {
  const currentDir = path.dirname(fileURLToPath(import.meta.url));
  const serverScript = path.join(currentDir, "mcp_crash_server.ts");

  return new MCPServerStdio({
    name: "debug-crash-mcp",
    command: process.execPath,
    args: ["--import", "tsx", serverScript],
    cacheToolsList: true,
    cwd: currentDir,
  });
}

async function runAgent() {
  const server = buildMcpServer();
  await server.connect();

  try {
    const agent = new Agent({
      name: "MCP crash probe",
      instructions:
        "You are testing MCP tool failure handling. " +
        "Call the always_fail tool exactly once with a short reason, then briefly explain that it failed.",
      mcpServers: [server],
      model: "gpt-4o-mini",
      modelSettings: {
        toolChoice: "required",
        temperature: 0,
      },
    });

    const result = await run(agent, "Trigger the MCP failure now.", { maxTurns: 3 });
    console.log(result.finalOutput);
  } finally {
    await server.close();
  }
}

const sovara_client = new SovaraClient({ projectName: "sovara-typescript-examples" });

await sovara_client.trace(
  "ts-openai-mcp-tool-crash",
  () => runAgent(),
);
