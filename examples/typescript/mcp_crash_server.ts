/// <reference types="node" />
import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { CallToolRequestSchema, ListToolsRequestSchema } from "@modelcontextprotocol/sdk/types.js";

const TOOL_NAME = "always_fail";

const server = new Server(
  {
    name: "Debug crash MCP",
    version: "0.1.0",
  },
  {
    capabilities: {
      tools: {},
    },
  },
);

server.setRequestHandler(ListToolsRequestSchema, async () => {
  return {
    tools: [
      {
        name: TOOL_NAME,
        description: "Always raises so MCP node failure handling can be inspected.",
        inputSchema: {
          type: "object",
          properties: {
            reason: {
              type: "string",
              description: "Reason for intentional failure.",
            },
          },
          required: ["reason"],
        },
      },
    ],
  };
});

server.setRequestHandler(CallToolRequestSchema, async (request) => {
  if (request.params.name !== TOOL_NAME) {
    return {
      content: [{ type: "text", text: `Unknown tool: ${request.params.name}` }],
      isError: true,
    };
  }

  const args = (request.params.arguments ?? {}) as { reason?: string };
  const reason = args.reason ?? "no reason provided";
  return {
    content: [{ type: "text", text: `intentional crash from MCP tool: ${reason}` }],
    isError: true,
  };
});

async function main() {
  const transport = new StdioServerTransport();
  await server.connect(transport);
}

main().catch((error) => {
  console.error("MCP crash server failed:", error);
  process.exit(1);
});
