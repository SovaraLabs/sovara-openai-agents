from mcp.server.fastmcp import FastMCP


server = FastMCP("Debug crash MCP")


@server.tool(description="Always raises so MCP node failure handling can be inspected.")
def always_fail(reason: str) -> str:
    raise RuntimeError(f"intentional crash from MCP tool: {reason}")


if __name__ == "__main__":
    server.run(transport="stdio")
