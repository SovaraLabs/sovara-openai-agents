import asyncio
import sys
from pathlib import Path

from agents import Agent, ModelSettings, Runner
from agents.mcp import MCPServerStdio
import sovara_openai_agents
from sovara import SovaraClient, auto_instrument

auto_instrument()
sovara_openai_agents.instrument()


sovara_client = SovaraClient(project_name="debug-openai")


def build_mcp_server() -> MCPServerStdio:
    server_script = Path(__file__).with_name("mcp_crash_server.py")
    return MCPServerStdio(
        params={"command": sys.executable, "args": [str(server_script)]},
        cache_tools_list=True,
        name="debug-crash-mcp",
    )


async def main():
    server = build_mcp_server()
    await server.connect()
    try:
        agent = Agent(
            name="MCP crash probe",
            instructions=(
                "You are testing MCP tool failure handling. "
                "Call the always_fail tool exactly once with a short reason, then briefly explain that it failed."
            ),
            mcp_servers=[server],
            model="gpt-4o-mini",
            model_settings=ModelSettings(tool_choice="required", temperature=0),
        )
        result = await Runner.run(agent, "Trigger the MCP failure now.", max_turns=3)
        print(result.final_output)
    finally:
        await server.cleanup()


if __name__ == "__main__":

    async def _run():
        async with sovara_client.trace("OpenAI MCP Tool Crash"):
            await main()

    asyncio.run(_run())
