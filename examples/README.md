# Examples

Scripts that record OpenAI Agents SDK runs in Sovara. They call real models, so set
`OPENAI_API_KEY` and sign in with `sovara login` first. `mcp_tool_crash` starts a local MCP
server whose tool fails on purpose.

Python, from `examples/python`:

```sh
uv sync
uv run python project_analyst.py
```

TypeScript, from `examples/typescript`, after building the integration package:

```sh
npm install
npx tsx project_analyst.ts
```
