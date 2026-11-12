# Sovara for the OpenAI Agents SDK (TypeScript)

Record OpenAI Agents function tool calls in Sovara traces.

```bash
npm install @sovara/openai-agents @sovara/sdk
```

```ts
import { autoInstrument } from "@sovara/sdk";
import { instrumentOpenAIAgents } from "@sovara/openai-agents";

await autoInstrument();         // model calls, through the HTTP client
await instrumentOpenAIAgents(); // function tools as tool steps
```

Await both once at startup, before creating agents. Inside a Sovara trace, function tools are
recorded as tool steps; outside one they run unrecorded. Requires Node.js 24.22.1+,
`@openai/agents`.

## Development

See [Development](https://github.com/SovaraLabs/sovara-openai-agents#development).

## License

MIT.
