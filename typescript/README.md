# Sovara for the OpenAI Agents SDK (TypeScript)

Sovara helps teams manage domain knowledge and put it to work in AI agents.
Capture business rules, expert guidance, and lessons from experience in a shared
knowledge base. Deliver relevant knowledge to agents as they work, review how
they use it, and improve it together.

Connect OpenAI Agents to Sovara to review agent runs and apply your team's domain knowledge.

```bash
npm install @sovara/openai-agents @sovara/sdk
```

```ts
import { autoInstrument } from "@sovara/sdk";
import { instrumentOpenAIAgents } from "@sovara/openai-agents";

await autoInstrument();
await instrumentOpenAIAgents();
```

Initialize both integrations once, before creating agents, and run your agent
inside a [Sovara trace](https://docs.sovara-labs.com/sdks/typescript/quickstart).
This package records tool calls; the Sovara SDK records supported model calls
and applies lessons when [knowledge injection](https://docs.sovara-labs.com/sovaradb/auto-injection)
is enabled. Calls outside a trace are not recorded.

Requires Node.js 24.22.1+ and `@openai/agents`. See the
[integration guide](https://docs.sovara-labs.com/integrations/openai-agents).

## License

MIT.
