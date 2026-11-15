# Sovara for the OpenAI Agents SDK (Python)

Sovara helps teams manage domain knowledge and put it to work in AI agents.
Capture business rules, expert guidance, and lessons from experience in a shared
knowledge base. Deliver relevant knowledge to agents as they work, review how
they use it, and improve it together.

Connect OpenAI Agents to Sovara to review agent runs and apply your team's domain knowledge.

```bash
uv add sovara-openai-agents
# or: pip install sovara-openai-agents
```

```python
import sovara
import sovara_openai_agents

sovara.auto_instrument()
sovara_openai_agents.instrument()
```

Initialize both integrations once, before creating agents, and run your agent
inside a [Sovara trace](https://docs.sovara-labs.com/sdks/python/quickstart).
This package records tool calls; the Sovara SDK records supported model calls
and applies lessons when [knowledge injection](https://docs.sovara-labs.com/sovaradb/auto-injection)
is enabled. Calls outside a trace are not recorded.

Requires `openai-agents` 0.13.6 (the supported 0.13 series). See the
[integration guide](https://docs.sovara-labs.com/integrations/openai-agents).

## License

MIT.
