# Sovara for the OpenAI Agents SDK (Python)

Record OpenAI Agents tool calls in Sovara traces.

```bash
uv add sovara-openai-agents
# or: pip install sovara-openai-agents
```

```python
import sovara
import sovara_openai_agents

sovara.auto_instrument()  # model calls, through the HTTP client
sovara_openai_agents.instrument()  # function tools as tool steps, agent tools as child traces
```

Call both once at startup, before creating agents. Inside a Sovara trace, function tools are
recorded as tool steps and agent tools (`agent.as_tool()`) open a child trace with their input
and output. Outside a trace, tools run unrecorded. Requires `openai-agents` 0.13.6 (the supported 0.13 series).

## License

MIT.
