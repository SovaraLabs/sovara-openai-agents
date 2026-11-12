import asyncio
import json
from contextlib import asynccontextmanager, contextmanager

from sovara import _recording, _steps
from sovara._recording import TraceWriteCredentials
from sovara.runner import context_manager
from sovara_openai_agents import _patch as openai_agents_patch_module


def _sdk():
    from agents import Agent, RunConfig, RunContextWrapper, function_tool
    from agents.lifecycle import RunHooks
    from agents.run_internal import tool_execution
    from agents.run_internal.run_steps import ToolRunFunction
    from openai.types.responses import ResponseFunctionToolCall

    return {
        "Agent": Agent,
        "RunConfig": RunConfig,
        "RunContextWrapper": RunContextWrapper,
        "RunHooks": RunHooks,
        "ToolRunFunction": ToolRunFunction,
        "ResponseFunctionToolCall": ResponseFunctionToolCall,
        "FunctionToolBatchExecutor": tool_execution._FunctionToolBatchExecutor,
        "AgentBindings": getattr(tool_execution, "AgentBindings", None),
        "function_tool": function_tool,
    }


def _build_executor(tool, tool_call):
    sdk = _sdk()
    agent = sdk["Agent"](
        name="Tool runner",
        instructions="Use the tool.",
        model="gpt-4o-mini",
        tools=[tool],
    )
    executor_args = dict(
        tool_runs=[sdk["ToolRunFunction"](tool_call=tool_call, function_tool=tool)],
        hooks=sdk["RunHooks"](),
        context_wrapper=sdk["RunContextWrapper"](context=None),
        config=sdk["RunConfig"](),
        isolate_parallel_failures=True,
    )
    if sdk["AgentBindings"] is None:
        executor_args["agent"] = agent
    else:
        executor_args["bindings"] = sdk["AgentBindings"](
            public_agent=agent,
            execution_agent=agent,
        )
    return sdk["FunctionToolBatchExecutor"](**executor_args)


class _FakeStep:
    recorded = True

    def __init__(self, call):
        self.call = call
        self.reports = []

    async def afinish(self, output=None, **kwargs):
        self.reports.append(("finish", output, kwargs))

    async def afail(self, error=None, **kwargs):
        self.reports.append(("fail", error, kwargs))


def _fake_steps(monkeypatch) -> list[_FakeStep]:
    steps = []

    async def astart_step(call, *, lessons=False, step_uuid=None):
        del lessons, step_uuid
        steps.append(_FakeStep(call))
        return steps[-1]

    monkeypatch.setattr(_steps, "astart_step", astart_step)
    return steps


@contextmanager
def _recording_context(trace_id: str):
    recording = context_manager.RecordingContext(
        owner=object(),
        credentials=TraceWriteCredentials(
            trace_id=trace_id,
            recording_id=trace_id,
            _trace_write_token=f"token-{trace_id}",
        ),
        lesson_scope_paths=None,
    )
    with context_manager.bind_recording_context(recording):
        yield


def test_openai_agents_contract_sdk_surface_available():
    sdk = _sdk()
    agent_tool = sdk["Agent"](name="Researcher", instructions="Research.").as_tool(
        tool_name="researcher",
        tool_description="Research a topic.",
    )

    assert callable(sdk["function_tool"])
    assert hasattr(sdk["FunctionToolBatchExecutor"], "_execute_single_tool_body")
    assert hasattr(sdk["FunctionToolBatchExecutor"], "_invoke_tool_and_run_post_invoke")
    assert agent_tool._is_agent_tool is True
    assert agent_tool._agent_instance.name == "Researcher"


def test_openai_agents_patch_bypasses_tracing_without_recording(monkeypatch):
    sdk = _sdk()
    captured = {}
    steps = _fake_steps(monkeypatch)

    @sdk["function_tool"]
    async def get_weather(city: str) -> dict[str, str]:
        captured["called_with"] = city
        return {"city": city}

    tool_call = sdk["ResponseFunctionToolCall"](
        arguments='{"city":"Zurich"}',
        call_id="call_bypass",
        name="get_weather",
        type="function_call",
    )

    openai_agents_patch_module.openai_agents_patch()

    executor = _build_executor(get_weather, tool_call)
    results, _, _ = asyncio.run(executor.execute())

    assert captured["called_with"] == "Zurich"
    assert results[0].output == {"city": "Zurich"}
    assert json.loads(tool_call.arguments) == {"city": "Zurich"}
    assert steps == []


def test_openai_agents_patch_records_function_tool_execution(monkeypatch):
    sdk = _sdk()
    captured = {}

    @sdk["function_tool"]
    async def get_weather(city: str) -> dict[str, str]:
        captured["called_with"] = city
        return {"forecast": f"sunny in {city}"}

    tool_call = sdk["ResponseFunctionToolCall"](
        arguments='{"city":"Zurich"}',
        call_id="call_1",
        name="get_weather",
        type="function_call",
    )

    steps = _fake_steps(monkeypatch)
    openai_agents_patch_module.openai_agents_patch()

    executor = _build_executor(get_weather, tool_call)
    with _recording_context("trace-1"):
        results, _, _ = asyncio.run(executor.execute())

    assert captured["called_with"] == "Zurich"
    assert steps[0].call.name == "get_weather" and steps[0].call.call_id == "call_1"
    assert steps[0].call.arguments == {"city": "Zurich"}
    assert steps[0].reports == [("finish", {"forecast": "sunny in Zurich"}, {})]
    assert results[0].output == {"forecast": "sunny in Zurich"}


def test_openai_agents_patch_records_agent_tool_as_subtrace(monkeypatch):
    sdk = _sdk()
    captured = {"subtraces": [], "io": []}
    steps = _fake_steps(monkeypatch)
    agent_tool = sdk["Agent"](name="Researcher", instructions="Research.").as_tool(
        tool_name="researcher",
        tool_description="Research a topic.",
    )

    async def run_researcher(_context, input_json):
        captured["trace_id"] = context_manager.get_trace_id()
        return f"researched {json.loads(input_json)['input']}"

    agent_tool.on_invoke_tool = run_researcher
    tool_call = sdk["ResponseFunctionToolCall"](
        arguments='{"input":"Zurich"}',
        call_id="call_agent",
        name="researcher",
        type="function_call",
    )

    @asynccontextmanager
    async def fake_subtrace(name):
        captured["subtraces"].append(name)
        with _recording_context("child-trace"):
            yield "child-trace"

    async def fake_log_trace_io(recording, kind, value):
        captured["io"].append((recording.trace_id, kind, value))

    monkeypatch.setattr(openai_agents_patch_module, "subtrace", fake_subtrace)
    monkeypatch.setattr(_recording, "alog_trace_io", fake_log_trace_io)

    openai_agents_patch_module.openai_agents_patch()
    executor = _build_executor(agent_tool, tool_call)

    async def execute():
        with _recording_context("parent-trace"):
            results, _, _ = await executor.execute()
            assert context_manager.get_trace_id() == "parent-trace"
            return results

    results = asyncio.run(execute())

    assert captured == {
        "subtraces": ["Researcher"],
        "io": [
            ("child-trace", "input", "Zurich"),
            ("child-trace", "output", "researched Zurich"),
        ],
        "trace_id": "child-trace",
    }
    assert results[0].output == "researched Zurich"
    assert steps == []
