import asyncio
import json
import sys
import types
from contextlib import asynccontextmanager
from types import SimpleNamespace

from sovara import _recording
from sovara.runner import context_manager
from sovara_openai_agents import _patch as openai_agents_patch_module
from tests.fake_steps import fake_steps, recording_context


class _FakeFunctionTool:
    def __init__(
        self,
        func,
        *,
        name: str = "get_weather",
        namespace: str | None = None,
        strict_json_schema: bool = False,
        is_agent_tool: bool = False,
        agent_name: str = "Research Agent",
        defer_loading: bool = False,
    ):
        self._func = func
        self.name = name
        self._tool_namespace = namespace
        self.strict_json_schema = strict_json_schema
        self._is_agent_tool = is_agent_tool
        self._agent_instance = SimpleNamespace(name=agent_name) if is_agent_tool else None
        self.defer_loading = defer_loading

    async def invoke(self, **kwargs):
        return await self._func(**kwargs)


def _make_tool_call(
    *,
    arguments: str = '{"city":"Zurich"}',
    call_id: str = "call_1",
    namespace: str | None = None,
):
    return SimpleNamespace(
        arguments=arguments,
        call_id=call_id,
        name="get_weather",
        namespace=namespace,
        type="function_call",
    )


def _decode_tool_input(arguments: str) -> dict[str, object] | None:
    if not arguments:
        return {}
    try:
        parsed = json.loads(arguments)
    except Exception:
        return None
    return parsed if isinstance(parsed, dict) else None


def _install_fake_agents_sdk(monkeypatch):
    tool_execution_module = types.ModuleType("agents.run_internal.tool_execution")

    async def invoke_function_tool(*, function_tool, context, arguments):
        tool_input = context.tool_input
        if tool_input is None:
            tool_input = _decode_tool_input(arguments) or {}
        return await function_tool.invoke(**tool_input)

    tool_execution_module.invoke_function_tool = invoke_function_tool

    class FakeBatchExecutor:
        def __init__(self, *, func_tool, tool_call):
            self.agent = SimpleNamespace(name="Tool runner")
            self.func_tool = func_tool
            self.tool_call = tool_call

        async def _execute_single_tool_body(
            self,
            *,
            outer_task,
            task_state,
            func_tool,
            tool_call,
            tool_context,
            agent_hooks,
        ):
            del outer_task, task_state, func_tool, tool_call, tool_context, agent_hooks
            return None

        async def _invoke_tool_and_run_post_invoke(
            self,
            *,
            outer_task,
            task_state,
            func_tool,
            tool_call,
            tool_context,
            agent_hooks,
        ):
            del outer_task, task_state, tool_call, agent_hooks
            return await tool_execution_module.invoke_function_tool(
                function_tool=func_tool,
                context=tool_context,
                arguments=tool_context.tool_arguments,
            )

        async def execute(self):
            task_state = SimpleNamespace()
            tool_context = SimpleNamespace(
                tool_arguments=str(getattr(self.tool_call, "arguments", "") or ""),
                tool_call=self.tool_call,
                tool_input=_decode_tool_input(str(getattr(self.tool_call, "arguments", "") or "")),
            )
            await self._execute_single_tool_body(
                outer_task=None,
                task_state=task_state,
                func_tool=self.func_tool,
                tool_call=self.tool_call,
                tool_context=tool_context,
                agent_hooks=None,
            )
            final_result = await self._invoke_tool_and_run_post_invoke(
                outer_task=None,
                task_state=task_state,
                func_tool=self.func_tool,
                tool_call=self.tool_call,
                tool_context=tool_context,
                agent_hooks=None,
            )
            return [SimpleNamespace(output=final_result)], None, None

    tool_execution_module._FunctionToolBatchExecutor = FakeBatchExecutor

    agents_module = types.ModuleType("agents")
    agents_module.__path__ = []
    trace_internal_module = types.ModuleType("agents.run_internal")
    trace_internal_module.__path__ = []
    trace_internal_module.tool_execution = tool_execution_module
    agents_module.run_internal = trace_internal_module

    monkeypatch.setitem(sys.modules, "agents", agents_module)
    monkeypatch.setitem(sys.modules, "agents.run_internal", trace_internal_module)
    monkeypatch.setitem(sys.modules, "agents.run_internal.tool_execution", tool_execution_module)

    if hasattr(FakeBatchExecutor, "__sovara_openai_agents_patched__"):
        delattr(FakeBatchExecutor, "__sovara_openai_agents_patched__")

    return FakeBatchExecutor


def test_openai_agents_patch_bypasses_tracing_without_recording(monkeypatch):
    BatchExecutor = _install_fake_agents_sdk(monkeypatch)
    steps = fake_steps(monkeypatch)
    captured = {}

    async def get_weather(city: str):
        captured["called_with"] = city
        return {"city": city}

    tool = _FakeFunctionTool(get_weather)
    tool_call = _make_tool_call(call_id="call_bypass")
    openai_agents_patch_module.openai_agents_patch()

    executor = BatchExecutor(func_tool=tool, tool_call=tool_call)
    results, _, _ = asyncio.run(executor.execute())

    assert captured["called_with"] == "Zurich"
    assert results[0].output == {"city": "Zurich"}
    assert json.loads(tool_call.arguments) == {"city": "Zurich"}
    assert steps == []


def test_openai_agents_patch_records_function_tool_execution(monkeypatch):
    BatchExecutor = _install_fake_agents_sdk(monkeypatch)
    steps = fake_steps(monkeypatch)

    async def get_weather(city: str):
        return {"forecast": f"sunny in {city}"}

    tool = _FakeFunctionTool(get_weather)
    tool_call = _make_tool_call(call_id="call_1")
    openai_agents_patch_module.openai_agents_patch()

    executor = BatchExecutor(func_tool=tool, tool_call=tool_call)
    with recording_context():
        results, _, _ = asyncio.run(executor.execute())

    assert results[0].output == {"forecast": "sunny in Zurich"}
    assert steps[0].call.name == "get_weather" and steps[0].call.call_id == "call_1"
    assert steps[0].call.arguments == {"city": "Zurich"}
    assert (steps[0].ending, steps[0].output) == ("finish", {"forecast": "sunny in Zurich"})


def test_openai_agents_patch_fails_the_step_when_the_tool_raises(monkeypatch):
    BatchExecutor = _install_fake_agents_sdk(monkeypatch)
    steps = fake_steps(monkeypatch)

    async def get_weather(city: str):
        raise ValueError(city)

    openai_agents_patch_module.openai_agents_patch()
    executor = BatchExecutor(func_tool=_FakeFunctionTool(get_weather), tool_call=_make_tool_call())
    with recording_context():
        try:
            asyncio.run(executor.execute())
        except ValueError:
            pass

    assert steps[0].ending == "fail"
    assert isinstance(steps[0].reports[0][1], ValueError)


def test_openai_agents_patch_records_agent_tool_as_subtrace(monkeypatch):
    BatchExecutor = _install_fake_agents_sdk(monkeypatch)
    steps = fake_steps(monkeypatch)
    captured = {"subtraces": [], "io": []}

    async def run_researcher(input: str):
        captured["trace_id"] = context_manager.get_trace_id()
        return f"researched {input}"

    tool = _FakeFunctionTool(
        run_researcher,
        name="researcher",
        is_agent_tool=True,
        agent_name="Researcher",
    )
    tool_call = _make_tool_call(
        arguments='{"input":"Zurich"}',
        call_id="call_agent",
    )

    @asynccontextmanager
    async def fake_subtrace(name):
        captured["subtraces"].append(name)
        with recording_context("child-trace"):
            yield "child-trace"

    async def fake_log_trace_io(recording, kind, value):
        captured["io"].append((recording.trace_id, kind, value))

    monkeypatch.setattr(openai_agents_patch_module, "subtrace", fake_subtrace)
    monkeypatch.setattr(_recording, "alog_trace_io", fake_log_trace_io)

    openai_agents_patch_module.openai_agents_patch()
    executor = BatchExecutor(func_tool=tool, tool_call=tool_call)

    async def execute():
        with recording_context("parent-trace"):
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
