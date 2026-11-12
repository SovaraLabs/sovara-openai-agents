"""Package tests that need no Sovara server; recording tests live in the Sovara repository."""

import asyncio
import json
import sys
import types
from types import SimpleNamespace

from sovara_openai_agents import _patch as openai_agents_patch_module


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

    tool_execution_module.invoke_function_tool = invoke_function_tool  # pyright: ignore[reportAttributeAccessIssue]

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

    tool_execution_module._FunctionToolBatchExecutor = FakeBatchExecutor  # pyright: ignore[reportAttributeAccessIssue]

    agents_module = types.ModuleType("agents")
    agents_module.__path__ = []
    trace_internal_module = types.ModuleType("agents.run_internal")
    trace_internal_module.__path__ = []
    trace_internal_module.tool_execution = tool_execution_module  # pyright: ignore[reportAttributeAccessIssue]
    agents_module.run_internal = trace_internal_module  # pyright: ignore[reportAttributeAccessIssue]

    monkeypatch.setitem(sys.modules, "agents", agents_module)
    monkeypatch.setitem(sys.modules, "agents.run_internal", trace_internal_module)
    monkeypatch.setitem(sys.modules, "agents.run_internal.tool_execution", tool_execution_module)

    if hasattr(FakeBatchExecutor, "__sovara_openai_agents_patched__"):
        delattr(FakeBatchExecutor, "__sovara_openai_agents_patched__")

    return FakeBatchExecutor


def test_tool_call_keeps_raw_arguments_when_not_json_object():
    async def get_weather(**kwargs):
        return kwargs

    tool = _FakeFunctionTool(get_weather)
    tool_call = _make_tool_call(arguments='["Zurich"]', call_id="call_invalid")

    call = openai_agents_patch_module._tool_call(tool, tool_call)

    assert (call.name, call.call_id, call.arguments) == ("get_weather", "call_invalid", None)
    assert call.metadata["tool_arguments_raw"] == '["Zurich"]'
    assert call.metadata["arguments_error"] == "Tool arguments must decode to a JSON object."


def test_tool_call_uses_tool_namespace_for_qualified_tool_name():
    async def get_weather(**kwargs):
        return kwargs

    tool = _FakeFunctionTool(get_weather, namespace="weather")
    tool_call = _make_tool_call(call_id="call_namespaced")

    call = openai_agents_patch_module._tool_call(tool, tool_call)

    assert call.name == "weather.get_weather"
    assert call.metadata["tool_namespace"] == "weather"
    assert call.arguments == {"city": "Zurich"}


def test_openai_agents_patch_bypasses_tracing_without_recording(monkeypatch):
    BatchExecutor = _install_fake_agents_sdk(monkeypatch)
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
