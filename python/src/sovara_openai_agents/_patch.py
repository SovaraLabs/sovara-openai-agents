"""Record OpenAI Agents function tools and nested agent-tool traces."""

from __future__ import annotations

import json
import logging
from functools import wraps
from typing import Any, Literal

from sovara import Step, ToolCall
from sovara import integration as instrumentation
from sovara.integration import json_value, subtrace

logger = logging.getLogger("sovara_openai_agents")

_STEP_ATTR = "_sovara_openai_agents_step"


def _is_agent_tool(func_tool: Any) -> bool:
    return bool(getattr(func_tool, "_is_agent_tool", False))


def _agent_tool_name(func_tool: Any) -> str:
    agent_name = getattr(getattr(func_tool, "_agent_instance", None), "name", None)
    return str(agent_name or func_tool.name)


def _agent_tool_input(arguments: str) -> str:
    parsed, parse_error = _parse_tool_arguments(arguments)
    if parse_error is None and parsed is not None:
        input_value = parsed.get("input")
        if set(parsed) == {"input"} and isinstance(input_value, str):
            return input_value
    return arguments


def _trace_io_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(json_value(value), ensure_ascii=False, sort_keys=True)


def _tool_namespace(tool_call: Any, func_tool: Any) -> str | None:
    namespace = getattr(tool_call, "namespace", None) or getattr(func_tool, "_tool_namespace", None)
    return namespace if isinstance(namespace, str) and namespace else None


def _qualified_tool_name(tool_name: str, namespace: str | None) -> str:
    return f"{namespace}.{tool_name}" if namespace else tool_name


def _parse_tool_arguments(arguments: str) -> tuple[dict[str, Any] | None, str | None]:
    if not arguments:
        return {}, None

    try:
        parsed = json.loads(arguments)
    except Exception as exc:
        return None, f"Invalid JSON arguments: {exc}"

    if not isinstance(parsed, dict):
        return None, "Tool arguments must decode to a JSON object."

    return parsed, None


def _tool_call(func_tool: Any, tool_call: Any) -> ToolCall:
    tool_name = str(func_tool.name)
    namespace = _tool_namespace(tool_call, func_tool)
    arguments = str(getattr(tool_call, "arguments", "") or "")
    parsed_input, parse_error = _parse_tool_arguments(arguments)
    metadata: dict[str, Any] = {
        "tool_class": func_tool.__class__.__name__,
        "strict_json_schema": bool(getattr(func_tool, "strict_json_schema", False)),
        "is_agent_tool": bool(getattr(func_tool, "_is_agent_tool", False)),
        "defer_loading": bool(getattr(func_tool, "defer_loading", False)),
    }
    if namespace:
        metadata["tool_namespace"] = namespace
    if parse_error is not None:
        metadata["tool_arguments_raw"] = arguments
        metadata["arguments_error"] = parse_error
    return ToolCall(
        name=_qualified_tool_name(tool_name, namespace),
        arguments=json_value(parsed_input),
        call_id=str(getattr(tool_call, "call_id", "") or "").strip() or None,
        metadata=metadata,
    )


def _step_of(task_state: Any) -> Step | None:
    return getattr(task_state, _STEP_ATTR, None)


async def _log_trace_io(kind: Literal["input", "output"], text: str) -> None:
    await instrumentation.alog_trace_io(kind, text)


def openai_agents_patch() -> None:
    """Install OpenAI Agents function-tool and agent-tool instrumentation once.

    Missing agents support is a no-op. Wrap the batch executor and supported invoker
    variant; capture is gated by the recording and tracing flag. Agent tools use child
    traces; ordinary function tools are recorded as tool steps. Tool arguments and
    results pass through unchanged.
    """
    try:
        from agents.run_internal import tool_execution
    except ImportError:
        logger.info("openai-agents not installed, skipping openai-agents patches")
        return

    batch_executor_cls = tool_execution._FunctionToolBatchExecutor
    if getattr(batch_executor_cls, "__sovara_openai_agents_patched__", False):
        return

    original_execute_single_tool_body = batch_executor_cls._execute_single_tool_body
    original_invoke_tool_and_run_post_invoke = batch_executor_cls._invoke_tool_and_run_post_invoke
    invoker_name = (
        "_invoke_function_tool_with_metadata"
        if hasattr(tool_execution, "_invoke_function_tool_with_metadata")
        else "invoke_function_tool"
    )
    original_invoke_function_tool = getattr(tool_execution, invoker_name)
    invoker_returns_metadata = invoker_name == "_invoke_function_tool_with_metadata"

    @wraps(original_invoke_function_tool)
    async def patched_invoke_function_tool(*, function_tool, context, arguments):
        if not instrumentation.capture_active() or not _is_agent_tool(function_tool):
            return await original_invoke_function_tool(
                function_tool=function_tool,
                context=context,
                arguments=arguments,
            )

        async with subtrace(_agent_tool_name(function_tool)) as child_trace_id:
            if child_trace_id:
                await _log_trace_io("input", _agent_tool_input(arguments))
            result = await original_invoke_function_tool(
                function_tool=function_tool,
                context=context,
                arguments=arguments,
            )
            output = result.output if invoker_returns_metadata else result
            if child_trace_id and output is not None:
                await _log_trace_io("output", _trace_io_text(output))
            return result

    @wraps(original_execute_single_tool_body)
    async def patched_execute_single_tool_body(
        self,
        *,
        outer_task,
        task_state,
        func_tool,
        tool_call,
        tool_context,
        agent_hooks,
    ):
        if not instrumentation.capture_active() or _is_agent_tool(func_tool):
            return await original_execute_single_tool_body(
                self,
                outer_task=outer_task,
                task_state=task_state,
                func_tool=func_tool,
                tool_call=tool_call,
                tool_context=tool_context,
                agent_hooks=agent_hooks,
            )

        step = await instrumentation.abegin(_tool_call(func_tool, tool_call))
        if step is not None:
            setattr(task_state, _STEP_ATTR, step)

        return await original_execute_single_tool_body(
            self,
            outer_task=outer_task,
            task_state=task_state,
            func_tool=func_tool,
            tool_call=tool_call,
            tool_context=tool_context,
            agent_hooks=agent_hooks,
        )

    @wraps(original_invoke_tool_and_run_post_invoke)
    async def patched_invoke_tool_and_run_post_invoke(
        self,
        *,
        outer_task,
        task_state,
        func_tool,
        tool_call,
        tool_context,
        agent_hooks,
    ):
        step = _step_of(task_state)
        if step is None:
            return await original_invoke_tool_and_run_post_invoke(
                self,
                outer_task=outer_task,
                task_state=task_state,
                func_tool=func_tool,
                tool_call=tool_call,
                tool_context=tool_context,
                agent_hooks=agent_hooks,
            )

        try:
            final_result = await original_invoke_tool_and_run_post_invoke(
                self,
                outer_task=outer_task,
                task_state=task_state,
                func_tool=func_tool,
                tool_call=tool_call,
                tool_context=tool_context,
                agent_hooks=agent_hooks,
            )
        except Exception as exc:
            await instrumentation.areport(step, "fail", exc, propagating=True)
            raise
        except BaseException as exc:
            instrumentation.report_in_background(step, "cancel", exc.__class__.__name__)
            raise
        await instrumentation.areport(step, "finish", json_value(final_result))
        return final_result

    batch_executor_cls._execute_single_tool_body = patched_execute_single_tool_body
    batch_executor_cls._invoke_tool_and_run_post_invoke = patched_invoke_tool_and_run_post_invoke
    setattr(tool_execution, invoker_name, patched_invoke_function_tool)
    batch_executor_cls.__sovara_openai_agents_patched__ = True
    logger.info("openai-agents patches applied")
