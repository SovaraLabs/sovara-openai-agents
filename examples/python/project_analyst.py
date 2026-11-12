"""OpenAI Agents SDK codebase-analysis example.

This example is intentionally more involved than a single weather tool. It gives
the agent several deterministic local tools so it can inspect part of the repo,
plan a short investigation, and then summarize what it found.

Requires `OPENAI_API_KEY` in the environment.
"""

from __future__ import annotations

import asyncio
import ast
import json
import os
from pathlib import Path
from typing import Any

from agents import Agent, ModelSettings, Runner, function_tool
from agents.items import MessageOutputItem, ToolCallItem, ToolCallOutputItem
import sovara_openai_agents
from sovara import SovaraClient, auto_instrument

auto_instrument()
sovara_openai_agents.instrument()


MODEL = os.environ.get("OPENAI_AGENTS_MODEL", "gpt-5.4-mini")
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TARGET_DIR = "python/src/sovara_openai_agents"
MAX_LINES_PER_READ = 120
sovara_client = SovaraClient(project_name="debug-openai-agents")


def _resolve_repo_path(relative_path: str) -> Path:
    candidate = (REPO_ROOT / relative_path).resolve()
    try:
        candidate.relative_to(REPO_ROOT)
    except ValueError as exc:
        raise ValueError(f"Path must stay inside the repository: {relative_path}") from exc
    return candidate


def _normalize_jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _normalize_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_normalize_jsonable(item) for item in value]
    return str(value)


def _clip_text(text: str, *, limit: int = 900) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def _symbol_signature(node: ast.AST) -> str:
    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return node.__class__.__name__
    if isinstance(node, ast.ClassDef):
        return node.name

    args = [arg.arg for arg in node.args.args]
    if node.args.vararg:
        args.append(f"*{node.args.vararg.arg}")
    args.extend(arg.arg for arg in node.args.kwonlyargs)
    if node.args.kwarg:
        args.append(f"**{node.args.kwarg.arg}")
    return f"{node.name}({', '.join(args)})"


def _parse_python_symbols(path: Path) -> dict[str, Any]:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    classes: list[dict[str, Any]] = []
    functions: list[dict[str, Any]] = []
    async_functions: list[dict[str, Any]] = []

    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            classes.append(
                {
                    "name": node.name,
                    "line": node.lineno,
                    "methods": [
                        _symbol_signature(child)
                        for child in node.body
                        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
                    ],
                }
            )
        elif isinstance(node, ast.FunctionDef):
            functions.append(
                {
                    "name": node.name,
                    "line": node.lineno,
                    "signature": _symbol_signature(node),
                }
            )
        elif isinstance(node, ast.AsyncFunctionDef):
            async_functions.append(
                {
                    "name": node.name,
                    "line": node.lineno,
                    "signature": _symbol_signature(node),
                }
            )

    module_docstring = ast.get_docstring(tree)
    return {
        "module_docstring": _clip_text(module_docstring or "", limit=300) or None,
        "classes": classes,
        "functions": functions,
        "async_functions": async_functions,
        "line_count": len(source.splitlines()),
    }


@function_tool
def list_python_files(relative_dir: str) -> dict[str, Any]:
    """List Python files under a repository directory.

    Args:
        relative_dir: Repository-relative directory to inspect.
    """

    try:
        target_dir = _resolve_repo_path(relative_dir)
    except ValueError as exc:
        return {"error": str(exc)}

    if not target_dir.exists():
        return {"error": f"Directory does not exist: {relative_dir}"}
    if not target_dir.is_dir():
        return {"error": f"Path is not a directory: {relative_dir}"}

    files = [
        str(path.relative_to(REPO_ROOT))
        for path in sorted(target_dir.rglob("*.py"))
        if path.is_file()
    ]
    return {
        "directory": relative_dir,
        "file_count": len(files),
        "files": files,
    }


@function_tool
def inspect_python_file(relative_path: str) -> dict[str, Any]:
    """Return a structured summary of a Python file.

    Args:
        relative_path: Repository-relative path to a Python file.
    """

    try:
        path = _resolve_repo_path(relative_path)
    except ValueError as exc:
        return {"error": str(exc)}

    if not path.exists():
        return {"error": f"File does not exist: {relative_path}"}
    if path.suffix != ".py":
        return {"error": f"Only Python files are supported: {relative_path}"}

    summary = _parse_python_symbols(path)
    summary["path"] = relative_path
    return summary


@function_tool
def read_file_excerpt(
    relative_path: str,
    start_line: int = 1,
    max_lines: int = 80,
) -> dict[str, Any]:
    """Read a bounded slice of a text file.

    Args:
        relative_path: Repository-relative file path.
        start_line: 1-based starting line number.
        max_lines: Maximum number of lines to read.
    """

    try:
        path = _resolve_repo_path(relative_path)
    except ValueError as exc:
        return {"error": str(exc)}

    if not path.exists():
        return {"error": f"File does not exist: {relative_path}"}

    safe_start = max(start_line, 1)
    safe_max = min(max(max_lines, 1), MAX_LINES_PER_READ)
    lines = path.read_text(encoding="utf-8").splitlines()
    excerpt = lines[safe_start - 1 : safe_start - 1 + safe_max]

    return {
        "path": relative_path,
        "start_line": safe_start,
        "end_line": safe_start + len(excerpt) - 1 if excerpt else safe_start,
        "line_count": len(lines),
        "content": "\n".join(excerpt),
    }


@function_tool
def search_repo_text(relative_dir: str, query: str, max_results: int = 10) -> dict[str, Any]:
    """Search for plain text in files under a repository directory.

    Args:
        relative_dir: Repository-relative directory to inspect.
        query: Plain text to search for.
        max_results: Maximum number of matching lines to return.
    """

    try:
        target_dir = _resolve_repo_path(relative_dir)
    except ValueError as exc:
        return {"error": str(exc)}

    if not target_dir.exists() or not target_dir.is_dir():
        return {"error": f"Directory does not exist: {relative_dir}"}

    normalized_query = query.strip()
    if not normalized_query:
        return {"error": "Query must not be empty."}

    matches: list[dict[str, Any]] = []
    for path in sorted(target_dir.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix not in {".py", ".md", ".toml"}:
            continue
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except UnicodeDecodeError:
            continue
        for line_number, line in enumerate(lines, start=1):
            if normalized_query in line:
                matches.append(
                    {
                        "path": str(path.relative_to(REPO_ROOT)),
                        "line": line_number,
                        "text": _clip_text(line.strip(), limit=220),
                    }
                )
                if len(matches) >= max(max_results, 1):
                    return {
                        "directory": relative_dir,
                        "query": normalized_query,
                        "matches": matches,
                    }

    return {
        "directory": relative_dir,
        "query": normalized_query,
        "matches": matches,
    }


def _print_trace(result: Any) -> None:
    print("\nTool activity:")
    saw_tool_activity = False

    for item in result.new_items:
        if isinstance(item, ToolCallItem):
            raw_item = item.raw_item
            tool_name = getattr(raw_item, "name", None)
            if tool_name is None and isinstance(raw_item, dict):
                tool_name = raw_item.get("name")
            arguments = getattr(raw_item, "arguments", None)
            if arguments is None and isinstance(raw_item, dict):
                arguments = raw_item.get("arguments")
            print(f"- call {tool_name}: {_clip_text(str(arguments), limit=240)}")
            saw_tool_activity = True
        elif isinstance(item, ToolCallOutputItem):
            output_text = json.dumps(_normalize_jsonable(item.output), sort_keys=True)
            print(f"  output: {_clip_text(output_text, limit=320)}")
            saw_tool_activity = True
        elif isinstance(item, MessageOutputItem):
            continue

    if not saw_tool_activity:
        print("- no tool items were captured")


async def main() -> None:
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("Set OPENAI_API_KEY before running this example.")

    agent = Agent(
        name="Codebase analyst",
        instructions=(
            "You are a careful codebase-analysis agent. "
            "You must inspect the repository with tools before answering. "
            "Start by listing files in the requested directory. "
            "You must use list_python_files, inspect_python_file, and read_file_excerpt before answering. "
            "Read excerpts from at least two files you identify as central. "
            "Use search_repo_text when you need cross-file confirmation. "
            "Do not give a final answer unless you have used read_file_excerpt at least twice. "
            "Keep the final answer compact but concrete, and cite file paths."
        ),
        model=MODEL,
        model_settings=ModelSettings(temperature=0),
        tools=[
            list_python_files,
            inspect_python_file,
            read_file_excerpt,
            search_repo_text,
        ],
    )

    prompt = f"""
    Analyze the `{DEFAULT_TARGET_DIR}` package in this repository.

    Tell me:
    1. Which Python files are in that directory.
    2. Which 3 files seem most central to how the patch gets installed and records tool calls.
    3. The main classes/functions in those files.
    4. A short summary of how the subsystem works.
    5. Two short quoted code snippets from the files you read.

    Use the tools. Read concrete excerpts from at least two files before you answer.
    Cite concrete file paths in the answer.
    """

    print("=" * 70)
    print("OPENAI AGENTS SDK - Running Codebase Analyst")
    print("=" * 70)
    print(f"Model: {MODEL}")
    print(f"Target directory: {DEFAULT_TARGET_DIR}")

    result = await Runner.run(agent, prompt, max_turns=8)

    _print_trace(result)
    print("\nFinal answer:\n")
    print(result.final_output)


if __name__ == "__main__":

    async def _run():
        async with sovara_client.trace("OpenAI Agents Project Analyst"):
            await main()

    asyncio.run(_run())
