/// <reference types="node" />
import { SovaraClient, autoInstrument } from "@sovara/sdk";
import { instrumentOpenAIAgents } from "@sovara/openai-agents";
import { Agent, run, tool } from "@openai/agents";
import * as fs from "node:fs/promises";
import * as path from "node:path";
import { z } from "zod";

await autoInstrument();
await instrumentOpenAIAgents();

const MODEL = process.env.OPENAI_AGENTS_MODEL ?? "gpt-5.4";
const REPO_ROOT = path.resolve(new URL("../..", import.meta.url).pathname);
const DEFAULT_TARGET_DIR = "python/src/sovara_openai_agents";
const MAX_LINES_PER_READ = 120;

function resolveRepoPath(relativePath: string): string {
  const candidate = path.resolve(REPO_ROOT, relativePath);
  const rel = path.relative(REPO_ROOT, candidate);
  if (rel.startsWith("..") || path.isAbsolute(rel)) {
    throw new Error(`Path must stay inside the repository: ${relativePath}`);
  }
  return candidate;
}

function clipText(text: string, limit = 900): string {
  return text.length <= limit ? text : `${text.slice(0, limit - 3)}...`;
}

async function listFilesRecursive(root: string, suffix: string): Promise<string[]> {
  const out: string[] = [];
  async function walk(dir: string) {
    const entries = await fs.readdir(dir, { withFileTypes: true });
    entries.sort((a, b) => a.name.localeCompare(b.name));
    for (const entry of entries) {
      const full = path.join(dir, entry.name);
      if (entry.isDirectory()) {
        await walk(full);
      } else if (entry.isFile() && entry.name.endsWith(suffix)) {
        out.push(full);
      }
    }
  }
  await walk(root);
  return out;
}

const listPythonFiles = tool({
  name: "list_python_files",
  description: "List Python files under a repository directory.",
  parameters: z.object({ relative_dir: z.string() }),
  execute: async ({ relative_dir }) => {
    try {
      const targetDir = resolveRepoPath(relative_dir);
      const stats = await fs.stat(targetDir).catch(() => null);
      if (!stats || !stats.isDirectory()) {
        return { error: `Directory does not exist: ${relative_dir}` };
      }
      const files = (await listFilesRecursive(targetDir, ".py")).map((p) =>
        path.relative(REPO_ROOT, p),
      );
      return { directory: relative_dir, file_count: files.length, files };
    } catch (error) {
      return { error: (error as Error).message };
    }
  },
});

const inspectPythonFile = tool({
  name: "inspect_python_file",
  description: "Return a coarse summary of a Python file (line count + leading docstring).",
  parameters: z.object({ relative_path: z.string() }),
  execute: async ({ relative_path }) => {
    try {
      const filePath = resolveRepoPath(relative_path);
      if (!relative_path.endsWith(".py")) {
        return { error: `Only Python files are supported: ${relative_path}` };
      }
      const source = await fs.readFile(filePath, "utf-8");
      const lines = source.split(/\r?\n/);
      const docMatch = source.match(/^"""([\s\S]*?)"""/m);
      return {
        path: relative_path,
        line_count: lines.length,
        module_docstring: docMatch ? clipText(docMatch[1].trim(), 300) : null,
      };
    } catch (error) {
      return { error: (error as Error).message };
    }
  },
});

const readFileExcerpt = tool({
  name: "read_file_excerpt",
  description: "Read a bounded slice of a text file.",
  parameters: z.object({
    relative_path: z.string(),
    start_line: z.number().int().min(1).default(1),
    max_lines: z.number().int().min(1).default(80),
  }),
  execute: async ({ relative_path, start_line, max_lines }) => {
    try {
      const filePath = resolveRepoPath(relative_path);
      const source = await fs.readFile(filePath, "utf-8");
      const lines = source.split(/\r?\n/);
      const safeStart = Math.max(start_line, 1);
      const safeMax = Math.min(Math.max(max_lines, 1), MAX_LINES_PER_READ);
      const excerpt = lines.slice(safeStart - 1, safeStart - 1 + safeMax);
      return {
        path: relative_path,
        start_line: safeStart,
        end_line: excerpt.length > 0 ? safeStart + excerpt.length - 1 : safeStart,
        line_count: lines.length,
        content: excerpt.join("\n"),
      };
    } catch (error) {
      return { error: (error as Error).message };
    }
  },
});

async function runAgent() {
  if (!process.env.OPENAI_API_KEY) {
    throw new Error("Set OPENAI_API_KEY before running this example.");
  }

  const agent = new Agent({
    name: "Codebase analyst",
    instructions:
      "You are a careful codebase-analysis agent. You must inspect the repository with " +
      "tools before answering. Start by listing files in the requested directory. " +
      "You must use list_python_files, inspect_python_file, and read_file_excerpt before " +
      "answering. Read excerpts from at least two files you identify as central. " +
      "Do not give a final answer unless you have used read_file_excerpt at least twice. " +
      "Keep the final answer compact but concrete, and cite file paths.",
    model: MODEL,
    modelSettings: { temperature: 0 },
    tools: [listPythonFiles, inspectPythonFile, readFileExcerpt],
  });

  const prompt = `
    Analyze the \`${DEFAULT_TARGET_DIR}\` package in this repository.

    Tell me:
    1. Which Python files are in that directory.
    2. Which 3 files seem most central to how the patch gets installed and records tool calls.
    3. The main classes/functions in those files.
    4. A short summary of how the subsystem works.
    5. Two short quoted code snippets from the files you read.

    Use the tools. Read concrete excerpts from at least two files before you answer.
    Cite concrete file paths in the answer.
  `;

  console.log("=".repeat(70));
  console.log("OPENAI AGENTS SDK - Running Codebase Analyst");
  console.log("=".repeat(70));
  console.log(`Model: ${MODEL}`);
  console.log(`Target directory: ${DEFAULT_TARGET_DIR}`);

  const result = await run(agent, prompt, { maxTurns: 8 });
  console.log("\nFinal answer:\n");
  console.log(result.finalOutput);
}

const sovara_client = new SovaraClient({ projectName: "sovara-typescript-examples" });

await sovara_client.trace(
  "ts-openai-agents-project-analyst",
  () => runAgent(),
);
