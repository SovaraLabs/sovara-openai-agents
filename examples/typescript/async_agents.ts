/// <reference types="node" />
import { SovaraClient, autoInstrument } from "@sovara/sdk";
import { instrumentOpenAIAgents } from "@sovara/openai-agents";
import { Agent, run } from "@openai/agents";

await autoInstrument();
await instrumentOpenAIAgents();

const spanishAgent = new Agent({
  name: "Spanish agent",
  instructions: "You only speak Spanish.",
});

const englishAgent = new Agent({
  name: "English agent",
  instructions: "You only speak English.",
});

const triageAgent = Agent.create({
  name: "Triage agent",
  instructions: "Handoff to the appropriate agent based on the language of the request.",
  handoffs: [spanishAgent, englishAgent],
});

async function runAgent() {
  const result = await run(triageAgent, "Hola, ¿cómo estás?");
  console.log(result.finalOutput);
}

const sovara_client = new SovaraClient({ projectName: "sovara-typescript-examples" });

await sovara_client.trace(
  "ts-openai-async-agents",
  () => runAgent(),
);
