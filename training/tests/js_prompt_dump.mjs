// Feeds rows through the REAL agent.js ask() with a mock host and prints the exact chat messages it would send to the LLM.
import { readFileSync, writeFileSync } from "node:fs";
const { Agent } = await import(process.argv[2]);
const rows = JSON.parse(readFileSync(process.argv[3], "utf8"));
const out = [];
for (const r of rows) {
  let cap = null;
  const host = { policy: () => "llm", llm: { ready: () => true, search: async () => r.knowledge, chat: async (m) => { cap = m; return "{}"; } } };
  const a = new Agent(host); a.ex = r.state.exercise;
  await a.ask(r.kind, r.state, r.allow, r.said || "");
  out.push(cap);
}
writeFileSync(process.argv[4], JSON.stringify(out));   // file, not stdout: piped stdout can truncate on exit
process.exit(0);   // ask() leaves its timeout timers running
