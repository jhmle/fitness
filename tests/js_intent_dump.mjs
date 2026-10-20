// Runs the REAL agent.js regexIntent for each {said, rest, target} item and writes the resulting calls.
import { readFileSync, writeFileSync } from "node:fs";
const { Agent } = await import(process.argv[2]);
const items = JSON.parse(readFileSync(process.argv[3], "utf8"));
const out = items.map((it) => {
  const a = new Agent({ policy: () => "rules", llm: null });
  a.rest = it.rest; a.target = it.target;
  return a.regexIntent(it.said);
});
writeFileSync(process.argv[4], JSON.stringify(out));
process.exit(0);
