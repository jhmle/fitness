// Coach brain: small instruct LLM (decisions) + MiniLM embeddings (RAG over the knowledge base).
import { TRANSFORMERS_URL, LLM_ID, EMBED_ID } from "./config.js";
let T, gen, emb, kb = [], kv = [], curId = null, curDevice = null, curDtype = null;
const post = (m) => self.postMessage(m);
const prog = (lo, hi) => (p) => p.status === "progress_total" && post({ type: "progress", pct: lo + (hi - lo) * p.progress / 100 });
// Pull the assistant's text out of whatever shape the pipeline returns (string, chat array, or nested content).
const pick = (out) => {
  let g = Array.isArray(out) ? out[0] : out;
  g = g?.generated_text ?? g;
  if (Array.isArray(g)) { const m = [...g].reverse().find((x) => x && x.role === "assistant") || g.at(-1); g = m?.content ?? m; }
  if (Array.isArray(g)) g = g.map((x) => x?.text ?? x).join("");
  return typeof g === "string" ? g : "";
};
const dot = (a, b) => { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * b[i]; return s; };

// Load the model on the first device/quantisation combo that can actually GENERATE. Each candidate gets a short warm-up run
// (prompt + a few decode steps), because some WebGPU builds load fine but crash inside OrtRun on the first real generation.
const WARM = [{ role: "system", content: "You are a helpful voice assistant. Answer in one short sentence." }, { role: "user", content: "Tell me one quick tip for doing a good squat." }];
async function loadLLM(id, wasmOnly = false) {
  const errs = [];
  const plans = [];
  if (self.navigator?.gpu && !wasmOnly) for (const d of ["q4", "q4f16", "fp16"]) plans.push(["webgpu", d]);
  for (const d of ["q8", "q4", "uint8"]) plans.push(["wasm", d]);
  for (const [device, dtype] of plans) {
    let p;
    try {
      p = await T.pipeline("text-generation", id, { dtype, device, progress_callback: prog(15, 100) });
      await p(WARM, { max_new_tokens: 6, do_sample: false, return_full_text: false });   // proves it really runs
      curId = id; curDevice = device; curDtype = dtype; return p;
    } catch (x) {
      errs.push(`${device}/${dtype}: ${String(x.message || x).slice(0, 110)}`);
      try { await p?.dispose?.(); } catch {}
    }
  }
  throw new Error(errs.join(" | "));
}

const handle = async ({ data }) => {
  try {
    if (data.type === "load") {
      T = await import(TRANSFORMERS_URL);
      let llm = false, e = false, why = "";
      try {                                                    // embeddings first (tiny), then index the KB
        emb = await T.pipeline("feature-extraction", EMBED_ID, { dtype: "q8", device: "wasm", progress_callback: prog(0, 15) });
        kb = data.kb; kv = [];
        for (const d of kb) kv.push((await emb(d.text, { pooling: "mean", normalize: true })).data);
        e = true;
      } catch (x) { post({ type: "warn", message: "Embeddings: " + (x.message || x) }); }
      let used = null;
      try { gen = await loadLLM(LLM_ID); llm = true; used = LLM_ID; }   // fine-tuned model only: no fallback
      catch (x) { why = `LLM ${LLM_ID}: ${x.message || x}`; post({ type: "warn", message: why }); }
      post({ type: "ready", llm, emb: e, model: used, why, run: curDevice + "/" + curDtype });
    } else if (data.type === "chat") {
      const run = () => gen(data.messages, { max_new_tokens: data.max || 70, do_sample: false, return_full_text: false });
      let out;
      try { out = await run(); }
      catch (x) {                                              // WebGPU generation failed: rebuild on WASM (slower, but reliable) and retry once
        if (curDevice === "wasm") throw x;
        post({ type: "warn", message: "WebGPU generation failed (" + String(x.message || x).slice(0, 120) + "), switching the coach model to WASM" });
        gen = await loadLLM(curId, true); out = await run();
      }
      post({ type: "result", id: data.id, text: pick(out), raw: JSON.stringify(out).slice(0, 300) });
    } else if (data.type === "search") {
      const q = (await emb(data.query, { pooling: "mean", normalize: true })).data;
      const ids = kb.map((d, i) => ({ id: d.id, ok: d.ex === data.ex || d.ex === "any", s: dot(q, kv[i]) }))
        .filter((d) => d.ok && d.s > 0.15).sort((a, b) => b.s - a.s).slice(0, data.k || 1).map((d) => d.id);
      post({ type: "result", id: data.id, ids });
    }
  } catch (e) { post({ type: data.id ? "result" : "error", id: data.id, error: true, message: String(e.message || e) }); }
};
// One job at a time: the ONNX session throws if two generations (e.g. a rep cue and a chat answer) overlap.
let queue = Promise.resolve();
self.onmessage = (ev) => { queue = queue.then(() => handle(ev)); };
