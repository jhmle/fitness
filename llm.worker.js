// Coach brain: small instruct LLM (decisions) + MiniLM embeddings (RAG over the knowledge base).
import { TRANSFORMERS_URL, LLM_ID, EMBED_ID } from "./config.js";
let T, gen, emb, kb = [], kv = [];
const post = (m) => self.postMessage(m);
const prog = (lo, hi) => (p) => p.status === "progress_total" && post({ type: "progress", pct: lo + (hi - lo) * p.progress / 100 });
const dot = (a, b) => { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * b[i]; return s; };

self.onmessage = async ({ data }) => {
  try {
    if (data.type === "load") {
      T = await import(TRANSFORMERS_URL);
      let llm = false, e = false;
      try {                                                    // embeddings first (tiny), then index the KB
        emb = await T.pipeline("feature-extraction", EMBED_ID, { dtype: "q8", device: "wasm", progress_callback: prog(0, 15) });
        kb = data.kb; kv = [];
        for (const d of kb) kv.push((await emb(d.text, { pooling: "mean", normalize: true })).data);
        e = true;
      } catch (x) { post({ type: "warn", message: "Embeddings: " + (x.message || x) }); }
      try {
        gen = await T.pipeline("text-generation", LLM_ID, { dtype: "q4", device: self.navigator?.gpu ? "webgpu" : "wasm", progress_callback: prog(15, 100) });
        llm = true;
      } catch (x) { post({ type: "warn", message: "LLM: " + (x.message || x) }); }
      post({ type: "ready", llm, emb: e });
    } else if (data.type === "chat") {
      const out = await gen(data.messages, { max_new_tokens: 70, do_sample: false, return_full_text: false });
      const g = out[0].generated_text;
      post({ type: "result", id: data.id, text: Array.isArray(g) ? g.at(-1).content : g });
    } else if (data.type === "search") {
      const q = (await emb(data.query, { pooling: "mean", normalize: true })).data;
      const ids = kb.map((d, i) => ({ id: d.id, ok: d.ex === data.ex || d.ex === "any", s: dot(q, kv[i]) }))
        .filter((d) => d.ok && d.s > 0.15).sort((a, b) => b.s - a.s).slice(0, data.k || 1).map((d) => d.id);
      post({ type: "result", id: data.id, ids });
    }
  } catch (e) { post({ type: data.id ? "result" : "error", id: data.id, error: true, message: String(e.message || e) }); }
};
