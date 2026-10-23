import { TRANSFORMERS_URL, MOONSHINE_ID } from "./config.js";
let T, asr;
const post = (m) => self.postMessage(m);
self.onmessage = async ({ data }) => {
  try {
    if (data.type === "load") {
      T = await import(TRANSFORMERS_URL);
      // Default dtype on purpose: Moonshine Tiny is small. Try { dtype: "q8" } later and check the transcripts still hold up.
      asr = await T.pipeline("automatic-speech-recognition", MOONSHINE_ID, {
        device: "wasm",
        progress_callback: (p) => p.status === "progress_total" && post({ type: "progress", pct: p.progress }),
      });
      post({ type: "ready" });
    } else if (data.type === "transcribe") {      // Float32Array @ 16 kHz from the VAD
      const out = await asr(data.audio);
      post({ type: "text", text: (out.text || "").trim() });
    }
  } catch (e) { post({ type: "error", message: String(e.message || e) }); }
};
