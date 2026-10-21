import { KOKORO_URL, KOKORO_ID } from "./config.js";
let tts;
const post = (m, t) => self.postMessage(m, t || []);
const log = (message) => post({ type: "log", message });
const withTimeout = (p, ms, what) => Promise.race([p, new Promise((_, rej) => setTimeout(() => rej(new Error(`${what} timed out after ${ms / 1000}s`)), ms))]);
const gen = async (text, voice) => { const t0 = performance.now(); const out = await tts.generate(text, { voice }); return { out, ms: Math.round(performance.now() - t0) }; };

async function handle({ data }) {
  try {
    if (data.type === "load") {
      log("Kokoro: loading library…");
      const { KokoroTTS } = await import(KOKORO_URL);
      log("Kokoro: loading model…");
      tts = await KokoroTTS.from_pretrained(KOKORO_ID, {
        dtype: "q8", device: "wasm",
        progress_callback: (p) => p.status === "progress" && post({ type: "progress", file: p.file, pct: p.progress }),
      });
      const voices = Object.keys(tts.voices);
      // Warm-up: the first generation pays for one-off setup (phonemizer + WASM compile), which is slow in Safari.
      // Doing it now, with nothing else competing, means the first coaching cue is fast, and tells us if Kokoro runs at all.
      log("Kokoro: warming up (first sentence is the slow one)…");
      let warm = 0, why = "";
      try { warm = (await withTimeout(gen("Ready.", voices.includes("af_heart") ? "af_heart" : voices[0]), 90000, "Kokoro warm-up")).ms; }
      catch (e) { why = String(e.message || e); }
      post({ type: "ready", voices, warm, bad: !!why, why });
    } else if (data.type === "speak") {
      const { out, ms } = await gen(data.text, data.voice);
      post({ type: "audio", id: data.id, samples: out.audio, rate: out.sampling_rate, ms }, [out.audio.buffer]);
    }
  } catch (e) { post({ type: "error", id: data && data.id, message: String(e.message || e) }); }
}
// One job at a time: an ONNX session throws if two generations overlap.
let queue = Promise.resolve();
self.onmessage = (ev) => { queue = queue.then(() => handle(ev)); };
