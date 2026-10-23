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
      post({ type: "ready", voices });                 // ready NOW: never make the page wait for the warm-up
      // Warm-up in the background: the first generation pays for one-off setup (phonemizer + WASM compile), slow in Safari.
      // The page uses the browser voice until this finishes, so a slow or hanging first run can't block anything.
      log("Kokoro: warming up (first sentence is the slow one)…");
      const run = gen("Ready.", voices.includes("af_heart") ? "af_heart" : voices[0]);
      try { post({ type: "warm", ms: (await (data.strict ? run : withTimeout(run, 60000, "Kokoro warm-up"))).ms }); }   // strict: wait as long as it takes
      catch (e) { post({ type: "warm", bad: true, why: String(e.message || e) }); }
    } else if (data.type === "speak") {
      log("Kokoro: generating…");
      const { out, ms } = await gen(data.text, data.voice);
      post({ type: "audio", id: data.id, samples: out.audio, rate: out.sampling_rate, ms }, [out.audio.buffer]);
    }
  } catch (e) { post({ type: "error", id: data && data.id, message: String(e.message || e) }); }
}
// One job at a time: an ONNX session throws if two generations overlap.
let queue = Promise.resolve();
self.onmessage = (ev) => { queue = queue.then(() => handle(ev)); };
