import { KOKORO_URL, KOKORO_ID } from "./config.js";
let tts;
const post = (m, t) => self.postMessage(m, t || []);

self.onmessage = async ({ data }) => {
  try {
    if (data.type === "load") {
      const { KokoroTTS } = await import(KOKORO_URL);
      tts = await KokoroTTS.from_pretrained(KOKORO_ID, {
        dtype: "q8", device: "wasm", // switch to "webgpu" if GPU memory allows
        progress_callback: (p) => p.status === "progress" && post({ type: "progress", file: p.file, pct: p.progress }),
      });
      post({ type: "ready", voices: Object.keys(tts.voices) });
    } else if (data.type === "speak") {
      const out = await tts.generate(data.text, { voice: data.voice });
      post({ type: "audio", id: data.id, samples: out.audio, rate: out.sampling_rate }, [out.audio.buffer]);
    }
  } catch (e) { post({ type: "error", id: data && data.id, message: String(e.message || e) }); }
};
