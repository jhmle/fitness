// VERIFY BEFORE DEPLOY: pin versions you have tested. Sizes/IDs below are from memory.
export const TRANSFORMERS_URL = "https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.3.0";
export const KOKORO_URL = "https://cdn.jsdelivr.net/npm/kokoro-js@1.2.1/+esm";
// Tried in order until one loads (some builds break on certain browsers/GPUs). Verify against npm.
export const MEDIAPIPE_VERSIONS = ["0.10.14", "0.10.22", "0.10.3"];
export const VAD_URL = "https://cdn.jsdelivr.net/npm/@ricky0123/vad-web@0.0.24/dist/";   // Silero VAD
export const ORT_URL = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/";
export const MOONSHINE_ID = "onnx-community/moonshine-tiny-ONNX";
export const LLM_ID = "onnx-community/SmolLM2-360M-Instruct";   // PRODUCTION coach brain. training/gate.py checks this exact line against metrics/production.json
export const TUNED_ID = "jhmle/trainer-coach-smollm2-360m-onnx";   // fine-tuned candidate (ONNX export of the merged QLoRA model on Hugging Face; the repo must be public)
export const LLM_CHOICES = [{ id: LLM_ID, label: "Stock SmolLM2-360M" }, { id: TUNED_ID, label: "Fine-tuned (jhmle)" }];
const okId = (v) => typeof v === "string" && /^[\w.-]+\/[\w.-]+$/.test(v);
// Which model to load: ?llm=owner/repo in the URL, else the saved dropdown choice, else the production default.
export function chosenLLM() {
  try { const u = new URLSearchParams(location.search).get("llm"); if (okId(u)) return u; } catch {}
  try { const v = localStorage.getItem("trainer.llm"); if (okId(v)) return v; } catch {}   // localStorage doesn't exist inside workers: caught
  return LLM_ID;
}
export const EMBED_ID = "Xenova/all-MiniLM-L6-v2";             // knowledge-base search
export const KOKORO_ID = "onnx-community/Kokoro-82M-v1.0-ONNX";

const pose = (v) => `https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_${v}/float16/1/pose_landmarker_${v}.task`;
const mobile = /Android|iPhone|iPad|iPod/i.test(navigator.userAgent) || (navigator.deviceMemory || 8) <= 4;

// Device profiles: desktop gets the fuller pose model + Kokoro; mobile gets lite pose + the browser's built-in voice.
export const PROFILE = mobile
  ? { name: "mobile", pose: pose("lite"), tts: "browser", llm: false }
  : { name: "desktop", pose: pose("full"), tts: "kokoro", llm: true };
