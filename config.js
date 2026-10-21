// VERIFY BEFORE DEPLOY: pin versions you have tested. Sizes/IDs below are from memory.
export const TRANSFORMERS_URL = "https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.3.0";
export const KOKORO_URL = "https://cdn.jsdelivr.net/npm/kokoro-js@1.2.1/+esm";
// Tried in order until one loads (some builds break on certain browsers/GPUs). Verify against npm.
export const MEDIAPIPE_VERSIONS = ["0.10.14", "0.10.22", "0.10.3"];
export const VAD_URL = "https://cdn.jsdelivr.net/npm/@ricky0123/vad-web@0.0.24/dist/";   // Silero VAD
export const ORT_URL = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/";
export const MOONSHINE_ID = "onnx-community/moonshine-tiny-ONNX";
export const LLM_ID = "jhmle/trainer-coach-smollm2-360m-onnx-20261021-0947";   // the ONLY coach model: fine-tuned ONNX export on Hugging Face (repo must be public)
export const EMBED_ID = "Xenova/all-MiniLM-L6-v2";             // knowledge-base search
export const KOKORO_ID = "onnx-community/Kokoro-82M-v1.0-ONNX";

const pose = (v) => `https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_${v}/float16/1/pose_landmarker_${v}.task`;
const mobile = /Android|iPhone|iPad|iPod/i.test(navigator.userAgent) || (navigator.deviceMemory || 8) <= 4;

// Device profiles: desktop gets the fuller pose model + Kokoro; mobile gets lite pose + the browser's built-in voice.
export const PROFILE = mobile
  ? { name: "mobile", pose: pose("lite"), tts: "browser", llm: false }
  : { name: "desktop", pose: pose("full"), tts: "kokoro", llm: true };
