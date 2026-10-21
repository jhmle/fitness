# 🏋️ AI Trainer

**A personal trainer that runs entirely in your browser.** It watches you through your webcam, counts your reps, checks your form, and coaches you out loud. No account, no server, and no video ever leaves your device.



## What it does

- **Counts reps and sets** for squats, push-ups and bicep curls using your camera
- **Checks your form** (depth, posture, speed) and flags what to fix
- **Talks to you** with short spoken cues like "Chest up" or "Go a bit deeper"
- **Listens to you**, so you can say things like "30 seconds rest" or "switch to push-ups" hands-free
- **Private by design:** every model runs locally on your device after a one-time download (~250 MB)

## How it works

```
Camera → Pose tracking → Rep counter + form rules → AI coach → Voice
```

| Step | What happens | Tech |
|---|---|---|
| **See** | Tracks 33 body points in real time | [MediaPipe Pose](https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker) |
| **Count and check** | Joint angles drive a rep counter and form checks | Custom rules in JavaScript |
| **Think** | A small language model picks the right cue from a coaching knowledge base | Fine-tuned [SmolLM2-360M](https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct), with retrieval (RAG) |
| **Hear** | Detects speech, then transcribes it | [Silero VAD](https://github.com/snakers4/silero-vad) + [Moonshine](https://github.com/moonshine-ai/moonshine) |
| **Speak** | Reads the coaching cue aloud | [Kokoro TTS](https://huggingface.co/hexgrad/Kokoro-82M) |

Normal reps get an instant, rule-based response. The language model only steps in when something needs judgment: a form error, the end of a set, or a spoken question. If the model ever gives an unusable answer, the app falls back to the rules, so the coach always responds.

## The ML side

- **Fine-tuned a small LLM** (QLoRA on SmolLM2-360M) to act as the "coach brain", outputting structured tool calls for the app to run
- **Synthetic training data** generated programmatically, with a frozen test set for fair comparison
- **Evaluation and quality gate** that compares each new model against the one in production and flags regressions
- **Exported to ONNX (4-bit)** so the model runs in the browser via WebGPU/WASM
- **Experiment tracking** with MLflow; training runs on a free Kaggle GPU
- **Device profiles:** desktop gets the full pipeline; mobile gets a lighter pose model and the browser's built-in voice

## Screenshots

<p align="center">
  <img src="assets/setup.png" alt="Model loading screen" width="45%">
  &nbsp;
  <img src="assets/controls.png" alt="Session controls" width="45%">
</p>

## Try it

```bash
cd web
python -m http.server 8000
```

Open http://localhost:8000 in desktop Chrome or Edge, click **Load models**, then **Start session**. Stand side-on to the camera with your whole body in view.

## Project layout

```
web/        The app: pose tracking, rep engine, agent, voice workers
training/   Data generation, fine-tuning, evaluation, quality gate, ONNX export
```

## Built with

MediaPipe · Moonshine · Kokoro · SmolLM2 · Silero VAD · ONNX Runtime Web · Hugging Face Transformers.js · PEFT / bitsandbytes · MLflow

---

*Coaching cues are general guidance, not medical advice. Stop if something hurts.*
