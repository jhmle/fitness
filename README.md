# 🏋️ AI Trainer

**A personal trainer that runs entirely in your browser.** It watches you through your webcam, counts your reps, checks your form, and coaches you out loud. No account, no server, and no video ever leaves your device.

### 🔗 [Try the live demo](https://jhmle.github.io/fitness/)

> **Heads up: the AI runs on your own device, so the demo may not work on yours.** All the models run locally in your browser (about 250 MB, downloaded once).
> - **Best on:** a laptop or desktop with a webcam, a microphone and plenty of free RAM.
> - ⚠️ **Not recommended on phones or tablets:** they will likely run out of memory and crash or reload.
> - If the voice is slow or missing, switch on **⚡ Fast voice** (uses your browser's built-in voice instead of Kokoro).

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


---

*Coaching cues are general guidance, not medical advice. Stop if something hurts.*
