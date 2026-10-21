# 🏋️ AI Trainer

**A personal trainer that runs entirely in your browser.** It watches you through your webcam, counts your reps, checks your form, and coaches you out loud. No account, no server, and no video ever leaves your device.

## What it does

- **Counts reps and sets** for squats, push-ups, and bicep curls using your camera
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

## Try it

```bash
cd fitness
python -m http.server 8000
```

Open http://localhost:8000 in desktop Chrome or Edge, click **Load models**, then **Start session**. Stand side-on to the camera with your whole body in view.

---

*Coaching cues are general guidance, not medical advice. Stop if something hurts.*
