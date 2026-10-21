# 🏋️ AI Trainer

**A personal trainer that runs entirely in your browser.** It watches you through your webcam, counts your reps, checks your form, and coaches you out loud. No account, no server, and no video ever leaves your device.

## What it does

- **Counts reps and sets** for squats, push-ups, and bicep curls using your camera
- **Checks your form** (depth, posture, speed) and flags what to fix
- **Talks to you** with short spoken cues like "Chest up" or "Go a bit deeper"
- **Listens to you**, so you can say things like "30 seconds rest" or "switch to push-ups" hands-free
- **Private by design:** every model runs locally on your device after a one-time download (~250 MB)

## Try it

```bash
cd fitness
python -m http.server 8000
```

Open http://localhost:8000 , click **Load models**, then **Start session**.

---

*Coaching cues are general guidance, not medical advice. Stop if something hurts.*
