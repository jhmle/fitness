# AI Trainer (on-device)

Pose (MediaPipe) -> rep state machine + form rules -> agent (tools + policy) -> voice (Silero VAD, Moonshine Tiny, Kokoro).

Run: `cd web && python -m http.server 8000`, open http://localhost:8000 in desktop Chrome/Edge (camera needs localhost or HTTPS).
Stand side-on to the camera with your whole body visible.

Profiles live in `web/config.js` (desktop: full pose + Kokoro; mobile: lite pose + browser voice).
Next: `training/` (GRU/TCN form model), `kb/` (RAG), `LLMPolicy` in `agent.js`, CI/MLflow.
