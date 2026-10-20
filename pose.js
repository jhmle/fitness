import { MEDIAPIPE_URL, PROFILE } from "./config.js";
let lm;

class OneEuro {                       // adaptive low-pass: smooth when still, responsive when moving
  constructor(mc, beta) { this.mc = mc; this.beta = beta; this.t = null; }
  a(c, dt) { const tau = 1 / (2 * Math.PI * c); return 1 / (1 + tau / dt); }
  f(x, t) {
    if (this.t == null) { this.x = x; this.dx = 0; this.t = t; return x; }
    const dt = Math.max((t - this.t) / 1000, 1e-3);
    this.dx += this.a(1, dt) * ((x - this.x) / dt - this.dx);
    this.x += this.a(this.mc + this.beta * Math.abs(this.dx), dt) * (x - this.x);
    this.t = t; return this.x;
  }
}
const mk = (n, mc, b) => Array.from({ length: 33 }, () => Array.from({ length: n }, () => new OneEuro(mc, b)));
const FW = mk(3, 1.5, 1), FI = mk(2, 2, 30);   // world (meters) and image (0..1) filters

export async function loadPose() {
  const { FilesetResolver, PoseLandmarker } = await import(MEDIAPIPE_URL);
  const fs = await FilesetResolver.forVisionTasks(MEDIAPIPE_URL + "/wasm");
  const make = (delegate) => PoseLandmarker.createFromOptions(fs, { baseOptions: { modelAssetPath: PROFILE.pose, delegate }, runningMode: "VIDEO", numPoses: 1 });
  try { lm = await make("GPU"); } catch { lm = await make("CPU"); }
}

export function detect(video, t) {
  const r = lm.detectForVideo(video, t);
  if (!r.landmarks.length) return null;
  const f = (L, F) => L.map((p, i) => ({ x: F[i][0].f(p.x, t), y: F[i][1].f(p.y, t), z: F[i][2] ? F[i][2].f(p.z, t) : 0, visibility: p.visibility }));
  return { img: f(r.landmarks[0], FI), world: f(r.worldLandmarks[0], FW) };
}
