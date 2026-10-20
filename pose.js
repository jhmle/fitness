import { MEDIAPIPE_VERSIONS, PROFILE } from "./config.js";
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
  const errs = [];
  for (const v of MEDIAPIPE_VERSIONS) {
    const base = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${v}`;
    let mod, fs;
    try { mod = await import(base + "/vision_bundle.mjs"); fs = await mod.FilesetResolver.forVisionTasks(base + "/wasm"); }
    catch (e) { errs.push(`${v} load: ${e.message || e}`); continue; }
    for (const delegate of ["GPU", "CPU"]) {
      try {
        lm = await mod.PoseLandmarker.createFromOptions(fs, { baseOptions: { modelAssetPath: PROFILE.pose, delegate }, runningMode: "VIDEO", numPoses: 1 });
        return `MediaPipe ${v} (${delegate})`;
      } catch (e) { errs.push(`${v} ${delegate}: ${e.message || e}`); }
    }
  }
  throw new Error(errs.join(" | "));
}

export function detect(video, t) {
  const r = lm.detectForVideo(video, t);
  if (!r.landmarks.length) return null;
  const f = (L, F) => L.map((p, i) => ({ x: F[i][0].f(p.x, t), y: F[i][1].f(p.y, t), z: F[i][2] ? F[i][2].f(p.z, t) : 0, visibility: p.visibility }));
  return { img: f(r.landmarks[0], FI), world: f(r.worldLandmarks[0], FW) };
}
