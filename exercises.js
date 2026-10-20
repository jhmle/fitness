// Per-exercise joint-angle rules + a rep state machine. All "extended" angles are high, "flexed" are low.
const sub = (a, b) => [a.x - b.x, a.y - b.y, a.z - b.z];
export const angle = (a, b, c) => {
  const u = sub(a, b), v = sub(c, b);
  const d = u[0] * v[0] + u[1] * v[1] + u[2] * v[2], n = Math.hypot(...u) * Math.hypot(...v) || 1;
  return Math.acos(Math.max(-1, Math.min(1, d / n))) * 180 / Math.PI;
};
const fromVertical = (a, b) => { const v = sub(b, a); return Math.atan2(Math.hypot(v[0], v[2]), Math.abs(v[1])) * 180 / Math.PI; };

export const EXERCISES = {
  squat: {
    label: "Squat", up: 160, fast: 1.2,
    angle: (p) => angle(p.hip, p.knee, p.ankle),
    feat: (p) => ({ lean: fromVertical(p.hip, p.shoulder) }),
    check: (m) => [m.min > 110 && "squat_depth", m.f.lean > 60 && "squat_lean", m.dur < 1.2 && "too_fast"],
  },
  pushup: {
    label: "Push-up", up: 155, fast: 0.8,
    angle: (p) => angle(p.shoulder, p.elbow, p.wrist),
    feat: (p) => ({ line: angle(p.shoulder, p.hip, p.ankle) }),
    check: (m) => [m.min > 100 && "pushup_depth", m.f.line < 160 && "pushup_sag", m.dur < 0.8 && "too_fast"],
  },
  curl: {
    label: "Curl", up: 150, fast: 0.8,
    angle: (p) => angle(p.shoulder, p.elbow, p.wrist),
    feat: (p) => ({ lean: fromVertical(p.hip, p.shoulder) }),
    check: (m) => [m.min > 70 && "curl_range", m.f.lean > 15 && "curl_swing", m.dur < 0.8 && "too_fast"],
  },
};
// which joints to flash red when an error fires
const ERR_JOINTS = { squat_depth: ["hip", "knee"], squat_lean: ["shoulder", "hip"], pushup_depth: ["elbow"], pushup_sag: ["hip"],
  curl_range: ["elbow", "wrist"], curl_swing: ["shoulder", "hip"], too_fast: ["knee", "elbow"] };

export const SKELETON = [[11,12],[11,13],[13,15],[12,14],[14,16],[11,23],[12,24],[23,24],[23,25],[25,27],[24,26],[26,28]];
const J = { shoulder: 11, elbow: 13, wrist: 15, hip: 23, knee: 25, ankle: 27 };   // +1 = right side
const vis = (w, s) => Object.values(J).reduce((a, i) => a + (w[i + s].visibility ?? 1), 0) / 6;

export class RepEngine {
  constructor(name = "squat") { this.set(name); }
  set(name) { this.name = name; this.ex = EXERCISES[name]; this.phase = "up"; this.reps = 0; this.m = null; }
  update(w, t) {                       // w = world landmarks (meters), t = ms. Uses whichever body side is more visible.
    const s = vis(w, 0) >= vis(w, 1) ? 0 : 1;
    if (vis(w, s) < 0.5) return { ok: false };
    const p = {}; for (const k in J) p[k] = w[J[k] + s];
    const a = this.ex.angle(p), ex = this.ex;
    if (this.phase === "up") {
      if (a < ex.up - 25) { this.phase = "down"; this.m = { min: a, t0: t, f: ex.feat(p) }; }
    } else {
      if (a < this.m.min) { this.m.min = a; this.m.f = ex.feat(p); }
      if (a > ex.up - 8) {             // back to extended = rep complete
        this.phase = "up"; const m = this.m; m.dur = (t - m.t0) / 1000; this.reps++;
        const errors = ex.check(m).filter(Boolean);
        const joints = [...new Set(errors.flatMap((e) => ERR_JOINTS[e] || []))].map((k) => J[k] + s);
        return { ok: true, angle: a, event: { type: "rep_end", exercise: this.name, rep: this.reps, errors, joints, min: Math.round(m.min), dur: +m.dur.toFixed(2) } };
      }
    }
    return { ok: true, angle: a };
  }
}
