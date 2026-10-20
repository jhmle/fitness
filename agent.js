// Agent layer: observe (rep events + voice) -> decide (policy) -> act (validated tool calls) -> measure (cue effectiveness).
// A policy is anything with decide(state) -> [{tool,args}]. TemplatePolicy now; an LLMPolicy can drop in later.

// Tiny knowledge base. Later: ship as kb.json with embeddings and swap retrieve() for cosine search.
export const KB = {
  squat_depth:  { ex: "squat",  sev: 2, cue: "Go a bit deeper.", alt: "Sit your hips back and down until your thighs are near parallel.", why: "Shallow squats train the muscles less and shift load to the knees." },
  squat_lean:   { ex: "squat",  sev: 3, cue: "Chest up.", alt: "Brace your core and keep your chest tall as you sit down.", why: "Leaning too far forward loads the lower back." },
  pushup_depth: { ex: "pushup", sev: 2, cue: "Lower your chest more.", alt: "Bend your elbows to about ninety degrees before pushing up.", why: "Partial push-ups build less strength." },
  pushup_sag:   { ex: "pushup", sev: 3, cue: "Hips up, stay straight.", alt: "Squeeze your glutes and abs so your body stays like a plank.", why: "A sagging hip strains the lower back." },
  curl_range:   { ex: "curl",   sev: 2, cue: "Curl all the way up.", alt: "Bring the weight up until your forearm is nearly vertical.", why: "Partial range leaves strength gains on the table." },
  curl_swing:   { ex: "curl",   sev: 3, cue: "Keep your torso still.", alt: "Pin your elbows at your sides and stop swinging.", why: "Swinging uses momentum instead of the biceps." },
  too_fast:     { ex: "any",    sev: 1, cue: "Slow down, control the lowering.", alt: "Take about two seconds on the way down.", why: "A controlled tempo is safer and builds more strength." },
  knee_pain:    { ex: "squat",  sev: 0, cue: "", alt: "Push your knees out over your toes and sit back. If the pain is sharp, stop.", why: "Knee discomfort often comes from knees collapsing inward or too much forward shift." },
};
const NAMES = ["squat", "pushup", "curl"];

export function retrieve(query, ex, k = 1) {      // metadata filter + keyword overlap (embeddings come later)
  const q = (query.toLowerCase().match(/[a-z]{4,}/g) || []);
  return Object.entries(KB).filter(([, e]) => e.ex === ex || e.ex === "any")
    .map(([id, e]) => ({ id, ...e, score: q.filter((w) => (id + e.cue + e.alt + e.why).toLowerCase().includes(w)).length }))
    .filter((e) => e.score > 0).sort((a, b) => b.score - a.score).slice(0, k);
}

export class TemplatePolicy {
  decide(s) {
    if (s.rep >= s.target) return [{ tool: "set_summary", args: {} }, { tool: "rest_timer", args: { seconds: 60 } }];
    const top = [...s.errors].sort((a, b) => b.sev - a.sev)[0];
    return [{ tool: "give_cue", args: { cue_id: top ? top.id : "rep_count", tone: "firm" } }];
  }
}

export class Agent {
  constructor(host) {
    this.h = host; this.policy = new TemplatePolicy();
    this.ex = "squat"; this.target = 8; this.rest = 60; this.set = 1; this.reps = []; this.last = null; this.lastCueRep = -9;
    try { this.stats = JSON.parse(localStorage.getItem("trainer.cueStats") || "{}"); } catch { this.stats = {}; }
    this.log = [];                                   // session trace: the raw material for later evals / fine-tuning
    this.tools = {
      give_cue: ({ cue_id }) => {
        const n = this.reps.length;
        if (cue_id === "rep_count") return this.h.say(String(n));
        const k = KB[cue_id]; if (!k) return;
        if (n - this.lastCueRep < 2) return this.h.say(String(n));          // rate limit: one correction per two reps
        const s = this.stats[cue_id];
        const text = s && s.n >= 4 && s.fixed / s.n < 0.3 ? k.alt : k.cue;  // cue isn't working -> rephrase
        this.lastCueRep = n; this.last = { id: cue_id, rep: n }; this.h.say(text);
      },
      retrieve_cues: ({ query }) => {
        const r = retrieve(query, this.ex)[0];
        this.h.say(r ? `${r.why} ${r.alt}` : "I don't have a tip for that yet.", { urgent: true });
      },
      set_summary: () => {
        const c = {}; this.reps.forEach((e) => e.errors.forEach((id) => (c[id] = (c[id] || 0) + 1)));
        const top = Object.entries(c).sort((a, b) => b[1] - a[1])[0], clean = this.reps.filter((e) => !e.errors.length).length;
        let t = `Set ${this.set}: ${this.reps.length} reps, ${clean} clean.`;
        if (top) t += ` Main issue: ${KB[top[0]].alt}`;
        if (top && top[1] >= this.reps.length / 2) t += this.tools.adjust_plan({ rest: 90, target: Math.max(4, this.target - 2) });
        this.h.say(t + ` Rest ${this.rest} seconds.`, { urgent: true });
        this.set++; this.reps = []; this.last = null;
      },
      adjust_plan: ({ rest, target }) => { this.rest = rest; this.target = target; return ` Let's drop to ${target} reps and rest longer.`; },
      rest_timer: ({ seconds }) => {
        clearTimeout(this.restT); const s = Math.min(300, Math.max(10, seconds | 0));
        this.restT = setTimeout(() => this.h.say("Rest is over. Ready when you are."), s * 1000);
      },
      start_exercise: ({ name }) => {
        if (!NAMES.includes(name)) return;
        this.ex = name; this.set = 1; this.reps = []; this.last = null; this.h.startExercise(name);
        this.h.say(`Starting ${name === "pushup" ? "push-ups" : name + "s"}. Get side-on to the camera.`, { urgent: true });
      },
      flag_pain_stop: () => {
        clearTimeout(this.restT); this.h.pause();
        this.h.say("Stop and rest now. If the pain is sharp or you feel dizzy, don't continue and get medical advice.", { urgent: true });
      },
    };
  }
  exec(call) {                                       // validate against the allowed-tool list before acting
    if (!call || typeof this.tools[call.tool] !== "function") return;
    this.log.push({ t: Date.now(), call });
    try { this.tools[call.tool](call.args || {}); } catch (e) { console.error(e); }
  }
  state(ev) {
    return { exercise: this.ex, set: this.set, rep: this.reps.length, target: this.target,
      errors: ev.errors.map((id) => ({ id, sev: KB[id].sev })),
      last_cue: this.last && { id: this.last.id, reps_ago: this.reps.length - this.last.rep, fixed: this.last.fixed } };
  }
  onRep(ev) {
    this.reps.push(ev);
    if (this.last && this.last.fixed === undefined && this.reps.length > this.last.rep) {   // did the cue work?
      this.last.fixed = !ev.errors.includes(this.last.id);
      const s = (this.stats[this.last.id] ??= { n: 0, fixed: 0 }); s.n++; s.fixed += this.last.fixed;
      try { localStorage.setItem("trainer.cueStats", JSON.stringify(this.stats)); } catch {}
    }
    const st = this.state(ev); this.log.push({ t: Date.now(), state: st });
    for (const c of this.policy.decide(st)) this.exec(c);
  }
  onVoice(text) {                                    // keyword intents now; an LLM maps text -> tool call later
    const t = text.toLowerCase();
    let c;
    if (/pain|sharp|dizzy|lightheaded|faint|can't breathe/.test(t)) c = { tool: "flag_pain_stop" };
    else if (/start|begin|switch|let's do/.test(t) && /squat|push|curl/.test(t)) c = { tool: "start_exercise", args: { name: /squat/.test(t) ? "squat" : /push/.test(t) ? "pushup" : "curl" } };
    else if (/how (was|did)|summary|finish|done|end (the )?set/.test(t)) c = { tool: "set_summary" };
    else if (/rest|break/.test(t)) c = { tool: "rest_timer", args: { seconds: this.rest } };
    else if (/\b(why|how|what|should|tips?|help)\b/.test(t)) c = { tool: "retrieve_cues", args: { query: text } };
    this.exec(c);
    return !!c;
  }
}
