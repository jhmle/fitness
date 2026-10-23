// Agent: observe (rep events + voice) -> retrieve -> decide (LLM or rules) -> validate -> act -> measure (did the cue fix it?).
// Clean reps stay on the instant reflex path; the LLM is consulted only on errors, set ends and voice.

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
const PAIN = /pain|sharp|dizzy|lightheaded|faint|can't breathe/i;
export const kbDocs = () => Object.entries(KB).map(([id, e]) => ({ id, ex: e.ex, text: `${id.replace(/_/g, " ")}. ${e.cue} ${e.alt} ${e.why}` }));
const avg = (a) => a.reduce((x, y) => x + y, 0) / (a.length || 1);
const NUM = { one: 1, two: 2, three: 3, four: 4, five: 5, six: 6, seven: 7, eight: 8, nine: 9, ten: 10, eleven: 11, twelve: 12, thirteen: 13, fourteen: 14, fifteen: 15 };
const WORDS = Object.keys(NUM).join("|");
const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));
export function parseSeconds(t) {                 // "30 seconds", "two minutes", "half a minute", "ninety seconds" -> seconds | null
  let m;
  if (/three quarters of a minute/.test(t)) return 45;
  if (/quarter of a minute/.test(t)) return 15;
  if ((m = t.match(/(\d+)\s*(?:seconds?|secs?)\b/))) return +m[1];
  if (/a minute and a half/.test(t)) return 90;
  if (/half a minute/.test(t)) return 30;
  if ((m = t.match(new RegExp(`\\b(\\d+|${WORDS})\\s*(?:minutes?|mins?)\\b`)))) return 60 * (NUM[m[1]] ?? +m[1]);
  if (/\b(?:a|one) minute\b/.test(t)) return 60;
  if ((m = t.match(/\b(fifteen|thirty|forty[ -]five|sixty|ninety)\s*(?:seconds?|secs?)\b/))) return { fifteen: 15, thirty: 30, "forty five": 45, sixty: 60, ninety: 90 }[m[1].replace("-", " ")];
  return null;
}
const repsIn = (t) => { const m = t.match(new RegExp(`\\b(\\d{1,2}|${WORDS})\\s+reps?\\b`)); return m ? (NUM[m[1]] ?? +m[1]) : null; };
const timeout = (ms) => new Promise((_, rej) => setTimeout(() => rej(new Error("timeout")), ms));

export function retrieve(query, ex, k = 1) {      // keyword fallback if the embedder isn't available
  const q = (query.toLowerCase().match(/[a-z]{4,}/g) || []);
  return Object.entries(KB).filter(([, e]) => e.ex === ex || e.ex === "any")
    .map(([id, e]) => ({ id, ...e, score: q.filter((w) => (id + e.cue + e.alt + e.why).toLowerCase().includes(w)).length }))
    .filter((e) => e.score > 0).sort((a, b) => b.score - a.score).slice(0, k);
}

export class TemplatePolicy {
  decide(s) {
    if (s.rep >= s.target) return [{ tool: "set_summary", args: {} }, ...(s.set < s.sets ? [{ tool: "rest_timer", args: { seconds: s.rest } }] : [])];
    const top = [...s.errors].sort((a, b) => b.sev - a.sev)[0];
    return [{ tool: "give_cue", args: { cue_id: top ? top.id : "rep_count", tone: "firm" } }];
  }
}

const SYSTEM = `You are the decision module of a workout coach. Reply with ONE line of JSON and nothing else.
Tools:
{"tool":"give_cue","args":{"cue_id":"<id from errors>","text":"<spoken cue, max 12 plain words>"}}
{"tool":"adjust_plan","args":{"rest":<10-300>,"target":<3-20>}}
{"tool":"retrieve_cues","args":{"query":"<question>"}}
{"tool":"rest_timer","args":{"seconds":<10-300>}}
{"tool":"start_exercise","args":{"name":"squat|pushup|curl"}}
{"tool":"set_summary","args":{}}
Rules: only cue an error listed in STATE. Use KNOWLEDGE to word the cue. If last_cue.fixed is false, word it differently. At set end, if trend.depth_loss_deg > 12 or STATE has 2 or more errors, call adjust_plan with rest+30 (max 300) and target-2 (min 3); otherwise call adjust_plan with the current rest and target unchanged. For VOICE events pick the tool that matches what the athlete said: start or switch exercise -> start_exercise; summary or done -> set_summary; a question or request for tips -> retrieve_cues; rest or break -> rest_timer with the seconds said, else the current rest; plan changes -> adjust_plan computed from STATE rest and target: easier = rest+30 and target-2; fewer reps = target-2; more reps = target+2 (max 20); longer rest = rest+30; shorter rest = rest-15 (min 10); "N reps" = target N; "set the rest to X" = rest X.
Examples:
EVENT REP STATE {"errors":[{"id":"squat_lean"}],"last_cue":null}
{"tool":"give_cue","args":{"cue_id":"squat_lean","text":"Chest up and brace your core."}}
EVENT VOICE SAID "why do my knees hurt"
{"tool":"retrieve_cues","args":{"query":"why do my knees hurt"}}
EVENT VOICE SAID "switch to push ups"
{"tool":"start_exercise","args":{"name":"pushup"}}
EVENT VOICE SAID "fewer reps please"
STATE {"rest":60,"target":8,"errors":[]}
{"tool":"adjust_plan","args":{"rest":60,"target":6}}
EVENT REP STATE {"rep":8,"target":8,"rest":60,"errors":[{"id":"squat_lean"},{"id":"too_fast"}],"trend":{"depth_loss_deg":4}}
{"tool":"adjust_plan","args":{"rest":90,"target":6}}`;

function validate(c, allow, errors) {              // never trust model output: whitelist tool + check every arg
  if (!c || typeof c.tool !== "string" || !allow.includes(c.tool)) return null;
  const a = c.args || {}, n = (v, lo, hi) => Number.isFinite(+v) && +v >= lo && +v <= hi;
  switch (c.tool) {
    case "give_cue": return errors.includes(a.cue_id) ? { tool: c.tool, args: { cue_id: a.cue_id, text: typeof a.text === "string" ? a.text.replace(/[^\w ,.'!?-]/g, "").trim().slice(0, 100) : "" } } : null;
    case "adjust_plan": return n(a.rest, 10, 300) && n(a.target, 3, 20) ? { tool: c.tool, args: { rest: +a.rest, target: +a.target } } : null;
    case "retrieve_cues": return typeof a.query === "string" && a.query.length < 200 ? c : null;
    case "rest_timer": return n(a.seconds, 10, 300) ? { tool: c.tool, args: { seconds: +a.seconds } } : null;
    case "start_exercise": return NAMES.includes(a.name) ? c : null;
    case "set_summary": return { tool: c.tool, args: {} };
  }
  return null;
}

export class Agent {
  constructor(host) {
    this.h = host; this.template = new TemplatePolicy(); this.seq = 0; this.src = "template";
    this.ex = "squat"; this.target = 8; this.sets = 3; this.rest = 60; this.set = 1; this.reps = []; this.last = null; this.lastCueRep = -9; this.adjNote = "";
    const load = (k, d) => { try { return JSON.parse(localStorage.getItem(k)) || d; } catch { return d; } };
    this.stats = load("trainer.cueStats", {}); this.pstats = load("trainer.pstats", {}); this.mem = load("trainer.mem", {});
    this.log = [];                                  // session trace: raw material for evals / fine-tuning
    this.tools = {
      give_cue: ({ cue_id, text }) => {
        const n = this.reps.length;
        if (cue_id === "rep_count") return this.h.say(String(n));
        const k = KB[cue_id]; if (!k) return;
        if (n - this.lastCueRep < 2) return this.h.say(String(n));            // one correction per two reps
        const s = this.stats[cue_id];
        const say = text || (s && s.n >= 4 && s.fixed / s.n < 0.3 ? k.alt : k.cue);   // failing cue -> rephrase
        this.lastCueRep = n; this.last = { id: cue_id, rep: n, policy: this.src }; this.h.say(say);
      },
      retrieve_cues: async ({ query }) => {
        const r = (await this.searchKB(query, 1))[0];
        this.h.say(r ? `${r.why} ${r.alt}` : "I don't have a tip for that yet.", { urgent: true });
      },
      set_summary: () => {
        const r = this.reps, c = {}; r.forEach((e) => e.errors.forEach((id) => (c[id] = (c[id] || 0) + 1)));
        const top = Object.entries(c).sort((a, b) => b[1] - a[1])[0], clean = r.filter((e) => !e.errors.length).length;
        let t = `Set ${this.set} of ${this.sets}: ${r.length} reps, ${clean} clean.`;
        if (top) t += ` Main issue: ${KB[top[0]].alt}`;
        if (!this.adjNote && top && top[1] >= r.length / 2) this.tools.adjust_plan({ rest: 90, target: Math.max(4, this.target - 2) });
        t += this.adjNote; this.adjNote = "";
        this.remember(r.length, clean);
        const finished = this.set >= this.sets;
        this.set = finished ? 1 : this.set + 1; this.reps = []; this.last = null;
        this.h.say(finished ? t + " Workout complete. Nice work." : t + ` Rest ${this.rest} seconds.`, { urgent: true }); this.h.hud();
      },
      adjust_plan: ({ rest, target }) => { this.rest = rest; this.target = target; this.adjNote = ` Next set: ${target} reps, ${rest} seconds rest.`; },
      rest_timer: ({ seconds }) => {
        clearTimeout(this.restT); const s = Math.min(300, Math.max(10, seconds | 0));
        this.restT = setTimeout(() => this.h.say("Rest is over. Ready when you are."), s * 1000);
      },
      start_exercise: ({ name }) => {
        if (!NAMES.includes(name)) return;
        this.ex = name; this.set = 1; this.reps = []; this.last = null; this.h.startExercise(name); this.h.hud();
        this.h.say(`Starting ${name === "pushup" ? "push-ups" : name + "s"}. Get side-on to the camera.`, { urgent: true });
      },
      flag_pain_stop: () => {                       // hard-coded safety path: never goes through the LLM
        clearTimeout(this.restT); this.h.pause();
        this.h.say("Stop and rest now. If the pain is sharp or you feel dizzy, don't continue and get medical advice.", { urgent: true });
      },
    };
  }
  exec(call) {
    if (!call || typeof this.tools[call.tool] !== "function") return;
    this.log.push({ t: Date.now(), src: this.src, call });
    Promise.resolve().then(() => this.tools[call.tool](call.args || {})).catch((e) => console.error(e));
  }
  remember(reps, clean) {                           // per-exercise memory across sessions
    const m = (this.mem[this.ex] ??= { sets: 0, reps: 0, clean: 0 });
    m.sets++; m.reps += reps; m.clean += clean; m.last = new Date().toISOString().slice(0, 10);
    try { localStorage.setItem("trainer.mem", JSON.stringify(this.mem)); } catch {}
  }
  state(ev) {
    const mins = this.reps.map((e) => e.min), k = Math.min(3, Math.floor(mins.length / 2));
    const m = this.mem[this.ex];
    return { exercise: this.ex, set: this.set, sets: this.sets, rep: this.reps.length, target: this.target, rest: this.rest,
      errors: ev.errors.map((id) => ({ id, sev: KB[id].sev })),
      last_cue: this.last && { id: this.last.id, reps_ago: this.reps.length - this.last.rep, fixed: this.last.fixed },
      trend: { depth_loss_deg: k ? Math.round(avg(mins.slice(-k)) - avg(mins.slice(0, k))) : 0 },
      history: m ? { sets: m.sets, clean_rate: +(m.clean / Math.max(1, m.reps)).toFixed(2), last: m.last } : null };
  }
  async searchKB(q, k = 1) {                        // embedding search, keyword fallback
    try { const ids = await Promise.race([this.h.llm.search(q, this.ex, k), timeout(1500)]); if (ids && ids.length) return ids.map((id) => ({ id, ...KB[id] })); } catch {}
    return retrieve(q, this.ex, k);
  }
  async ask(kind, st, allow, said = "") {           // LLM decision: retrieve -> prompt -> parse -> validate, retry once, 4.5 s budget
    if (this.h.policy() !== "llm" || !this.h.llm?.ready()) return null;
    const until = Date.now() + 4500;
    try {
      const docs = await this.searchKB(said || st.errors.map((e) => e.id.replace(/_/g, " ")).join(" ") || this.ex, 2);
      const user = `EVENT ${kind}${said ? ` SAID "${said}"` : ""}\nSTATE ${JSON.stringify(st)}\nKNOWLEDGE ${docs.map((d) => `${d.id}: ${d.cue} ${d.alt}`).join(" | ")}`;
      for (let i = 0; i < 2 && Date.now() < until; i++) {
        const text = await Promise.race([this.h.llm.chat([{ role: "system", content: SYSTEM }, { role: "user", content: user }]), timeout(until - Date.now())]);
        const m = String(text || "").match(/\{[\s\S]*\}/);
        let c = null; try { c = m && JSON.parse(m[0]); } catch {}
        c = validate(c, allow, st.errors.map((e) => e.id)); if (c) return c;
      }
    } catch {}
    return null;
  }
  async onRep(ev) {
    this.reps.push(ev);
    if (this.last && this.last.fixed === undefined && this.reps.length > this.last.rep) {   // did the cue work?
      this.last.fixed = !ev.errors.includes(this.last.id);
      const s = (this.stats[this.last.id] ??= { n: 0, fixed: 0 }); s.n++; s.fixed += this.last.fixed;
      const p = (this.pstats[this.last.policy] ??= { n: 0, fixed: 0 }); p.n++; p.fixed += this.last.fixed;
      try { localStorage.setItem("trainer.cueStats", JSON.stringify(this.stats)); localStorage.setItem("trainer.pstats", JSON.stringify(this.pstats)); } catch {}
      this.h.stats(this.statsText());
    }
    const st = this.state(ev), seq = ++this.seq, setEnd = st.rep >= st.target;
    this.log.push({ t: Date.now(), state: st });
    let calls = null, c = null;
    if (ev.errors.length || setEnd) c = await this.ask("REP", st, setEnd ? ["adjust_plan"] : ["give_cue"]);
    if (c && seq !== this.seq && !setEnd) return;   // a newer rep arrived while the LLM was thinking
    this.src = c ? "llm" : "template";
    calls = c ? (setEnd ? [c, ...this.template.decide(st)] : [c]) : this.template.decide(st);
    for (const x of calls) this.exec(x);
    this.h.hud();
  }
  async onVoice(text) {
    if (PAIN.test(text)) { this.src = "rule"; this.exec({ tool: "flag_pain_stop" }); return true; }
    const st = this.state({ errors: [] });
    let c = await this.ask("VOICE", st, ["retrieve_cues", "adjust_plan", "rest_timer", "start_exercise", "set_summary"], text);
    this.src = c ? "llm" : "rule";
    c ??= this.regexIntent(text);
    this.exec(c);
    if (c && c.tool === "adjust_plan") Promise.resolve().then(() => { this.adjNote = ""; this.h.say(`Okay. ${this.target} reps, ${this.rest} seconds rest.`, { urgent: true }); });
    else if (c && c.tool === "rest_timer") Promise.resolve().then(() => this.h.say(`Resting ${Math.round(c.args.seconds)} seconds.`, { urgent: true }));
    return !!c;
  }
  regexIntent(text) {                               // fallback when the LLM is off, late or invalid. Mirrored by training/eval.py predict_rules (a test keeps them identical)
    const t = text.toLowerCase(), secs = parseSeconds(t), n = repsIn(t);
    const plan = (rest, target) => ({ tool: "adjust_plan", args: { rest: clamp(rest, 10, 300), target: clamp(target, 3, 20) } });
    if (/start|begin|switch|let's do|change|want to do/.test(t) && /squat|push|curl/.test(t)) return { tool: "start_exercise", args: { name: /squat/.test(t) ? "squat" : /push/.test(t) ? "pushup" : "curl" } };
    if (/\b(set|change|make|update)\b.*\brest\b/.test(t) && secs) return plan(secs, this.target);          // "set my rest time to 90 seconds"
    if (/\b(easier|too hard|too difficult)\b/.test(t)) return plan(this.rest + 30, this.target - 2);
    if (/\b(fewer|reduce|decrease|less|lower)\b.*\breps?\b/.test(t)) return plan(this.rest, this.target - 2);
    if (/\b(more|increase|add|extra)\b.*\breps?\b/.test(t)) return plan(this.rest, this.target + 2);
    if (/\b(longer|more|extra|increase|lengthen)\b.*\brest\b/.test(t)) return plan(this.rest + 30, this.target);
    if (/\b(shorter|less|reduce|decrease|lower)\b.*\brest\b/.test(t)) return plan(this.rest - 15, this.target);
    if (n) return plan(this.rest, n);                                                                        // "do 10 reps per set"
    if (/how (was|did)|summary|finish|done|end (the )?set/.test(t)) return { tool: "set_summary" };
    if (/rest|break/.test(t)) return { tool: "rest_timer", args: { seconds: clamp(secs ?? this.rest, 10, 300) } };
    if (/\b(why|how|what|should|tips?|help)\b/.test(t)) return { tool: "retrieve_cues", args: { query: text } };
    return null;
  }
  statsText() {
    const f = (k) => { const p = this.pstats[k]; return p && p.n ? `${Math.round(100 * p.fixed / p.n)}% (${p.n})` : "n/a"; };
    return `Cue fix rate: rules ${f("template")} · LLM ${f("llm")}`;
  }
}
