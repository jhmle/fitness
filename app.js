import { PROFILE, VAD_URL, ORT_URL, LLM_ID, LLM_CHOICES, chosenLLM } from "./config.js";
import { loadPose, detect } from "./pose.js";
import { RepEngine, SKELETON } from "./exercises.js";
import { Agent, kbDocs } from "./agent.js";

const $ = (id) => document.getElementById(id);
const setStatus = (text, kind = "idle") => { const s = $("status"); s.textContent = text; s.className = "pill pill-" + kind; };

const moon = new Worker("moonshine.worker.js", { type: "module" });
const kokoro = PROFILE.tts === "kokoro" ? new Worker("kokoro.worker.js", { type: "module" }) : null;
const llmW = PROFILE.llm ? new Worker("llm.worker.js", { type: "module" }) : null;
const need = 2 + (kokoro ? 1 : 0) + (llmW ? 1 : 0);                       // pose+VAD, Moonshine, (Kokoro)
let llmModel = null, ready = 0, starting = false, S = null, state = "off", vad = null, paused = false, busyTTS = 0, epoch = 0;
let lastMeter = 0;
let micStream = null;                                                       // mic stream we open ourselves and hand to the VAD
const engine = new RepEngine("squat");
const PTS = [...new Set(SKELETON.flat())];

if (!llmW) { $("lLabel").hidden = true; $("policy").parentElement.hidden = true; $("mLabel").hidden = true; $("modelNote").hidden = true; }
else {                                                                       // coach-model picker: stock vs the fine-tuned Hugging Face model
  const cur = chosenLLM(), opts = LLM_CHOICES.some((c) => c.id === cur) ? LLM_CHOICES : [...LLM_CHOICES, { id: cur, label: cur }];
  $("llmModel").innerHTML = opts.map((c) => `<option value="${c.id}">${c.label}</option>`).join("");
  $("llmModel").value = cur;
  $("llmModel").onchange = () => { try { localStorage.setItem("trainer.llm", $("llmModel").value); } catch {} if (ready) $("modelNote").textContent = "Reload the page to switch the coach model."; };
}
if (!kokoro) { $("kLabel").hidden = true; $("voice").parentElement.hidden = true; }

// ---------------------------------------------------------------- states (what the stage shows)
const STATES = {
  off:       ["Not started", "Ready when you are", "Start a session and set the camera so your whole body is in view."],
  listening: ["Tracking", "Go ahead, I'm watching", "Stand side-on to the camera. Say a command any time."],
  hearing:   ["Voice", "Hearing you…", "Pause when you're done."],
  thinking:  ["Coach", "Thinking…", "Working out a reply."],
  speaking:  ["Coach", "Speaking", "Press Skip cue to cut it short."],
};
function setState(s) {
  state = s;
  const [turn, title, hint] = STATES[s];
  $("stage").dataset.state = s;
  $("turnText").textContent = turn; $("title").textContent = title; $("hint").textContent = hint;
  if (s === "off") { if (ready === need) setStatus("Ready", "ready"); } else setStatus("Live · " + turn.toLowerCase(), s === "listening" || s === "hearing" ? "ready" : "busy");
  refresh();
}
function refresh() {
  $("mic").disabled = ready < need || starting;
  $("skip").disabled = !S || !(state === "thinking" || state === "speaking");
}
const hud = () => { $("repN").textContent = agent.reps.length + "/" + agent.target; $("setN").textContent = agent.set; };

// ---------------------------------------------------------------- loading
const fail = (e, who = "") => {                      // always say which component failed
  const msg = String(e.message || e); console.error(who, e);
  $("diag").textContent = who + ": " + msg;
  setStatus(`Error${who ? " (" + who + ")" : ""}: ` + msg.slice(0, 180), "error");
};
const onReady = () => { if (++ready < need) return; $("setup").classList.add("done"); setState("off"); };
function loadScript(src) {
  return new Promise((res, rej) => { const s = document.createElement("script"); s.src = src; s.onload = res; s.onerror = () => rej(new Error("failed to load " + src)); document.head.append(s); });
}
$("load").onclick = async () => {
  $("load").disabled = true; setStatus("Downloading models…", "loading");
  moon.postMessage({ type: "load" });
  if (kokoro) kokoro.postMessage({ type: "load" });
  if (llmW) { $("llmModel").disabled = true; llmW.postMessage({ type: "load", kb: kbDocs(), model: $("llmModel").value }); }
  try {
    const info = await loadPose(); $("pp").value = 60; $("diag").textContent = "pose: " + info;
    try { await loadScript(ORT_URL + "ort.wasm.min.js"); await loadScript(VAD_URL + "bundle.min.js"); }
    catch (e) { return fail(e, "Voice detection"); }
    $("pp").value = 100; onReady();
  } catch (e) { fail(e, "Pose"); }
};
moon.onmessage = async ({ data: d }) => {
  if (d.type === "progress") $("pm").value = d.pct;
  else if (d.type === "ready") { $("pm").value = 100; onReady(); }
  else if (d.type === "text") {
    $("diag").textContent = "heard: " + d.text;
    const handled = d.text && await utter(d.text);
    if (d.text && !handled) $("cueOut").textContent = "Heard: " + d.text;
    hud(); if (!busyTTS && !chatBusy && S) setState("listening");
  }
  else if (d.type === "error") { fail(d, "Moonshine"); if (S) setState("listening"); }
};

// ---------------------------------------------------------------- speech out (Kokoro on desktop, browser voice on mobile)
let actx, nextT = 0, sources = [];
function play(samples, rate) {
  actx ??= new AudioContext();
  const b = actx.createBuffer(1, samples.length, rate); b.copyToChannel(samples, 0);
  const s = actx.createBufferSource(); s.buffer = b; s.connect(actx.destination);
  nextT = Math.max(nextT, actx.currentTime + 0.05); s.start(nextT); nextT += b.duration;
  sources.push(s);
  s.onended = () => { sources = sources.filter((x) => x !== s); done(); };
}
function stopSpeech() {
  epoch++; busyTTS = 0; nextT = 0;
  sources.forEach((s) => { s.onended = null; try { s.stop(); } catch {} }); sources = [];
  speechSynthesis.cancel();
}
function done() {
  busyTTS = Math.max(0, busyTTS - 1);
  if (!busyTTS && S) setTimeout(() => { if (!busyTTS && S) setState("listening"); }, 300);   // gap so the mic doesn't catch our own tail
}
function say(text, { urgent = false } = {}) {
  $("cueOut").textContent = text;
  if (!S || !$("speakOn").checked) return;
  if (busyTTS && !urgent) return;                  // don't talk over ourselves: drop routine cues
  if (urgent) stopSpeech();
  busyTTS++; setState("speaking");
  { const e = epoch; setTimeout(() => { if (e === epoch && busyTTS && S) { busyTTS = 0; setState("listening"); } }, 20000); }   // never stay deaf if audio fails to finish
  if (kokoro) kokoro.postMessage({ type: "speak", text, voice: $("voice").value, id: epoch });
  else { const e = epoch, u = new SpeechSynthesisUtterance(text); u.onend = u.onerror = () => e === epoch && done(); speechSynthesis.speak(u); }
}
kokoro && (kokoro.onmessage = ({ data: d }) => {
  if (d.type === "progress") $("pk").value = d.pct;
  else if (d.type === "ready") {
    $("voice").innerHTML = d.voices.map((v) => `<option>${v}</option>`).join("");
    $("voice").value = d.voices.includes("af_heart") ? "af_heart" : d.voices[0];
    onReady();
  }
  else if (d.type === "audio") { if (d.id === epoch) play(d.samples, d.rate); }
  else if (d.type === "error") { if (d.id === epoch) done(); fail(d, "Kokoro"); }
});
$("skip").onclick = () => { stopSpeech(); if (S) setState("listening"); };

// ---------------------------------------------------------------- coach brain worker (LLM + embeddings)
const pend = new Map(); let rid = 0, llmOK = false, embOK = false, llmCounted = false;
const llmDone = () => { if (!llmCounted) { llmCounted = true; onReady(); } };
const call = (msg) => new Promise((res) => { const id = ++rid; pend.set(id, res); llmW.postMessage({ ...msg, id }); });
const llm = llmW && {
  ready: () => llmOK,
  chat: async (messages) => (await call({ type: "chat", messages })).text,
  search: async (query, ex, k) => (embOK ? (await call({ type: "search", query, ex, k })).ids : []),
};
if (llmW) llmW.onmessage = ({ data: d }) => {
  if (d.type === "progress") $("pl").value = d.pct;
  else if (d.type === "warn") { console.warn(d.message); $("diag").textContent = d.message; }
  else if (d.type === "ready") {
    llmOK = d.llm; embOK = d.emb; llmModel = d.model; $("pl").value = 100;
    $("modelNote").textContent = d.llm ? "Coach brain: " + d.model + (d.model !== $("llmModel").value ? " (fallback: the selected model failed to load)" : "") : "Coach brain: NOT loaded, rules only. " + (d.why || "");
    llmDone();
  }
  else if (d.type === "result") { pend.get(d.id)?.(d); pend.delete(d.id); }
  else if (d.type === "error") { fail(d, "Coach brain"); llmDone(); }
};

// ---------------------------------------------------------------- general chat fallback
// The agent only knows workout commands + a small tips knowledge base. When it has no tip, ask the LLM instead.
let lastHeard = "", chatBusy = false;
const chatHist = [];
const CHAT_SYS = "You are a friendly voice assistant and fitness coach. Answer the user's question directly in one to three short spoken sentences. " +
  "Plain words only: no markdown, lists, emojis or stage directions. If you are not sure, say so briefly.";
async function chatAnswer(q) {
  if (!llm || !llm.ready()) { say("My language model didn't load, so I can only handle workout commands."); return; }
  chatBusy = true; setState("thinking");
  try {
    let t = ((await llm.chat([{ role: "system", content: CHAT_SYS }, ...chatHist.slice(-6), { role: "user", content: q }])) || "").trim();
    if (t && !/[.!?]$/.test(t)) { const i = Math.max(t.lastIndexOf("."), t.lastIndexOf("!"), t.lastIndexOf("?")); if (i > 20) t = t.slice(0, i + 1); }   // don't end mid-sentence
    if (!t) t = "Sorry, I couldn't come up with an answer to that.";
    chatHist.push({ role: "user", content: q }, { role: "assistant", content: t });
    chatBusy = false; say(t);
  } catch (e) { chatBusy = false; say("Sorry, something went wrong answering that."); }
  if (!busyTTS && S) setState("listening");
}
// Only workout controls go to the rule-based agent; everything else is a free-form question for the LLM.
const CMD = /\b(start|begin|switch to|change to|stop|pause|resume|rest|break|reps?|sets?|easier|harder|longer|shorter|next set|how was|how did|how many|count)\b/i;
function utter(t) { lastHeard = t; if (CMD.test(t) || !llm?.ready()) return agent.onVoice(t); chatAnswer(t); return true; }
const route = (text, o) => (lastHeard && llm?.ready() && /don'?t have a tip/i.test(text) ? chatAnswer(lastHeard) : say(text, o));

// ---------------------------------------------------------------- agent
const agent = new Agent({
  say: route, llm, hud: () => hud(), policy: () => $("policy").value, stats: (t) => ($("stats").textContent = t),
  startExercise(name) { engine.set(name); $("ex").value = name; paused = false; },
  pause() { paused = true; },
});
$("ex").onchange = () => agent.exec({ tool: "start_exercise", args: { name: $("ex").value } });
$("txt").onclick = () => { const t = $("prompt").value.trim(); if (t) { utter(t); hud(); $("prompt").value = ""; } };

// ---------------------------------------------------------------- pose loop: camera -> landmarks -> reps -> agent
function draw(g, lm, w, h) {
  const hot = S.flash && S.flash.until > performance.now() ? S.flash.j : new Set();
  g.lineWidth = 4;
  for (const [a, b] of SKELETON) {
    const A = lm[a], B = lm[b]; if ((A.visibility ?? 1) < 0.4 || (B.visibility ?? 1) < 0.4) continue;
    g.strokeStyle = hot.has(a) || hot.has(b) ? "#ef4444" : "#22c55e";
    g.beginPath(); g.moveTo(A.x * w, A.y * h); g.lineTo(B.x * w, B.y * h); g.stroke();
  }
  for (const i of PTS) {
    const P = lm[i]; if ((P.visibility ?? 1) < 0.4) continue;
    g.fillStyle = hot.has(i) ? "#ef4444" : "#fff"; g.beginPath(); g.arc(P.x * w, P.y * h, 5, 0, 7); g.fill();
  }
}
function loop() {
  if (!S) return;
  S.raf = requestAnimationFrame(loop);
  const v = $("cam"); if (v.readyState < 2 || v.currentTime === S.t) return; S.t = v.currentTime;
  const c = $("ov"); if (c.width !== v.videoWidth) { c.width = v.videoWidth; c.height = v.videoHeight; }
  const g = c.getContext("2d"); g.clearRect(0, 0, c.width, c.height);
  const r = detect(v, performance.now());
  if (!r) return warn("I can't see you. Step back so your whole body is in frame.");
  draw(g, r.img, c.width, c.height);
  if (paused) return;
  const res = engine.update(r.world, performance.now());
  if (!res.ok) return warn("Turn side-on so I can see your hips, knees and ankles.");
  if (S.warn) { S.warn = false; setState(state); }
  if (res.event) {
    const ev = res.event;
    if (ev.errors.length) S.flash = { j: new Set(ev.joints), until: performance.now() + 1500 };
    $("diag").textContent = JSON.stringify(ev);
    agent.onRep(ev); hud();
  }
}
function warn(t) { S.warn = true; $("hint").textContent = t; }

// ---------------------------------------------------------------- microphone
// Open the mic ourselves (with fallbacks) so a picky default doesn't kill voice commands.
async function openMic() {
  let err;
  for (const audio of [{ channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: true }, true]) {
    try { return await navigator.mediaDevices.getUserMedia({ audio }); } catch (e) { err = e; }
  }
  const devs = (await navigator.mediaDevices.enumerateDevices().catch(() => [])).filter((d) => d.kind === "audioinput");
  for (const d of devs) {
    if (!d.deviceId) continue;
    try { return await navigator.mediaDevices.getUserMedia({ audio: { deviceId: { exact: d.deviceId } } }); } catch (e) { err = e; }
  }
  throw new Error(`${err?.name}: ${err?.message} (audio inputs found: ${devs.length})`);
}

// ---------------------------------------------------------------- session
async function begin() {
  if (S || starting || ready < need) return;
  starting = true; refresh(); stopSpeech();
  let stream;
  try { stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "user", width: { ideal: 640 }, height: { ideal: 480 } }, audio: false }); }
  catch (e) { starting = false; refresh(); return setStatus("Camera access failed: " + e.message, "error"); }
  const v = $("cam"); v.srcObject = stream; try { await v.play(); } catch {}
  $("stage").classList.add("cam");
  S = { stream, t: -1, flash: null, warn: false, raf: 0 };
  engine.set($("ex").value); agent.ex = $("ex").value; paused = false;
  let voiceErr = "";
  if ($("voiceOn").checked) {
    try {
      micStream = await openMic();
      vad = await window.vad.MicVAD.new({                       // Silero VAD: only wake Moonshine on real speech
        baseAssetPath: VAD_URL, onnxWASMBasePath: ORT_URL,
        positiveSpeechThreshold: 0.4, negativeSpeechThreshold: 0.25,   // easier to trigger on quiet or distant speech
        redemptionMs: 600,                                       // end your turn after 0.6 s of quiet (default is longer)
        minSpeechMs: 150, preSpeechPadMs: 300,                   // catch short commands, keep the first syllable
        onFrameProcessed: (p) => {                               // live meter so you can SEE the mic is being heard
          const now = performance.now();
          if (S && !S.warn && state === "listening" && now - lastMeter > 120) {
            lastMeter = now; const n = Math.round(p.isSpeech * 10);
            $("hint").textContent = "Mic live  " + "▮".repeat(n) + "▯".repeat(10 - n);
          }
        },
        getStream: async () => micStream,                        // use our stream instead of the VAD's own getUserMedia
        pauseStream: async () => {},
        resumeStream: async () => micStream,
        onSpeechStart: () => { if (S && !busyTTS) setState("hearing"); },
        onVADMisfire: () => { if (S && !busyTTS) setState("listening"); },
        onSpeechEnd: (audio) => {                                // Float32Array @ 16 kHz
          if (!S) return;
          if (busyTTS) return;                                   // ignore our own voice
          setState("thinking"); moon.postMessage({ type: "transcribe", audio }, [audio.buffer]);
        },
      });
      vad.start();
    } catch (e) { voiceErr = "Voice commands unavailable: " + (e.message || e); }
  }
  starting = false;
  $("mic").classList.add("on"); $("micLabel").textContent = "End session";
  setState("listening"); hud();
  if (voiceErr) setStatus(voiceErr, "error");
  loop();
}
function end() {
  const s = S; if (!s) return; S = null;
  cancelAnimationFrame(s.raf); s.stream.getTracks().forEach((t) => t.stop());
  try { vad?.destroy(); } catch {} vad = null;
  micStream?.getTracks().forEach((t) => t.stop()); micStream = null;
  stopSpeech(); clearTimeout(agent.restT);
  $("cam").srcObject = null; $("stage").classList.remove("cam"); $("ov").getContext("2d").clearRect(0, 0, 9999, 9999);
  $("mic").classList.remove("on"); $("micLabel").textContent = "Start session";
  setState("off");
}
$("mic").onclick = () => (S ? end() : begin());
setState("off");

$("dl").onclick = () => {                           // session trace = future training / eval data
  const a = document.createElement("a");
  const meta = { meta: { model: llmModel, selected: llmW ? $("llmModel").value : null, default_model: LLM_ID, profile: PROFILE.name, saved: new Date().toISOString() } };   // entries without "state" are ignored by gen_data.py --logs
  a.href = URL.createObjectURL(new Blob([JSON.stringify([meta, ...agent.log], null, 1)], { type: "application/json" }));
  a.download = "trainer-session.json"; a.click(); URL.revokeObjectURL(a.href);
};
